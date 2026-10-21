"""Inference wrapper for the trained (mask-and-reconstruct) CRCDNet.

    fused = fuse_trained(s1_hwc, s2_hwc)          # (H, W, 4) float32, reflectance scale

Loads weights AND the frozen normalization statistics from the checkpoint, so
callers cannot forget them. Same shape/dtype contract as fusion.baseline.fuse
(HWC in, HWC out, finite), and the same output radiometry on every date -- the
normalization constants are dataset-level, not per-image, so two dates of the
same ground do not drift apart and fake change in compare().

torch is imported lazily: importing this module never requires torch; only
loading a checkpoint does. The dashboard/backend keep working without it.

Nodata / cloud-masked pixels (-9999 or non-finite) are passed to the network
as "missing optical", so the model fills them in from SAR + context instead of
having them zeroed out.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

NODATA = -9999.0
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "crcdnet_maskrec.pt")

_CACHE: Dict[str, "TrainedFuser"] = {}


class TrainedFuser:
    def __init__(self, path: str = DEFAULT_WEIGHTS, device: str = "cpu"):
        import torch

        from fusion.train_fusion import build_model

        ck = torch.load(path, map_location=device)
        self.norm = ck["norm"]
        self.s2_scale = ck["config"]["s2_scale"]
        self.meta = ck.get("meta", {})
        self.device = device
        self.model = build_model(ck["config"]["feature_dim"]).to(device)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()

    def fuse(self, s1_hwc: np.ndarray, s2_hwc: np.ndarray) -> np.ndarray:
        import torch

        if s1_hwc.ndim != 3 or s2_hwc.ndim != 3 or s1_hwc.shape[:2] != s2_hwc.shape[:2]:
            raise ValueError(f"expected aligned (H,W,C) arrays, got {s1_hwc.shape} and {s2_hwc.shape}")
        valid = (
            (s1_hwc != NODATA).all(-1) & (s2_hwc != NODATA).all(-1)
            & np.isfinite(s1_hwc).all(-1) & np.isfinite(s2_hwc).all(-1)
        )
        n = self.norm
        s1n = (s1_hwc - np.array(n["s1_mean"], np.float32)) / np.array(n["s1_std"], np.float32)
        s2n = (s2_hwc - np.array(n["s2_mean"], np.float32)) / np.array(n["s2_std"], np.float32)
        s1n = np.where(valid[..., None], s1n, 0.0).astype(np.float32)
        s2n = np.where(valid[..., None], s2n, 0.0).astype(np.float32)
        from fusion.train_fusion import context_prior

        x_opt = np.concatenate(
            [s2n, valid[..., None].astype(np.float32), context_prior(np.where(valid[..., None], s2_hwc, 0.0), valid, n)],
            axis=-1,
        )
        chw = lambda a: torch.from_numpy(np.ascontiguousarray(np.transpose(a, (2, 0, 1))))[None].to(self.device)
        with torch.no_grad():
            out = self.model(chw(s1n), chw(x_opt))[0].cpu().numpy()
        return (np.transpose(out, (1, 2, 0)) * self.s2_scale).astype(np.float32)


def fuse_trained(s1_hwc: np.ndarray, s2_hwc: np.ndarray, weights: Optional[str] = None, device: str = "cpu") -> np.ndarray:
    """Drop-in analogue of fuse(s1, s2, data_layout='HWC'). Checkpoint is loaded once and cached."""
    path = weights or DEFAULT_WEIGHTS
    key = f"{path}|{device}"
    if key not in _CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No trained fusion weights at {path}. Train with: python -m fusion.train_fusion")
        _CACHE[key] = TrainedFuser(path, device)
    return _CACHE[key].fuse(s1_hwc, s2_hwc)
