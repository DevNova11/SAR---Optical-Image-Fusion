"""Fusion/reconstruction metrics (numpy only; no torch needed).

All functions take (H, W, C) arrays plus an optional boolean (H, W) `mask`
(True = evaluate this pixel), so scores can be restricted to e.g. the hidden
region of the mask-and-reconstruct task and never include nodata pixels.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from skimage.metrics import structural_similarity


def _select(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray]):
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if mask is None:
        mask = np.ones(a.shape[:2], dtype=bool)
    if mask.sum() == 0:
        raise ValueError("empty evaluation mask")
    return a[mask].astype(np.float64), b[mask].astype(np.float64)  # (N, C)


def rmse(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    p, t = _select(pred, target, mask)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def psnr(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None, data_range: float = 1.0) -> float:
    p, t = _select(pred, target, mask)
    mse = np.mean((p - t) ** 2)
    return float("inf") if mse == 0 else float(10.0 * np.log10(data_range**2 / mse))


def sam(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Mean spectral angle in degrees between per-pixel band vectors (lower is better)."""
    p, t = _select(pred, target, mask)
    num = np.sum(p * t, axis=1)
    den = np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1) + 1e-12
    return float(np.degrees(np.mean(np.arccos(np.clip(num / den, -1.0, 1.0)))))


def ssim(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None, data_range: float = 1.0) -> float:
    """Mean SSIM over the masked pixels (SSIM map computed on the full image, then averaged under the mask)."""
    if mask is None:
        mask = np.ones(pred.shape[:2], dtype=bool)
    _, smap = structural_similarity(
        target.astype(np.float64), pred.astype(np.float64),
        data_range=data_range, channel_axis=-1, full=True,
    )
    return float(smap.mean(axis=-1)[mask].mean())


def all_metrics(pred, target, mask=None, data_range: float = 1.0) -> Dict[str, float]:
    return {
        "psnr": psnr(pred, target, mask, data_range),
        "ssim": ssim(pred, target, mask, data_range),
        "rmse": rmse(pred, target, mask),
        "sam_deg": sam(pred, target, mask),
    }
