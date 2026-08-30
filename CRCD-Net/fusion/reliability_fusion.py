"""
Reliability-Aware SAR-Optical Fusion & Modality Attribution Module.

Computes adaptive, cross-modal fused representations along with explicit
per-pixel modality weight maps:
    W_SAR(x, y) + W_OPT(x, y) = 1.0

Integrates with deep spatial-channel gating attention (CRCDNet) while maintaining
physics-grounded backscatter-reflectance coherence.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np

from fusion.baseline import _sanitize_array, _standardize_per_channel, validate_and_format_inputs

if TYPE_CHECKING:
    # Only needed for the type hint below; the actual runtime import is
    # deferred into the `if model is not None:` branch in
    # fuse_reliability_aware(), since the default (model=None) path --
    # what every caller in this codebase actually uses -- is pure
    # numpy/scipy and must not require torch to be installed.
    from fusion.deep_model import CRCDNet


def compute_reliability_maps(
    s1_formatted: np.ndarray,  # (C1, H, W)
    s2_formatted: np.ndarray,  # (C2, H, W)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes spatial-spectral reliability maps for SAR and Optical modalities.
    
    SAR reliability is highest in high-roughness/structural regions (built-up, forest canopy)
    and robust against cloud attenuation.
    Optical reliability is highest in clear-sky spectral variation (NDVI, visible gradients).
    
    Returns:
    --------
    w_sar : np.ndarray of shape (H, W), values in [0, 1]
    w_opt : np.ndarray of shape (H, W), values in [0, 1]
    where w_sar + w_opt = 1.0
    """
    c1, h, w = s1_formatted.shape
    c2, _, _ = s2_formatted.shape

    # 1. SAR local structural signal (gradient energy & backscatter variance)
    # VV is typically channel 0, VH channel 1
    sar_vv = s1_formatted[0]
    sar_vh = s1_formatted[1] if c1 > 1 else s1_formatted[0]
    
    # Local variance filter (5x5 neighborhood via uniform filter)
    def _local_std(img, size=5):
        from scipy.ndimage import uniform_filter
        mean = uniform_filter(img, size=size)
        sq_mean = uniform_filter(img ** 2, size=size)
        var = np.maximum(sq_mean - mean ** 2, 0.0)
        return np.sqrt(var)

    sar_detail = _local_std(sar_vv, 5) + _local_std(sar_vh, 5)
    sar_detail_norm = (sar_detail - sar_detail.min()) / (sar_detail.max() - sar_detail.min() + 1e-6)

    # 2. Optical local spectral signal (visible/NIR color variance)
    opt_std = np.mean([_local_std(s2_formatted[c], 5) for c in range(c2)], axis=0)
    opt_std_norm = (opt_std - opt_std.min()) / (opt_std.max() - opt_std.min() + 1e-6)

    # Softmax / Sigmoid combination for smooth bounded weighting
    sar_score = np.exp(sar_detail_norm * 2.0)
    opt_score = np.exp(opt_std_norm * 2.0)

    w_sar = sar_score / (sar_score + opt_score + 1e-8)
    w_opt = 1.0 - w_sar

    return w_sar.astype(np.float32), w_opt.astype(np.float32)


def fuse_reliability_aware(
    s1_image: np.ndarray,
    s2_image: np.ndarray,
    model: Optional[CRCDNet] = None,
    device: str = "cpu",
    data_layout: str = "HWC",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Public interface for Reliability-Aware SAR-Optical Fusion.

    Parameters:
    -----------
    s1_image : np.ndarray
        Sentinel-1 SAR array (H, W, 2) or (2, H, W).
    s2_image : np.ndarray
        Sentinel-2 Optical array (H, W, 4) or (4, H, W).
    model : CRCDNet, optional
        Pretrained/instantiated CRCDNet deep attention model.
    device : str
        Compute device ('cpu' or 'cuda').
    data_layout : str
        Data layout ('HWC' or 'CHW').

    Returns:
    --------
    fused : np.ndarray
        Fused multi-modal representation matching data_layout.
    w_sar : np.ndarray
        SAR reliability weight map (H, W).
    w_opt : np.ndarray
        Optical reliability weight map (H, W).
    metrics : Dict[str, float]
        Mean SAR weight, mean Optical weight, entropy of weights.
    """
    s1_formatted, s2_formatted, was_hwc = validate_and_format_inputs(
        s1_image, s2_image, data_layout=data_layout
    )
    c1, h, w = s1_formatted.shape
    c2, _, _ = s2_formatted.shape

    # Standardize inputs
    s1_std = _standardize_per_channel(s1_formatted)
    s2_std = _standardize_per_channel(s2_formatted)

    if model is not None:
        import torch  # deferred: only the optional deep-model path needs this

        model = model.to(device)
        model.eval()
        t_sar = torch.from_numpy(s1_formatted).unsqueeze(0).to(device)
        t_opt = torch.from_numpy(s2_formatted).unsqueeze(0).to(device)

        with torch.no_grad():
            # Extract features and gating attention
            f_sar = model.sar_encoder(t_sar)
            f_opt = model.opt_encoder(t_opt)
            
            # Spatial and channel attention gates
            concat_feat = torch.cat([f_sar, f_opt], dim=1)
            spatial_weights = model.gating_fusion.spatial_gate(concat_feat)  # (1, 2, H, W)
            w_sar_s = spatial_weights[0, 0].cpu().numpy()
            w_opt_s = spatial_weights[0, 1].cpu().numpy()
            
            # Normalization to sum to 1.0
            sum_w = w_sar_s + w_opt_s + 1e-8
            w_sar = (w_sar_s / sum_w).astype(np.float32)
            w_opt = (1.0 - w_sar).astype(np.float32)

            f_fused = model.gating_fusion(f_sar, f_opt)
            out_tensor = model.decoder(f_fused)
            fused_chw = out_tensor.squeeze(0).cpu().numpy().astype(np.float32)
    else:
        # Physics-grounded spatial-spectral reliability weighting
        w_sar, w_opt = compute_reliability_maps(s1_formatted, s2_formatted)
        
        # Project S1 to C2 channels
        repeats = (c2 + c1 - 1) // c1
        s1_proj = np.tile(s1_std, (repeats, 1, 1))[:c2]
        s2_proj = s2_std

        # Modality weighted combination
        fused_raw = w_sar[np.newaxis, ...] * s1_proj + w_opt[np.newaxis, ...] * s2_proj

        # Min-max normalization per channel into [0, 1]
        fused_chw = np.zeros_like(fused_raw, dtype=np.float32)
        for c in range(c2):
            c_min, c_max = fused_raw[c].min(), fused_raw[c].max()
            if c_max > c_min:
                fused_chw[c] = (fused_raw[c] - c_min) / (c_max - c_min)
            else:
                fused_chw[c] = fused_raw[c]

    # Metrics
    mean_w_sar = float(np.mean(w_sar))
    mean_w_opt = float(np.mean(w_opt))
    # Entropy: measure of cross-sensor information balance
    eps = 1e-8
    entropy = float(-np.mean(w_sar * np.log2(w_sar + eps) + w_opt * np.log2(w_opt + eps)))

    metrics = {
        "mean_sar_weight": round(mean_w_sar, 4),
        "mean_optical_weight": round(mean_w_opt, 4),
        "modality_balance_entropy": round(entropy, 4),
    }

    if was_hwc or data_layout.upper() == "HWC":
        fused_out = np.transpose(fused_chw, (1, 2, 0))
    else:
        fused_out = fused_chw

    return fused_out.astype(np.float32), w_sar, w_opt, metrics
