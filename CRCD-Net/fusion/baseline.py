"""
Baseline Fusion Module for SAR (Sentinel-1) and Optical (Sentinel-2) Imagery.

Provides weighted and PCA-based image fusion without requiring deep model training.
Robust against differing channel counts, NaN/Inf values, and shape variations.
"""

from typing import Optional, Tuple, Union
import numpy as np


def validate_and_format_inputs(
    s1_image: np.ndarray,
    s2_image: np.ndarray,
    data_layout: str = "CHW",
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Validates S1 and S2 arrays, checks spatial dimension compatibility, cleans invalid numbers

    (NaN/Inf), and standardizes format to (C, H, W).

    Parameters
    ----------
    s1_image : np.ndarray
        Sentinel-1 SAR image array.
    s2_image : np.ndarray
        Sentinel-2 Optical image array.
    data_layout : str
        Input layout hint: 'CHW' (default) or 'HWC'.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, bool]
        (s1_formatted, s2_formatted, was_hwc)
        where formatted arrays are shape (C, H, W), float32, with NaNs/Infs sanitized.
    """
    if not isinstance(s1_image, np.ndarray) or not isinstance(s2_image, np.ndarray):
        raise TypeError(
            f"Expected numpy ndarrays, got {type(s1_image)} and {type(s2_image)}"
        )

    if s1_image.ndim != 3 or s2_image.ndim != 3:
        raise ValueError(
            f"Expected 3D arrays (C, H, W) or (H, W, C). Got S1 shape {s1_image.shape} "
            f"and S2 shape {s2_image.shape}."
        )

    was_hwc = False
    # Determine layout based on parameters or shape heuristics
    layout = data_layout.upper()

    if layout == "HWC":
        was_hwc = True
        s1 = np.transpose(s1_image, (2, 0, 1))
        s2 = np.transpose(s2_image, (2, 0, 1))
    elif layout == "CHW":
        s1 = s1_image.copy()
        s2 = s2_image.copy()
    else:
        raise ValueError(f"Unsupported data_layout: '{data_layout}'. Must be 'CHW' or 'HWC'.")

    # Spatial dimensions check
    c1, h1, w1 = s1.shape
    c2, h2, w2 = s2.shape

    if (h1, w1) != (h2, w2):
        raise ValueError(
            f"Spatial dimensions mismatch! S1 spatial dimensions (H={h1}, W={w1}) "
            f"do not match S2 spatial dimensions (H={h2}, W={w2})."
        )

    # Clean NaNs and Infs safely
    s1 = _sanitize_array(s1, "S1")
    s2 = _sanitize_array(s2, "S2")

    return s1.astype(np.float32), s2.astype(np.float32), was_hwc


def _sanitize_array(arr: np.ndarray, name: str) -> np.ndarray:
    """Replaces NaNs with zero and clips infinite values to min/max finite values."""
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)

    if np.isinf(arr).any():
        finite_vals = arr[np.isfinite(arr)]
        min_val = float(np.min(finite_vals)) if len(finite_vals) > 0 else 0.0
        max_val = float(np.max(finite_vals)) if len(finite_vals) > 0 else 1.0
        arr = np.clip(arr, min_val, max_val)

    return arr


def _weighted_fusion(
    s1: np.ndarray,
    s2: np.ndarray,
    weights: Tuple[float, float] = (0.5, 0.5),
    target_channels: Optional[int] = None,
) -> np.ndarray:
    """Performs weighted fusion of S1 (SAR) and S2 (Optical) arrays.

    Handles differing channel counts (C1 != C2) by projecting/mapping channels to target_channels.
    """
    c1, h, w = s1.shape
    c2, _, _ = s2.shape

    w1, w2 = weights
    total_w = w1 + w2
    if total_w == 0:
        w1, w2 = 0.5, 0.5
    else:
        w1, w2 = w1 / total_w, w2 / total_w

    if target_channels is None:
        target_channels = c2  # Default to Optical channel count

    # Project S1 to target_channels
    if c1 == target_channels:
        s1_proj = s1
    elif c1 < target_channels:
        # Repeat or tile channels to match target
        repeats = (target_channels + c1 - 1) // c1
        s1_proj = np.tile(s1, (repeats, 1, 1))[:target_channels]
    else:
        # Average down extra channels
        s1_proj = np.array_split(s1, target_channels, axis=0)
        s1_proj = np.stack([chunk.mean(axis=0) for chunk in s1_proj], axis=0)

    # Project S2 to target_channels
    if c2 == target_channels:
        s2_proj = s2
    elif c2 < target_channels:
        repeats = (target_channels + c2 - 1) // c2
        s2_proj = np.tile(s2, (repeats, 1, 1))[:target_channels]
    else:
        s2_proj = np.array_split(s2, target_channels, axis=0)
        s2_proj = np.stack([chunk.mean(axis=0) for chunk in s2_proj], axis=0)

    fused = w1 * s1_proj + w2 * s2_proj
    return fused


def _pca_fusion(
    s1: np.ndarray,
    s2: np.ndarray,
    target_channels: Optional[int] = None,
) -> np.ndarray:
    """Performs PCA-based spectral-spatial fusion across combined SAR and Optical bands."""
    c1, h, w = s1.shape
    c2, _, _ = s2.shape

    if target_channels is None:
        target_channels = c2

    # Stack all input bands along channel dimension: shape (C1+C2, H*W)
    combined = np.concatenate([s1, s2], axis=0)  # shape (C1+C2, H, W)
    num_total_channels = c1 + c2
    flat_data = combined.reshape(num_total_channels, -1)  # shape (num_total_channels, N)

    # Mean center
    mean_vec = flat_data.mean(axis=1, keepdims=True)
    centered_data = flat_data - mean_vec

    # Covariance matrix (num_total_channels x num_total_channels)
    cov_matrix = np.cov(centered_data)

    # Eigen decomposition
    if cov_matrix.ndim == 0:
        eigenvalues, eigenvectors = np.array([1.0]), np.array([[1.0]])
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvectors by eigenvalue descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Select top principal component projection weights
    n_components = min(target_channels, num_total_channels)
    top_eigenvectors = eigenvectors[:, :n_components]  # shape (num_total_channels, n_components)

    # Project centered data onto principal components
    pca_features = np.dot(top_eigenvectors.T, centered_data)  # shape (n_components, H*W)

    # If target_channels > n_components, pad with repeated features
    if target_channels > n_components:
        repeats = (target_channels + n_components - 1) // n_components
        pca_features = np.tile(pca_features, (repeats, 1))[:target_channels]

    fused = pca_features.reshape(target_channels, h, w)

    # Normalize output per channel to [0, 1] range for stability
    fused_norm = np.zeros_like(fused)
    for c in range(target_channels):
        c_min, c_max = fused[c].min(), fused[c].max()
        if c_max > c_min:
            fused_norm[c] = (fused[c] - c_min) / (c_max - c_min)
        else:
            fused_norm[c] = fused[c]

    return fused_norm


def fuse(
    s1_image: np.ndarray,
    s2_image: np.ndarray,
    method: str = "weighted",
    weights: Tuple[float, float] = (0.5, 0.5),
    target_channels: Optional[int] = None,
    data_layout: str = "CHW",
) -> np.ndarray:
    """Public interface for baseline SAR-Optical image fusion.

    Parameters
    ----------
    s1_image : np.ndarray
        Sentinel-1 SAR image array (C1, H, W) or (H, W, C1).
    s2_image : np.ndarray
        Sentinel-2 Optical image array (C2, H, W) or (H, W, C2).
    method : str
        Fusion method: 'weighted' (default) or 'pca'.
    weights : Tuple[float, float]
        (weight_s1, weight_s2) for weighted fusion. Defaults to (0.5, 0.5).
    target_channels : int, optional
        Target channel count of output array. Defaults to Optical channel count.
    data_layout : str
        Input data layout hint: 'CHW' (default) or 'HWC'. Output will match this layout.

    Returns
    -------
    np.ndarray
        Fused image array matching the specified data_layout.
    """
    s1_formatted, s2_formatted, was_hwc = validate_and_format_inputs(
        s1_image, s2_image, data_layout=data_layout
    )

    method_clean = method.lower().strip()

    if method_clean == "weighted":
        fused_chw = _weighted_fusion(
            s1_formatted, s2_formatted, weights=weights, target_channels=target_channels
        )
    elif method_clean == "pca":
        fused_chw = _pca_fusion(
            s1_formatted, s2_formatted, target_channels=target_channels
        )
    else:
        raise ValueError(
            f"Unknown fusion method '{method}'. Supported methods are 'weighted' and 'pca'."
        )

    # Ensure output layout matches input convention
    if was_hwc or data_layout.upper() == "HWC":
        fused_out = np.transpose(fused_chw, (1, 2, 0))
    else:
        fused_out = fused_chw

    return fused_out.astype(np.float32)
