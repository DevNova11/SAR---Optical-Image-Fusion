"""Local preprocessing: load validated GeoTIFF exports, stack, and patch them
into training tensors. See DATA_CONTRACT.md for the exact shapes/dtypes this
guarantees.

Also re-exports `suggest_best_dates` from gee_data_collection so callers only
need to import this one module for the "pick dates -> load arrays" workflow.
"""
from __future__ import annotations

import numpy as np
import rasterio

from gee_data_collection import suggest_best_dates  # noqa: F401 (re-exported)
from validate_export import NODATA, validate_geotiff_pair

DEFAULT_PATCH_SIZE = 256
DEFAULT_STRIDE = 256
DEFAULT_MAX_PATCH_NODATA_FRAC = 0.3


def load_geotiff(path: str) -> np.ndarray:
    """Reads a GeoTIFF into a channels-last (H, W, C) float32 array."""
    with rasterio.open(path) as src:
        arr = src.read().astype("float32")  # (C, H, W)
    return np.moveaxis(arr, 0, -1)  # -> (H, W, C)


def build_date_stack(s1_path: str, s2_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Validates an S1/S2 GeoTIFF pair, then returns (s1_image, s2_image) per the data contract."""
    validate_geotiff_pair(s1_path, s2_path)
    return load_geotiff(s1_path), load_geotiff(s2_path)


def stack_6channel(s1_image: np.ndarray, s2_image: np.ndarray) -> np.ndarray:
    """[VV, VH, B2, B3, B4, B8] concatenation. s1_image/s2_image must share (H, W)."""
    if s1_image.shape[:2] != s2_image.shape[:2]:
        raise ValueError(f"Spatial shape mismatch: s1={s1_image.shape[:2]} vs s2={s2_image.shape[:2]}")
    return np.concatenate([s1_image, s2_image], axis=-1)


def extract_patches(
    image: np.ndarray,
    patch_size: int = DEFAULT_PATCH_SIZE,
    stride: int = DEFAULT_STRIDE,
    max_nodata_frac: float = DEFAULT_MAX_PATCH_NODATA_FRAC,
) -> list[np.ndarray]:
    """Tiles a (H, W, C) array into (patch_size, patch_size, C) patches, dropping mostly-nodata ones."""
    h, w = image.shape[:2]
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y : y + patch_size, x : x + patch_size]
            nodata_frac = float(np.mean(patch == NODATA))
            if nodata_frac <= max_nodata_frac:
                patches.append(patch)
    return patches
