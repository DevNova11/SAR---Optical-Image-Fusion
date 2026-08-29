"""Post-processing utilities for binary change maps."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology


def morphological_opening(binary_map: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply morphology opening to remove isolated noisy pixels."""
    binary_map = np.asarray(binary_map, dtype=bool)
    if kernel_size <= 1:
        return binary_map.astype(np.uint8)
    footprint = morphology.disk(kernel_size)
    processed = morphology.opening(binary_map, footprint=footprint)
    return processed.astype(np.uint8)


def morphological_closing(binary_map: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply morphology closing to fill small gaps in detected change regions."""
    binary_map = np.asarray(binary_map, dtype=bool)
    if kernel_size <= 1:
        return binary_map.astype(np.uint8)
    footprint = morphology.disk(kernel_size)
    processed = morphology.closing(binary_map, footprint=footprint)
    return processed.astype(np.uint8)


def remove_small_objects(binary_map: np.ndarray, min_size: int = 20, connectivity: int = 2) -> np.ndarray:
    """Remove small connected components below a given minimum size."""
    binary_map = np.asarray(binary_map, dtype=bool)
    if min_size <= 0:
        return binary_map.astype(np.uint8)
    processed = morphology.remove_small_objects(binary_map, max_size=min_size, connectivity=connectivity)
    return processed.astype(np.uint8)


def remove_small_holes(binary_map: np.ndarray, min_size: int = 20, connectivity: int = 2) -> np.ndarray:
    """Remove small holes within detected regions."""
    binary_map = np.asarray(binary_map, dtype=bool)
    if min_size <= 0:
        return binary_map.astype(np.uint8)
    processed = morphology.remove_small_holes(binary_map, max_size=min_size, connectivity=connectivity)
    return processed.astype(np.uint8)


def connected_components(binary_map: np.ndarray, connectivity: int = 2) -> tuple[np.ndarray, int, int]:
    """Return labels, number of regions, and largest region size."""
    binary_map = np.asarray(binary_map, dtype=bool)
    if not np.any(binary_map):
        return np.zeros_like(binary_map, dtype=np.int32), 0, 0

    labels = measure.label(binary_map, connectivity=connectivity)
    region_sizes = np.bincount(labels.ravel())[1:]
    num_regions = int(len(region_sizes))
    largest_region = int(region_sizes.max()) if region_sizes.size > 0 else 0
    return labels.astype(np.int32), num_regions, largest_region


def postprocess_change_map(binary_map: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Apply a configurable post-processing pipeline to the binary change map."""
    binary_map = np.asarray(binary_map, dtype=np.uint8)
    cfg = config or {}

    kernel_size = int(cfg.get("morphology_kernel_size", 3))
    min_region_size = int(cfg.get("min_region_size", 20))
    remove_holes = bool(cfg.get("remove_small_holes", True))
    remove_objects = bool(cfg.get("remove_small_objects", True))

    processed = binary_map.copy().astype(np.uint8)

    if kernel_size > 0:
        processed = morphological_opening(processed, kernel_size=kernel_size)
        processed = morphological_closing(processed, kernel_size=max(1, kernel_size))

    if remove_objects:
        processed = remove_small_objects(processed, min_size=min_region_size)

    if remove_holes:
        processed = remove_small_holes(processed, min_size=max(1, min_region_size // 2))

    return processed.astype(np.uint8)


__all__ = [
    "morphological_opening",
    "morphological_closing",
    "remove_small_objects",
    "remove_small_holes",
    "connected_components",
    "postprocess_change_map",
]
