"""Statistics and area calculation for detected changes."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from skimage import measure


def compute_statistics(
    change_map: np.ndarray,
    difference_map: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Calculate change statistics and optional real-world area metrics."""
    change_map = np.asarray(change_map)
    difference_map = np.asarray(difference_map, dtype=np.float32)

    total_pixels = int(change_map.size)
    changed_pixels = int(np.sum(change_map > 0))
    unchanged_pixels = total_pixels - changed_pixels
    change_percentage = float((changed_pixels / total_pixels) * 100.0) if total_pixels else 0.0

    binary = change_map > 0
    labels = measure.label(binary, connectivity=2)
    if labels.max() == 0:
        num_change_regions = 0
        largest_region_pixels = 0
    else:
        region_sizes = np.bincount(labels.ravel())[1:]
        num_change_regions = int(region_sizes.size)
        largest_region_pixels = int(region_sizes.max()) if region_sizes.size > 0 else 0

    mean_difference = float(np.mean(difference_map)) if difference_map.size else 0.0
    max_difference = float(np.max(difference_map)) if difference_map.size else 0.0

    stats: Dict[str, Any] = {
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "unchanged_pixels": unchanged_pixels,
        "change_percentage": round(change_percentage, 2),
        "num_change_regions": num_change_regions,
        "largest_region_pixels": largest_region_pixels,
        "mean_difference": round(mean_difference, 6),
        "max_difference": round(max_difference, 6),
    }

    metadata = dict(metadata or {})
    pixel_size = metadata.get("pixel_size")
    if pixel_size is None:
        stats["pixel_area_m2"] = None
        stats["changed_area_m2"] = None
        stats["changed_area_hectares"] = None
        stats["changed_area_km2"] = None
        stats["area_note"] = "Real-world area unavailable because pixel_size metadata was not provided."
        return stats

    try:
        pixel_size_value = float(pixel_size)
    except (TypeError, ValueError):
        stats["pixel_area_m2"] = None
        stats["changed_area_m2"] = None
        stats["changed_area_hectares"] = None
        stats["changed_area_km2"] = None
        stats["area_note"] = "Real-world area unavailable because pixel_size metadata was not provided."
        return stats

    pixel_area_m2 = pixel_size_value * pixel_size_value
    changed_area_m2 = changed_pixels * pixel_area_m2
    changed_area_hectares = changed_area_m2 / 10000.0
    changed_area_km2 = changed_area_m2 / 1_000_000.0

    stats["pixel_area_m2"] = float(pixel_area_m2)
    stats["changed_area_m2"] = float(changed_area_m2)
    stats["changed_area_hectares"] = float(changed_area_hectares)
    stats["changed_area_km2"] = float(changed_area_km2)
    stats["area_note"] = "Real-world area computed from pixel_size metadata."
    return stats


__all__ = ["compute_statistics"]
