"""Baseline change detection pipeline for fused satellite imagery."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from skimage.filters import threshold_otsu

from .interfaces import ChangeDetectionConfig, validate_fused_pair
from .postprocessing import postprocess_change_map


def _normalize_minmax(image: np.ndarray) -> np.ndarray:
    """Apply a safe min-max normalization to a numeric image."""
    image = np.asarray(image, dtype=np.float32)
    if image.size == 0:
        raise ValueError("Cannot normalize an empty image")
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value == min_value:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_value) / (max_value - min_value)


def aggregate_difference(difference_map: np.ndarray, method: str = "mean") -> np.ndarray:
    """Aggregate per-channel absolute differences into a single spatial map."""
    difference_map = np.asarray(difference_map, dtype=np.float32)
    if difference_map.ndim == 2:
        return difference_map
    if difference_map.ndim == 3:
        method = method.lower()
        if method == "mean":
            return np.mean(difference_map, axis=-1)
        if method == "max":
            return np.max(difference_map, axis=-1)
        if method == "euclidean":
            return np.sqrt(np.sum(np.square(difference_map), axis=-1))
        raise ValueError(f"Unsupported aggregation method: {method}")
    raise ValueError(f"Difference map must be 2D or 3D; got shape {difference_map.shape}")


def estimate_threshold(
    difference_map: np.ndarray,
    method: str = "otsu",
    threshold_value: Optional[float] = None,
    percent_threshold: float = 95.0,
) -> float:
    """Estimate a threshold for the spatial difference map."""
    difference_map = np.asarray(difference_map, dtype=np.float32)
    if difference_map.size == 0:
        raise ValueError("Difference map is empty")

    method = method.lower()
    if method == "manual":
        if threshold_value is None:
            raise ValueError("A manual threshold_value must be provided when threshold_method='manual'")
        return float(threshold_value)
    if method == "percentile":
        return float(np.percentile(difference_map, percent_threshold))
    if method == "otsu":
        unique_values = np.unique(difference_map)
        if unique_values.size < 2:
            return float(unique_values[0]) if unique_values.size else 0.0
        return float(threshold_otsu(difference_map))
    raise ValueError(f"Unsupported threshold method: {method}")


def detect_changes(
    fused_1: np.ndarray,
    fused_2: np.ndarray,
    config: Optional[ChangeDetectionConfig | Dict[str, Any]] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the baseline change detection pipeline.

    Pipeline:
    1. normalize both images safely
    2. absolute difference
    3. aggregate per channel differences
    4. thresholding
    5. binary change map
    6. post-processing cleanup
    """
    fused_1_std, fused_2_std = validate_fused_pair(fused_1, fused_2)

    if config is None:
        config_obj = ChangeDetectionConfig()
    elif isinstance(config, ChangeDetectionConfig):
        config_obj = config
    else:
        config_obj = ChangeDetectionConfig(**config)

    normalized_1 = _normalize_minmax(fused_1_std)
    normalized_2 = _normalize_minmax(fused_2_std)

    absolute_difference = np.abs(normalized_1 - normalized_2)
    aggregated = aggregate_difference(absolute_difference, method=config_obj.aggregation)
    normalized_difference = _normalize_minmax(aggregated)

    threshold = estimate_threshold(
        normalized_difference,
        method=config_obj.threshold_method,
        threshold_value=config_obj.threshold_value,
        percent_threshold=config_obj.percent_threshold,
    )

    binary_change_map = (normalized_difference >= threshold).astype(np.uint8)
    processed = postprocess_change_map(
        binary_change_map,
        config={
            "morphology_kernel_size": config_obj.morphology_kernel_size,
            "min_region_size": config_obj.min_region_size,
            "remove_small_holes": config_obj.remove_small_holes,
            "remove_small_objects": config_obj.remove_small_objects,
        },
    )

    return processed.astype(np.uint8), normalized_difference.astype(np.float32), float(threshold)


def generate_direction_labels(change_map: np.ndarray, difference_map: np.ndarray) -> Dict[str, Any]:
    """Generate optional heuristic labels for change direction.

    This is intentionally heuristic-only and is disabled by default. It does not
    claim ground-truth semantic classification.
    """
    change_map = np.asarray(change_map)
    difference_map = np.asarray(difference_map, dtype=np.float32)
    changed_pixels = np.sum(change_map > 0)

    if changed_pixels == 0:
        return {
            "enabled": True,
            "label": "No Change",
            "confidence": "low",
            "note": "Heuristic labels are estimates only and no significant change was detected.",
        }

    mean_diff = float(np.mean(difference_map[change_map > 0]))
    if mean_diff >= 0.6:
        label = "Possible Urbanization"
    elif mean_diff >= 0.3:
        label = "Possible Vegetation Loss"
    else:
        label = "Other Change"

    return {
        "enabled": True,
        "label": label,
        "confidence": "heuristic",
        "note": "This is a heuristic estimate for directionality only, not a scientific classification.",
        "mean_difference_changed_pixels": round(mean_diff, 6),
    }


__all__ = [
    "_normalize_minmax",
    "aggregate_difference",
    "estimate_threshold",
    "detect_changes",
    "generate_direction_labels",
]
