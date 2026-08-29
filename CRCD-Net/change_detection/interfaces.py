"""Public integration interfaces for the CRCD-Net change detection module.

This module defines the stable contract used by the rest of the project.
Person B is expected to provide two NumPy arrays representing fused images for
Date 1 and Date 2. This module intentionally does not depend on any external
satellite data acquisition or fusion implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ChangeDetectionConfig:
    """Configuration used by the compare pipeline.

    This dataclass acts as a clear, explicit configuration object for the
    change detection pipeline. It is intentionally simple and deterministic so
    that future integration remains stable.
    """

    threshold_method: str = "otsu"
    threshold_value: Optional[float] = None
    percent_threshold: float = 95.0
    aggregation: str = "mean"
    normalization: str = "minmax"
    enable_direction_heuristics: bool = False
    morphology_kernel_size: int = 3
    min_region_size: int = 20
    remove_small_holes: bool = True
    remove_small_objects: bool = True
    connected_components: bool = True


@dataclass
class ChangeDetectionResult:
    """Structured result returned by the compare() API.

    Attributes:
        change_map: Binary change mask with 1 for changed pixels and 0 for no
            change.
        difference_map: Continuous difference map before thresholding.
        statistics: JSON-serializable statistics dictionary.
        metadata: Dictionary containing processing metadata and optional
            integration metadata.
    """

    change_map: np.ndarray
    difference_map: np.ndarray
    statistics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return the result in a JSON-serializable dictionary form."""
        return {
            "change_map": np.asarray(self.change_map),
            "difference_map": np.asarray(self.difference_map),
            "statistics": self.statistics,
            "metadata": self.metadata,
        }


def _ensure_numpy_array(array: Any, name: str) -> np.ndarray:
    """Validate that an object is a NumPy array and return it."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains invalid values (NaN or inf)")
    return array


def _normalize_image_shape(image: np.ndarray) -> np.ndarray:
    """Normalize supported image layouts to a common internal format.

    Internal format: (H, W, C) for multi-channel images and (H, W) for
    grayscale images. This keeps channel handling explicit and makes downstream
    processing predictable.
    """
    if image.ndim == 2:
        return image.astype(np.float32, copy=False)
    if image.ndim == 3:
        return image.astype(np.float32, copy=False)
    raise ValueError(
        "Input arrays must be 2D (H, W) or 3D (H, W, C) or (C, H, W); "
        f"received shape {image.shape}"
    )


def _standardize_layout(image: np.ndarray) -> np.ndarray:
    """Convert arrays to a consistent internal layout.

    Supported input layouts are:
    - (H, W)
    - (H, W, C)
    - (C, H, W)

    Internal format: (H, W, C) for multi-channel images, (H, W) for grayscale.
    """
    image = np.asarray(image)
    if image.ndim == 2:
        return image.astype(np.float32, copy=False)
    if image.ndim == 3:
        # Common channel-first arrays usually have a small leading dimension,
        # such as C=1, 3, 4, 8. We detect that only when both spatial dims are
        # clearly larger than the leading candidate channel count.
        if (
            image.shape[0] <= 8
            and image.shape[0] < image.shape[1]
            and image.shape[0] < image.shape[2]
        ):
            return np.moveaxis(image, 0, -1).astype(np.float32, copy=False)
        return image.astype(np.float32, copy=False)
    raise ValueError(
        "Unsupported image dimensionality. Expected 2D or 3D arrays only; "
        f"received shape {image.shape}"
    )


def validate_fused_pair(fused_1: np.ndarray, fused_2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate compatibility of the two fused images and normalize their layout.

    Supported input layouts:
    - (H, W)
    - (H, W, C)
    - (C, H, W)

    The function converts both inputs into a consistent internal representation
    and raises clear errors if they are incompatible.
    """
    fused_1 = _ensure_numpy_array(fused_1, "fused_1")
    fused_2 = _ensure_numpy_array(fused_2, "fused_2")

    if fused_1.ndim not in (2, 3):
        raise ValueError(
            "fused_1 must be a 2D or 3D array; "
            f"received shape {fused_1.shape}"
        )
    if fused_2.ndim not in (2, 3):
        raise ValueError(
            "fused_2 must be a 2D or 3D array; "
            f"received shape {fused_2.shape}"
        )

    fused_1_std = _standardize_layout(fused_1)
    fused_2_std = _standardize_layout(fused_2)

    if fused_1_std.shape[:2] != fused_2_std.shape[:2]:
        raise ValueError(
            "fused_1 and fused_2 must have identical spatial dimensions; "
            f"got {fused_1_std.shape[:2]} and {fused_2_std.shape[:2]}"
        )

    if fused_1_std.ndim == 2 and fused_2_std.ndim == 2:
        return fused_1_std, fused_2_std

    if fused_1_std.shape[2] != fused_2_std.shape[2]:
        raise ValueError(
            "fused_1 and fused_2 must have identical channel counts; "
            f"got {fused_1_std.shape[2]} and {fused_2_std.shape[2]}"
        )

    return fused_1_std, fused_2_std


def compare(
    fused_1: np.ndarray,
    fused_2: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    config: Optional[ChangeDetectionConfig | Dict[str, Any]] = None,
) -> ChangeDetectionResult:
    """Compare two fused satellite images and return a structured result.

    This is the public integration API and must remain stable for future
    integration with Person B's fused outputs.

    Parameters:
        fused_1: Array for image Date 1.
        fused_2: Array for image Date 2.
        metadata: Optional metadata dict with fields such as date1, date2, aoi,
            pixel_size, crs, etc.
        config: Optional configuration dictionary or ChangeDetectionConfig.

    Returns:
        ChangeDetectionResult with change_map, difference_map, statistics, and
        metadata.
    """
    from .change_detector import detect_changes
    from .statistics import compute_statistics

    fused_1_std, fused_2_std = validate_fused_pair(fused_1, fused_2)

    if config is None:
        config_obj = ChangeDetectionConfig()
    elif isinstance(config, ChangeDetectionConfig):
        config_obj = config
    else:
        config_obj = ChangeDetectionConfig(**config)

    metadata_dict: Dict[str, Any] = dict(metadata or {})
    metadata_dict.setdefault("input_format", "internal_standard=(H, W, C)")
    metadata_dict.setdefault("input_shape_1", tuple(fused_1_std.shape))
    metadata_dict.setdefault("input_shape_2", tuple(fused_2_std.shape))

    change_map, difference_map, threshold_used = detect_changes(fused_1_std, fused_2_std, config=config_obj)
    statistics = compute_statistics(change_map, difference_map, metadata=metadata_dict)

    metadata_dict["threshold_method"] = config_obj.threshold_method
    metadata_dict["threshold_used"] = float(threshold_used)
    metadata_dict["difference_map_shape"] = tuple(difference_map.shape)
    metadata_dict["change_map_shape"] = tuple(change_map.shape)

    if config_obj.enable_direction_heuristics:
        from .change_detector import generate_direction_labels

        metadata_dict["direction_heuristics"] = generate_direction_labels(change_map, difference_map)
    else:
        metadata_dict["direction_heuristics"] = {
            "enabled": False,
            "note": "Optional heuristic labels are disabled by default; they are estimates only, not ground-truth classification.",
        }

    result = ChangeDetectionResult(
        change_map=change_map.astype(np.uint8),
        difference_map=difference_map.astype(np.float32),
        statistics=statistics,
        metadata=metadata_dict,
    )

    return result


__all__ = [
    "ChangeDetectionConfig",
    "ChangeDetectionResult",
    "compare",
    "validate_fused_pair",
]
