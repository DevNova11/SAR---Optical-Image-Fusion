"""Visualization utilities for fused images and change detection outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _prepare_display_array(image: np.ndarray) -> np.ndarray:
    """Normalize image arrays for display in Matplotlib."""
    image = np.asarray(image)
    if image.ndim == 2:
        normalized = image.astype(np.float32)
        if normalized.size == 0:
            return normalized
        if not np.isfinite(normalized).all():
            normalized = np.nan_to_num(normalized)
        min_value = np.min(normalized)
        max_value = np.max(normalized)
        if max_value > min_value:
            normalized = (normalized - min_value) / (max_value - min_value)
        return normalized

    if image.ndim == 3:
        if image.shape[0] in (1, 3, 4) and image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
            image = np.moveaxis(image, 0, -1)
        if image.shape[-1] == 1:
            return _prepare_display_array(image[:, :, 0])
        if image.shape[-1] >= 3:
            rgb = image[:, :, :3].astype(np.float32)
            rgb = np.nan_to_num(rgb)
            rgb_min = np.min(rgb)
            rgb_max = np.max(rgb)
            if rgb_max > rgb_min:
                rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
            return rgb
        return image.astype(np.float32)

    raise ValueError(f"Unsupported image shape for display: {image.shape}")


def save_image_visualization(
    image: np.ndarray,
    title: str,
    output_path: str | Path,
    cmap: Optional[str] = None,
) -> str:
    """Save a single image to disk."""
    img = _prepare_display_array(image)
    fig, ax = plt.subplots(figsize=(6, 6))
    if img.ndim == 2:
        ax.imshow(img, cmap=cmap or "gray")
    else:
        ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def save_difference_map(difference_map: np.ndarray, output_path: str | Path) -> str:
    """Save a continuous difference map as a PNG."""
    diff = _prepare_display_array(difference_map)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(diff, cmap="viridis")
    ax.set_title("Difference Map")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def save_change_map(change_map: np.ndarray, output_path: str | Path) -> str:
    """Save a binary change map as a PNG."""
    cmap = plt.get_cmap("binary")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.asarray(change_map, dtype=np.uint8), cmap=cmap)
    ax.set_title("Final Change Map")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def save_side_by_side(
    fused_1: np.ndarray,
    fused_2: np.ndarray,
    difference_map: np.ndarray,
    change_map: np.ndarray,
    output_path: str | Path,
) -> str:
    """Create a 2x2 panel showing the input images and the detection outputs."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(_prepare_display_array(fused_1), cmap="gray" if np.asarray(fused_1).ndim == 2 else None)
    axes[0, 0].set_title("Date 1")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(_prepare_display_array(fused_2), cmap="gray" if np.asarray(fused_2).ndim == 2 else None)
    axes[0, 1].set_title("Date 2")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(_prepare_display_array(difference_map), cmap="viridis")
    axes[1, 0].set_title("Difference Map")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(np.asarray(change_map, dtype=np.uint8), cmap="binary")
    axes[1, 1].set_title("Final Change Map")
    axes[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def generate_visualizations(
    fused_1: np.ndarray,
    fused_2: np.ndarray,
    difference_map: np.ndarray,
    change_map: np.ndarray,
    output_dir: str | Path = "outputs/visualizations",
) -> Dict[str, str]:
    """Generate all required visual outputs and return the saved paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "date1": save_image_visualization(fused_1, "Date 1", output_dir / "date1.png"),
        "date2": save_image_visualization(fused_2, "Date 2", output_dir / "date2.png"),
        "difference": save_difference_map(difference_map, output_dir / "difference_map.png"),
        "change_map": save_change_map(change_map, output_dir / "change_map.png"),
        "comparison": save_side_by_side(fused_1, fused_2, difference_map, change_map, output_dir / "comparison.png"),
    }
    return paths


__all__ = [
    "save_image_visualization",
    "save_difference_map",
    "save_change_map",
    "save_side_by_side",
    "generate_visualizations",
]
