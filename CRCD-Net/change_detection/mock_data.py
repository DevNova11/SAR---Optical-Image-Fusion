"""Deterministic mock fused image generation for CRCD-Net Person C.

This module provides synthetic fused image pairs that can be used for local
prototyping, unit tests, and deterministic core validation before Person B's
real fused arrays are available.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def _make_base_image(height: int, width: int, channels: int, rng: np.random.Generator) -> np.ndarray:
    """Create a smooth base image with realistic spectral structure."""
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    gradient = 0.35 * (y + x)
    pattern = np.sin((y * 8.0 + x * 7.0) * np.pi) * 0.12
    noise = rng.normal(0.0, 0.05, size=(height, width))
    base = gradient + pattern + noise
    base = np.clip(base, 0.0, 1.0)
    return np.repeat(base[:, :, None], channels, axis=2)



def _apply_rectangular_change(image: np.ndarray, region: tuple[int, int, int, int], signal: float) -> np.ndarray:
    """Apply a rectangular change to an image."""
    y0, y1, x0, x1 = region
    image[y0:y1, x0:x1, :] += signal
    return np.clip(image, 0.0, 1.0)


def _apply_circular_change(image: np.ndarray, center_y: int, center_x: int, radius: int, signal: float) -> np.ndarray:
    """Apply a circular change to an image."""
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    dist_sq = (yy - center_y) ** 2 + (xx - center_x) ** 2
    mask = dist_sq <= radius**2
    image[mask, :] += signal
    return np.clip(image, 0.0, 1.0)


def _add_small_noisy_regions(image: np.ndarray, rng: np.random.Generator, count: int = 8) -> np.ndarray:
    """Add small, noisy regions to simulate minor artifacts."""
    height, width, channels = image.shape
    for _ in range(count):
        y = int(rng.integers(0, height))
        x = int(rng.integers(0, width))
        radius = int(rng.integers(2, 6))
        yy, xx = np.ogrid[:height, :width]
        dist_sq = (yy - y) ** 2 + (xx - x) ** 2
        mask = dist_sq <= radius**2
        image[mask, :] += rng.uniform(0.06, 0.18, size=(channels,))
    return np.clip(image, 0.0, 1.0)


def generate_mock_fused_pair(
    height: int = 256,
    width: int = 256,
    channels: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate a deterministic pair of fused images with known synthetic changes.

    Parameters:
        height: Height of the synthetic image.
        width: Width of the synthetic image.
        channels: Number of channels in each fused image.
        seed: Fixed random seed for deterministic generation.

    Returns:
        fused_1: baseline fused image for Date 1.
        fused_2: modified fused image for Date 2.
        metadata: a dictionary describing generated changes and settings.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive integers")
    if channels <= 0:
        raise ValueError("channels must be a positive integer")

    rng = np.random.default_rng(seed)

    fused_1 = _make_base_image(height, width, channels, rng)
    fused_2 = fused_1.copy()

    # Large rectangular change.
    rect_y0 = max(0, int(height * 0.18))
    rect_y1 = max(0, int(height * 0.52))
    rect_x0 = max(0, int(width * 0.22))
    rect_x1 = min(width, int(width * 0.62))
    fused_2 = _apply_rectangular_change(fused_2, (rect_y0, rect_y1, rect_x0, rect_x1), 0.30)

    # Circular change.
    center_y = int(height * 0.72)
    center_x = int(width * 0.78)
    radius = int(min(height, width) * 0.12)
    fused_2 = _apply_circular_change(fused_2, center_y, center_x, radius, -0.28)

    # Minor noisy changes.
    fused_2 = _add_small_noisy_regions(fused_2, rng, count=max(5, min(10, height // 25)))

    metadata: Dict[str, Any] = {
        "seed": seed,
        "height": height,
        "width": width,
        "channels": channels,
        "date1": "synthetic_date_1",
        "date2": "synthetic_date_2",
        "aoi": "synthetic_demo_area",
        "pixel_size": 10.0,
        "crs": "EPSG:4326",
        "generated_changes": {
            "rectangular_region": {
                "y0": rect_y0,
                "y1": rect_y1,
                "x0": rect_x0,
                "x1": rect_x1,
                "signal": 0.30,
                "type": "increase",
            },
            "circular_region": {
                "center_y": center_y,
                "center_x": center_x,
                "radius": radius,
                "signal": -0.28,
                "type": "decrease",
            },
            "noise_regions": "small synthetic artifacts",
        },
    }

    return fused_1, fused_2, metadata


__all__ = ["generate_mock_fused_pair"]
