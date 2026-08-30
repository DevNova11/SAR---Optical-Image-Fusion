"""
Multi-Modal Land-Cover Classification Module.

Produces pixel-level semantic land-cover classifications and class probability distributions
combining Sentinel-1 SAR backscatter and Sentinel-2 multispectral indices.

Classes:
--------
0: Forest / Dense Vegetation (Color: #2ca02c)
1: Agriculture / Low Vegetation (Color: #98df8a)
2: Urban / Built-up (Color: #d62728)
3: Bare Land / Soil / Quarry (Color: #ff7f0e)
4: Water (Color: #1f77b4)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


LAND_COVER_CLASSES = {
    0: {"name": "Forest", "color": "#2ca02c", "rgb": (44, 160, 44), "description": "Dense tree canopy and perennial forest"},
    1: {"name": "Agriculture", "color": "#98df8a", "rgb": (152, 223, 138), "description": "Cropland, grassland, and seasonal vegetation"},
    2: {"name": "Urban", "color": "#d62728", "rgb": (214, 39, 40), "description": "Built-up structures, impervious surfaces, infrastructure"},
    3: {"name": "Bare Land", "color": "#ff7f0e", "rgb": (255, 127, 14), "description": "Exposed soil, cleared terrain, rocks, and quarries"},
    4: {"name": "Water", "color": "#1f77b4", "rgb": (31, 119, 180), "description": "Rivers, reservoirs, lakes, and coastal water bodies"},
}

CLASS_NAMES = [LAND_COVER_CLASSES[i]["name"] for i in range(len(LAND_COVER_CLASSES))]
CLASS_COLORS = [LAND_COVER_CLASSES[i]["color"] for i in range(len(LAND_COVER_CLASSES))]


@dataclass
class SemanticClassificationResult:
    class_map: np.ndarray  # Shape (H, W), dtype int32
    probabilities: np.ndarray  # Shape (H, W, 5), dtype float32
    confidence_margin: np.ndarray  # Shape (H, W), dtype float32 (P_top1 - P_top2)
    class_counts: Dict[str, int]
    class_fractions: Dict[str, float]


def compute_spectral_indices(s2_hwc: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes spectral remote sensing indices from Sentinel-2 [B2, B3, B4, B8].
    """
    b2 = s2_hwc[..., 0]  # Blue
    b3 = s2_hwc[..., 1]  # Green
    b4 = s2_hwc[..., 2]  # Red
    b8 = s2_hwc[..., 3]  # NIR
    eps = 1e-6

    # NDVI: (NIR - Red) / (NIR + Red)
    ndvi = (b8 - b4) / (b8 + b4 + eps)
    ndvi = np.clip(ndvi, -1.0, 1.0)

    # NDWI (McFeeters): (Green - NIR) / (Green + NIR)
    ndwi = (b3 - b8) / (b3 + b8 + eps)
    ndwi = np.clip(ndwi, -1.0, 1.0)

    # NDBI proxy (Built-up index / Spectral contrast): (Red - Green) / (Red + Green)
    ndbi = (b4 - b3) / (b4 + b3 + eps)
    ndbi = np.clip(ndbi, -1.0, 1.0)

    # Brightness (Albedo proxy)
    brightness = (b2 + b3 + b4 + b8) / 4.0

    return {
        "ndvi": ndvi.astype(np.float32),
        "ndwi": ndwi.astype(np.float32),
        "ndbi": ndbi.astype(np.float32),
        "brightness": brightness.astype(np.float32),
    }


def compute_sar_features(s1_hwc: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes radar backscatter features from Sentinel-1 [VV, VH] (in dB).
    """
    vv = s1_hwc[..., 0]
    vh = s1_hwc[..., 1] if s1_hwc.shape[-1] > 1 else s1_hwc[..., 0]
    
    # Backscatter ratio (Volume / Roughness proxy)
    vh_vv_ratio = vh - vv  # In dB, difference is the ratio in linear power
    
    # Mean radar cross-section
    total_power = (vv + vh) / 2.0

    return {
        "vv": vv.astype(np.float32),
        "vh": vh.astype(np.float32),
        "vh_vv_ratio": vh_vv_ratio.astype(np.float32),
        "total_power": total_power.astype(np.float32),
    }


class LandCoverClassifier:
    """
    Rule-based land-cover classifier for SAR-Optical inputs -- a placeholder,
    not a trained model. Combines radar backscatter response and spectral
    vegetation/water indices via hand-tuned linear scores per class, then a
    softmax to get a bounded, class-comparable output per pixel.

    Honesty note: these per-class score weights and thresholds were chosen
    from domain knowledge, not fit or validated against any labeled ground
    truth -- there's no labeled land-cover dataset in this project. Treat
    the softmax output as a relative ranking across the 5 classes, not as
    a calibrated probability in the statistical sense (i.e. "60% forest"
    here is not the same claim as a calibrated model's 60%).
    """

    def __init__(self):
        self.num_classes = len(LAND_COVER_CLASSES)

    def predict(
        self,
        s1_image: np.ndarray,  # (H, W, 2)
        s2_image: np.ndarray,  # (H, W, 4)
        fused_image: Optional[np.ndarray] = None,
    ) -> SemanticClassificationResult:
        """
        Classifies each pixel into one of the 5 canonical land-cover classes.
        """
        h, w = s1_image.shape[:2]
        indices = compute_spectral_indices(s2_image)
        sar = compute_sar_features(s1_image)

        ndvi = indices["ndvi"]
        ndwi = indices["ndwi"]
        brightness = indices["brightness"]
        vv = sar["vv"]
        vh = sar["vh"]
        tot_sar = sar["total_power"]

        # Logit score accumulators for each class
        # Class 0: Forest (High NDVI > 0.5, high VH volume scattering, moderate VV)
        score_forest = (
            2.5 * (ndvi - 0.35)
            + 1.0 * np.clip((vh + 16.0) / 10.0, -1.0, 1.5)
            - 1.5 * np.maximum(ndwi, 0.0)
            - 1.0 * np.clip((vv + 8.0) / 10.0, 0.0, 2.0)  # Penalize high double bounce
        )

        # Class 1: Agriculture / Low Vegetation (Moderate NDVI 0.25..0.55, lower radar roughness)
        score_agri = (
            2.0 * (1.0 - np.abs(ndvi - 0.35) * 3.5)
            - 1.0 * np.maximum(ndwi, 0.0)
            - 0.5 * np.abs(tot_sar + 14.0) / 5.0
        )

        # Class 2: Urban / Built-up (High SAR double-bounce VV > -10 dB, high VH, low NDVI)
        score_urban = (
            2.8 * np.clip((vv + 12.0) / 8.0, -1.0, 2.0)
            + 1.8 * np.clip((vh + 18.0) / 8.0, -1.0, 2.0)
            - 2.5 * np.maximum(ndvi - 0.2, 0.0)
            - 2.0 * np.maximum(ndwi, 0.0)
        )

        # Class 3: Bare Land / Quarry (Low NDVI < 0.2, high visible brightness, moderate-low SAR)
        score_bare = (
            2.0 * (0.25 - ndvi)
            + 1.5 * np.clip((brightness - 0.15) / 0.2, -1.0, 1.5)
            - 1.5 * np.maximum(ndwi, 0.0)
            - 1.0 * np.clip((vv + 10.0) / 10.0, 0.0, 2.0)
        )

        # Class 4: Water (High NDWI > 0.0, very low SAR specular reflection VV < -18 dB, low NIR)
        score_water = (
            3.0 * (ndwi + 0.1)
            + 2.5 * np.clip((-16.0 - vv) / 6.0, -1.0, 2.0)
            - 3.0 * np.maximum(ndvi, 0.0)
            - 2.0 * np.clip(brightness / 0.1, 0.0, 2.0)
        )

        # Stack logits: shape (H, W, 5)
        logits = np.stack(
            [score_forest, score_agri, score_urban, score_bare, score_water],
            axis=-1,
        )

        # Softmax to get calibrated probabilities
        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-8)
        probs = probs.astype(np.float32)

        # Argmax discrete classification
        class_map = np.argmax(probs, axis=-1).astype(np.int32)

        # Confidence margin: difference between top-1 and top-2 probabilities
        sorted_probs = np.sort(probs, axis=-1)
        margin = (sorted_probs[..., -1] - sorted_probs[..., -2]).astype(np.float32)

        # Statistics
        total_pixels = h * w
        counts = {}
        fractions = {}
        for c_idx in range(self.num_classes):
            c_name = CLASS_NAMES[c_idx]
            count = int(np.sum(class_map == c_idx))
            counts[c_name] = count
            fractions[c_name] = round(float(count / total_pixels), 4)

        return SemanticClassificationResult(
            class_map=class_map,
            probabilities=probs,
            confidence_margin=margin,
            class_counts=counts,
            class_fractions=fractions,
        )


def colorize_class_map(class_map: np.ndarray) -> np.ndarray:
    """
    Renders integer class map to RGB uint8 image using standardized color palette.
    """
    h, w = class_map.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c_idx, meta in LAND_COVER_CLASSES.items():
        mask = class_map == c_idx
        rgb[mask] = meta["rgb"]
    return rgb
