"""
Uncertainty & Confidence Estimation Engine.

Computes a mathematically grounded, multi-factor confidence score for each detected change
combining classification certainty, temporal trajectory persistence, cross-sensor agreement,
and physical change magnitude.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


CONFIDENCE_LEVELS = {
    0: {"name": "LOW", "color": "#e74c3c", "rgb": (231, 76, 60), "description": "Uncertain or conflicting multi-sensor signal"},
    1: {"name": "MEDIUM", "color": "#f39c12", "rgb": (243, 156, 18), "description": "Moderate confidence with single-sensor dominance or emerging status"},
    2: {"name": "HIGH", "color": "#2ecc71", "rgb": (46, 204, 113), "description": "High-confidence persistent change verified across multiple sensors"},
}


@dataclass
class ConfidenceResult:
    confidence_score_map: np.ndarray  # (H, W) float32 in [0, 1]
    confidence_level_map: np.ndarray  # (H, W) int32 {0: LOW, 1: MEDIUM, 2: HIGH}
    confidence_label_map: np.ndarray  # (H, W) str
    component_breakdown: Dict[str, float]
    level_summary: Dict[str, Dict[str, Union[int, float]]]


class ConfidenceEngine:
    """
    Evaluates prediction confidence and uncertainty.
    
    Formulation:
    Confidence = w_prob * P_margin + w_temp * S_persist + w_sens * A_sensor + w_mag * M_change
    where weights sum to 1.0.
    """

    def __init__(
        self,
        w_prob: float = 0.35,
        w_temp: float = 0.30,
        w_sens: float = 0.20,
        w_mag: float = 0.15,
        high_threshold: float = 0.75,
        med_threshold: float = 0.50,
    ):
        total_w = w_prob + w_temp + w_sens + w_mag
        self.w_prob = w_prob / total_w
        self.w_temp = w_temp / total_w
        self.w_sens = w_sens / total_w
        self.w_mag = w_mag / total_w
        self.high_threshold = high_threshold
        self.med_threshold = med_threshold

    def compute_confidence(
        self,
        classifier_margin: np.ndarray,       # (H, W) in [0, 1]
        persistence_score: np.ndarray,       # (H, W) in [0, 1]
        sensor_agreement: np.ndarray,        # (H, W) in [0, 1]
        sar_evidence: np.ndarray,            # (H, W) in [0, 1]
        optical_evidence: np.ndarray,        # (H, W) in [0, 1]
        changed_mask: Optional[np.ndarray] = None,
    ) -> ConfidenceResult:
        """
        Computes the composite confidence score and categorizes into LOW / MEDIUM / HIGH.
        """
        h, w = classifier_margin.shape[:2]

        # Change magnitude is average of physical sensor evidence
        change_magnitude = np.clip((sar_evidence + optical_evidence) / 2.0, 0.0, 1.0)

        # Composite formulation
        score_map = (
            self.w_prob * classifier_margin
            + self.w_temp * persistence_score
            + self.w_sens * sensor_agreement
            + self.w_mag * change_magnitude
        ).astype(np.float32)

        score_map = np.clip(score_map, 0.0, 1.0)

        # Categorize
        level_map = np.zeros((h, w), dtype=np.int32)
        label_map = np.full((h, w), "LOW", dtype=object)

        med_mask = (score_map >= self.med_threshold) & (score_map < self.high_threshold)
        high_mask = score_map >= self.high_threshold

        level_map[med_mask] = 1
        label_map[med_mask] = "MEDIUM"

        level_map[high_mask] = 2
        label_map[high_mask] = "HIGH"

        # Component breakdown averages for changed pixels
        mask = changed_mask if changed_mask is not None else np.ones((h, w), dtype=bool)
        if np.any(mask):
            comp_breakdown = {
                "mean_classifier_margin": round(float(np.mean(classifier_margin[mask])), 4),
                "mean_persistence_score": round(float(np.mean(persistence_score[mask])), 4),
                "mean_sensor_agreement": round(float(np.mean(sensor_agreement[mask])), 4),
                "mean_change_magnitude": round(float(np.mean(change_magnitude[mask])), 4),
                "mean_composite_confidence": round(float(np.mean(score_map[mask])), 4),
            }
        else:
            comp_breakdown = {
                "mean_classifier_margin": 0.0,
                "mean_persistence_score": 0.0,
                "mean_sensor_agreement": 0.0,
                "mean_change_magnitude": 0.0,
                "mean_composite_confidence": 0.0,
            }

        # Level summary
        total_pixels = h * w
        summary = {}
        for lvl_id, meta in CONFIDENCE_LEVELS.items():
            cnt = int(np.sum(level_map == lvl_id))
            summary[meta["name"]] = {
                "count": cnt,
                "percentage": round(float(cnt / total_pixels) * 100.0, 2),
                "area_km2": round(cnt * (10.0 * 10.0) / 1e6, 4),
            }

        return ConfidenceResult(
            confidence_score_map=score_map,
            confidence_level_map=level_map,
            confidence_label_map=label_map,
            component_breakdown=comp_breakdown,
            level_summary=summary,
        )


def colorize_confidence_map(level_map: np.ndarray) -> np.ndarray:
    """
    Renders confidence map to RGB uint8 image.
    """
    h, w = level_map.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for lvl_id, meta in CONFIDENCE_LEVELS.items():
        mask = level_map == lvl_id
        rgb[mask] = meta["rgb"]
    return rgb
