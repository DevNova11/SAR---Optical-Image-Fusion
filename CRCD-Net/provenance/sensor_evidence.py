"""
Sensor Evidence & Modality Attribution Engine.

Quantifies independent evidence from Sentinel-1 SAR (structural/backscatter shifts)
and Sentinel-2 Optical (spectral/vegetation shifts), evaluates cross-sensor agreement,
and determines modality attribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from semantics.land_cover_classifier import compute_sar_features, compute_spectral_indices


EVIDENCE_TYPES = {
    0: {"name": "Both-sensor supported", "color": "#2ecc71", "rgb": (46, 204, 113), "description": "Corroborated by both SAR backscatter and Optical reflectance"},
    1: {"name": "SAR-supported", "color": "#3498db", "rgb": (52, 152, 219), "description": "Primarily driven by radar structural and roughness change"},
    2: {"name": "Optical-supported", "color": "#e67e22", "rgb": (230, 126, 34), "description": "Primarily driven by multispectral/NDVI reflectance change"},
    3: {"name": "Low-evidence / Ambiguous", "color": "#95a5a6", "rgb": (149, 165, 166), "description": "Weak or contradictory multi-sensor signal"},
}


@dataclass
class SensorEvidenceResult:
    sar_evidence_map: np.ndarray       # (H, W) float32 in [0, 1]
    optical_evidence_map: np.ndarray   # (H, W) float32 in [0, 1]
    sensor_agreement_map: np.ndarray   # (H, W) float32 in [0, 1]
    attribution_id_map: np.ndarray     # (H, W) int32 {0: Both, 1: SAR, 2: Optical, 3: Ambiguous}
    attribution_label_map: np.ndarray  # (H, W) str
    evidence_summary: Dict[str, Dict[str, Union[int, float]]]


class SensorEvidenceEngine:
    """
    Computes cross-sensor evidence metrics and attribution for detected changes.
    """

    def __init__(self, sar_scale_db: float = 8.0, opt_scale: float = 0.4):
        self.sar_scale_db = sar_scale_db
        self.opt_scale = opt_scale

    def evaluate_evidence(
        self,
        s1_t1: np.ndarray,  # (H, W, 2)
        s2_t1: np.ndarray,  # (H, W, 4)
        s1_tn: np.ndarray,  # (H, W, 2)
        s2_tn: np.ndarray,  # (H, W, 4)
        changed_mask: Optional[np.ndarray] = None,
    ) -> SensorEvidenceResult:
        """
        Calculates SAR and Optical change evidence and classifies attribution.
        """
        h, w = s1_t1.shape[:2]

        # 1. SAR evidence calculation (dB domain Euclidean shift)
        sar_diff_vv = np.abs(s1_tn[..., 0] - s1_t1[..., 0])
        sar_diff_vh = np.abs(s1_tn[..., 1] - s1_t1[..., 1]) if s1_t1.shape[-1] > 1 else sar_diff_vv
        
        sar_mag = np.sqrt(sar_diff_vv ** 2 + sar_diff_vh ** 2)
        sar_evidence = np.clip(sar_mag / self.sar_scale_db, 0.0, 1.0).astype(np.float32)

        # 2. Optical evidence calculation (Euclidean spectral distance + NDVI delta)
        opt_diff_rgb = np.sqrt(np.sum((s2_tn[..., :3] - s2_t1[..., :3]) ** 2, axis=-1))
        
        idx_t1 = compute_spectral_indices(s2_t1)
        idx_tn = compute_spectral_indices(s2_tn)
        ndvi_diff = np.abs(idx_tn["ndvi"] - idx_t1["ndvi"])

        opt_mag = opt_diff_rgb + 0.8 * ndvi_diff
        opt_evidence = np.clip(opt_mag / self.opt_scale, 0.0, 1.0).astype(np.float32)

        # 3. Sensor Agreement
        # Agreement is high when both sensors agree (both high or both low)
        sensor_agreement = (1.0 - np.abs(sar_evidence - opt_evidence)).astype(np.float32)

        # 4. Modality Attribution
        attribution_id = np.full((h, w), 3, dtype=np.int32)
        attribution_label = np.full((h, w), "Low-evidence / Ambiguous", dtype=object)

        # Masks for classification
        both_high = (sar_evidence >= 0.40) & (opt_evidence >= 0.40)
        sar_dominant = (sar_evidence >= 0.40) & (sar_evidence > opt_evidence + 0.15)
        opt_dominant = (opt_evidence >= 0.40) & (opt_evidence > sar_evidence + 0.15)

        attribution_id[both_high] = 0
        attribution_label[both_high] = "Both-sensor supported"

        attribution_id[sar_dominant] = 1
        attribution_label[sar_dominant] = "SAR-supported"

        attribution_id[opt_dominant] = 2
        attribution_label[opt_dominant] = "Optical-supported"

        if changed_mask is not None:
            # For unchanged regions, keep neutral
            unchanged = ~changed_mask
            attribution_id[unchanged] = 3
            attribution_label[unchanged] = "Stable / Unchanged"

        # Summary statistics
        total_pixels = h * w
        summary = {}
        for ev_id, meta in EVIDENCE_TYPES.items():
            cnt = int(np.sum(attribution_id == ev_id))
            summary[meta["name"]] = {
                "count": cnt,
                "percentage": round(float(cnt / total_pixels) * 100.0, 2),
                "mean_sar_evidence": round(float(np.mean(sar_evidence[attribution_id == ev_id])) if cnt > 0 else 0.0, 4),
                "mean_opt_evidence": round(float(np.mean(opt_evidence[attribution_id == ev_id])) if cnt > 0 else 0.0, 4),
            }

        return SensorEvidenceResult(
            sar_evidence_map=sar_evidence,
            optical_evidence_map=opt_evidence,
            sensor_agreement_map=sensor_agreement,
            attribution_id_map=attribution_id,
            attribution_label_map=attribution_label,
            evidence_summary=summary,
        )


def colorize_sensor_evidence_map(attribution_id_map: np.ndarray) -> np.ndarray:
    """
    Renders sensor evidence map to RGB uint8 image.
    """
    h, w = attribution_id_map.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for ev_id, meta in EVIDENCE_TYPES.items():
        mask = attribution_id_map == ev_id
        rgb[mask] = meta["rgb"]
    return rgb
