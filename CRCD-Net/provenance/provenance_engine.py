"""
Change Provenance Engine & Early-Warning Priority Ranking.

Extracts spatially coherent change hotspots, aggregates multi-temporal trajectory
and multi-sensor evidence, computes actionable Early-Warning Priority Scores,
and generates structured, explainable provenance records.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from skimage import measure

from provenance.confidence_engine import ConfidenceResult
from provenance.persistence_verifier import PersistenceResult
from provenance.sensor_evidence import SensorEvidenceResult
from provenance.trajectory_engine import TrajectoryResult
from semantics.land_cover_classifier import CLASS_NAMES, LAND_COVER_CLASSES


PRIORITY_LEVELS = {
    0: {"name": "LOW", "color": "#95a5a6", "rgb": (149, 165, 166), "description": "Minor or low-confidence change"},
    1: {"name": "MEDIUM", "color": "#f39c12", "rgb": (243, 156, 18), "description": "Moderate change warranting routine observation"},
    2: {"name": "HIGH", "color": "#e67e22", "rgb": (230, 126, 34), "description": "Significant persistent transition requiring field verification"},
    3: {"name": "CRITICAL", "color": "#c0392b", "rgb": (192, 57, 43), "description": "Severe rapid deforestation, wetland loss, or critical urban encroachment"},
}


@dataclass
class HotspotRecord:
    rank: int
    hotspot_id: str
    centroid_rc: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)
    area_pixels: int
    area_hectares: float
    area_km2: float
    previous_class: str
    current_class: str
    transition: str
    transition_category: str
    first_detected: str
    last_confirmed: str
    observations: int
    persistence_score: float
    persistence_level: str
    change_magnitude: float
    sar_evidence: float
    optical_evidence: float
    sensor_agreement: float
    sensor_attribution: str
    confidence: float
    confidence_level: str
    priority_score: float
    priority_level: str
    trajectory: List[str]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProvenanceResult:
    priority_score_map: np.ndarray       # (H, W) float32 in [0, 1]
    priority_level_map: np.ndarray       # (H, W) int32 {0: LOW, 1: MED, 2: HIGH, 3: CRITICAL}
    priority_label_map: np.ndarray       # (H, W) str
    hotspots: List[HotspotRecord]
    hotspot_label_matrix: np.ndarray    # (H, W) int32 labeled regions
    summary: Dict[str, Any]


class ProvenanceEngine:
    """
    Computes priority ranking and structured change provenance.
    """

    def __init__(
        self,
        w_sev: float = 0.30,
        w_mag: float = 0.20,
        w_per: float = 0.25,
        w_conf: float = 0.15,
        w_area: float = 0.10,
        crit_threshold: float = 0.80,
        high_threshold: float = 0.65,
        med_threshold: float = 0.45,
        pixel_size_m: float = 10.0,
    ):
        total = w_sev + w_mag + w_per + w_conf + w_area
        self.w_sev = w_sev / total
        self.w_mag = w_mag / total
        self.w_per = w_per / total
        self.w_conf = w_conf / total
        self.w_area = w_area / total
        self.crit_threshold = crit_threshold
        self.high_threshold = high_threshold
        self.med_threshold = med_threshold
        self.pixel_size_m = pixel_size_m

    def generate_provenance(
        self,
        trajectory: TrajectoryResult,
        persistence: PersistenceResult,
        evidence: SensorEvidenceResult,
        confidence: ConfidenceResult,
        dates: List[str],
        min_region_pixels: int = 15,
    ) -> ProvenanceResult:
        """
        Executes provenance synthesis and hotspot priority ranking.
        """
        h, w = trajectory.changed_mask.shape
        pixel_area_ha = (self.pixel_size_m * self.pixel_size_m) / 10000.0
        pixel_area_km2 = (self.pixel_size_m * self.pixel_size_m) / 1e6

        # Continuous change magnitude map
        change_magnitude = np.clip(
            (evidence.sar_evidence_map + evidence.optical_evidence_map) / 2.0, 0.0, 1.0
        )

        # Region labeling on changed pixels
        labeled_regions = measure.label(trajectory.changed_mask, connectivity=2)
        region_props = measure.regionprops(labeled_regions)

        # Build area score map (log-scaled normalized area)
        area_score_map = np.zeros((h, w), dtype=np.float32)
        for prop in region_props:
            a_score = float(np.clip(np.log10(max(1, prop.area)) / 4.0, 0.0, 1.0))
            area_score_map[labeled_regions == prop.label] = a_score

        # Multi-factor priority score map
        priority_score_map = (
            self.w_sev * trajectory.severity_map
            + self.w_mag * change_magnitude
            + self.w_per * persistence.persistence_score_map
            + self.w_conf * confidence.confidence_score_map
            + self.w_area * area_score_map
        ).astype(np.float32)

        # Zero out unchanged regions
        priority_score_map[~trajectory.changed_mask] = 0.0

        # Categorize priority levels
        priority_level_map = np.zeros((h, w), dtype=np.int32)
        priority_label_map = np.full((h, w), "LOW", dtype=object)

        med_mask = (priority_score_map >= self.med_threshold) & (priority_score_map < self.high_threshold)
        high_mask = (priority_score_map >= self.high_threshold) & (priority_score_map < self.crit_threshold)
        crit_mask = priority_score_map >= self.crit_threshold

        priority_level_map[med_mask] = 1
        priority_label_map[med_mask] = "MEDIUM"

        priority_level_map[high_mask] = 2
        priority_label_map[high_mask] = "HIGH"

        priority_level_map[crit_mask] = 3
        priority_label_map[crit_mask] = "CRITICAL"

        # Extract structured hotspot records
        hotspots: List[HotspotRecord] = []
        raw_hotspots = []

        for prop in region_props:
            if prop.area < min_region_pixels:
                continue

            r_mask = labeled_regions == prop.label
            mean_priority = float(np.mean(priority_score_map[r_mask]))
            mean_persist = float(np.mean(persistence.persistence_score_map[r_mask]))
            mean_conf = float(np.mean(confidence.confidence_score_map[r_mask]))
            mean_mag = float(np.mean(change_magnitude[r_mask]))
            mean_sar = float(np.mean(evidence.sar_evidence_map[r_mask]))
            mean_opt = float(np.mean(evidence.optical_evidence_map[r_mask]))
            mean_agree = float(np.mean(evidence.sensor_agreement_map[r_mask]))

            # Dominant transition in this region
            transitions, t_counts = np.unique(trajectory.transition_label_map[r_mask], return_counts=True)
            dom_t_idx = int(np.argmax(t_counts))
            dominant_transition = str(transitions[dom_t_idx])

            # Dominant previous and current classes
            c_init_vals, c_init_counts = np.unique(trajectory.initial_class_map[r_mask], return_counts=True)
            prev_c_id = int(c_init_vals[np.argmax(c_init_counts)])
            prev_class = CLASS_NAMES[prev_c_id]

            c_fin_vals, c_fin_counts = np.unique(trajectory.final_class_map[r_mask], return_counts=True)
            curr_c_id = int(c_fin_vals[np.argmax(c_fin_counts)])
            curr_class = CLASS_NAMES[curr_c_id]

            # Dominant attribution
            attr_vals, attr_counts = np.unique(evidence.attribution_label_map[r_mask], return_counts=True)
            dom_attr = str(attr_vals[np.argmax(attr_counts)])

            # Dominant persistence label
            per_vals, per_counts = np.unique(persistence.persistence_label_map[r_mask], return_counts=True)
            dom_per_label = str(per_vals[np.argmax(per_counts)])

            # First detected date
            step_vals, step_counts = np.unique(persistence.first_detected_step_map[r_mask], return_counts=True)
            first_step = int(step_vals[np.argmax(step_counts)])
            first_date = dates[min(first_step, len(dates) - 1)]
            latest_date = dates[-1]

            # Temporal trajectory string list
            cy, cx = int(prop.centroid[0]), int(prop.centroid[1])
            traj_classes = [CLASS_NAMES[int(trajectory.trajectory_matrix[k, cy, cx])] for k in range(len(dates))]

            # Priority level string
            if mean_priority >= self.crit_threshold:
                p_level_str = "CRITICAL"
            elif mean_priority >= self.high_threshold:
                p_level_str = "HIGH"
            elif mean_priority >= self.med_threshold:
                p_level_str = "MEDIUM"
            else:
                p_level_str = "LOW"

            # Confidence level string
            if mean_conf >= 0.75:
                c_level_str = "HIGH"
            elif mean_conf >= 0.50:
                c_level_str = "MEDIUM"
            else:
                c_level_str = "LOW"

            # Dynamic natural language explanation
            explanation = _generate_explanation(
                prev_class=prev_class,
                curr_class=curr_class,
                transition=dominant_transition,
                first_date=first_date,
                latest_date=latest_date,
                persistence_score=mean_persist,
                persistence_level=dom_per_label,
                sar_evidence=mean_sar,
                optical_evidence=mean_opt,
                sensor_agreement=mean_agree,
                confidence_score=mean_conf,
                priority_level=p_level_str,
                area_ha=prop.area * pixel_area_ha,
            )

            raw_hotspots.append({
                "centroid_rc": (int(prop.centroid[0]), int(prop.centroid[1])),
                "bounding_box": (int(prop.bbox[0]), int(prop.bbox[1]), int(prop.bbox[2]), int(prop.bbox[3])),
                "area_pixels": int(prop.area),
                "area_hectares": round(float(prop.area * pixel_area_ha), 2),
                "area_km2": round(float(prop.area * pixel_area_km2), 4),
                "previous_class": prev_class,
                "current_class": curr_class,
                "transition": dominant_transition,
                "transition_category": dominant_transition.split(" (")[0],
                "first_detected": first_date,
                "last_confirmed": latest_date,
                "observations": len(dates),
                "persistence_score": round(mean_persist, 3),
                "persistence_level": dom_per_label,
                "change_magnitude": round(mean_mag, 3),
                "sar_evidence": round(mean_sar, 3),
                "optical_evidence": round(mean_opt, 3),
                "sensor_agreement": round(mean_agree, 3),
                "sensor_attribution": dom_attr,
                "confidence": round(mean_conf, 3),
                "confidence_level": c_level_str,
                "priority_score": round(mean_priority, 3),
                "priority_level": p_level_str,
                "trajectory": traj_classes,
                "explanation": explanation,
            })

        # Sort hotspots descending by priority score
        raw_hotspots.sort(key=lambda x: x["priority_score"], reverse=True)

        for i, item in enumerate(raw_hotspots):
            hotspot_id = f"HS-{i+1:03d}"
            record = HotspotRecord(
                rank=i + 1,
                hotspot_id=hotspot_id,
                centroid_rc=item["centroid_rc"],
                bounding_box=item["bounding_box"],
                area_pixels=item["area_pixels"],
                area_hectares=item["area_hectares"],
                area_km2=item["area_km2"],
                previous_class=item["previous_class"],
                current_class=item["current_class"],
                transition=item["transition"],
                transition_category=item["transition_category"],
                first_detected=item["first_detected"],
                last_confirmed=item["last_confirmed"],
                observations=item["observations"],
                persistence_score=item["persistence_score"],
                persistence_level=item["persistence_level"],
                change_magnitude=item["change_magnitude"],
                sar_evidence=item["sar_evidence"],
                optical_evidence=item["optical_evidence"],
                sensor_agreement=item["sensor_agreement"],
                sensor_attribution=item["sensor_attribution"],
                confidence=item["confidence"],
                confidence_level=item["confidence_level"],
                priority_score=item["priority_score"],
                priority_level=item["priority_level"],
                trajectory=item["trajectory"],
                explanation=item["explanation"],
            )
            hotspots.append(record)

        # Priority summary
        total_pixels = h * w
        summary = {
            "total_hotspots_detected": len(hotspots),
            "critical_hotspots": sum(1 for h in hotspots if h.priority_level == "CRITICAL"),
            "high_hotspots": sum(1 for h in hotspots if h.priority_level == "HIGH"),
            "medium_hotspots": sum(1 for h in hotspots if h.priority_level == "MEDIUM"),
            "low_hotspots": sum(1 for h in hotspots if h.priority_level == "LOW"),
            "total_changed_area_hectares": round(float(np.sum(trajectory.changed_mask) * pixel_area_ha), 2),
            "total_changed_area_km2": round(float(np.sum(trajectory.changed_mask) * pixel_area_km2), 4),
        }

        return ProvenanceResult(
            priority_score_map=priority_score_map,
            priority_level_map=priority_level_map,
            priority_label_map=priority_label_map,
            hotspots=hotspots,
            hotspot_label_matrix=labeled_regions,
            summary=summary,
        )


def _generate_explanation(
    prev_class: str,
    curr_class: str,
    transition: str,
    first_date: str,
    latest_date: str,
    persistence_score: float,
    persistence_level: str,
    sar_evidence: float,
    optical_evidence: float,
    sensor_agreement: float,
    confidence_score: float,
    priority_level: str,
    area_ha: float,
) -> str:
    """Generates a concise, evidence-grounded scientific provenance explanation."""
    sar_desc = "Strong" if sar_evidence >= 0.6 else ("Moderate" if sar_evidence >= 0.35 else "Subtle")
    opt_desc = "Strong" if optical_evidence >= 0.6 else ("Moderate" if optical_evidence >= 0.35 else "Subtle")
    agree_desc = "High" if sensor_agreement >= 0.7 else ("Moderate" if sensor_agreement >= 0.4 else "Low")

    text = (
        f"Verified transition from {prev_class} to {curr_class} spanning {area_ha:.1f} ha. "
        f"First detected on {first_date} and confirmed at {latest_date} with {persistence_level.lower()} "
        f"temporal stability (persistence score {persistence_score:.2f}). "
        f"Cross-sensor validation shows {sar_desc.lower()} SAR backscatter shift (score: {sar_evidence:.2f}) "
        f"and {opt_desc.lower()} Optical spectral shift (score: {optical_evidence:.2f}) with {agree_desc.lower()} "
        f"cross-sensor agreement ({sensor_agreement:.2f}). Overall confidence is {confidence_score*100:.1f}% "
        f"yielding a {priority_level} early-warning investigation priority."
    )
    return text


def colorize_priority_map(priority_level_map: np.ndarray) -> np.ndarray:
    """
    Renders priority map to RGB uint8 image.
    """
    h, w = priority_level_map.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for p_id, meta in PRIORITY_LEVELS.items():
        mask = priority_level_map == p_id
        rgb[mask] = meta["rgb"]
    return rgb
