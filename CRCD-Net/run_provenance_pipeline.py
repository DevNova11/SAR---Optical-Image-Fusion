"""
End-to-End Explainable Multi-Temporal Change Provenance & Early-Warning Pipeline.

Executes the full research pipeline:
1. Multi-Temporal Data Ingestion (T1, T2, ..., TN)
2. Reliability-Aware SAR-Optical Fusion & Modality Attribution
3. Pixel-Level Semantic Land-Cover Classification (Forest, Agri, Urban, Bare, Water)
4. Semantic Change Trajectory Modeling & Transition Matrix
5. Persistence Verification (Temporary, Emerging, Persistent, Confirmed)
6. Cross-Sensor Evidence & Uncertainty Estimation (SAR vs Optical)
7. Multi-Factor Early-Warning Priority Ranking (Critical, High, Medium, Low)
8. Structured JSON Provenance Record Generation
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.temporal_dataset import MultiTemporalDataset, get_default_temporal_dates
from fusion.reliability_fusion import fuse_reliability_aware
from provenance.confidence_engine import ConfidenceEngine, ConfidenceResult, colorize_confidence_map
from provenance.persistence_verifier import PersistenceVerifier, PersistenceResult, colorize_persistence_map
from provenance.provenance_engine import (
    HotspotRecord,
    ProvenanceEngine,
    ProvenanceResult,
    colorize_priority_map,
)
from provenance.sensor_evidence import SensorEvidenceEngine, SensorEvidenceResult, colorize_sensor_evidence_map
from provenance.trajectory_engine import ChangeTrajectoryEngine, TrajectoryResult, colorize_transition_map
from semantics.land_cover_classifier import (
    CLASS_NAMES,
    LAND_COVER_CLASSES,
    LandCoverClassifier,
    SemanticClassificationResult,
    colorize_class_map,
)


@dataclass
class MultiTemporalProvenancePipelineResult:
    aoi_name: str
    dates: List[str]
    fused_series: List[np.ndarray]             # List of N fused images (H, W, C)
    sar_weight_series: List[np.ndarray]        # List of N SAR weight maps (H, W)
    opt_weight_series: List[np.ndarray]        # List of N Optical weight maps (H, W)
    class_map_series: List[np.ndarray]         # List of N discrete class maps (H, W)
    probability_series: List[np.ndarray]       # List of N probability tensors (H, W, 5)
    trajectory: TrajectoryResult
    persistence: PersistenceResult
    sensor_evidence: SensorEvidenceResult
    confidence: ConfidenceResult
    provenance: ProvenanceResult
    
    # Pre-rendered visualization maps (RGB uint8)
    initial_class_rgb: np.ndarray
    final_class_rgb: np.ndarray
    transition_rgb: np.ndarray
    persistence_rgb: np.ndarray
    evidence_rgb: np.ndarray
    confidence_rgb: np.ndarray
    priority_rgb: np.ndarray
    
    # High-level metrics
    metrics: Dict[str, Any]

    def export_summary_json(self, out_path: Optional[str] = None) -> Dict[str, Any]:
        """Exports complete machine-readable provenance JSON."""
        data = {
            "aoi_name": self.aoi_name,
            "temporal_dates": self.dates,
            "observations_count": len(self.dates),
            "summary_metrics": self.metrics,
            "transition_summary": self.trajectory.transition_summary,
            "persistence_summary": self.persistence.category_summary,
            "evidence_summary": self.sensor_evidence.evidence_summary,
            "confidence_breakdown": self.confidence.component_breakdown,
            "priority_summary": self.provenance.summary,
            "hotspots": [h.to_dict() for h in self.provenance.hotspots],
        }
        if out_path:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
        return data


def run_provenance_pipeline(
    aoi_name: str,
    dates: Optional[List[str]] = None,
    aoi_geometry=None,
    data_dir: str = "data",
    pixel_size: float = 10.0,
    min_region_size: int = 15,
) -> MultiTemporalProvenancePipelineResult:
    """
    Executes the full multi-temporal semantic change provenance pipeline.
    """
    if dates is None or len(dates) == 0:
        dates = get_default_temporal_dates(aoi_name, n_dates=4)

    # 1. Load multi-temporal dataset
    dataset = MultiTemporalDataset(
        aoi_name=aoi_name,
        dates=dates,
        data_dir=data_dir,
        aoi_geometry=aoi_geometry,
        use_cache=True,
    )

    # Which dates are real satellite observations vs a sigmoidal interpolation
    # between the two real anchor dates (temporal_dataset.py's honest fallback
    # when a middle date has no cached/live GEE data). This must stay visible
    # downstream -- a hotspot's "First Seen" date is not a real observation if
    # it lands on one of these.
    interpolated_dates = [obs.date for obs in dataset.observations if obs.is_interpolated]
    real_dates = [obs.date for obs in dataset.observations if not obs.is_interpolated]

    n_obs = dataset.count
    fused_series = []
    sar_weights = []
    opt_weights = []
    class_maps = []
    prob_series = []
    margin_series = []

    classifier = LandCoverClassifier()

    # 2. Fuse and classify each temporal observation
    for i in range(n_obs):
        obs = dataset.get_observation(i)
        fused_img, w_sar, w_opt, _ = fuse_reliability_aware(
            obs.s1_image, obs.s2_image, data_layout="HWC"
        )
        fused_series.append(fused_img)
        sar_weights.append(w_sar)
        opt_weights.append(w_opt)

        sem_res = classifier.predict(obs.s1_image, obs.s2_image, fused_image=fused_img)
        class_maps.append(sem_res.class_map)
        prob_series.append(sem_res.probabilities)
        margin_series.append(sem_res.confidence_margin)

    # 3. Multi-temporal Change Trajectory
    traj_engine = ChangeTrajectoryEngine()
    trajectory = traj_engine.analyze_trajectories(class_maps, dates)

    # 4. Persistence Verification
    persist_verifier = PersistenceVerifier()
    persistence = persist_verifier.verify_persistence(
        trajectory.trajectory_matrix, dates, probability_series=prob_series
    )

    # 5. Sensor Evidence & Modality Attribution
    obs_t1 = dataset.get_observation(0)
    obs_tn = dataset.get_observation(-1)
    evidence_engine = SensorEvidenceEngine()
    evidence = evidence_engine.evaluate_evidence(
        obs_t1.s1_image,
        obs_t1.s2_image,
        obs_tn.s1_image,
        obs_tn.s2_image,
        changed_mask=trajectory.changed_mask,
    )

    # 6. Uncertainty & Confidence Estimation
    # Use final observation margin combined with trajectory persistence
    conf_engine = ConfidenceEngine()
    final_margin = margin_series[-1]
    confidence = conf_engine.compute_confidence(
        classifier_margin=final_margin,
        persistence_score=persistence.persistence_score_map,
        sensor_agreement=evidence.sensor_agreement_map,
        sar_evidence=evidence.sar_evidence_map,
        optical_evidence=evidence.optical_evidence_map,
        changed_mask=trajectory.changed_mask,
    )

    # 7. Change Provenance & Early-Warning Priority Ranking
    prov_engine = ProvenanceEngine(pixel_size_m=pixel_size)
    provenance = prov_engine.generate_provenance(
        trajectory=trajectory,
        persistence=persistence,
        evidence=evidence,
        confidence=confidence,
        dates=dates,
        min_region_pixels=min_region_size,
    )

    # 8. Render Visual RGB Maps
    initial_class_rgb = colorize_class_map(trajectory.initial_class_map)
    final_class_rgb = colorize_class_map(trajectory.final_class_map)
    transition_rgb = colorize_transition_map(trajectory.transition_label_map)
    persistence_rgb = colorize_persistence_map(persistence.persistence_level_map)
    evidence_rgb = colorize_sensor_evidence_map(evidence.attribution_id_map)
    confidence_rgb = colorize_confidence_map(confidence.confidence_level_map)
    priority_rgb = colorize_priority_map(provenance.priority_level_map)

    # High level overall metrics
    h, w = dataset.shape
    total_pixels = h * w
    changed_pixels = int(np.sum(trajectory.changed_mask))
    pixel_area_ha = (pixel_size * pixel_size) / 10000.0

    metrics = {
        "aoi": aoi_name,
        "total_observations": n_obs,
        "real_observation_dates": real_dates,
        "interpolated_observation_dates": interpolated_dates,
        "first_observation_date": dates[0],
        "final_observation_date": dates[-1],
        "total_area_hectares": round(total_pixels * pixel_area_ha, 2),
        "total_area_km2": round(total_pixels * (pixel_size * pixel_size) / 1e6, 4),
        "changed_area_hectares": round(changed_pixels * pixel_area_ha, 2),
        "changed_area_km2": round(changed_pixels * (pixel_size * pixel_size) / 1e6, 4),
        "change_percentage": round((changed_pixels / total_pixels) * 100.0, 2),
        "total_hotspots": len(provenance.hotspots),
        "critical_priority_hotspots": provenance.summary["critical_hotspots"],
        "high_priority_hotspots": provenance.summary["high_hotspots"],
        "mean_confidence": confidence.component_breakdown["mean_composite_confidence"],
        "mean_persistence": persistence.category_summary["Persistent"]["percentage"]
        + persistence.category_summary["Confirmed"]["percentage"],
    }

    return MultiTemporalProvenancePipelineResult(
        aoi_name=aoi_name,
        dates=dates,
        fused_series=fused_series,
        sar_weight_series=sar_weights,
        opt_weight_series=opt_weights,
        class_map_series=class_maps,
        probability_series=prob_series,
        trajectory=trajectory,
        persistence=persistence,
        sensor_evidence=evidence,
        confidence=confidence,
        provenance=provenance,
        initial_class_rgb=initial_class_rgb,
        final_class_rgb=final_class_rgb,
        transition_rgb=transition_rgb,
        persistence_rgb=persistence_rgb,
        evidence_rgb=evidence_rgb,
        confidence_rgb=confidence_rgb,
        priority_rgb=priority_rgb,
        metrics=metrics,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multi-Temporal Change Provenance Pipeline")
    parser.add_argument("aoi_name", nargs="?", default="chimakurthy_quarry", help="Area of interest name")
    parser.add_argument("--dates", nargs="+", default=None, help="List of observation dates")
    parser.add_argument("--out_json", default=None, help="Path to export provenance JSON report")
    args = parser.parse_args()

    print(f"Executing Multi-Temporal Change Provenance Pipeline for: {args.aoi_name}")
    res = run_provenance_pipeline(args.aoi_name, dates=args.dates)

    interpolated = set(res.metrics.get("interpolated_observation_dates", []))
    if interpolated:
        print(
            f"\n*** {len(interpolated)} of {res.metrics['total_observations']} observation dates are "
            f"INTERPOLATED, not real satellite data: {sorted(interpolated)} ***\n"
            "*** (sigmoidal blend between the real anchor dates -- no cached or live GEE data existed "
            "for these middle dates). 'First Seen' timestamps landing on these dates are estimates, "
            "not observations. ***"
        )

    print("\n--- Pipeline Execution Summary ---")
    for k, v in res.metrics.items():
        print(f"  {k}: {v}")

    print(f"\n--- Detected Change Hotspots (Top 5 of {len(res.provenance.hotspots)}) ---")
    for hs in res.provenance.hotspots[:5]:
        interp_flag = " [INTERPOLATED DATE, not a real observation]" if hs.first_detected in interpolated else ""
        print(f"  [{hs.hotspot_id}] Priority: {hs.priority_level} (Score: {hs.priority_score:.2f}) | {hs.transition}")
        print(f"         First Seen: {hs.first_detected}{interp_flag} | Persistence: {hs.persistence_level} ({hs.persistence_score:.2f})")
        print(f"         Evidence: {hs.sensor_attribution} (SAR: {hs.sar_evidence:.2f}, Opt: {hs.optical_evidence:.2f}) | Conf: {hs.confidence_level} ({hs.confidence*100:.1f}%)")
        print(f"         Explanation: {hs.explanation[:110]}...")

    if args.out_json:
        res.export_summary_json(args.out_json)
        print(f"\nSaved structured provenance report to: {args.out_json}")
