"""
Comprehensive Automated Test Suite for CRCD-Net Explainable Provenance System.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from change_detection.postprocessing import postprocess_change_map, remove_small_holes, remove_small_objects
from data.temporal_dataset import MultiTemporalDataset, get_default_temporal_dates
from evaluation.benchmark_suite import BenchmarkSuite
from fusion.baseline import fuse
from fusion.reliability_fusion import compute_reliability_maps, fuse_reliability_aware
from provenance.confidence_engine import ConfidenceEngine
from provenance.persistence_verifier import PersistenceVerifier
from provenance.provenance_engine import ProvenanceEngine
from provenance.sensor_evidence import SensorEvidenceEngine
from provenance.trajectory_engine import ChangeTrajectoryEngine
from run_provenance_pipeline import run_provenance_pipeline
from semantics.land_cover_classifier import LandCoverClassifier, compute_sar_features, compute_spectral_indices


def test_postprocessing_skimage_compatibility():
    """Verify that skimage morphology functions run without keyword argument errors."""
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[20:40, 20:40] = 1
    arr[30, 30] = 0  # small hole
    arr[80:82, 80:82] = 1  # small object

    cleaned = postprocess_change_map(
        arr,
        config={
            "morphology_kernel_size": 3,
            "min_region_size": 10,
            "remove_small_holes": True,
            "remove_small_objects": True,
        }
    )
    assert cleaned.shape == (100, 100)
    assert cleaned.dtype == np.uint8
    # Small object should have been removed
    assert cleaned[80, 80] == 0
    # Small hole should have been filled
    assert cleaned[30, 30] == 1


def test_reliability_aware_fusion():
    """Verify that reliability fusion generates valid normalized output and weight maps that sum to 1."""
    h, w = 64, 64
    s1 = np.random.uniform(-25.0, 0.0, size=(h, w, 2)).astype(np.float32)
    s2 = np.random.uniform(0.0, 1.0, size=(h, w, 4)).astype(np.float32)

    fused, w_sar, w_opt, metrics = fuse_reliability_aware(s1, s2, data_layout="HWC")
    
    assert fused.shape == (h, w, 4)
    assert fused.min() >= 0.0 and fused.max() <= 1.0
    assert w_sar.shape == (h, w)
    assert w_opt.shape == (h, w)
    # Sum of weights must equal 1.0
    np.testing.assert_allclose(w_sar + w_opt, np.ones((h, w)), rtol=1e-5)
    assert 0.0 <= metrics["mean_sar_weight"] <= 1.0
    assert 0.0 <= metrics["mean_optical_weight"] <= 1.0


def test_semantic_land_cover_classifier():
    """Verify semantic classification output shapes, classes, and probability simplex."""
    h, w = 50, 50
    s1 = np.random.uniform(-20.0, -5.0, size=(h, w, 2)).astype(np.float32)
    s2 = np.random.uniform(0.1, 0.8, size=(h, w, 4)).astype(np.float32)

    classifier = LandCoverClassifier()
    res = classifier.predict(s1, s2)

    assert res.class_map.shape == (h, w)
    assert res.probabilities.shape == (h, w, 5)
    assert res.confidence_margin.shape == (h, w)
    # Probabilities must sum to 1.0 across channels
    prob_sums = np.sum(res.probabilities, axis=-1)
    np.testing.assert_allclose(prob_sums, np.ones((h, w)), rtol=1e-5)
    assert np.all(res.class_map >= 0) and np.all(res.class_map < 5)


def test_change_trajectory_and_persistence():
    """Verify trajectory mapping and persistence verification across temporal observations."""
    h, w = 30, 30
    dates = ["2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01"]

    # Trajectory 1: Stable Forest (0 -> 0 -> 0 -> 0)
    t_stable = np.zeros((4, h, w), dtype=np.int32)

    # Trajectory 2: Persistent Deforestation (0 -> 3 -> 3 -> 3)
    t_defor = np.zeros((4, h, w), dtype=np.int32)
    t_defor[1:, :15, :] = 3  # Class 3: Bare Land

    # Trajectory 3: Temporary Blip (0 -> 3 -> 0 -> 0)
    t_temp = np.zeros((4, h, w), dtype=np.int32)
    t_temp[1, 15:, :] = 3

    traj_engine = ChangeTrajectoryEngine()
    persist_verifier = PersistenceVerifier()

    # Test deforestation
    traj_res = traj_engine.analyze_trajectories([t_defor[k] for k in range(4)], dates)
    persist_res = persist_verifier.verify_persistence(traj_res.trajectory_matrix, dates)

    assert np.all(traj_res.changed_mask[:15, :] == True)
    assert np.all(traj_res.changed_mask[15:, :] == False)
    # Deforestation region should have high persistence score
    assert np.all(persist_res.persistence_score_map[:15, :] >= 0.70)
    assert np.all(persist_res.persistence_level_map[:15, :] >= 2)  # Persistent or Confirmed


def test_sensor_evidence_and_confidence():
    """Verify sensor evidence quantification and composite confidence bounds."""
    h, w = 40, 40
    s1_t1 = np.full((h, w, 2), -15.0, dtype=np.float32)
    s2_t1 = np.full((h, w, 4), 0.3, dtype=np.float32)
    
    # Induce strong SAR and Optical shift
    s1_tn = s1_t1.copy()
    s2_tn = s2_t1.copy()
    s1_tn[..., 0] += 10.0  # +10 dB shift
    s2_tn[..., 2] += 0.4   # +0.4 reflectance shift

    evidence_engine = SensorEvidenceEngine()
    ev_res = evidence_engine.evaluate_evidence(s1_t1, s2_t1, s1_tn, s2_tn)

    assert ev_res.sar_evidence_map.shape == (h, w)
    assert ev_res.optical_evidence_map.shape == (h, w)
    assert np.all(ev_res.sar_evidence_map > 0.5)
    assert np.all(ev_res.optical_evidence_map > 0.5)
    assert np.all(ev_res.attribution_id_map == 0)  # Both-sensor supported

    conf_engine = ConfidenceEngine()
    margin = np.full((h, w), 0.8, dtype=np.float32)
    persist_score = np.full((h, w), 0.9, dtype=np.float32)
    conf_res = conf_engine.compute_confidence(
        margin, persist_score, ev_res.sensor_agreement_map,
        ev_res.sar_evidence_map, ev_res.optical_evidence_map
    )
    assert np.all(conf_res.confidence_score_map >= 0.75)
    assert np.all(conf_res.confidence_level_map == 2)  # HIGH


def test_full_pipeline_cached_aois():
    """Verify end-to-end execution on real cached satellite datasets."""
    aois = ["chimakurthy_quarry", "bengaluru_sarjapur", "chennai_oragadam", "dubai_islands_v2"]
    for aoi in aois:
        res = run_provenance_pipeline(aoi)
        assert res.aoi_name == aoi
        assert len(res.dates) == 4
        assert len(res.fused_series) == 4
        assert len(res.class_map_series) == 4
        assert res.provenance.priority_score_map.shape == res.trajectory.changed_mask.shape
        assert res.metrics["total_hotspots"] == len(res.provenance.hotspots)
        
        # Verify JSON export
        summary = res.export_summary_json()
        assert summary["aoi_name"] == aoi
        assert "hotspots" in summary


def test_benchmark_suite_execution():
    """Verify that the benchmarking and ablation framework runs properly."""
    suite = BenchmarkSuite()
    res = suite.run_benchmark("bengaluru_sarjapur")
    assert res.baseline_a_metrics["changed_area_km2"] >= 0.0
    assert res.baseline_b_metrics["changed_area_km2"] >= 0.0
    assert res.proposed_method_metrics["total_hotspots_detected"] >= 0
    assert len(res.ablation_results) == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
