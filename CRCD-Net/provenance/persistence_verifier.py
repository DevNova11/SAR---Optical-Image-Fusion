"""
Persistence Verification Engine for Multi-Temporal Change Detection.

Evaluates temporal trajectories across N observations to distinguish persistent
land-cover changes from transient noise, cloud artifacts, or seasonal oscillations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


PERSISTENCE_LEVELS = {
    0: {"name": "Temporary", "color": "#7f7f7f", "rgb": (127, 127, 127), "description": "Single-date transient anomaly or noise"},
    1: {"name": "Emerging", "color": "#f1c40f", "rgb": (241, 196, 15), "description": "Recently detected transition in latest observation"},
    2: {"name": "Persistent", "color": "#e67e22", "rgb": (230, 126, 34), "description": "Multi-temporal transition sustained across observations"},
    3: {"name": "Confirmed", "color": "#e74c3c", "rgb": (231, 76, 60), "description": "High-confidence permanent conversion with post-transition stability"},
}


@dataclass
class PersistenceResult:
    persistence_score_map: np.ndarray  # (H, W) float32 in [0.0, 1.0]
    persistence_level_map: np.ndarray  # (H, W) int32 {0: Temporary, 1: Emerging, 2: Persistent, 3: Confirmed}
    persistence_label_map: np.ndarray  # (H, W) str
    first_detected_step_map: np.ndarray  # (H, W) int32 step index (0..N-1)
    support_ratio_map: np.ndarray      # (H, W) float32 fraction of dates showing post-transition class
    category_summary: Dict[str, Dict[str, Union[int, float]]]


class PersistenceVerifier:
    """
    Evaluates temporal consistency and post-transition stability across multi-temporal steps.
    """

    def __init__(self, confirmed_threshold: float = 0.80, persistent_threshold: float = 0.60):
        self.confirmed_threshold = confirmed_threshold
        self.persistent_threshold = persistent_threshold

    def verify_persistence(
        self,
        trajectory_matrix: np.ndarray,  # Shape (N, H, W) of class IDs
        dates: List[str],               # N date strings
        probability_series: Optional[List[np.ndarray]] = None,  # Optional N arrays of shape (H, W, 5)
    ) -> PersistenceResult:
        """
        Calculates per-pixel persistence metrics from the trajectory matrix.
        """
        n_obs, h, w = trajectory_matrix.shape
        t_initial = trajectory_matrix[0]
        t_final = trajectory_matrix[-1]

        score_map = np.zeros((h, w), dtype=np.float32)
        level_map = np.zeros((h, w), dtype=np.int32)
        label_map = np.full((h, w), "Stable", dtype=object)
        first_step_map = np.zeros((h, w), dtype=np.int32)
        support_ratio_map = np.zeros((h, w), dtype=np.float32)

        changed_pixels = t_initial != t_final

        if n_obs == 2:
            # For 2-date observations: direct transition is marked as persistent or emerging
            score_map[changed_pixels] = 0.75
            level_map[changed_pixels] = 2  # Persistent
            label_map[changed_pixels] = "Persistent"
            first_step_map[changed_pixels] = 1
            support_ratio_map[changed_pixels] = 1.0
        else:
            # Multi-temporal trajectory (N >= 3)
            for y in range(h):
                for x in range(w):
                    c_init = t_initial[y, x]
                    c_fin = t_final[y, x]

                    if c_init == c_fin:
                        # Check if intermediate variation occurred
                        series = trajectory_matrix[:, y, x]
                        deviations = np.sum(series != c_init)
                        if deviations > 0:
                            # Reverted change -> Temporary / Seasonal
                            score_map[y, x] = 0.20
                            level_map[y, x] = 0
                            label_map[y, x] = "Temporary"
                            support_ratio_map[y, x] = float(deviations / n_obs)
                        continue

                    # Change occurred from T1 to TN
                    series = trajectory_matrix[:, y, x]
                    
                    # Find first step where class shifted from c_init
                    onset_idx = 1
                    for k in range(1, n_obs):
                        if series[k] != c_init:
                            onset_idx = k
                            break
                    first_step_map[y, x] = onset_idx

                    # Check how many subsequent steps after onset stay equal to c_fin
                    steps_after_onset = n_obs - onset_idx
                    matches_final = np.sum(series[onset_idx:] == c_fin)
                    post_stability = matches_final / float(steps_after_onset)

                    # Total support across entire series
                    support = np.sum(series == c_fin) / float(n_obs - 1)
                    support_ratio_map[y, x] = round(float(support), 3)

                    # Calculate continuous persistence score
                    # Score combines post-onset stability and total temporal support
                    if onset_idx == n_obs - 1:
                        # Emerging change (only visible in the very latest observation)
                        p_score = 0.50 * post_stability
                        p_level = 1  # Emerging
                        p_label = "Emerging"
                    elif post_stability >= 0.90 and support >= 0.60:
                        # Sustained confirmed transition
                        p_score = 0.85 + 0.15 * (support)
                        p_level = 3  # Confirmed
                        p_label = "Confirmed"
                    elif post_stability >= 0.60:
                        p_score = 0.60 + 0.25 * post_stability
                        p_level = 2  # Persistent
                        p_label = "Persistent"
                    else:
                        # Erratic / Oscillating
                        p_score = 0.30 * post_stability
                        p_level = 0  # Temporary
                        p_label = "Temporary"

                    score_map[y, x] = round(float(np.clip(p_score, 0.0, 1.0)), 4)
                    level_map[y, x] = p_level
                    label_map[y, x] = p_label

        # Category summary
        total_pixels = h * w
        summary = {}
        for lvl_id, meta in PERSISTENCE_LEVELS.items():
            cnt = int(np.sum(level_map == lvl_id))
            summary[meta["name"]] = {
                "count": cnt,
                "percentage": round(float(cnt / total_pixels) * 100.0, 2),
                "area_km2": round(cnt * (10.0 * 10.0) / 1e6, 4),
            }

        return PersistenceResult(
            persistence_score_map=score_map,
            persistence_level_map=level_map,
            persistence_label_map=label_map,
            first_detected_step_map=first_step_map,
            support_ratio_map=support_ratio_map,
            category_summary=summary,
        )


def colorize_persistence_map(level_map: np.ndarray) -> np.ndarray:
    """
    Renders persistence map to RGB uint8 array.
    """
    h, w = level_map.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for lvl_id, meta in PERSISTENCE_LEVELS.items():
        mask = level_map == lvl_id
        rgb[mask] = meta["rgb"]
    return rgb
