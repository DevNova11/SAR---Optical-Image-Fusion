"""
Multi-Temporal Semantic Change Trajectory Engine.

Tracks land-cover class transitions across temporal observations (T1 -> T2 -> ... -> TN),
identifies transition types, and computes transition severity and trajectory dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from semantics.land_cover_classifier import CLASS_NAMES, LAND_COVER_CLASSES


# Ecological and land-use transition severity matrix (0.0 to 1.0)
# (from_class_idx, to_class_idx) -> (transition_name, severity_weight, category)
TRANSITION_DEFINITIONS = {
    (0, 3): ("Forest -> Bare Land (Deforestation / Clearing)", 0.95, "DEFORESTATION"),
    (0, 1): ("Forest -> Agriculture (Vegetation Conversion)", 0.75, "AGRICULTURAL_EXPANSION"),
    (0, 2): ("Forest -> Urban (Direct Urban Encroachment)", 1.00, "URBANIZATION"),
    (1, 2): ("Agriculture -> Urban (Urban Expansion)", 0.85, "URBANIZATION"),
    (3, 2): ("Bare Land -> Urban (Built-up Construction)", 0.80, "URBANIZATION"),
    (4, 3): ("Water -> Bare Land (Water Body Desiccation / Reclamation)", 0.90, "WATER_LOSS"),
    (4, 2): ("Water -> Urban (Coastal/Wetland Reclamation)", 0.95, "URBANIZATION"),
    (1, 3): ("Agriculture -> Bare Land (Fallow / Land Degradation)", 0.60, "LAND_DEGRADATION"),
    (3, 0): ("Bare Land -> Forest (Reforestation / Regrowth)", 0.30, "REVEGETATION"),
    (3, 1): ("Bare Land -> Agriculture (Cultivation)", 0.40, "REVEGETATION"),
    (1, 0): ("Agriculture -> Forest (Afforestation / Regrowth)", 0.20, "REVEGETATION"),
    (3, 4): ("Bare Land -> Water (Inundation / Flooding)", 0.70, "INUNDATION"),
    (0, 4): ("Forest -> Water (Flooding / Submergence)", 0.85, "INUNDATION"),
    (2, 3): ("Urban -> Bare Land (Demolition / Redevelopment)", 0.50, "REDEVELOPMENT"),
    (2, 0): ("Urban -> Forest (Urban Greening)", 0.20, "GREENING"),
}


@dataclass
class TrajectoryResult:
    initial_class_map: np.ndarray  # (H, W) class at T1
    final_class_map: np.ndarray    # (H, W) class at TN
    transition_id_map: np.ndarray  # (H, W) unique transition code
    transition_label_map: np.ndarray  # (H, W) object array of transition names
    severity_map: np.ndarray       # (H, W) float32 [0.0, 1.0]
    changed_mask: np.ndarray       # (H, W) bool, True where T1 != TN
    trajectory_matrix: np.ndarray  # (N, H, W) full temporal sequence of class ids
    transition_summary: Dict[str, Dict[str, Union[int, float]]]


class ChangeTrajectoryEngine:
    """
    Evaluates multi-temporal classification sequences to extract change trajectories.
    """

    def __init__(self):
        self.transitions = TRANSITION_DEFINITIONS

    def analyze_trajectories(
        self,
        class_maps: List[np.ndarray],  # List of N class maps of shape (H, W)
        dates: List[str],              # List of N date strings
    ) -> TrajectoryResult:
        """
        Analyzes the trajectory of class predictions across N temporal observations.
        """
        n_obs = len(class_maps)
        if n_obs < 2:
            raise ValueError(f"At least 2 temporal observations required; got {n_obs}")

        h, w = class_maps[0].shape[:2]
        trajectory_matrix = np.stack(class_maps, axis=0)  # Shape (N, H, W)

        t_initial = trajectory_matrix[0]
        t_final = trajectory_matrix[-1]

        changed_mask = t_initial != t_final

        transition_id_map = np.zeros((h, w), dtype=np.int32)
        transition_label_map = np.full((h, w), "Stable", dtype=object)
        severity_map = np.zeros((h, w), dtype=np.float32)

        # Iterate over all possible from->to transitions
        for (c_from, c_to), (t_name, sev, cat) in self.transitions.items():
            mask = (t_initial == c_from) & (t_final == c_to)
            if np.any(mask):
                code = c_from * 10 + c_to
                transition_id_map[mask] = code
                transition_label_map[mask] = t_name
                severity_map[mask] = sev

        # Handle any unlisted transitions
        unlisted_mask = changed_mask & (transition_label_map == "Stable")
        if np.any(unlisted_mask):
            for y, x in zip(*np.where(unlisted_mask)):
                cf = int(t_initial[y, x])
                ct = int(t_final[y, x])
                name = f"{CLASS_NAMES[cf]} -> {CLASS_NAMES[ct]}"
                transition_label_map[y, x] = name
                severity_map[y, x] = 0.50
                transition_id_map[y, x] = cf * 10 + ct

        # Check for temporary oscillations: e.g. T1 == TN but intermediate date differed
        if n_obs >= 3:
            interm_changed = np.zeros((h, w), dtype=bool)
            for k in range(1, n_obs - 1):
                interm_changed |= (trajectory_matrix[k] != t_initial)
            
            seasonal_mask = (~changed_mask) & interm_changed
            if np.any(seasonal_mask):
                transition_label_map[seasonal_mask] = "Seasonal / Transient Variation"
                severity_map[seasonal_mask] = 0.15

        # Compute summary statistics
        total_pixels = h * w
        summary = {}
        unique_labels = np.unique(transition_label_map)
        for lbl in unique_labels:
            if lbl == "Stable":
                continue
            cnt = int(np.sum(transition_label_map == lbl))
            frac = float(cnt / total_pixels)
            summary[str(lbl)] = {
                "pixel_count": cnt,
                "percentage": round(frac * 100.0, 2),
                "area_km2": round(cnt * (10.0 * 10.0) / 1e6, 4),  # 10m pixel standard
            }

        return TrajectoryResult(
            initial_class_map=t_initial,
            final_class_map=t_final,
            transition_id_map=transition_id_map,
            transition_label_map=transition_label_map,
            severity_map=severity_map,
            changed_mask=changed_mask,
            trajectory_matrix=trajectory_matrix,
            transition_summary=summary,
        )


def colorize_transition_map(transition_label_map: np.ndarray) -> np.ndarray:
    """
    Renders transition map to distinct color-coded RGB image.
    """
    h, w = transition_label_map.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Stable: Charcoal dark grey
    rgb[transition_label_map == "Stable"] = [35, 40, 35]
    
    # Deforestation: Vivid Crimson Red
    mask_defor = np.char.startswith(transition_label_map.astype(str), "Forest -> Bare Land")
    rgb[mask_defor] = [220, 20, 60]

    # Agricultural conversion: Orange
    mask_agri = np.char.startswith(transition_label_map.astype(str), "Forest -> Agriculture")
    rgb[mask_agri] = [255, 140, 0]

    # Urbanization: Magenta / Purple
    mask_urb = (
        np.char.find(transition_label_map.astype(str), "-> Urban") >= 0
    )
    rgb[mask_urb] = [186, 85, 211]

    # Water loss: Cyan
    mask_water = (
        np.char.find(transition_label_map.astype(str), "Water ->") >= 0
    )
    rgb[mask_water] = [0, 206, 209]

    # Seasonal: Amber yellow
    mask_seas = (
        np.char.find(transition_label_map.astype(str), "Seasonal") >= 0
    )
    rgb[mask_seas] = [240, 230, 140]

    return rgb
