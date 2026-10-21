"""Offline tests for the Dynamic World weak-label logic (no GEE needed)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semantics.dw_labels import DW_CLASS_BANDS, IGNORE_INDEX, merge_probs, probs_to_labels


def _one_hot(name, p=1.0):
    v = np.zeros((1, 1, 9), np.float32)
    v[0, 0, DW_CLASS_BANDS.index(name)] = p
    return v


def test_every_dw_class_maps_to_expected_crcd_class():
    expected = {"trees": 0, "shrub_and_scrub": 0, "grass": 1, "crops": 1, "flooded_vegetation": 1,
                "built": 2, "bare": 3, "snow_and_ice": 3, "water": 4}
    for name, cls in expected.items():
        labels, _ = probs_to_labels(_one_hot(name), min_prob=0.5)
        assert labels[0, 0] == cls, name


def test_merged_probabilities_are_summed_not_argmaxed():
    # trees 0.3 + shrub 0.3 = Forest 0.6 beats crops 0.4, even though crops is the single top DW class
    v = np.zeros((1, 1, 9), np.float32)
    v[0, 0, DW_CLASS_BANDS.index("trees")] = 0.3
    v[0, 0, DW_CLASS_BANDS.index("shrub_and_scrub")] = 0.3
    v[0, 0, DW_CLASS_BANDS.index("crops")] = 0.4
    assert merge_probs(v)[0, 0].argmax() == 0
    assert probs_to_labels(v, min_prob=0.5)[0][0, 0] == 0


def test_low_confidence_nodata_and_invalid_are_ignored():
    low = np.full((1, 1, 9), 1 / 9, np.float32)
    assert probs_to_labels(low, min_prob=0.5)[0][0, 0] == IGNORE_INDEX
    assert probs_to_labels(np.zeros((1, 1, 9), np.float32))[0][0, 0] == IGNORE_INDEX  # DW had no data
    assert probs_to_labels(_one_hot("built"), valid=np.zeros((1, 1), bool))[0][0, 0] == IGNORE_INDEX
