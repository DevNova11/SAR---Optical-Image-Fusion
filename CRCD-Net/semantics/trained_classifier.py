"""Drop-in trained replacement for semantics.land_cover_classifier.LandCoverClassifier.

    clf = get_classifier()                 # trained if weights exist, else the rule-based fallback
    res = clf.predict(s1_hwc, s2_hwc)      # -> SemanticClassificationResult (same fields/dtypes)

Weak-supervision caveat: the model is trained on Dynamic World labels (see
semantics/dw_labels.py). Its probabilities are softmax outputs of that
distillation, not calibrated probabilities of ground truth.

torch is imported lazily so the dashboard/backend still run without it.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from semantics.land_cover_classifier import CLASS_NAMES, LandCoverClassifier, SemanticClassificationResult

NODATA = -9999.0
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "classifier_dw.pt")


class TrainedLandCoverClassifier:
    def __init__(self, path: str = DEFAULT_WEIGHTS, device: str = "cpu"):
        import torch

        from semantics.train_classifier import build_model

        ck = torch.load(path, map_location=device)
        self.norm, self.meta, self.device = ck["norm"], ck.get("meta", {}), device
        self.model = build_model(ck["config"]["base"]).to(device)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.num_classes = len(CLASS_NAMES)

    def predict(self, s1_image: np.ndarray, s2_image: np.ndarray, fused_image: Optional[np.ndarray] = None) -> SemanticClassificationResult:
        # fused_image is accepted for interface compatibility and intentionally ignored: the model is trained on raw S1+S2.
        from semantics.train_classifier import predict_probs

        valid = (
            (s1_image != NODATA).all(-1) & (s2_image != NODATA).all(-1)
            & np.isfinite(s1_image).all(-1) & np.isfinite(s2_image).all(-1)
        )
        probs = predict_probs(self.model, self.norm, s1_image, s2_image, valid, device=self.device).astype(np.float32)
        class_map = probs.argmax(-1).astype(np.int32)
        top2 = np.sort(probs, axis=-1)
        margin = (top2[..., -1] - top2[..., -2]).astype(np.float32)
        total = class_map.size
        counts = {n: int((class_map == i).sum()) for i, n in enumerate(CLASS_NAMES)}
        fractions = {n: round(c / total, 4) for n, c in counts.items()}
        return SemanticClassificationResult(class_map, probs, margin, counts, fractions)


def get_classifier(weights: Optional[str] = None, device: str = "cpu"):
    """Trained classifier if its weights exist, else the rule-based one. Use this instead of LandCoverClassifier()."""
    path = weights or DEFAULT_WEIGHTS
    if os.path.exists(path):
        try:
            return TrainedLandCoverClassifier(path, device)
        except ImportError:  # torch not installed here: fall back rather than break the dashboard
            pass
    return LandCoverClassifier()
