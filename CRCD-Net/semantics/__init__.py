"""
CRCD-Net Semantic Land Cover Classification Package.
"""

from semantics.land_cover_classifier import (
    CLASS_COLORS,
    CLASS_NAMES,
    LAND_COVER_CLASSES,
    LandCoverClassifier,
    SemanticClassificationResult,
    colorize_class_map,
    compute_sar_features,
    compute_spectral_indices,
)

__all__ = [
    "LAND_COVER_CLASSES",
    "CLASS_NAMES",
    "CLASS_COLORS",
    "LandCoverClassifier",
    "SemanticClassificationResult",
    "colorize_class_map",
    "compute_spectral_indices",
    "compute_sar_features",
]
