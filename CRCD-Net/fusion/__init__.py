"""
CRCD-Net Fusion Module.
Provides baseline fusion, reliability-aware fusion with sensor attribution,
and (optionally, if torch is installed) a deep PyTorch fusion model.
"""

from fusion.baseline import fuse
from fusion.reliability_fusion import compute_reliability_maps, fuse_reliability_aware

__all__ = [
    "fuse",
    "fuse_reliability_aware",
    "compute_reliability_maps",
    "fuse_deep",
    "CRCDNet",
    "train_fusion_model",
]

__version__ = "0.2.0"


def __getattr__(name):
    # fusion.baseline / fusion.reliability_fusion never need torch -- the
    # whole point of the baseline path is that it works without it. Only
    # touch fusion.deep_model (which imports torch) if someone actually
    # asks for one of its names, so `from fusion.baseline import fuse` and
    # `import fusion` keep working in environments without torch installed.
    if name in ("fuse_deep", "CRCDNet", "train_fusion_model"):
        from fusion.deep_model import CRCDNet, fuse_deep, train_fusion_model
        globals().update(
            fuse_deep=fuse_deep, CRCDNet=CRCDNet, train_fusion_model=train_fusion_model
        )
        return globals()[name]
    raise AttributeError(f"module 'fusion' has no attribute {name!r}")
