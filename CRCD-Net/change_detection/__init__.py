"""Change detection package for CRCD-Net.

This package contains the Person C module responsible for change detection,
statistics, visualization, and the public integration interface.
"""

from .interfaces import ChangeDetectionResult, compare

__all__ = ["ChangeDetectionResult", "compare"]
