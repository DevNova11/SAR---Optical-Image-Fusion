"""End-to-end CRCD-Net pipeline: data -> fusion -> change detection.

Wires together Person A's handoff.get_training_pair(), Person B's
fusion.baseline.fuse(), and Person C's change_detection.compare() into a
single call. This is the function the Streamlit demo (or anything else)
should call rather than re-gluing the three modules by hand.
"""
from __future__ import annotations

from typing import Optional

from change_detection import ChangeDetectionResult, compare
from fusion.baseline import fuse
from handoff import get_training_pair


def run_pipeline(
    date_1: str,
    date_2: str,
    aoi_name: str,
    aoi: "ee.Geometry | None" = None,
    data_dir: str = "data",
    pixel_size: Optional[float] = 10.0,
    fusion_method: str = "weighted",
) -> ChangeDetectionResult:
    """Runs the full data -> fusion -> change-detection pipeline for one AOI/date pair.

    Uses cached GeoTIFFs in `data_dir` when present (the 3 demo AOIs need no
    GEE call). Pass `aoi` (an ee.Geometry) to fall back to a live GEE pull
    for anything not already cached. See DATA_CONTRACT.md for the array
    shapes/dtypes this depends on.
    """
    s1_1, s2_1, s1_2, s2_2 = get_training_pair(aoi, date_1, date_2, aoi_name, data_dir=data_dir)

    fused_1 = fuse(s1_1, s2_1, method=fusion_method, data_layout="HWC")
    fused_2 = fuse(s1_2, s2_2, method=fusion_method, data_layout="HWC")

    metadata = {"aoi": aoi_name, "date1": date_1, "date2": date_2, "pixel_size": pixel_size}
    return compare(fused_1, fused_2, metadata=metadata)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("usage: python run_pipeline.py <aoi_name> <date_1> <date_2>")
        print("  e.g.: python run_pipeline.py chimakurthy_quarry 2018-02-03 2023-01-28")
        sys.exit(2)

    aoi_name, date_1, date_2 = sys.argv[1], sys.argv[2], sys.argv[3]
    result = run_pipeline(date_1, date_2, aoi_name)
    print(f"change_map: {result.change_map.shape} {result.change_map.dtype}")
    print(f"difference_map: {result.difference_map.shape} {result.difference_map.dtype}")
    print("statistics:")
    for k, v in result.statistics.items():
        print(f"  {k}: {v}")
