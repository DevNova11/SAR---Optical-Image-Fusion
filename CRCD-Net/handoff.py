"""The Person B / Person C entry point. One function, one stable signature.

    s1_date1, s2_date1, s1_date2, s2_date2 = get_training_pair(aoi, date_1, date_2, aoi_name)

Uses cached GeoTIFFs in `data_dir` when present (the normal demo path — no
GEE call, no auth needed at demo time). Only falls back to a live
`gee_data_collection.collect_pair()` call when a file is missing, which
requires `ee.Initialize()` to already be authenticated.
"""
from __future__ import annotations

import os

import numpy as np

from local_preprocessing import build_date_stack


def cache_paths(aoi_name: str, date: str, data_dir: str) -> tuple[str, str]:
    """(s1_path, s2_path) for one aoi_name/date under data_dir. Public so callers
    (e.g. data/temporal_dataset.py) can check/build cache paths without reaching
    into a private helper."""
    return (
        os.path.join(data_dir, f"{aoi_name}_{date}_S1.tif"),
        os.path.join(data_dir, f"{aoi_name}_{date}_S2.tif"),
    )


def get_training_pair(
    aoi: "ee.Geometry | None",
    date_1: str,
    date_2: str,
    aoi_name: str,
    data_dir: str = "data",
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (s1_date1, s2_date1, s1_date2, s2_date2) numpy arrays — see DATA_CONTRACT.md."""
    s1_p1, s2_p1 = cache_paths(aoi_name, date_1, data_dir)
    s1_p2, s2_p2 = cache_paths(aoi_name, date_2, data_dir)
    have_cache = use_cache and all(os.path.exists(p) for p in (s1_p1, s2_p1, s1_p2, s2_p2))

    if not have_cache:
        if aoi is None:
            raise FileNotFoundError(
                f"No cached GeoTIFFs for aoi_name={aoi_name!r} dates={date_1},{date_2} in {data_dir}, "
                "and no `aoi` was given to fall back to a live GEE pull."
            )
        import gee_data_collection as gdc
        gdc.collect_pair(aoi, date_1, date_2, aoi_name, out_dir=data_dir)

    s1_date1, s2_date1 = build_date_stack(s1_p1, s2_p1)
    s1_date2, s2_date2 = build_date_stack(s1_p2, s2_p2)
    return s1_date1, s2_date1, s1_date2, s2_date2


def get_temporal_series(
    aoi: "ee.Geometry | None",
    dates: list[str],
    aoi_name: str,
    data_dir: str = "data",
    use_cache: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Returns a list of (s1_image, s2_image) pairs, one per date in `dates`.

    Same cache-first/live-GEE-fallback contract as get_training_pair(), just
    for N dates instead of 2 -- used by the multi-temporal provenance pipeline.
    """
    series = []
    for d in dates:
        s1_p, s2_p = cache_paths(aoi_name, d, data_dir)
        if not (use_cache and os.path.exists(s1_p) and os.path.exists(s2_p)):
            if aoi is None:
                raise FileNotFoundError(
                    f"Missing cached GeoTIFFs for aoi_name={aoi_name!r} date={d} in {data_dir}"
                )
            import gee_data_collection as gdc
            gdc.collect_date(aoi, d, aoi_name, out_dir=data_dir)
        s1_img, s2_img = build_date_stack(s1_p, s2_p)
        series.append((s1_img, s2_img))
    return series
