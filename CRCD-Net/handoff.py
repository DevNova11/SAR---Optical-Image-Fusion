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

import gee_data_collection as gdc
from local_preprocessing import build_date_stack


def _paths(aoi_name: str, date: str, data_dir: str) -> tuple[str, str]:
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
    s1_p1, s2_p1 = _paths(aoi_name, date_1, data_dir)
    s1_p2, s2_p2 = _paths(aoi_name, date_2, data_dir)
    have_cache = use_cache and all(os.path.exists(p) for p in (s1_p1, s2_p1, s1_p2, s2_p2))

    if not have_cache:
        if aoi is None:
            raise FileNotFoundError(
                f"No cached GeoTIFFs for aoi_name={aoi_name!r} dates={date_1},{date_2} in {data_dir}, "
                "and no `aoi` was given to fall back to a live GEE pull."
            )
        gdc.collect_pair(aoi, date_1, date_2, aoi_name, out_dir=data_dir)

    s1_date1, s2_date1 = build_date_stack(s1_p1, s2_p1)
    s1_date2, s2_date2 = build_date_stack(s1_p2, s2_p2)
    return s1_date1, s2_date1, s1_date2, s2_date2
