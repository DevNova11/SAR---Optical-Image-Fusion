"""Pretrained land-cover classification via Google Dynamic World.

Dynamic World (GOOGLE/DYNAMICWORLD/V1) is a pretrained deep learning model
published by Google/WRI on Earth Engine -- no training required, free under
the same GEE access already used for Sentinel-1/2 in gee_data_collection.py.
It gives per-pixel land-cover class probabilities (including `built` and
`trees`) for Sentinel-2 scenes. We use the built/trees probability shift
between two dates as a real, model-backed urbanization/deforestation signal,
instead of a hand-tuned magnitude threshold on the fused difference map.

Per Google's guidance, Dynamic World probabilities are per-scene and should
not be composited/averaged over a wide window like S2 reflectance is --
that blurs the land-cover signal. We instead pick the single scene nearest
the target date within the window.
"""
from __future__ import annotations

import datetime as dt
import json
import os

import ee

from gee_data_collection import DEFAULT_WINDOW_DAYS, date_window, init

DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
DW_BANDS = ["built", "trees"]


def build_dw_snapshot(
    aoi: "ee.Geometry", target_date: str, window_days: int = DEFAULT_WINDOW_DAYS, scale: int = 10
) -> "ee.Image":
    """The nearest-to-`target_date` Dynamic World scene that actually has coverage over `aoi`.

    A single scene can be cloud-masked over a small AOI even when the wider
    tile has data (seen in practice: the literal nearest-by-date scene had 0
    valid pixels here due to a cloud directly over this AOI on that day).
    Walks candidates in ascending time-distance order and skips any with
    less than half the AOI's theoretical pixel count of valid data, instead
    of blindly trusting whichever scene is chronologically closest.
    """
    start, end = date_window(target_date, window_days)
    target_ms = ee.Date(target_date).millis()
    dw = ee.ImageCollection(DW_COLLECTION).filterBounds(aoi).filterDate(start, end)
    n = dw.size().getInfo()
    if n == 0:
        raise RuntimeError(
            f"No Dynamic World scenes for {start}..{end} over this AOI. Widen the window."
        )

    def _with_time_diff(img: "ee.Image") -> "ee.Image":
        diff = ee.Number(img.get("system:time_start")).subtract(target_ms).abs()
        return img.set("time_diff", diff)

    def _annotate(img: "ee.Image") -> "ee.Feature":
        diff = ee.Number(img.get("system:time_start")).subtract(target_ms).abs()
        count = ee.Number(
            img.select("built").reduceRegion(
                reducer=ee.Reducer.count(), geometry=aoi, scale=scale, maxPixels=1e9
            ).get("built")
        )
        return ee.Feature(None, {"index": img.get("system:index"), "time_diff": diff, "valid_count": count})

    # One round trip for every candidate's (time_diff, valid_count), not N sequential ones.
    candidates = ee.FeatureCollection(dw.map(_annotate)).getInfo()["features"]
    full_count = aoi.area(1).divide(scale * scale).getInfo()

    ranked = sorted(candidates, key=lambda f: f["properties"]["time_diff"])
    chosen_index = None
    for f in ranked:
        if f["properties"]["valid_count"] >= 0.5 * full_count:
            chosen_index = f["properties"]["index"]
            break
    if chosen_index is None:
        chosen_index = max(candidates, key=lambda f: f["properties"]["valid_count"])["properties"]["index"]

    chosen = ee.Image(dw.filter(ee.Filter.eq("system:index", chosen_index)).first())
    return chosen.select(DW_BANDS).clip(aoi)


def get_land_cover_stats(aoi: "ee.Geometry", date: str, window_days: int = DEFAULT_WINDOW_DAYS, scale: int = 10) -> dict:
    """Mean built/trees probability (0..1) over the AOI for the scene nearest `date`."""
    init()
    snapshot = build_dw_snapshot(aoi, date, window_days)
    means = snapshot.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=scale, maxPixels=1e9).getInfo()
    return {"date": date, "built": float(means["built"]), "trees": float(means["trees"])}


def get_land_cover_delta(
    aoi: "ee.Geometry",
    date_1: str,
    date_2: str,
    aoi_name: str,
    out_dir: str = "data",
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> dict:
    """Built/trees probability for both dates plus their delta; caches to `data/{aoi_name}_land_cover.json`."""
    stats_1 = get_land_cover_stats(aoi, date_1, window_days)
    stats_2 = get_land_cover_stats(aoi, date_2, window_days)
    result = {
        "aoi_name": aoi_name,
        "date_1": stats_1,
        "date_2": stats_2,
        "built_delta": stats_2["built"] - stats_1["built"],
        "trees_delta": stats_2["trees"] - stats_1["trees"],
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{aoi_name}_land_cover.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def load_land_cover_delta(aoi_name: str, data_dir: str = "data") -> dict | None:
    """Read a cached land-cover delta with no GEE call; None if not cached yet."""
    path = os.path.join(data_dir, f"{aoi_name}_land_cover.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def label_from_delta(delta: dict, built_threshold: float = 0.05, trees_threshold: float = -0.05) -> str:
    """A real, model-backed label -- not a magnitude heuristic on the fused difference."""
    built_up = delta["built_delta"] >= built_threshold
    tree_loss = delta["trees_delta"] <= trees_threshold
    if built_up and tree_loss:
        return "Likely urbanization with associated vegetation/tree loss"
    if built_up:
        return "Likely urbanization (built-up area increased)"
    if tree_loss:
        return "Likely deforestation/vegetation loss (tree cover decreased)"
    return "No strong built-up or tree-cover shift detected"
