"""Per-pixel weak labels from Google's Dynamic World (a pretrained model).

IMPORTANT (say this in the report): these are WEAK labels from a pretrained
model, not human ground truth. A classifier trained on them can at best match
Dynamic World plus add SAR robustness (clouds/monsoon); never describe the
result as "accuracy vs ground truth".

On-disk format  data/{aoi}_{date}_DW.tif
    9 bands, uint8 = round(prob * 200), nodata = 255, in Dynamic World band order,
    on EXACTLY the same grid (CRS, transform, shape) as {aoi}_{date}_S2.tif.
    The 9 raw probabilities are stored (not the mapped labels) so the class
    mapping / confidence threshold can be revisited without re-downloading.
    Tags: dw_scene (the GEE scene id used), prob_scale (=200).

Class mapping (FROZEN for the team; see docs/ML_TRAINING_GUIDE.md section 3.3):
    DW class               -> CRCD-Net class (semantics.land_cover_classifier.LAND_COVER_CLASSES)
    trees, shrub_and_scrub -> 0 Forest
    grass, crops,
      flooded_vegetation   -> 1 Agriculture
    built                  -> 2 Urban
    bare, snow_and_ice     -> 3 Bare Land
    water                  -> 4 Water
Probabilities of merged DW classes are SUMMED (not argmax-then-map), so e.g.
Forest = P(trees) + P(shrub_and_scrub).
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

DW_CLASS_BANDS = [
    "water", "trees", "grass", "flooded_vegetation", "crops",
    "shrub_and_scrub", "built", "bare", "snow_and_ice",
]  # Dynamic World's own band order (class ids 0..8)
DW_TO_CRCD = {
    "water": 4, "trees": 0, "grass": 1, "flooded_vegetation": 1, "crops": 1,
    "shrub_and_scrub": 0, "built": 2, "bare": 3, "snow_and_ice": 3,
}
NUM_CLASSES = 5
IGNORE_INDEX = 255
PROB_SCALE = 200.0
DW_NODATA = 255
DEFAULT_MIN_PROB = 0.5  # 0.6 ignored ~60% of pixels on chimakurthy 2018 (DW is uncertain there); tune per experiment


def merge_probs(dw_probs: np.ndarray) -> np.ndarray:
    """(H, W, 9) DW probabilities -> (H, W, 5) CRCD-Net class probabilities (merged classes summed)."""
    if dw_probs.shape[-1] != len(DW_CLASS_BANDS):
        raise ValueError(f"expected 9 DW bands, got {dw_probs.shape}")
    out = np.zeros(dw_probs.shape[:-1] + (NUM_CLASSES,), dtype=np.float32)
    for i, name in enumerate(DW_CLASS_BANDS):
        out[..., DW_TO_CRCD[name]] += dw_probs[..., i]
    return out


def probs_to_labels(dw_probs: np.ndarray, valid: Optional[np.ndarray] = None, min_prob: float = DEFAULT_MIN_PROB):
    """-> (labels (H,W) uint8 in 0..4 with 255=ignore, conf (H,W) float32 = top merged-class probability).

    A pixel is ignored (255) when DW has no data there, the S1/S2 pair is invalid
    there, or the top merged-class probability is below `min_prob`.
    """
    merged = merge_probs(dw_probs)
    total = merged.sum(-1)
    conf = merged.max(-1)
    labels = merged.argmax(-1).astype(np.uint8)
    ignore = (total < 0.5) | (conf < min_prob)  # total ~0 means DW had no data
    if valid is not None:
        ignore |= ~valid
    labels[ignore] = IGNORE_INDEX
    return labels, conf.astype(np.float32)


def dw_path(aoi: str, date: str, data_dir: str) -> str:
    return os.path.join(data_dir, f"{aoi}_{date}_DW.tif")


def load_dw_probs(aoi: str, date: str, data_dir: str = "data") -> Optional[np.ndarray]:
    """(H, W, 9) float32 probabilities, NaN where DW had no data; None if not exported yet."""
    import rasterio

    p = dw_path(aoi, date, data_dir)
    if not os.path.exists(p):
        return None
    with rasterio.open(p) as src:
        arr = src.read().astype(np.float32)  # (9, H, W)
    nodata = (arr == DW_NODATA).any(0)
    probs = np.moveaxis(arr / PROB_SCALE, 0, -1)
    probs[nodata] = np.nan
    return probs


def load_dw_labels(aoi: str, date: str, data_dir: str = "data", valid: Optional[np.ndarray] = None,
                   min_prob: float = DEFAULT_MIN_PROB) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    probs = load_dw_probs(aoi, date, data_dir)
    if probs is None:
        return None
    return probs_to_labels(np.nan_to_num(probs, nan=0.0), valid=valid, min_prob=min_prob)


# ------------------------------------------------------------------ GEE export
def export_dw_for_scene(aoi: str, date: str, data_dir: str = "data", overwrite: bool = False) -> str:
    """Export the Dynamic World class probabilities for one cached scene, aligned to its S2 grid.

    Needs an authenticated GEE session (gee_data_collection.init). Downloads on the
    scene's own CRS at 10 m, then reprojects onto the exact S2 pixel grid so label
    pixel (i, j) is the same ground location as S2 pixel (i, j).
    """
    import ee
    import geemap
    import rasterio
    from rasterio.warp import Resampling, reproject

    from gee_data_collection import init
    from land_cover import build_dw_snapshot

    out_path = dw_path(aoi, date, data_dir)
    if os.path.exists(out_path) and not overwrite:
        return out_path
    s2_path = os.path.join(data_dir, f"{aoi}_{date}_S2.tif")
    with rasterio.open(s2_path) as ref:
        ref_crs, ref_transform, ref_w, ref_h, bounds = ref.crs, ref.transform, ref.width, ref.height, ref.bounds

    init()
    region = ee.Geometry.Rectangle(
        [bounds.left, bounds.bottom, bounds.right, bounds.top], proj=ref_crs.to_string(), geodesic=False
    )
    snap = build_dw_snapshot(region, date, bands=DW_CLASS_BANDS)
    try:
        scene_id = ee.Image(snap).get("system:index").getInfo() or "unknown"
    except Exception:
        scene_id = "unknown"
    img = snap.multiply(PROB_SCALE).round().toUint8().unmask(DW_NODATA)

    tmp = out_path + ".download.tif"
    geemap.ee_export_image(img, filename=tmp, scale=10, crs=ref_crs.to_string(), region=region, file_per_band=False)

    with rasterio.open(tmp) as src:
        dst_arr = np.full((len(DW_CLASS_BANDS), ref_h, ref_w), DW_NODATA, dtype=np.uint8)
        for b in range(len(DW_CLASS_BANDS)):
            reproject(
                source=rasterio.band(src, b + 1), destination=dst_arr[b],
                src_transform=src.transform, src_crs=src.crs, src_nodata=DW_NODATA,
                dst_transform=ref_transform, dst_crs=ref_crs, dst_nodata=DW_NODATA,
                resampling=Resampling.nearest,
            )
    profile = {"driver": "GTiff", "dtype": "uint8", "count": len(DW_CLASS_BANDS), "height": ref_h, "width": ref_w,
               "crs": ref_crs, "transform": ref_transform, "nodata": DW_NODATA, "compress": "deflate"}
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(dst_arr)
        dst.update_tags(dw_scene=scene_id, prob_scale=str(int(PROB_SCALE)), bands=",".join(DW_CLASS_BANDS))
    os.remove(tmp)
    return out_path


def _export_all(data_dir: str = "data") -> None:
    """Export DW probabilities for every cached S1+S2 scene (skips existing, keeps going on errors)."""
    import time

    from fusion.train_fusion import list_scenes

    scenes = [(a, d) for a, ds in sorted(list_scenes(data_dir).items()) for d in ds]
    ok = fail = 0
    for i, (aoi, date) in enumerate(scenes, 1):
        t = time.time()
        try:
            export_dw_for_scene(aoi, date, data_dir)
            ok += 1
            print(f"[{i}/{len(scenes)}] OK   {aoi} {date} ({time.time() - t:.0f}s)", flush=True)
        except Exception as exc:  # network / no DW coverage: log and continue
            fail += 1
            print(f"[{i}/{len(scenes)}] FAIL {aoi} {date}: {str(exc)[:120]}", flush=True)
    print(f"done: {ok} ok, {fail} failed", flush=True)


if __name__ == "__main__":
    _export_all()
