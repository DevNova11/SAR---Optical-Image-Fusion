"""GEE data acquisition for CRCD-Net.

Pulls cloud-masked Sentinel-2 (B2,B3,B4,B8) and speckle-filtered Sentinel-1
(VV,VH), reprojected onto a shared, AOI-appropriate UTM grid, and exports
each as a GeoTIFF. See DATA_CONTRACT.md for the exact band order/dtype this
guarantees downstream.

Two supported auth paths, both handled by init() below -- this module still
never calls ee.Authenticate() itself:

  1. Local development: an already-authenticated `earthengine-api` session
     via `ee.Authenticate()` (interactive, run once yourself), which caches
     a credential ee.Initialize() picks up with no extra step.
  2. Headless/hosted (e.g. Streamlit Community Cloud): a GEE service-account
     key stored in Streamlit secrets (st.secrets["gee_service_account"]),
     never on disk. init() prefers this path when it's present, and falls
     back to the interactive-credential path otherwise -- so the same code
     works unchanged on a laptop and on a server with no browser.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os

import ee
import geemap
import rasterio

EE_PROJECT = "autonomous-star-504310-c1"


def _service_account_info_from_secrets() -> dict | None:
    """The GEE service-account key dict from Streamlit secrets, or None.

    Deliberately soft-fails (returns None) rather than raising: this module
    is also used outside Streamlit (run_pipeline.py, tests, scripts), where
    `streamlit` may not even be installed, and where falling through to the
    interactive-credential path below is the correct behavior anyway.
    """
    try:
        import streamlit as st
    except ImportError:
        return None
    try:
        if "gee_service_account" not in st.secrets:
            return None
        return dict(st.secrets["gee_service_account"])
    except Exception:
        # No secrets.toml at all (e.g. local run without one configured)
        # raises rather than just missing a key -- same fallback either way.
        return None

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUD_COLLECTION = "COPERNICUS/S2_CLOUD_PROBABILITY"
S1_COLLECTION = "COPERNICUS/S1_GRD"

S2_BANDS = ["B2", "B3", "B4", "B8"]  # Blue, Green, Red, NIR — the data-contract order
S1_BANDS = ["VV", "VH"]

NODATA = -9999.0
DEFAULT_SCALE = 10  # meters
DEFAULT_WINDOW_DAYS = 45  # composite window on each side of a single requested date


def init():
    """Initialize EE. Never calls ee.Authenticate().

    Prefers a service-account key from Streamlit secrets (headless/hosted
    path); falls back to the interactive-auth cached credential (local dev
    path) when no service account is configured.
    """
    service_account_info = _service_account_info_from_secrets()
    if service_account_info is not None:
        try:
            credentials = ee.ServiceAccountCredentials(
                service_account_info["client_email"],
                key_data=json.dumps(service_account_info),
            )
            ee.Initialize(credentials, project=service_account_info.get("project_id", EE_PROJECT))
            return
        except Exception as exc:
            raise RuntimeError(
                "ee.Initialize() with the service account from "
                "st.secrets['gee_service_account'] failed -- check the key is "
                "a valid GEE-enabled service account and every field from the "
                "downloaded JSON was copied into secrets.toml."
            ) from exc

    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception as exc:
        raise RuntimeError(
            "ee.Initialize() failed — no cached credential found and no "
            "st.secrets['gee_service_account'] is configured. For local dev, "
            "run `ee.Authenticate()` interactively yourself (this module will "
            "not do it for you); for a hosted deploy, add a GEE service "
            "account key to Streamlit secrets."
        ) from exc


def get_utm_epsg(aoi: "ee.Geometry") -> str:
    """Auto-detect the UTM EPSG code for an AOI from its centroid."""
    lon, lat = aoi.centroid(1).coordinates().getInfo()
    zone = int(math.floor((lon + 180) / 6) + 1)
    return f"EPSG:{32600 + zone}" if lat >= 0 else f"EPSG:{32700 + zone}"


def date_window(center_date: str, days: int = DEFAULT_WINDOW_DAYS) -> tuple[str, str]:
    """A ±`days` acquisition window around a single YYYY-MM-DD date."""
    center = dt.date.fromisoformat(center_date)
    start = center - dt.timedelta(days=days)
    end = center + dt.timedelta(days=days)
    return start.isoformat(), end.isoformat()


def mask_s2_clouds(image: "ee.Image") -> "ee.Image":
    clouds = ee.Image(image.get("cloud_mask")).select("probability")
    is_not_cloud = clouds.lt(40)
    scl = image.select("SCL")
    is_not_shadow_or_cirrus = scl.neq(3).And(scl.neq(10))  # 3=shadow, 10=cirrus
    return (
        image.updateMask(is_not_cloud.And(is_not_shadow_or_cirrus))
        .divide(10000)
        .copyProperties(image, ["system:time_start"])
    )


def build_s2_composite(aoi: "ee.Geometry", start: str, end: str) -> "ee.Image":
    """Cloud-masked, reflectance-scaled S2 median composite, S2_BANDS only."""
    s2_sr = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
    )
    s2_clouds = ee.ImageCollection(S2_CLOUD_COLLECTION).filterBounds(aoi).filterDate(start, end)
    s2_joined = ee.Join.saveFirst("cloud_mask").apply(
        primary=s2_sr,
        secondary=s2_clouds,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )
    s2_clean = ee.ImageCollection(s2_joined).map(mask_s2_clouds)
    if s2_clean.size().getInfo() == 0:
        raise RuntimeError(
            f"No cloud-free Sentinel-2 scenes for {start}..{end} over this AOI. "
            "Widen the window or try a different date (see suggest_best_dates)."
        )
    return s2_clean.select(S2_BANDS).median().clip(aoi)


def _to_natural(img: "ee.Image") -> "ee.Image":
    return ee.Image(10.0).pow(img.divide(10.0))


def _to_db(img: "ee.Image") -> "ee.Image":
    return ee.Image(img).log10().multiply(10.0)


def speckle_filter(image: "ee.Image") -> "ee.Image":
    smoothed = _to_natural(image).focalMean(7, "square", "pixels")
    return _to_db(smoothed)


def build_s1_composite(aoi: "ee.Geometry", start: str, end: str) -> "ee.Image":
    """Speckle-filtered S1 median composite, S1_BANDS only, single orbit pass."""
    s1_base = (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    n_asc = s1_base.filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING")).size().getInfo()
    n_desc = s1_base.filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING")).size().getInfo()
    if n_asc == 0 and n_desc == 0:
        raise RuntimeError(
            f"No Sentinel-1 IW scenes for {start}..{end} over this AOI. "
            "Widen the date window or check the AOI is over land."
        )
    orbit_pass = "ASCENDING" if n_asc >= n_desc else "DESCENDING"
    s1 = s1_base.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
    s1_median = s1.select(S1_BANDS).median().clip(aoi)
    return speckle_filter(s1_median).rename(S1_BANDS)


def _export_geotiff(image: "ee.Image", aoi: "ee.Geometry", crs: str, scale: int, out_path: str) -> str:
    """Synchronous local download (small AOIs only) with an explicit nodata tag."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    filled = image.unmask(NODATA)
    geemap.ee_export_image(filled, filename=out_path, scale=scale, crs=crs, region=aoi, file_per_band=False)
    with rasterio.open(out_path, "r+") as dst:
        dst.nodata = NODATA
    return out_path


def collect_date(
    aoi: "ee.Geometry",
    date: str,
    aoi_name: str,
    out_dir: str = "data",
    crs: str | None = None,
    scale: int = DEFAULT_SCALE,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> dict:
    """Pull S1+S2 for a single date over one AOI, export 2 GeoTIFFs, return their paths.

    Returns {'s1': path, 's2': path, 'crs': crs}. Shared by collect_pair() (two dates)
    and handoff.get_temporal_series() (N dates), so there's exactly one place that
    knows how to export one date's imagery.
    """
    init()
    crs = crs or get_utm_epsg(aoi)
    start, end = date_window(date, window_days)
    s2 = build_s2_composite(aoi, start, end).reproject(crs=crs, scale=scale)
    s1 = build_s1_composite(aoi, start, end).reproject(crs=crs, scale=scale)
    s2_path = os.path.join(out_dir, f"{aoi_name}_{date}_S2.tif")
    s1_path = os.path.join(out_dir, f"{aoi_name}_{date}_S1.tif")
    return {
        "s2": _export_geotiff(s2, aoi, crs, scale, s2_path),
        "s1": _export_geotiff(s1, aoi, crs, scale, s1_path),
        "crs": crs,
    }


def collect_pair(
    aoi: "ee.Geometry",
    date_1: str,
    date_2: str,
    aoi_name: str,
    out_dir: str = "data",
    crs: str | None = None,
    scale: int = DEFAULT_SCALE,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> dict:
    """Pull S1+S2 for two dates over one AOI, export 4 GeoTIFFs, return their paths.

    Returns {'s1_date1':path, 's2_date1':path, 's1_date2':path, 's2_date2':path, 'crs':crs}.
    """
    crs = crs or get_utm_epsg(aoi)
    paths = {}
    for key, date in (("date1", date_1), ("date2", date_2)):
        one = collect_date(aoi, date, aoi_name, out_dir=out_dir, crs=crs, scale=scale, window_days=window_days)
        paths[f"s2_{key}"] = one["s2"]
        paths[f"s1_{key}"] = one["s1"]
    paths["crs"] = crs
    return paths


def suggest_best_dates(aoi: "ee.Geometry", year_range: range, month_range: tuple[int, int] = (1, 3)) -> list[str]:
    """Lowest-cloud S2 acquisition date per year, restricted to `month_range` (dry season default: Jan-Mar).

    Returns one 'YYYY-MM-DD' string per year in `year_range` that has at least
    one scene in that window (years with none are skipped, not raised on).
    """
    init()
    start_month, end_month = month_range
    dates = []
    for year in year_range:
        start = f"{year}-{start_month:02d}-01"
        end_year, end_month_adj = (year, end_month + 1) if end_month < 12 else (year + 1, 1)
        end = f"{end_year}-{end_month_adj:02d}-01"
        s2 = (
            ee.ImageCollection(S2_COLLECTION)
            .filterBounds(aoi)
            .filterDate(start, end)
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )
        if s2.size().getInfo() == 0:
            continue
        best = ee.Image(s2.first())
        ts = best.get("system:time_start").getInfo()
        cloud_pct = best.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
        date_str = dt.datetime.utcfromtimestamp(ts / 1000).date().isoformat()
        print(f"  {year}: {date_str}  ({cloud_pct:.1f}% cloud)")
        dates.append(date_str)
    return dates
