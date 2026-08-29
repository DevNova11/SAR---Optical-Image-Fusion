"""Validation gate for any pair of exported GeoTIFFs (S1+S2, same date, same AOI).

Everyone downstream depends on this pipeline, so a bad export must fail
loudly here rather than silently propagate. Run standalone:

    python validate_export.py data/aoi1_2024-02-01_S1.tif data/aoi1_2024-02-01_S2.tif

or import `validate_geotiff_pair` and call it from local_preprocessing.py.
"""
from __future__ import annotations

import sys

import numpy as np
import rasterio

NODATA = -9999.0
DEFAULT_MAX_NODATA_FRAC = 0.6


class DataValidationError(Exception):
    """Raised when an exported GeoTIFF pair fails a contract check. Never swallow this."""


def validate_geotiff_pair(path_a: str, path_b: str, max_nodata_frac: float = DEFAULT_MAX_NODATA_FRAC) -> dict:
    """Checks CRS, grid alignment, shape, and nodata fraction for one GeoTIFF pair.

    Raises DataValidationError with a specific reason on the first failure.
    Returns a stats dict (per-file nodata fraction, shape, crs) on success.
    """
    with rasterio.open(path_a) as ra, rasterio.open(path_b) as rb:
        if ra.crs != rb.crs:
            raise DataValidationError(f"CRS mismatch: {path_a}={ra.crs} vs {path_b}={rb.crs}")

        if not ra.transform.almost_equals(rb.transform, precision=1e-6):
            raise DataValidationError(
                f"Pixel grid mismatch (resolution/origin differ): {path_a}={ra.transform} "
                f"vs {path_b}={rb.transform}"
            )

        if (ra.width, ra.height) != (rb.width, rb.height):
            raise DataValidationError(
                f"Extent mismatch: {path_a}={ra.width}x{ra.height} vs {path_b}={rb.width}x{rb.height}"
            )

        stats = {"crs": str(ra.crs), "width": ra.width, "height": ra.height, "files": {}}
        for path, src in ((path_a, ra), (path_b, rb)):
            arr = src.read().astype("float64")
            nodata = src.nodata if src.nodata is not None else NODATA
            valid_mask = arr != nodata
            nodata_frac = 1.0 - valid_mask.mean()
            if nodata_frac > max_nodata_frac:
                raise DataValidationError(
                    f"{path}: {nodata_frac:.1%} nodata exceeds the {max_nodata_frac:.0%} threshold "
                    "— cloud masking likely failed to reduce cloud cover enough, or the AOI/date has no coverage."
                )
            if np.isnan(arr[valid_mask]).any():
                raise DataValidationError(f"{path}: unexpected NaNs inside the valid (non-nodata) region.")

            per_band = {}
            for b in range(arr.shape[0]):
                band = arr[b][valid_mask[b]]
                if band.size == 0:
                    raise DataValidationError(f"{path}: band {b + 1} is entirely nodata.")
                if np.all(band == band.flat[0]):
                    raise DataValidationError(
                        f"{path}: band {b + 1} is a constant value ({band.flat[0]}) — likely a broken export."
                    )
                per_band[b + 1] = {
                    "min": float(band.min()),
                    "max": float(band.max()),
                    "mean": float(band.mean()),
                }
            stats["files"][path] = {"nodata_frac": nodata_frac, "bands": per_band}

    return stats


def _print_stats(stats: dict) -> None:
    print(f"CRS: {stats['crs']}  |  grid: {stats['width']}x{stats['height']}")
    for path, file_stats in stats["files"].items():
        print(f"  {path}  (nodata: {file_stats['nodata_frac']:.1%})")
        for band, band_stats in file_stats["bands"].items():
            print(
                f"    band {band}: min={band_stats['min']:.4f} max={band_stats['max']:.4f} "
                f"mean={band_stats['mean']:.4f}"
            )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python validate_export.py <path_a.tif> <path_b.tif> [max_nodata_frac]")
        sys.exit(2)
    frac = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_MAX_NODATA_FRAC
    try:
        result = validate_geotiff_pair(sys.argv[1], sys.argv[2], max_nodata_frac=frac)
    except DataValidationError as e:
        print(f"VALIDATION FAILED: {e}")
        sys.exit(1)
    print("VALIDATION OK")
    _print_stats(result)
