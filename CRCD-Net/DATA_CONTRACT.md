# CRCD-Net Data Contract (Person A → Person B / Person C)

Owner: Person A (data acquisition & preprocessing). Frozen for the hackathon —
change only after syncing with Person B (fusion) and Person C (change
detection + demo), since both build directly against this shape.

## 1. What you get, per date

Every call to `get_training_pair(aoi, date_1, date_2)` (in
[`handoff.py`](handoff.py)) returns exactly four numpy arrays:

```python
s1_date1, s2_date1, s1_date2, s2_date2 = get_training_pair(aoi, date_1, date_2)
```

S1 and S2 are **not** pre-stacked — this matches the fusion interface from
`03_DESIGN_DOC.md` (`fuse(s1_image, s2_image)`), which takes the two
modalities separately.

| Array       | Shape      | Bands (in order)              | dtype     | Value range |
|-------------|------------|--------------------------------|-----------|-------------|
| `s1_dateN`  | `(H, W, 2)` | `[VV, VH]`                     | `float32` | dB, typically `-25..0`; `NODATA = -9999.0` |
| `s2_dateN`  | `(H, W, 4)` | `[B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)]` | `float32` | reflectance `0..1`; `NODATA = -9999.0` |

**Channel axis is last** (`H, W, C`), not `C, H, W`. Transpose on your side if
your framework (e.g. PyTorch) wants channels-first.

`s1_dateN` and `s2_dateN` for the same date always share the same `(H, W)` —
both are reprojected onto one common UTM grid before export (see §3), so they
are pixel-aligned and safe to `np.concatenate(..., axis=-1)` directly if you
need a single 6-channel tensor:

```python
stacked = np.concatenate([s1_image, s2_image], axis=-1)
# stacked.shape[-1] == 6, order: [VV, VH, B2, B3, B4, B8]
```

This `[VV, VH, B2, B3, B4, B8]` order is **the** canonical 6-channel order
referenced elsewhere (feature doc, training patches) — do not silently
reorder it.

## 2. Nodata

Masked/cloudy/no-observation pixels are **not** `NaN` — they are the
sentinel value `-9999.0`, stored in the GeoTIFF's nodata tag so any GIS
reader (rasterio, QGIS, GDAL) honors it automatically. Always mask on this
value before computing stats or feeding patches to a model, e.g.:

```python
valid = s2_image[..., 0] != -9999.0
```

`validate_export.py` (see §4) enforces an upper bound on nodata fraction so
you should rarely see a patch that's mostly nodata, but always check.

## 3. Grid / CRS

- Projection: **UTM, auto-detected per AOI** from the AOI centroid (`gee_data_collection.get_utm_epsg`), not a hardcoded zone. Both dates and both sensors for one AOI share the same CRS.
- Resolution: **10 m** pixels.
- S1 and S2 are reprojected onto the *same* grid (identical transform/origin/shape) before export — `validate_export.py` fails loudly if they ever drift apart.

## 4. Validation (run on any exported GeoTIFF pair)

```bash
python validate_export.py path/to/s1.tif path/to/s2.tif
```

Checks, and raises `DataValidationError` (non-zero exit) on first failure:
1. Same CRS.
2. Same transform (resolution + pixel grid origin, `1e-6`-tolerance).
3. Same `(width, height)`.
4. Nodata fraction below threshold (default 60%; cloud masking is what
   produces S2 nodata, so this doubles as the "cloud mask actually worked"
   check).
5. No band is 100% constant (catches silent all-zero / all-nodata exports).

This is intentionally the loudest, least forgiving part of the pipeline —
bad exports must never pass silently downstream.

## 5. Source GeoTIFF naming (on disk, `data/`)

```
{aoi_name}_{YYYY-MM-DD}_S1.tif   # bands: VV, VH
{aoi_name}_{YYYY-MM-DD}_S2.tif   # bands: B2, B3, B4, B8
```

`handoff.py` reads these by convention; if a pair isn't cached on disk it
falls back to a live GEE pull (requires `ee.Initialize()` to already be
authenticated — see `gee_data_collection.py`).

## 6. Demo AOIs (pre-exported, checked into `data/`)

All three validated end-to-end (`validate_export.py` passes, 0% nodata,
plausible band stats) via the real pipeline against live GEE data — not
synthetic. Two show urbanization, one shows deforestation/land-clearing, so
the demo can tell either story.

| `aoi_name`           | date_1       | date_2       | Grid      | Story |
|-----------------------|--------------|--------------|-----------|-------|
| `bengaluru_sarjapur`  | 2019-02-01   | 2024-02-01   | 330x336   | Sarjapur Road corridor, Bengaluru — rapid residential/IT-park urbanization |
| `chennai_oragadam`    | 2018-02-25   | 2023-02-14   | 437x445   | Oragadam industrial corridor, Chennai — large-scale industrial land conversion |
| `chimakurthy_quarry`  | 2018-02-03   | 2023-01-28   | 540x557   | Granite quarry belt, Chimakurthy (Prakasam, AP) — visible deforestation/quarry expansion (16.1% of pixels show a strong NDVI drop between dates) |

Both date pairs were chosen with `suggest_best_dates()` (dry season,
lowest-cloud date per year) and are ready to feed straight into
`get_training_pair()` with `use_cache=True` — no GEE call needed at demo
time.

## 7. Setup

```bash
pip install earthengine-api geemap rasterio numpy
```

`gee_data_collection.py` expects `ee.Initialize()` to succeed against an
already-authenticated credential — run `ee.Authenticate()` yourself,
interactively, once, before using it.

## 8. What upstream processing already happened (so you don't redo it)

- **S2**: cloud + cloud-shadow + cirrus masked (s2cloudless probability +
  SCL), then median-composited over the date's acquisition window, then
  scaled to reflectance (`/10000`).
- **S1**: IW mode, VV+VH only, single orbit pass (whichever has more scenes
  in the window), median-composited, then speckle-filtered (7x7 focal mean
  in natural units, converted back to dB).
- Both reprojected to the shared UTM grid (§3) as the last step.

Don't re-cloud-mask, re-speckle-filter, or reproject again downstream —
it's already done and doing it twice will just soften/distort the data
further.
