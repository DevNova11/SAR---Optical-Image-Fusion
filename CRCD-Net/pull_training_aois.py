"""Pull extra AOIs for classifier/fusion training, targeted at class coverage.

The first 15 cached AOIs were skewed: most are almost single-class (urban Bengaluru/Dubai,
all-forest, all-water) and Bare Land was ~0.3% of training pixels. These AOIs add mixed
landscapes with real Bare/farmland/forest/water so every split can contain all 5 classes.

Resumable: a (aoi, date) whose S1+S2 GeoTIFFs already exist is skipped; failures are logged and the
loop continues (GEE can time out, or a date can have no cloud-free scene).

    python -u pull_training_aois.py            # all AOIs
    python -u pull_training_aois.py thar_jaisalmer kodagu_forest   # just these

After pulling:  python -m semantics.dw_labels   (exports labels for new scenes, skips existing)
Then check class composition per AOI before freezing evaluation/splits.json.
"""
from __future__ import annotations

import os
import sys
import time

import ee

import gee_data_collection as gdc

HALF = 0.02  # degrees, about 4.4 km square, same as the demo AOIs
# Dry-season dates (Jan-Mar) 2 years apart; collect_pair uses a +/-45 day median window around each.
DATES = ["2018-02-01", "2020-02-01", "2022-02-01", "2024-02-01"]

# name -> (lat, lon, intended class mix).
# Every coordinate here was screened against ESA WorldCover (a >=40% share of the intended class) BEFORE pulling.
# Lesson learned: the first batch used well-known CITY coordinates (Ludhiana, Guntur, Thanjavur, Mandya, Anantapur,
# Ooty, Madikeri, Kolar, Jaisalmer) labeled as farmland/forest/desert; they came out 50-95% Built. Those scenes were
# renamed *_city / *_town and kept as urban/mixed examples. Always screen new points with WorldCover first.
AOIS = {
    "punjab_rural": (30.60, 75.45, "intensive cropland"),
    "cauvery_rural": (10.95, 79.50, "paddy cropland"),
    "parbhani_rural": (19.50, 76.50, "cotton/soy cropland"),
    "telangana_rural": (18.10, 78.90, "cropland + grass + trees"),
    "godavari_delta": (16.60, 81.90, "trees + cropland + water (mixed)"),
    "thar_dunes": (26.80, 70.55, "bare desert dunes"),
    "nagarhole_forest": (12.05, 76.15, "dense forest"),
    "chilika_lagoon": (19.70, 85.30, "lagoon water + cropland"),
}


def _have(out_dir, name, date):
    return all(os.path.exists(os.path.join(out_dir, f"{name}_{date}_{b}.tif")) for b in ("S1", "S2"))


def main(names, out_dir="data"):
    gdc.init()
    todo = names or list(AOIS)
    ok = fail = skip = 0
    for name in todo:
        lat, lon, note = AOIS[name]
        aoi = ee.Geometry.Rectangle([lon - HALF, lat - HALF, lon + HALF, lat + HALF])
        print(f"== {name} ({lat}, {lon}) {note}", flush=True)
        for d1, d2 in ((DATES[0], DATES[1]), (DATES[2], DATES[3])):
            pending = [d for d in (d1, d2) if not _have(out_dir, name, d)]
            if not pending:
                skip += 2
                continue
            t = time.time()
            try:
                gdc.collect_pair(aoi, d1, d2, name, out_dir=out_dir)
                ok += 2
                print(f"   OK   {d1},{d2} ({time.time() - t:.0f}s)", flush=True)
            except Exception as exc:
                fail += 2
                print(f"   FAIL {d1},{d2}: {str(exc)[:140]}", flush=True)
    print(f"done: {ok} scenes pulled, {skip} already cached, {fail} failed", flush=True)


if __name__ == "__main__":
    unknown = [a for a in sys.argv[1:] if a not in AOIS]
    if unknown:
        sys.exit(f"unknown AOI(s) {unknown}; choices: {list(AOIS)}")
    main(sys.argv[1:])
