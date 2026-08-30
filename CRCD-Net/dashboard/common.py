"""Shared setup for every page of the CRCD-Net multi-page Streamlit app.

Every page (Home.py and everything in pages/) imports this module first,
before any other project import -- it puts CRCD-Net's root on sys.path
(so `from fusion.baseline import fuse` etc. resolve the same way from
Home.py and from pages/*.py, which sit one directory deeper) and centralizes
the constants, dark theme CSS, and AOI helpers every page needs.
"""
from __future__ import annotations

import os
import sys

import streamlit as st
from pyproj import Geod

# common.py always lives at CRCD-Net/dashboard/common.py, so this path is
# stable regardless of which page (Home.py, or one directory deeper in
# pages/) imports it -- that's what lets every page reach project modules
# with the same `from fusion.baseline import fuse`-style imports.
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
CRCD_NET_ROOT = os.path.dirname(DASHBOARD_DIR)
if CRCD_NET_ROOT not in sys.path:
    sys.path.insert(0, CRCD_NET_ROOT)

DATA_DIR = os.path.join(CRCD_NET_ROOT, "data")
MIN_DAYS_OLD = 10

# AOI drawn on the map above this size (km^2) gets a warning, not a block --
# the user can still run it. Adjustable.
AOI_AREA_WARNING_KM2 = 500.0

# Deployed-server guard: strangers hitting live GEE from a public demo can
# hit rate limits or slow the app down unpredictably (see deploy notes).
# Demo Area is always the default selection either way; this only controls
# whether "Custom Location" (live GEE pull) is offered at all. Override by
# setting ALLOW_LIVE_GEE = false under [general] in Streamlit secrets --
# no code change needed to flip it after seeing the deployed app live.
try:
    ALLOW_LIVE_GEE = bool(st.secrets.get("ALLOW_LIVE_GEE", True))
except Exception:
    ALLOW_LIVE_GEE = True

DEMO_AOIS = {
    "chimakurthy_quarry (mining/deforestation)": {
        "dates": ["2018-02-03", "2019-10-03", "2021-05-31", "2023-01-28"],
        "date_1": "2018-02-03", "date_2": "2023-01-28",
        "latitude": 15.550, "longitude": 79.850,
        "story": "Granite quarry belt (Prakasam, AP) showing substantial vegetation clearing and quarry expansion.",
    },
    "bengaluru_sarjapur (rapid urbanization)": {
        "dates": ["2019-02-01", "2020-10-02", "2022-06-02", "2024-02-01"],
        "date_1": "2019-02-01", "date_2": "2024-02-01",
        "latitude": 12.845, "longitude": 77.645,
        "story": "Sarjapur Road corridor, Bengaluru showing conversion of agricultural land to residential IT complexes.",
    },
    "chennai_oragadam (industrial corridor)": {
        "dates": ["2018-02-25", "2019-10-23", "2021-06-19", "2023-02-14"],
        "date_1": "2018-02-25", "date_2": "2023-02-14",
        "latitude": 12.770, "longitude": 80.000,
        "story": "Oragadam industrial belt, Chennai showing large-scale industrial built-up expansion.",
    },
    "dubai_islands_v2 (coastal reclamation)": {
        "dates": ["2016-02-01", "2018-06-02", "2020-10-02", "2023-02-01"],
        "date_1": "2016-02-01", "date_2": "2023-02-01",
        "latitude": 25.290, "longitude": 55.330,
        "story": "Dubai Islands offshore project showing marine reclamation and urban infrastructure growth.",
    },
}

_GEOD = Geod(ellps="WGS84")


def bbox_from_drawn_geojson(feature: dict) -> tuple[float, float, float, float]:
    """(west, south, east, north) from a folium Draw rectangle's GeoJSON Feature.

    A drawn rectangle comes back as a 4-corner (5 with the closing point)
    Polygon ring, not a clean bbox -- so we take the min/max of its corners
    rather than assuming a corner order.
    """
    ring = feature["geometry"]["coordinates"][0]
    lons = [c[0] for c in ring]
    lats = [c[1] for c in ring]
    return min(lons), min(lats), max(lons), max(lats)


def bbox_area_km2(west: float, south: float, east: float, north: float) -> float:
    """Geodesic area of a lat/lon bbox in km^2, via pyproj -- no UTM lookup needed."""
    lons = [west, east, east, west]
    lats = [south, south, north, north]
    area_m2, _perimeter = _GEOD.polygon_area_perimeter(lons, lats)
    return abs(area_m2) / 1e6


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500&display=swap');

:root {
    --void: #0a0c0a;
    --panel: #12160f;
    --panel-raised: #161c13;
    --hairline: #2a3226;
    --hairline-soft: #1c2318;
    --phosphor: #7fffa0;
    --phosphor-dim: #3f7a55;
    --optical: #5aa8ff;
    --paper: #eaf2e6;
    --paper-dim: #8fa08c;
}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"] {
    background-color: var(--void) !important;
    font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
}
[data-testid="stAppViewContainer"] * { color: var(--paper); }
[data-testid="stHeader"] { background-color: transparent !important; }
.block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; }

section[data-testid="stSidebar"] {
    background-color: var(--panel) !important;
    border-right: 1px solid var(--hairline);
}
section[data-testid="stSidebar"] * { color: var(--paper) !important; }
section[data-testid="stSidebar"] h1 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.15rem !important;
    letter-spacing: 0.05em;
    color: var(--phosphor) !important;
}

h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--paper) !important;
}

.hero {
    padding: 2.2rem 0;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--hairline);
}
.hero .badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--phosphor) !important;
    margin-bottom: 0.8rem;
}
.hero .badge::before {
    content: "";
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--phosphor);
    box-shadow: 0 0 6px 2px rgba(127,255,160,.6);
}
.hero h1 {
    font-size: 2.5rem;
    margin-bottom: 0.6rem;
    font-weight: 700;
    line-height: 1.15;
}
.hero h1 em {
    font-style: normal;
    color: var(--phosphor) !important;
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    margin-top: 1.8rem;
    margin-bottom: 0.3rem;
}
.section-underline {
    width: 45px;
    height: 2px;
    background-color: var(--phosphor);
    margin: 0 0 1.4rem 0;
}

.card {
    padding: 1.2rem;
    background-color: var(--panel);
    border: 1px solid var(--hairline);
    min-height: 140px;
}
.card h4 { margin-bottom: 0.4rem; font-size: 1.05rem; }
.card p { color: var(--paper-dim) !important; font-size: 0.86rem; line-height: 1.5; }

.metric-card {
    padding: 1.1rem;
    background-color: var(--panel);
    text-align: center;
    border: 1px solid var(--hairline);
}
.metric-card h4 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--paper-dim) !important;
    margin-bottom: 0.4rem;
}
.metric-card p {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--phosphor) !important;
    font-weight: 700;
}

.badge-critical { background-color: #c0392b; color: #fff !important; padding: 3px 8px; border-radius: 2px; font-weight: 700; }
.badge-high { background-color: #e67e22; color: #fff !important; padding: 3px 8px; border-radius: 2px; font-weight: 700; }
.badge-medium { background-color: #f39c12; color: #fff !important; padding: 3px 8px; border-radius: 2px; font-weight: 700; }
.badge-low { background-color: #7f8c8d; color: #fff !important; padding: 3px 8px; border-radius: 2px; font-weight: 700; }

.stButton > button {
    background-color: transparent;
    color: var(--phosphor) !important;
    border: 1px solid var(--phosphor);
    border-radius: 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
    padding: 0.65rem 1.3rem;
    transition: 0.2s;
}
.stButton > button:hover {
    background-color: var(--phosphor);
    color: var(--void) !important;
    box-shadow: 0 0 20px rgba(127,255,160,.4);
}

/* Radio button selection dots: Streamlit's default red accent
   (rgb(255,75,75)) lives on the dot div, which sits at the same DOM
   depth (2 div-ancestors deep) as the sibling div that wraps the
   option's *label text* -- both match a plain "div > div > div"
   selector, which is what turned the whole "Home" label green
   instead of just the dot. The text wrapper always carries
   data-testid="stMarkdownContainer"; the dot never does, so
   excluding that testid is what actually disambiguates them.
   Confirmed via computed-style + bounding-rect inspection. */
[data-testid="stRadio"] label[data-selected="true"] div > div > div:not([data-testid]) {
    background-color: var(--phosphor) !important;
}
[data-testid="stRadio"] svg { fill: var(--phosphor) !important; }
*:focus, *:focus-visible {
    outline-color: var(--phosphor) !important;
    box-shadow: none !important;
}
</style>
"""


def configure_page(page_title: str) -> None:
    """Call first thing in every page: sets tab title/layout, injects the
    shared dark theme CSS, and renders the sidebar branding block. Streamlit
    auto-builds the page-to-page navigation list from Home.py + pages/*.py,
    so this no longer needs (or should have) its own nav radio."""
    st.set_page_config(
        page_title=f"CRCD-Net | {page_title}",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.title("CRCD-Net")
        st.caption("Explainable SAR-Optical Provenance Console")
        st.markdown("---")
        st.markdown(
            """
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:var(--paper-dim);">
            <b>CORE RESEARCH MODULES</b><br>
            &bull; Multi-Temporal Alignment (T1..TN)<br>
            &bull; Reliability Fusion (W_SAR, W_OPT)<br>
            &bull; Semantic Land-Cover (5 Classes)<br>
            &bull; Trajectory & Persistence Engine<br>
            &bull; Sensor Evidence Attribution<br>
            &bull; Grounded Confidence Model<br>
            &bull; Early-Warning Priority Ranking
            </div>
            """,
            unsafe_allow_html=True,
        )
