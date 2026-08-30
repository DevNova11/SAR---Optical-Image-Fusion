"""Shared setup for every page of the CRCD-Net multi-page Streamlit app.

Every page (Home.py and everything in pages/) imports this module first,
before any other project import -- it puts CRCD-Net's root on sys.path
(so `from fusion.baseline import fuse` etc. resolve the same way from
Home.py and from pages/*.py, which sit one directory deeper) and centralizes
the constants, dark theme CSS, and AOI helpers every page needs.
"""
from __future__ import annotations

import base64
import io
import os
import sys

import numpy as np
import streamlit as st
from PIL import Image
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
    --ok: #7fffa0;
    --warn: #f39c12;
    --crit: #c0392b;
    --info: #5aa8ff;
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

/* ---- System status pill (sidebar) ---- */
.status-pill {
    display: inline-flex; align-items: center; gap: 0.45rem;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--ok) !important; margin: 0.4rem 0 1rem 0;
}
.status-pill::before {
    content: ""; width: 7px; height: 7px; border-radius: 50%;
    background: var(--ok); box-shadow: 0 0 6px 2px rgba(52,199,123,.6);
}

/* ---- Command-center action cards ---- */
.action-card {
    background: var(--panel);
    border: 1px solid var(--hairline);
    border-top: 2px solid var(--phosphor);
    padding: 1.3rem 1.2rem 1.1rem 1.2rem;
    min-height: 128px;
    margin-bottom: 0.6rem;
}
.action-card .tag {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--phosphor) !important; margin-bottom: 0.35rem;
}
.action-card h4 { margin: 0 0 0.35rem 0; font-size: 1.15rem; }
.action-card p { color: var(--paper-dim) !important; font-size: 0.85rem; margin: 0; }

/* ---- Workflow stepper ---- */
.stepper-wrap {
    display: flex; align-items: center; width: 100%;
    margin: 0.6rem 0 1.6rem 0; flex-wrap: wrap;
    font-family: 'IBM Plex Mono', monospace;
}
.step-node { display: flex; align-items: center; }
.step-dot {
    width: 22px; height: 22px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.68rem; font-weight: 700; flex-shrink: 0;
    border: 1px solid var(--hairline); color: var(--paper-dim);
    background: var(--panel);
}
.step-dot.done { background: var(--phosphor); border-color: var(--phosphor); color: var(--void); }
.step-dot.active { border-color: var(--phosphor); color: var(--phosphor); box-shadow: 0 0 0 3px rgba(54,224,240,.15); }
.step-label {
    font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase;
    margin-left: 0.5rem; margin-right: 0.9rem; color: var(--paper-dim);
    white-space: nowrap;
}
.step-label.done, .step-label.active { color: var(--paper); }
.step-connector { width: 34px; height: 1px; background: var(--hairline); margin-right: 0.9rem; }
.step-connector.done { background: var(--phosphor); }

/* ---- Reliability / evidence bars ---- */
.rbar-row { margin-bottom: 0.65rem; }
.rbar-label {
    display: flex; justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    color: var(--paper-dim); margin-bottom: 3px;
}
.rbar-label b { color: var(--paper); font-weight: 600; }
.rbar-track { height: 7px; background: var(--hairline-soft); width: 100%; }
.rbar-fill { height: 7px; background: var(--phosphor); }

/* ---- Before / After comparison slider ---- */
.cmp-wrap {
    position: relative; width: 100%;
    border: 1px solid var(--hairline); user-select: none; line-height: 0;
}
.cmp-wrap img { display: block; width: 100%; height: auto; pointer-events: none; }
.cmp-after-img { position: relative; z-index: 1; }
.cmp-before-img { position: absolute; top: 0; left: 0; z-index: 2; }
.cmp-divider {
    position: absolute; top: 0; bottom: 0; width: 2px;
    background: var(--phosphor); z-index: 3; box-shadow: 0 0 8px rgba(54,224,240,.7);
}
.cmp-tag {
    position: absolute; top: 8px; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase;
    background: rgba(6,10,19,.72); padding: 3px 8px; color: var(--paper); z-index: 4;
}

/* ---- Explanation / evidence card ---- */
.explain-card {
    background: var(--panel); border: 1px solid var(--hairline);
    border-left: 3px solid var(--ok); padding: 0.9rem 1.1rem; margin: 0.6rem 0 1rem 0;
}
.explain-card .line { font-size: 0.86rem; color: var(--paper); margin: 3px 0; }
.explain-card .line::before { content: "\2713  "; color: var(--ok); font-weight: 700; }

/* ---- Motion / hover polish ---- */
@keyframes crcdFadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes crcdPulse {
    0% { box-shadow: 0 0 0 0 rgba(127,255,160,.55); }
    70% { box-shadow: 0 0 0 8px rgba(127,255,160,0); }
    100% { box-shadow: 0 0 0 0 rgba(127,255,160,0); }
}
.hero, .action-card, .card, .metric-card { animation: crcdFadeUp 0.45s ease-out both; }
.action-card, .card {
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}
.action-card:hover, .card:hover {
    transform: translateY(-3px);
    border-color: var(--phosphor);
    box-shadow: 0 8px 22px rgba(0,0,0,.35);
}
.metric-card { transition: border-color 0.18s ease, transform 0.18s ease; }
.metric-card:hover { border-color: var(--phosphor); transform: translateY(-2px); }
.status-pill::before { animation: crcdPulse 2s infinite; }
.step-dot.active { animation: crcdPulse 1.8s infinite; }

/* ---- Live pipeline checklist ---- */
.live-stage-box {
    background: var(--panel); border: 1px solid var(--hairline);
    border-left: 3px solid var(--phosphor); padding: 0.9rem 1.1rem;
    margin: 0.5rem 0 1rem 0; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem; animation: crcdFadeUp 0.3s ease-out both;
}
.live-stage-box .stage-title {
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--phosphor);
    font-size: 0.68rem; margin-bottom: 0.55rem;
}
.live-stage-box .stage-row { display: flex; align-items: center; gap: 0.5rem; padding: 2px 0; color: var(--paper-dim); }
.live-stage-box .stage-row.working { color: var(--paper); }
.live-stage-box .dotpending { width: 6px; height: 6px; border-radius: 50%; border: 1px solid var(--hairline); flex-shrink: 0; }
.live-stage-box .dotworking { width: 6px; height: 6px; border-radius: 50%; background: var(--phosphor); flex-shrink: 0; animation: crcdPulse 1.2s infinite; }

/* ---- Gauge / donut (conic-gradient, real values only) ---- */
.gauge-wrap { display: flex; flex-direction: column; align-items: center; }
.gauge-ring {
    width: 92px; height: 92px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
}
.gauge-inner {
    width: 70px; height: 70px; border-radius: 50%; background: var(--panel);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.gauge-inner .val { font-family: 'IBM Plex Mono', monospace; font-weight: 700; color: var(--phosphor); font-size: 1rem; }
.gauge-inner .lbl { font-size: 0.58rem; color: var(--paper-dim); letter-spacing: 0.05em; text-transform: uppercase; }
.gauge-caption { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.06em; text-transform: uppercase; color: var(--paper-dim); margin-top: 0.4rem; }

/* ---- Priority distribution bar ---- */
.priority-stack { display: flex; width: 100%; height: 14px; overflow: hidden; border: 1px solid var(--hairline); }
.priority-legend { display: flex; gap: 14px; flex-wrap: wrap; font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--paper-dim); margin-top: 6px; }
.priority-legend span.sw { display: inline-block; width: 8px; height: 8px; margin-right: 4px; vertical-align: middle; }

/* ---- Data-source chips (SAR / Optical / Fused) ---- */
.source-chip {
    display: inline-flex; align-items: center; gap: 6px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--paper);
    background: var(--panel); border: 1px solid var(--hairline);
    border-top: 2px solid var(--phosphor); padding: 6px 12px; margin-bottom: 6px;
}
</style>
"""


# --------------------------------------------------
# REUSABLE UI COMPONENTS
# --------------------------------------------------
def render_status_pill(text: str = "System Ready") -> None:
    st.markdown(f'<div class="status-pill">{text}</div>', unsafe_allow_html=True)


def render_workflow_stepper(labels: list[str], current_index: int) -> None:
    """current_index: steps before it are 'done', that index is 'active', after are pending."""
    parts = ['<div class="stepper-wrap">']
    for i, label in enumerate(labels):
        state = "done" if i < current_index else ("active" if i == current_index else "")
        mark = "✓" if state == "done" else str(i + 1)
        parts.append(
            f'<div class="step-node"><div class="step-dot {state}">{mark}</div>'
            f'<div class="step-label {state}">{label}</div></div>'
        )
        if i < len(labels) - 1:
            conn_state = "done" if i < current_index else ""
            parts.append(f'<div class="step-connector {conn_state}"></div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_gauge(value_0_1: float, label: str, caption: str = "") -> None:
    """Conic-gradient ring gauge -- purely a styled readout of a real 0-1 value."""
    pct = max(0.0, min(1.0, float(value_0_1))) * 100
    st.markdown(
        f"""
        <div class="gauge-wrap">
          <div class="gauge-ring" style="background: conic-gradient(var(--phosphor) {pct}%, var(--hairline-soft) {pct}% 100%);">
            <div class="gauge-inner"><div class="val">{pct:.0f}%</div><div class="lbl">{label}</div></div>
          </div>
          {f'<div class="gauge-caption">{caption}</div>' if caption else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_priority_distribution(counts: dict) -> None:
    """counts: {'CRITICAL': n, 'HIGH': n, 'MEDIUM': n, 'LOW': n} -- computed from real hotspot data."""
    colors = {"CRITICAL": "#c0392b", "HIGH": "#e67e22", "MEDIUM": "#f39c12", "LOW": "#7f8c8d"}
    total = sum(counts.values()) or 1
    segments = "".join(
        f'<div style="width:{counts.get(k,0)/total*100:.2f}%; background:{colors[k]};"></div>'
        for k in ["CRITICAL", "HIGH", "MEDIUM", "LOW"] if counts.get(k, 0) > 0
    )
    legend = "".join(
        f'<span><span class="sw" style="background:{colors[k]};"></span>{k} ({counts.get(k,0)})</span>'
        for k in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    )
    st.markdown(
        f'<div class="priority-stack">{segments}</div><div class="priority-legend">{legend}</div>',
        unsafe_allow_html=True,
    )


def render_live_stage_checklist(stages: list[str], working_index: int) -> None:
    """Static, honestly-labeled list of the pipeline stages actually run -- no fabricated %."""
    rows = []
    for i, s in enumerate(stages):
        if i < working_index:
            rows.append(f'<div class="stage-row"><span class="dotpending" style="background:var(--phosphor); border-color:var(--phosphor);"></span>{s} -- done</div>')
        elif i == working_index:
            rows.append(f'<div class="stage-row working"><span class="dotworking"></span>{s} -- running...</div>')
        else:
            rows.append(f'<div class="stage-row"><span class="dotpending"></span>{s}</div>')
    st.markdown(
        f'<div class="live-stage-box"><div class="stage-title">Pipeline Stages</div>{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def render_source_chip(text: str) -> None:
    st.markdown(f'<div class="source-chip">{text}</div>', unsafe_allow_html=True)


def render_reliability_bar(label: str, value_0_1: float) -> None:
    pct = max(0.0, min(1.0, float(value_0_1))) * 100
    st.markdown(
        f"""
        <div class="rbar-row">
          <div class="rbar-label"><span>{label}</span><b>{pct:.1f}%</b></div>
          <div class="rbar-track"><div class="rbar-fill" style="width:{pct:.1f}%;"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _np_to_png_b64(img: np.ndarray) -> str:
    """Convert a HxWx3 float [0,1] or uint8 array to a base64 PNG data URI."""
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render_compare_slider(before_img: np.ndarray, after_img: np.ndarray,
                           before_label: str = "BEFORE", after_label: str = "AFTER",
                           key: str = "cmp") -> None:
    """Drag-to-reveal before/after image comparison slider (pure CSS/HTML, no data change)."""
    before_uri = _np_to_png_b64(before_img)
    after_uri = _np_to_png_b64(after_img)
    slider_val = st.slider(
        f"Drag to compare -- {before_label} vs {after_label}", 0, 100, 50,
        key=f"slider_{key}", label_visibility="collapsed",
    )
    clip_right = 100 - slider_val
    st.markdown(
        f"""
        <div class="cmp-wrap">
          <div class="cmp-tag" style="right:8px;">{after_label}</div>
          <div class="cmp-tag" style="left:8px;">{before_label}</div>
          <div class="cmp-after-img"><img src="{after_uri}" /></div>
          <div class="cmp-before-img" style="clip-path: inset(0 {clip_right}% 0 0);">
            <img src="{before_uri}" />
          </div>
          <div class="cmp-divider" style="left:{slider_val}%;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        render_status_pill("System Ready")
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
