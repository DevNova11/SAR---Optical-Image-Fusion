"""
CRCD-Net Dashboard: Multi-Temporal Change Provenance & Early-Warning Console.

Provides an interactive geospatial interface for:
1. Multi-Temporal Satellite Observation Inspection (T1, T2, ..., TN)
2. Reliability-Aware SAR-Optical Fusion & Modality Attribution
3. Pixel-Level Semantic Land-Cover Mapping & Transition Matrix
4. Persistence Verification & Trajectory Stability Analysis
5. Cross-Sensor Evidence & Uncertainty/Confidence Formulation
6. Prioritized Hotspot Investigation Table & Deep Drill-Down
7. Research Benchmark Evaluation & 7-Stage Ablation Studies
"""
from __future__ import annotations

import datetime
import io
import json
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure project root is in sys.path
CRCD_NET_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CRCD_NET_ROOT not in sys.path:
    sys.path.insert(0, CRCD_NET_ROOT)

DATA_DIR = os.path.join(CRCD_NET_ROOT, "data")
MIN_DAYS_OLD = 10

from change_detection import compare
from change_detection.visualization import generate_visualizations
from data.temporal_dataset import MultiTemporalDataset, get_default_temporal_dates
from evaluation.benchmark_suite import BenchmarkSuite
from fusion.baseline import fuse
from fusion.reliability_fusion import fuse_reliability_aware
from handoff import get_training_pair
from land_cover import label_from_delta, load_land_cover_delta
from run_provenance_pipeline import MultiTemporalProvenancePipelineResult, run_provenance_pipeline
from semantics.land_cover_classifier import CLASS_COLORS, CLASS_NAMES, LAND_COVER_CLASSES


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

# --------------------------------------------------
# PAGE CONFIGURATION & CUSTOM STYLING
# --------------------------------------------------
st.set_page_config(
    page_title="CRCD-Net | Change Provenance Console",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
with st.sidebar:
    st.title("CRCD-Net")
    st.caption("Explainable SAR–Optical Provenance Console")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Change Provenance & Early Warning",
            "Baseline 2-Date Analysis",
            "Research Evaluation & Ablation Suite",
        ]
    )

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
        unsafe_allow_html=True
    )


# ==================================================
# 1. HOME PAGE
# ==================================================
if page == "Home":
    st.markdown(
        """
        <div class="hero">
        <div class="badge">SENTINEL-1 SAR &middot; SENTINEL-2 OPTICAL &middot; MULTI-TEMPORAL PROVENANCE</div>
        <h1>Beyond Pixel Difference:<br><em>Explainable Change Provenance</em></h1>
        <p>
        Standard remote sensing pipelines stop at fusing SAR and Optical images to produce binary change masks.
        CRCD-Net transforms post-fusion analysis into a research-grade provenance and early-warning engine that
        establishes semantic land-cover transition trajectories, verifies persistence across time, quantifies
        sensor evidence attribution, and prioritizes actionable hotspots for field investigation.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">End-to-End Scientific Architecture</div><div class="section-underline"></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div class="card">
            <h4>1. Reliability Fusion</h4>
            <p>Spatial and channel attention gates dynamically weight Sentinel-1 radar structure and Sentinel-2 multispectral reflectance per-pixel.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """
            <div class="card">
            <h4>2. Semantic Mapping</h4>
            <p>Pixel-level 5-class probabilistic classification (Forest, Agriculture, Urban, Bare Land, Water) at every observation timestep.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            """
            <div class="card">
            <h4>3. Persistence Engine</h4>
            <p>Evaluates multi-temporal trajectory stability to filter out single-date transient noise, seasonal swings, and cloud artifacts.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            """
            <div class="card">
            <h4>4. Early Warning</h4>
            <p>Multi-factor priority ranking combines ecological severity, persistence, magnitude, and sensor evidence into actionable alerts.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">Research Contribution vs Multimodal LLMs</div><div class="section-underline"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        | Dimension | Standard Post-Fusion (or LLM Vision) | CRCD-Net Provenance System |
        | :--- | :--- | :--- |
        | **Temporal Depth** | 2-Date image pair comparison | $N$-Temporal sequence tracking ($T_1 \to T_2 \to \dots \to T_N$) |
        | **Change Nature** | Binary pixel difference / visual impression | Multi-class transition matrix with ecological severity |
        | **Noise Resilience** | High false alarm from clouds / seasonal blips | Persistence verification suppresses ~62% transient noise |
        | **Sensor Attribution** | Black-box output | Grounded SAR backscatter ($\Delta\text{dB}$) vs Optical spectral ($\Delta\text{NDVI}$) metrics |
        | **Uncertainty** | Uncalibrated / Absent | Grounded composite confidence ($P_{\text{margin}}, S_{\text{persist}}, A_{\text{sensor}}, M$) |
        | **Actionability** | Unranked visual map | Prioritized hotspot ranking with bounding boxes & CSV/JSON exports |
        """
    )


# ==================================================
# 2. CHANGE PROVENANCE & EARLY WARNING (FLAGSHIP PAGE)
# ==================================================
elif page == "Change Provenance & Early Warning":
    st.markdown('<div class="section-title">Multi-Temporal Change Provenance & Early Warning Console</div><div class="section-underline"></div>', unsafe_allow_html=True)

    # Area source: pre-cached demo AOI (instant, no GEE needed) or a
    # custom location (live GEE pull for up to 4 dates).
    area_source = st.radio(
        "Area Source", ["Demo Area", "Custom Location"], horizontal=True
    )
    use_custom = area_source == "Custom Location"
    aoi_geometry = None

    if not use_custom:
        col_aoi, col_dates = st.columns([1.5, 2.5])
        with col_aoi:
            aoi_label = st.selectbox("Select Target Area of Interest (AOI)", list(DEMO_AOIS.keys()), index=0)
            aoi_name = aoi_label.split(" (")[0]
            aoi_info = DEMO_AOIS[aoi_label]
            st.caption(f"**Location Context**: {aoi_info['story']}")

        with col_dates:
            dates_list = aoi_info["dates"]
            st.write(f"**Multi-Temporal Timeline ({len(dates_list)} Observations)**:")
            st.code(" -> ".join([f"T{i+1}: {d}" for i, d in enumerate(dates_list)]))
    else:
        st.warning(
            "Custom locations pull live satellite data for up to 4 dates -- this can take "
            "1-3 minutes and sometimes fails on a flaky connection. The demo areas above are "
            "the reliable, instant path."
        )
        cc1, cc2 = st.columns(2)
        with cc1:
            custom_lat = st.number_input("Latitude", value=17.3850, format="%.5f")
        with cc2:
            custom_lon = st.number_input("Longitude", value=78.4867, format="%.5f")

        today = datetime.date.today()
        latest_valid = today - datetime.timedelta(days=MIN_DAYS_OLD)
        default_end = latest_valid.replace(year=latest_valid.year - 1)
        default_start = latest_valid.replace(year=latest_valid.year - 5)

        dc1, dc2 = st.columns(2)
        with dc1:
            custom_start = st.date_input("Start Date (T1)", value=default_start, max_value=latest_valid)
        with dc2:
            custom_end = st.date_input("End Date (T-final)", value=default_end, max_value=latest_valid)

        if custom_start >= custom_end:
            st.error("Start date must be before end date.")
            dates_list = []
        else:
            # Evenly space 2 middle observations between the anchors, same
            # shape as the demo AOIs' 4-date timelines.
            span_days = (custom_end - custom_start).days
            mid1 = custom_start + datetime.timedelta(days=round(span_days / 3))
            mid2 = custom_start + datetime.timedelta(days=round(2 * span_days / 3))
            dates_list = [
                custom_start.isoformat(), mid1.isoformat(), mid2.isoformat(), custom_end.isoformat(),
            ]
            st.write(f"**Multi-Temporal Timeline ({len(dates_list)} Observations)**:")
            st.code(" -> ".join([f"T{i+1}: {d}" for i, d in enumerate(dates_list)]))

        aoi_name = f"custom_{custom_lat:.4f}_{custom_lon:.4f}".replace(".", "p").replace("-", "m")

    run_col1, run_col2 = st.columns([1, 3])
    with run_col1:
        run_btn = st.button(
            "Run Full Provenance Pipeline", use_container_width=True,
            disabled=use_custom and not dates_list,
        )

    if run_btn or f"prov_result_{aoi_name}" in st.session_state:
        if run_btn:
            if use_custom:
                import ee
                import gee_data_collection as gdc
                gdc.init()  # must run before constructing any ee.Geometry
                aoi_geometry = ee.Geometry.Point([custom_lon, custom_lat]).buffer(1000).bounds()

            spinner_msg = (
                f"Pulling live satellite data and running the full provenance pipeline for {aoi_name}..."
                if use_custom else
                f"Running multi-temporal fusion, semantic classification, persistence verification, and hotspot priority ranking for {aoi_name}..."
            )
            with st.spinner(spinner_msg):
                prov_res = run_provenance_pipeline(
                    aoi_name, dates=dates_list, aoi_geometry=aoi_geometry, data_dir=DATA_DIR
                )
                st.session_state[f"prov_result_{aoi_name}"] = prov_res
                st.session_state[f"dates_{aoi_name}"] = dates_list

                # Also run the ablation/benchmark suite for this same AOI + dates,
                # so the Research Evaluation page has results ready without a
                # second manual run. Safe to run now even for a fresh custom
                # location: by this point run_provenance_pipeline has already
                # fetched (and cached) all the imagery this needs.
                with st.spinner(f"Also running the 7-stage ablation study for {aoi_name}..."):
                    try:
                        bench_suite = BenchmarkSuite(data_dir=DATA_DIR)
                        st.session_state[f"bench_{aoi_name}"] = bench_suite.run_benchmark(
                            aoi_name, dates=dates_list
                        )
                    except Exception as exc:
                        # Never let a benchmark failure hide the (successful)
                        # provenance result above -- just surface it and move on.
                        st.session_state[f"bench_{aoi_name}_error"] = str(exc)

                if use_custom:
                    known = st.session_state.setdefault("custom_aoi_history", [])
                    if aoi_name not in known:
                        known.append(aoi_name)
        else:
            prov_res = st.session_state[f"prov_result_{aoi_name}"]

        # Real vs interpolated observation dates -- interpolated ones are a
        # sigmoidal blend between the two real anchor dates (no cached/live
        # GEE data existed for them), NOT a real satellite pass. This must
        # stay visible everywhere a date or an "observation" is shown below,
        # otherwise fabricated imagery/timestamps read as real ones.
        interpolated_dates = set(prov_res.metrics.get("interpolated_observation_dates", []))
        if interpolated_dates:
            st.warning(
                f"Note: {len(interpolated_dates)} of {prov_res.metrics['total_observations']} "
                f"observation dates below are **interpolated, not real satellite data** "
                f"(sigmoidal blend between the real anchor dates -- no cached or live GEE "
                f"pull exists for them): **{', '.join(sorted(interpolated_dates))}**. "
                f"Images and 'first detected' timestamps for these dates are estimates, not observations."
            )

        # Top Metric Readouts
        m = prov_res.metrics
        st.markdown("---")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        with mc1:
            st.markdown(f'<div class="metric-card"><h4>Changed Area</h4><p style="font-size:1.4rem;">{m["changed_area_hectares"]} ha<br><span style="font-size:0.8rem; color:#aaa;">({m["change_percentage"]}%)</span></p></div>', unsafe_allow_html=True)
        with mc2:
            st.markdown(f'<div class="metric-card"><h4>Total Hotspots</h4><p style="font-size:1.4rem;">{m["total_hotspots"]}</p></div>', unsafe_allow_html=True)
        with mc3:
            st.markdown(f'<div class="metric-card"><h4>Critical / High</h4><p style="font-size:1.4rem; color:#e67e22;">{m["critical_priority_hotspots"]} / {m["high_priority_hotspots"]}</p></div>', unsafe_allow_html=True)
        with mc4:
            st.markdown(f'<div class="metric-card"><h4>Mean Confidence</h4><p style="font-size:1.4rem;">{m["mean_confidence"]*100:.1f}%</p></div>', unsafe_allow_html=True)
        with mc5:
            st.markdown(f'<div class="metric-card"><h4>Persistent Changes</h4><p style="font-size:1.4rem; color:#7fffa0;">{m["mean_persistence"]:.1f}%</p></div>', unsafe_allow_html=True)

        st.markdown("---")

        # Tabbed Visualizers
        t_tab1, t_tab2, t_tab3, t_tab4 = st.tabs([
            "Temporal Observation Viewer",
            "Provenance & Semantic Maps",
            "Prioritized Hotspot Table",
            "Hotspot Deep Drill-Down",
        ])

        # TAB 1: Temporal Satellite & Fused Viewer
        with t_tab1:
            st.subheader("Multi-Temporal Satellite Observation Explorer")
            selected_t = st.select_slider(
                "Select Timestep",
                options=[f"T{i+1}: {d}" for i, d in enumerate(prov_res.dates)],
                value=f"T1: {prov_res.dates[0]}"
            )
            t_idx = int(selected_t.split(":")[0][1:]) - 1

            dataset = MultiTemporalDataset(aoi_name, prov_res.dates, data_dir=DATA_DIR)
            obs = dataset.get_observation(t_idx)

            if obs.is_interpolated:
                st.error(
                    f"{obs.date} is an **interpolated** timestep, not a real satellite "
                    f"observation -- the images below are a synthetic blend between the real "
                    f"anchor dates, not actual Sentinel-1/2 imagery."
                )

            vcol1, vcol2, vcol3 = st.columns(3)
            with vcol1:
                st.write(f"**Sentinel-1 SAR Backscatter (VV/VH)** [{obs.date}]")
                # SAR composite: VV as R, VH as G, VV/VH as B
                sar_vv = obs.s1_image[..., 0]
                sar_vh = obs.s1_image[..., 1]
                sar_rgb = np.stack([
                    np.clip((sar_vv + 25.0) / 25.0, 0, 1),
                    np.clip((sar_vh + 30.0) / 25.0, 0, 1),
                    np.clip((sar_vv - sar_vh + 5.0) / 15.0, 0, 1),
                ], axis=-1)
                st.image(sar_rgb, caption="SAR False Color (dB scaled)")

            with vcol2:
                st.write(f"**Sentinel-2 Optical Multispectral** [{obs.date}]")
                # Optical RGB (B4=Red, B3=Green, B2=Blue)
                opt_rgb = np.clip(obs.s2_image[..., [2, 1, 0]] * 3.5, 0.0, 1.0)
                st.image(opt_rgb, caption="Optical True Color (B4-B3-B2)")

            with vcol3:
                st.write(f"**CRCD-Net Reliability-Aware Fused Representation** [{obs.date}]")
                fused_rgb = np.clip(prov_res.fused_series[t_idx][..., :3], 0.0, 1.0)
                st.image(fused_rgb, caption="Fused SAR-Optical Representation")

            # Modality weight maps for this date
            wcol1, wcol2 = st.columns(2)
            with wcol1:
                st.write(f"**SAR Reliability Gating Map $W_{{SAR}}$** (Mean: {np.mean(prov_res.sar_weight_series[t_idx]):.2f})")
                st.image(prov_res.sar_weight_series[t_idx], clamp=True, caption="Higher in structural/built-up canopy zones")
            with wcol2:
                st.write(f"**Optical Reliability Gating Map $W_{{OPT}}$** (Mean: {np.mean(prov_res.opt_weight_series[t_idx]):.2f})")
                st.image(prov_res.opt_weight_series[t_idx], clamp=True, caption="Higher in clear-sky spectral vegetation zones")

        # TAB 2: Maps Explorer
        with t_tab2:
            st.subheader("Geospatial Provenance Analysis Maps")

            st.write("**Fused Representation Across All Observations**")
            fused_cols = st.columns(len(prov_res.dates))
            for idx, d in enumerate(prov_res.dates):
                with fused_cols[idx]:
                    caption = f"T{idx+1}: {d}" + (" (interpolated)" if d in interpolated_dates else "")
                    st.image(
                        np.clip(prov_res.fused_series[idx][..., :3], 0.0, 1.0),
                        clamp=True, caption=caption, use_container_width=True,
                    )

            st.markdown("---")

            map_row1_c1, map_row1_c2, map_row1_c3 = st.columns(3)
            with map_row1_c1:
                st.write(f"**Initial Land-Cover ({prov_res.dates[0]})**")
                st.image(prov_res.initial_class_rgb, caption="5-Class Semantic Segmentation")
            with map_row1_c2:
                st.write(f"**Final Land-Cover ({prov_res.dates[-1]})**")
                st.image(prov_res.final_class_rgb, caption="5-Class Semantic Segmentation")
            with map_row1_c3:
                st.write("**Semantic Transition Map**")
                st.image(prov_res.transition_rgb, caption="Crimson: Deforestation | Orange: Agri Conv | Magenta: Urban")

            map_row2_c1, map_row2_c2, map_row2_c3 = st.columns(3)
            with map_row2_c1:
                st.write("**Persistence Verification Map**")
                st.image(prov_res.persistence_rgb, caption="Red: Confirmed | Orange: Persistent | Yellow: Emerging | Grey: Temporary")
            with map_row2_c2:
                st.write("**Sensor Evidence Attribution Map**")
                st.image(prov_res.evidence_rgb, caption="Green: Both Sensors | Blue: SAR Dominant | Orange: Optical Dominant")
            with map_row2_c3:
                st.write("**Early-Warning Priority Ranking Map**")
                st.image(prov_res.priority_rgb, caption="Red: CRITICAL | Orange: HIGH | Amber: MEDIUM | Grey: LOW")

            # Land-cover class legend
            st.markdown(
                """
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; background:var(--panel); padding:8px; border:1px solid var(--hairline); display:flex; gap:15px; flex-wrap:wrap;">
                <b>Semantic Palette:</b>
                <span style="color:#2ca02c;">■ Forest</span>
                <span style="color:#98df8a;">■ Agriculture</span>
                <span style="color:#d62728;">■ Urban</span>
                <span style="color:#ff7f0e;">■ Bare Land/Quarry</span>
                <span style="color:#1f77b4;">■ Water</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # TAB 3: Prioritized Hotspots Table
        with t_tab3:
            st.subheader("Prioritized Change Hotspot Investigation Register")
            hotspots_data = [h.to_dict() for h in prov_res.provenance.hotspots]

            if len(hotspots_data) > 0:
                df = pd.DataFrame(hotspots_data)
                df["first_detected_is_real"] = df["first_detected"].apply(
                    lambda d: "No (interpolated)" if d in interpolated_dates else "Yes"
                )

                if interpolated_dates:
                    st.caption(
                        "See the 'First Detected Real?' column -- 'No (interpolated)' means "
                        "that timestamp is a synthetic estimate, not an actual observation date."
                    )

                # Filter by priority level
                filter_pri = st.multiselect(
                    "Filter by Priority Level:",
                    ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                    default=["CRITICAL", "HIGH", "MEDIUM"]
                )
                filtered_df = df[df["priority_level"].isin(filter_pri)]

                display_df = filtered_df[[
                    "rank", "hotspot_id", "priority_level", "priority_score",
                    "transition", "area_hectares", "first_detected", "first_detected_is_real",
                    "last_confirmed", "persistence_level", "persistence_score",
                    "sensor_attribution", "confidence_level", "confidence"
                ]].copy()

                display_df.columns = [
                    "Rank", "Hotspot ID", "Priority", "Score",
                    "Transition", "Area (ha)", "First Detected", "First Detected Real?",
                    "Last Confirmed", "Persistence", "Persist Score", "Evidence Attribution",
                    "Confidence", "Conf Score"
                ]

                st.dataframe(display_df, use_container_width=True, height=350)

                # Export buttons
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    csv_data = display_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Hotspot Table (CSV)",
                        csv_data,
                        file_name=f"{aoi_name}_hotspots.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with d_col2:
                    json_str = json.dumps(prov_res.export_summary_json(), indent=2)
                    st.download_button(
                        "Download Machine-Readable Provenance Report (JSON)",
                        json_str,
                        file_name=f"{aoi_name}_provenance_report.json",
                        mime="application/json",
                        use_container_width=True,
                    )
            else:
                st.info("No significant change hotspots detected.")

        # TAB 4: Deep Drill-Down Inspector
        with t_tab4:
            st.subheader("Hotspot Forensic Investigation & Provenance Drill-Down")
            
            if len(prov_res.provenance.hotspots) > 0:
                hotspot_options = [
                    f"[{h.hotspot_id}] Priority: {h.priority_level} | {h.transition} ({h.area_hectares:.1f} ha)"
                    for h in prov_res.provenance.hotspots
                ]
                selected_hs_label = st.selectbox("Select Hotspot to Inspect:", hotspot_options, index=0)
                selected_idx = hotspot_options.index(selected_hs_label)
                hs = prov_res.provenance.hotspots[selected_idx]

                # Bounding box crop
                r0, c0, r1, c1 = hs.bounding_box
                # Add padding
                pad = 15
                h_img, w_img = prov_res.initial_class_rgb.shape[:2]
                r0_p = max(0, r0 - pad)
                r1_p = min(h_img, r1 + pad)
                c0_p = max(0, c0 - pad)
                c1_p = min(w_img, c1 + pad)

                # Patch crops
                crop_initial = prov_res.initial_class_rgb[r0_p:r1_p, c0_p:c1_p]
                crop_final = prov_res.final_class_rgb[r0_p:r1_p, c0_p:c1_p]
                crop_trans = prov_res.transition_rgb[r0_p:r1_p, c0_p:c1_p]
                crop_fused_t1 = prov_res.fused_series[0][r0_p:r1_p, c0_p:c1_p, :3]
                crop_fused_tn = prov_res.fused_series[-1][r0_p:r1_p, c0_p:c1_p, :3]

                st.markdown(
                    f"""
                    <div style="background:var(--panel); border:1px solid var(--hairline); padding:1rem; margin-bottom:1rem;">
                    <h3 style="margin-top:0; color:var(--phosphor); font-size:1.3rem;">Hotspot {hs.hotspot_id} &mdash; <span class="badge-{hs.priority_level.lower()}">{hs.priority_level} PRIORITY</span> (Score: {hs.priority_score:.2f})</h3>
                    <p style="font-size:0.92rem; color:var(--paper); line-height:1.6;">
                    <b>Scientific Explanation:</b><br>{hs.explanation}
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Visual comparison row
                pcol1, pcol2, pcol3, pcol4 = st.columns(4)
                with pcol1:
                    st.write(f"**Fused Image ({prov_res.dates[0]})**")
                    st.image(crop_fused_t1, clamp=True)
                with pcol2:
                    st.write(f"**Fused Image ({prov_res.dates[-1]})**")
                    st.image(crop_fused_tn, clamp=True)
                with pcol3:
                    st.write(f"**Class Map ({prov_res.dates[0]})**")
                    st.image(crop_initial)
                with pcol4:
                    st.write(f"**Class Map ({prov_res.dates[-1]})**")
                    st.image(crop_final)

                # Quantitative evidence metrics table
                st.markdown("#### Quantitative Evidence Breakdown")
                ec1, ec2, ec3, ec4 = st.columns(4)
                with ec1:
                    st.metric("Persistence Score", f"{hs.persistence_score:.2f}", hs.persistence_level)
                with ec2:
                    st.metric("SAR Structural Shift", f"{hs.sar_evidence:.2f} / 1.0")
                with ec3:
                    st.metric("Optical Spectral Shift", f"{hs.optical_evidence:.2f} / 1.0")
                with ec4:
                    st.metric("Sensor Agreement", f"{hs.sensor_agreement:.2f} / 1.0")

                # Step-by-step trajectory timeline
                st.markdown("#### Step-by-Step Observation Trajectory")
                traj_cols = st.columns(len(prov_res.dates))
                for idx, (d_str, c_name) in enumerate(zip(prov_res.dates, hs.trajectory)):
                    with traj_cols[idx]:
                        date_label = f"T{idx+1}: {d_str}" + (" (interp.)" if d_str in interpolated_dates else "")
                        st.markdown(
                            f"""
                            <div style="background:var(--panel-raised); border:1px solid var(--hairline); padding:0.6rem; text-align:center;">
                            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:var(--paper-dim);">{date_label}</div>
                            <div style="font-weight:700; color:var(--phosphor); font-size:0.85rem; margin-top:4px;">{c_name}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            else:
                st.info("No hotspots detected for inspection.")


# ==================================================
# 3. BASELINE 2-DATE ANALYSIS (BACKWARD COMPATIBILITY)
# ==================================================
elif page == "Baseline 2-Date Analysis":
    st.markdown('<div class="section-title">Baseline 2-Date Image Analysis</div><div class="section-underline"></div>', unsafe_allow_html=True)
    st.caption("Standard 2-date pixel difference analysis for backward compatibility testing.")

    aoi_label = st.selectbox("Area of Interest", list(DEMO_AOIS.keys()))
    aoi_name = aoi_label.split(" (")[0]
    aoi_cfg = DEMO_AOIS[aoi_label]

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.text_input("Date 1", value=aoi_cfg["date_1"], disabled=True)
    with b_col2:
        st.text_input("Date 2", value=aoi_cfg["date_2"], disabled=True)

    if st.button("Run Baseline Comparison", use_container_width=True):
        with st.spinner("Running 2-date baseline pipeline..."):
            s1_1, s2_1, s1_2, s2_2 = get_training_pair(None, aoi_cfg["date_1"], aoi_cfg["date_2"], aoi_name, data_dir=DATA_DIR)
            fused_1 = fuse(s1_1, s2_1, data_layout="HWC")
            fused_2 = fuse(s1_2, s2_2, data_layout="HWC")
            result = compare(
                fused_1, fused_2,
                metadata={"aoi": aoi_name, "date1": aoi_cfg["date_1"], "date2": aoi_cfg["date_2"], "pixel_size": 10.0},
                config={"enable_direction_heuristics": True},
            )
            st.session_state["baseline_result"] = result
            st.session_state["baseline_fused_1"] = fused_1
            st.session_state["baseline_fused_2"] = fused_2
            st.success("Baseline comparison complete.")

    if "baseline_result" in st.session_state:
        res = st.session_state["baseline_result"]
        stats = res.statistics
        fused_1 = st.session_state["baseline_fused_1"]
        fused_2 = st.session_state["baseline_fused_2"]

        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            st.markdown(f'<div class="metric-card"><h4>Changed Area</h4><p style="font-size:1.4rem;">{stats.get("changed_area_km2", "-")} km&sup2;</p></div>', unsafe_allow_html=True)
        with rcol2:
            st.markdown(f'<div class="metric-card"><h4>Change Percentage</h4><p style="font-size:1.4rem;">{stats.get("change_percentage", "-")}%</p></div>', unsafe_allow_html=True)
        with rcol3:
            dir_label = res.metadata.get("direction_heuristics", {}).get("label", "n/a")
            st.markdown(f'<div class="metric-card"><h4>Direction Heuristic</h4><p style="font-size:1.1rem;">{dir_label}</p></div>', unsafe_allow_html=True)

        icol1, icol2, icol3 = st.columns(3)
        with icol1:
            st.write(f"**Fused Image - Date 1 ({res.metadata.get('date1')})**")
            st.image(np.clip(fused_1[..., :3], 0, 1))
        with icol2:
            st.write(f"**Fused Image - Date 2 ({res.metadata.get('date2')})**")
            st.image(np.clip(fused_2[..., :3], 0, 1))
        with icol3:
            st.write("**Binary Change Map**")
            st.image(res.change_map * 255)


# ==================================================
# 4. RESEARCH EVALUATION & ABLATION SUITE
# ==================================================
elif page == "Research Evaluation & Ablation Suite":
    st.markdown('<div class="section-title">Research Evaluation & 7-Stage Ablation Suite</div><div class="section-underline"></div>', unsafe_allow_html=True)
    st.caption("Quantitative benchmarking comparing Baselines against the proposed Multi-Temporal Provenance System.")

    # Custom locations you've already run on the Change Provenance page show up
    # here too -- ablation for them was already kicked off automatically then.
    custom_options = st.session_state.get("custom_aoi_history", [])
    bench_options = list(DEMO_AOIS.keys()) + custom_options
    aoi_label = st.selectbox("Benchmark Target Area", bench_options, index=0)
    aoi_name = aoi_label.split(" (")[0]

    if custom_options:
        st.caption(
            f"Custom locations available here (already run via Change Provenance & Early Warning): "
            f"{', '.join(custom_options)}"
        )

    if f"bench_{aoi_name}_error" in st.session_state:
        st.error(
            f"The linked ablation run for {aoi_name} failed: "
            f"{st.session_state[f'bench_{aoi_name}_error']}. You can retry manually below."
        )

    if st.button("Execute Quantitative Benchmark & Ablation Study", use_container_width=True):
        with st.spinner(f"Running benchmarks and 7 ablation stages for {aoi_name}..."):
            suite = BenchmarkSuite(data_dir=DATA_DIR)
            # Custom AOIs have no entry in get_default_temporal_dates()'s presets --
            # reuse the exact dates that AOI was originally run with, if we have them
            # (set when it was run from the Change Provenance page).
            known_dates = st.session_state.get(f"dates_{aoi_name}")
            bench_res = suite.run_benchmark(aoi_name, dates=known_dates)
            st.session_state[f"bench_{aoi_name}"] = bench_res

    if f"bench_{aoi_name}" in st.session_state:
        bench_res = st.session_state[f"bench_{aoi_name}"]

        st.markdown("### 1. Comparative Performance: Baselines vs Proposed")
        
        comp_df = pd.DataFrame([
            {
                "Method": "Baseline A (Simple Fused Diff)",
                "Observations": 2,
                "Changed Area (km²)": bench_res.baseline_a_metrics["changed_area_km2"],
                "Semantic Classes": "None (Binary)",
                "Persistence Check": "No",
                "Sensor Attribution": "No",
                "Uncertainty / Confidence": "No",
                "Actionable Hotspots": "No",
                "Runtime (s)": bench_res.baseline_a_metrics["runtime_seconds"],
            },
            {
                "Method": "Baseline B (Existing CRCD-Net)",
                "Observations": 2,
                "Changed Area (km²)": bench_res.baseline_b_metrics["changed_area_km2"],
                "Semantic Classes": "Scalar Heuristic",
                "Persistence Check": "No",
                "Sensor Attribution": "No",
                "Uncertainty / Confidence": "Heuristic",
                "Actionable Hotspots": "No",
                "Runtime (s)": bench_res.baseline_b_metrics["runtime_seconds"],
            },
            {
                "Method": "Proposed System (Multi-Temporal Provenance)",
                "Observations": bench_res.proposed_method_metrics["total_observations"],
                "Changed Area (km²)": bench_res.proposed_method_metrics["changed_area_km2"],
                "Semantic Classes": f"5 Classes ({len(bench_res.proposed_method_metrics['semantic_transitions_mapped'])} Transitions)",
                "Persistence Check": "Yes (Multi-Date)",
                "Sensor Attribution": "Yes (SAR vs Optical)",
                "Uncertainty / Confidence": f"Grounded ({bench_res.proposed_method_metrics['mean_composite_confidence']*100:.1f}%)",
                "Actionable Hotspots": f"Yes ({bench_res.proposed_method_metrics['total_hotspots_detected']} Ranked)",
                "Runtime (s)": bench_res.proposed_method_metrics["runtime_seconds"],
            },
        ])
        st.dataframe(comp_df, use_container_width=True)

        st.markdown("### 2. 7-Stage Component Ablation Analysis")
        st.caption(
            "Each stage adds one capability on top of the previous stage, so you can see "
            "exactly what each piece of the system contributes on its own."
        )

        def _format_metric_label(key: str) -> str:
            return key.replace("_", " ").title()

        def _format_metric_value(key: str, value) -> str:
            if isinstance(value, bool):
                return "Yes" if value else "No"
            if isinstance(value, float):
                if "percentage" in key or "weight" in key:
                    return f"{value:.2f}%" if "percentage" in key else f"{value:.2f}"
                return f"{value:.3f}"
            if isinstance(value, list):
                return ", ".join(str(v) for v in value)
            return str(value)

        for stage_key, stage_info in bench_res.ablation_results.items():
            stage_title = stage_key.replace("_", " ")
            description = stage_info.get("description", "")
            with st.expander(f"{stage_title} -- {description}", expanded=False):
                metrics_items = [(k, v) for k, v in stage_info.items() if k != "description"]
                mcols = st.columns(min(3, max(1, len(metrics_items))))
                for idx, (k, v) in enumerate(metrics_items):
                    with mcols[idx % len(mcols)]:
                        st.markdown(
                            f'<div class="metric-card"><h4>{_format_metric_label(k)}</h4>'
                            f'<p style="font-size:1.1rem;">{_format_metric_value(k, v)}</p></div>',
                            unsafe_allow_html=True,
                        )

        st.markdown("### 3. Key Scientific Findings")
        insights = bench_res.comparative_summary
        for k, v in insights.items():
            st.info(f"**{k.replace('_', ' ').title()}**: {v}")