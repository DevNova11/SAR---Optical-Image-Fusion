"""
CRCD-Net Dashboard: Multi-Temporal Change Provenance & Early-Warning Console.

Home page: pick an AOI (a pre-cached demo area, a drawn rectangle, or a
quick lat/lon point) and a date range, then run the full provenance
pipeline. Results are stored in st.session_state and shown on the
Results page (pages/1_Results.py), not here.
"""
from __future__ import annotations

import datetime

import folium
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

import common
from common import ALLOW_LIVE_GEE, AOI_AREA_WARNING_KM2, DATA_DIR, DEMO_AOIS, MIN_DAYS_OLD
from common import render_live_stage_checklist, render_workflow_stepper

common.configure_page("Home")

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
    unsafe_allow_html=True,
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
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="card">
        <h4>2. Semantic Mapping</h4>
        <p>Pixel-level 5-class probabilistic classification (Forest, Agriculture, Urban, Bare Land, Water) at every observation timestep.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="card">
        <h4>3. Persistence Engine</h4>
        <p>Evaluates multi-temporal trajectory stability to filter out single-date transient noise, seasonal swings, and cloud artifacts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        """
        <div class="card">
        <h4>4. Early Warning</h4>
        <p>Multi-factor priority ranking combines ecological severity, persistence, magnitude, and sensor evidence into actionable alerts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">Jump To</div><div class="section-underline"></div>', unsafe_allow_html=True)
jc1, jc2 = st.columns(2)
with jc1:
    st.markdown(
        '<div class="action-card"><div class="tag">Compatibility</div>'
        '<h4>Baseline 2-Date</h4><p>Run the standard fused pixel-difference '
        'comparison across two dates only.</p></div>',
        unsafe_allow_html=True,
    )
    if st.button("Open Baseline Analysis", use_container_width=True, key="cc_baseline"):
        st.switch_page("pages/2_Baseline_2Date_Analysis.py")
with jc2:
    st.markdown(
        '<div class="action-card"><div class="tag">Research</div>'
        '<h4>Evaluation Suite</h4><p>Benchmark the proposed system against '
        'baselines with a 7-stage component ablation study.</p></div>',
        unsafe_allow_html=True,
    )
    if st.button("Open Evaluation Suite", use_container_width=True, key="cc_eval"):
        st.switch_page("pages/3_Research_Ablation_Suite.py")

# --------------------------------------------------
# AOI + DATE SELECTION -> RUN ANALYSIS
# --------------------------------------------------
st.markdown('<div class="section-title">Multi-Temporal Change Provenance & Early Warning Console</div><div class="section-underline"></div>', unsafe_allow_html=True)

render_workflow_stepper(
    ["AOI", "Data", "Fusion", "Change", "Insights"],
    current_index=5 if any(k.startswith("prov_result_") for k in st.session_state) else 0,
)

area_source_options = ["Demo Area"] + (["Custom Location"] if ALLOW_LIVE_GEE else [])
area_source = st.radio("Area Source", area_source_options, horizontal=True)
if not ALLOW_LIVE_GEE:
    st.caption(
        "Custom Location (live GEE pull) is disabled on this deployment -- "
        "set ALLOW_LIVE_GEE = true in Streamlit secrets to re-enable it."
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
    aoi_input_method = st.radio(
        "AOI Input Method", ["Draw Rectangle on Map", "Point (quick)"], horizontal=True
    )
    drawn_bbox = None  # (west, south, east, north) in WGS84, set only by the map path

    if aoi_input_method == "Point (quick)":
        cc1, cc2 = st.columns(2)
        with cc1:
            custom_lat = st.number_input("Latitude", value=17.3850, format="%.5f")
        with cc2:
            custom_lon = st.number_input("Longitude", value=78.4867, format="%.5f")
    else:
        st.caption(
            "Draw a rectangle with the box tool (top-left of the map). "
            "Only rectangles are supported -- other draw tools are disabled."
        )
        aoi_map = folium.Map(
            location=[17.3850, 78.4867], zoom_start=11,
            # CartoDB's "dark_matter" preset now watermarks anonymous
            # (no API key) requests -- Esri's World_Dark_Gray_Base tile
            # service is genuinely key-free, confirmed by loading it live.
            tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            attr="Esri, HERE, Garmin, FAO, NOAA, USGS, OpenStreetMap contributors, GIS User Community",
        )
        Draw(
            export=False,
            draw_options={
                "rectangle": True,
                "polygon": False,
                "circle": False,
                "marker": False,
                "polyline": False,
                "circlemarker": False,
            },
            edit_options={"edit": True, "remove": True},
        ).add_to(aoi_map)
        map_state = st_folium(
            aoi_map, key="aoi_draw_map", height=420, use_container_width=True,
            returned_objects=["all_drawings"],
        )

        drawings = (map_state or {}).get("all_drawings") or []
        if drawings:
            drawn_bbox = common.bbox_from_drawn_geojson(drawings[-1])
            west, south, east, north = drawn_bbox
            area_km2 = common.bbox_area_km2(west, south, east, north)
            custom_lat, custom_lon = (south + north) / 2, (west + east) / 2
            st.info(f"**Drawn AOI area: {area_km2:,.2f} km^2**")
            if area_km2 > AOI_AREA_WARNING_KM2:
                st.warning(
                    f"This AOI is larger than the {AOI_AREA_WARNING_KM2:,.0f} km^2 guideline -- "
                    "the GEE pull and pipeline will take longer and may hit export limits. "
                    "You can still run it, or draw a smaller rectangle."
                )
        else:
            st.warning("Draw a rectangle on the map above to select an AOI before running.")
            custom_lat = custom_lon = None

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

    if drawn_bbox is not None:
        west, south, east, north = drawn_bbox
        aoi_name = (
            f"custom_rect_{south:.4f}_{west:.4f}_{north:.4f}_{east:.4f}"
            .replace(".", "p").replace("-", "m")
        )
    elif custom_lat is not None:
        aoi_name = f"custom_{custom_lat:.4f}_{custom_lon:.4f}".replace(".", "p").replace("-", "m")
    else:
        aoi_name = "custom_pending"

run_col1, run_col2 = st.columns([1, 3])
with run_col1:
    aoi_not_ready = use_custom and (not dates_list or custom_lat is None)
    run_btn = st.button(
        "Run Full Provenance Pipeline", use_container_width=True,
        disabled=aoi_not_ready,
    )
with run_col2:
    if f"prov_result_{aoi_name}" in st.session_state:
        st.caption(f"Already have a result for **{aoi_name}** -- go to the Results page, or re-run to refresh it.")

if run_btn:
    # Imported here, not at module load: keeps Home.py's import cost low
    # when nothing has been run yet, and this is the only place run_btn's
    # branch actually needs the pipeline/evaluation stack.
    from evaluation.benchmark_suite import BenchmarkSuite
    from run_provenance_pipeline import run_provenance_pipeline

    if use_custom:
        import ee
        import gee_data_collection as gdc
        gdc.init()  # must run before constructing any ee.Geometry
        # Both AOI input paths converge to the same ee.Geometry rectangle
        # shape before hitting the pipeline: a drawn rectangle's own
        # corners, or a Point buffered out and boxed to its bounds.
        if drawn_bbox is not None:
            west, south, east, north = drawn_bbox
            aoi_geometry = ee.Geometry.Rectangle([west, south, east, north])
        else:
            aoi_geometry = ee.Geometry.Point([custom_lon, custom_lat]).buffer(1000).bounds()

    spinner_msg = (
        f"Pulling live satellite data and running the full provenance pipeline for {aoi_name}..."
        if use_custom else
        f"Running multi-temporal fusion, semantic classification, persistence verification, and hotspot priority ranking for {aoi_name}..."
    )
    _stage_labels = [
        "Multi-temporal provenance pipeline (AOI -> fusion -> change -> hotspots)",
        "7-stage benchmark & ablation suite",
    ]
    _stage_box = st.empty()
    with _stage_box.container():
        render_live_stage_checklist(_stage_labels, working_index=0)

    with st.spinner(spinner_msg):
        prov_res = run_provenance_pipeline(
            aoi_name, dates=dates_list, aoi_geometry=aoi_geometry, data_dir=DATA_DIR
        )
        st.session_state[f"prov_result_{aoi_name}"] = prov_res
        st.session_state[f"dates_{aoi_name}"] = dates_list
        # The pointer the Results page reads to find "the last run" --
        # everything else here is unchanged, per-AOI cached results, same
        # as before the multi-page split.
        st.session_state["last_run_aoi_name"] = aoi_name

        with _stage_box.container():
            render_live_stage_checklist(_stage_labels, working_index=1)

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

        _stage_box.empty()

        if use_custom:
            known = st.session_state.setdefault("custom_aoi_history", [])
            if aoi_name not in known:
                known.append(aoi_name)

    st.success(f"Analysis complete for {aoi_name}. Opening the Results page...")
    st.switch_page("pages/1_Results.py")
