import datetime
import os
import sys

import streamlit as st

# Sentinel-2 L2A products have a processing lag of a few days after
# acquisition; anything more recent than this may not exist yet regardless
# of how wide the search window is.
MIN_DAYS_OLD = 10

# dashboard.py lives in CRCD-Net/dashboard/ -- put CRCD-Net/ on sys.path so
# handoff.py, fusion/, change_detection/ (siblings of dashboard/) import,
# and resolve data/ absolutely since Streamlit's cwd depends on how/where
# it was launched, not on this file's location.
CRCD_NET_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CRCD_NET_ROOT)
DATA_DIR = os.path.join(CRCD_NET_ROOT, "data")

from change_detection import compare
from change_detection.visualization import generate_visualizations
from fusion.baseline import fuse
from handoff import get_training_pair
from land_cover import get_land_cover_delta, label_from_delta, load_land_cover_delta

# The 3 pre-exported demo AOIs -- no live GEE call needed, see DATA_CONTRACT.md.
DEMO_AOIS = {
    "bengaluru_sarjapur (urbanization)": {
        "date_1": "2019-02-01", "date_2": "2024-02-01",
        "latitude": 12.845, "longitude": 77.645,
    },
    "chennai_oragadam (industrial expansion)": {
        "date_1": "2018-02-25", "date_2": "2023-02-14",
        "latitude": 12.770, "longitude": 80.000,
    },
    "chimakurthy_quarry (deforestation/mining)": {
        "date_1": "2018-02-03", "date_2": "2023-01-28",
        "latitude": 15.550, "longitude": 79.850,
    },
}

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CRCD-Net",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS ("Signal console" dark theme: void black, phosphor green,
# monospace readouts -- see dashboard/SIGNAL_REFERENCE.html for the
# original design reference this is adapted from)
# --------------------------------------------------

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
        --fused: #e8ffb0;
        --paper: #eaf2e6;
        --paper-dim: #8fa08c;
    }

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stMain"],
    .main, .block-container {
        background-color: var(--void) !important;
        font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
    }

    [data-testid="stAppViewContainer"] * {
        color: var(--paper);
    }

    [data-testid="stHeader"] { background-color: transparent !important; }

    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--panel) !important;
        border-right: 1px solid var(--hairline);
    }

    section[data-testid="stSidebar"] * { color: var(--paper) !important; }

    section[data-testid="stSidebar"] h1 {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.04em;
        color: var(--phosphor) !important;
    }

    section[data-testid="stSidebar"] hr { border-color: var(--hairline-soft); }

    section[data-testid="stSidebar"] .stRadio label {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: var(--paper-dim) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.7rem !important;
    }

    /* Headings in general */
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--paper) !important;
    }

    /* Hero section */
    .hero {
        padding: 3rem 0;
        text-align: left;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--hairline);
    }

    .hero .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--phosphor) !important;
        margin-bottom: 1.2rem;
    }

    .hero .badge::before {
        content: "";
        width: 6px; height: 6px; border-radius: 50%;
        background: var(--phosphor);
        box-shadow: 0 0 6px 2px rgba(127,255,160,.6);
    }

    .hero h1 {
        font-size: 3rem;
        margin-bottom: 0.8rem;
        color: var(--paper) !important;
        font-weight: 700;
        letter-spacing: -0.01em;
        line-height: 1.08;
    }

    .hero h1 em {
        font-style: normal;
        color: var(--phosphor) !important;
    }

    .hero p {
        font-size: 1rem;
        color: var(--paper-dim) !important;
        max-width: 620px;
        line-height: 1.7;
    }

    /* Section titles */
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.7rem;
        font-weight: 700;
        margin-top: 2.5rem;
        margin-bottom: 0.4rem;
        color: var(--paper) !important;
        text-align: left;
    }

    .band-tag {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--phosphor-dim) !important;
        display: flex;
        align-items: center;
        gap: 0.4rem;
        margin-top: 2.2rem;
    }

    .band-tag::before {
        content: "";
        width: 5px; height: 5px;
        background: var(--phosphor);
    }

    .section-underline {
        width: 40px;
        height: 2px;
        background-color: var(--phosphor);
        margin: 0 0 1.8rem 0;
    }

    /* Cards */
    .card {
        padding: 1.5rem;
        background-color: var(--panel);
        border: 1px solid var(--hairline);
        text-align: left;
        min-height: 160px;
    }

    .card h4 {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--paper) !important;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .card p {
        color: var(--paper-dim) !important;
        font-size: 0.88rem;
        margin: 0;
        line-height: 1.6;
    }

    .metric-card {
        padding: 1.4rem;
        background-color: var(--panel);
        text-align: center;
        border: 1px solid var(--hairline);
    }

    .metric-card h4 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--paper-dim) !important;
        font-weight: 500;
        margin-bottom: 0.6rem;
    }

    .metric-card p {
        font-family: 'IBM Plex Mono', monospace !important;
        color: var(--phosphor) !important;
    }

    .placeholder-box {
        background-color: var(--panel);
        border: 1px dashed var(--hairline);
        height: 220px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--paper-dim) !important;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.8rem;
    }

    /* Buttons */
    .stButton > button {
        background-color: transparent;
        color: var(--phosphor) !important;
        border: 1px solid var(--phosphor);
        border-radius: 0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-weight: 500;
        padding: 0.7rem 1.4rem;
        transition: 0.15s;
    }

    .stButton > button:hover {
        background-color: var(--phosphor);
        color: var(--void) !important;
        box-shadow: 0 0 24px rgba(127,255,160,.3);
        border-color: var(--phosphor);
    }

    .stButton > button * { color: inherit !important; }

    /* Info / success / error / warning boxes */
    .stAlert, [data-testid="stAlert"] {
        background-color: var(--panel) !important;
        border: 1px solid var(--hairline);
        border-left: 3px solid var(--phosphor);
        border-radius: 0;
    }

    .stAlert *, [data-testid="stAlert"] * {
        color: var(--paper) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.85rem !important;
    }

    /* Number / date / select inputs */
    [data-testid="stNumberInput"],
    [data-testid="stDateInput"],
    [data-testid="stSelectbox"] {
        color: var(--paper) !important;
    }

    [data-testid="stNumberInput"] *,
    [data-testid="stDateInput"] *,
    [data-testid="stSelectbox"] * {
        color: var(--paper) !important;
        -webkit-text-fill-color: var(--paper) !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }

    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input,
    [data-testid="stSelectbox"] input,
    .stNumberInput input,
    .stDateInput input {
        background-color: var(--panel-raised) !important;
        color: var(--paper) !important;
        -webkit-text-fill-color: var(--paper) !important;
        border: 1px solid var(--hairline) !important;
        border-radius: 0 !important;
        caret-color: var(--phosphor) !important;
    }

    [data-testid="stNumberInput"] div,
    [data-testid="stDateInput"] div,
    [data-testid="stSelectbox"] div,
    [data-testid="stNumberInput"] div[data-baseweb="input"],
    [data-testid="stDateInput"] div[data-baseweb="input"],
    [data-testid="stNumberInput"] div[data-baseweb="base-input"],
    [data-testid="stDateInput"] div[data-baseweb="base-input"],
    [data-testid="stSelectbox"] div[data-baseweb="select"] {
        background-color: var(--panel-raised) !important;
        border-color: var(--hairline) !important;
        border-radius: 0 !important;
    }

    [data-testid="stNumberInput"] button {
        background-color: var(--panel-raised) !important;
        color: var(--phosphor) !important;
        border-color: var(--hairline) !important;
    }

    [data-testid="stDateInput"] svg, [data-testid="stSelectbox"] svg {
        fill: var(--phosphor) !important;
    }

    div[data-baseweb="calendar"],
    div[data-baseweb="popover"],
    ul[data-testid="stSelectboxVirtualDropdown"] {
        background-color: var(--panel-raised) !important;
        border: 1px solid var(--hairline) !important;
    }

    div[data-baseweb="calendar"] *,
    ul[data-testid="stSelectboxVirtualDropdown"] * {
        color: var(--paper) !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }

    li[aria-selected="true"] {
        background-color: var(--hairline-soft) !important;
    }

    /* Widget labels */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {
        color: var(--phosphor-dim) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    /* Radio group (Area source, Navigation) */
    [data-testid="stRadio"] label {
        font-family: 'IBM Plex Mono', monospace !important;
    }

    [data-testid="stCaptionContainer"] {
        color: var(--paper-dim) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.78rem !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background-color: var(--panel) !important;
        border: 1px solid var(--hairline) !important;
        border-radius: 0 !important;
    }

    /* Images: hairline frame to match the console aesthetic */
    [data-testid="stImage"] img {
        border: 1px solid var(--hairline);
    }

    hr { border-color: var(--hairline-soft) !important; }

    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("CRCD-Net")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Analysis",
            "Fusion & Results"
        ]
    )

    st.markdown("---")

    st.caption(
        "SAR-Optical Image Fusion "
        "and Change Detection"
    )


# ==================================================
# HOME PAGE
# ==================================================

if page == "Home":

    st.markdown(
        """
        <div class="hero">

        <div class="badge">SENTINEL-1 &middot; SENTINEL-2 // DUAL-CHANNEL</div>

        <h1>Two signals.<br>One <em>ground truth</em>.</h1>

        <p>
        CRCD-Net reads the radar backscatter of Sentinel-1 against the spectral
        detail of Sentinel-2, standardizes both to a common scale, and resolves
        where the two disagree between two dates &mdash; that disagreement is change.
        Cross-checked against a real pretrained land-cover model, not just a
        pixel-difference guess.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="band-tag">Signal Chain</div>'
        '<div class="section-title">What each channel contributes</div>'
        '<div class="section-underline"></div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="card">
            <h4>SAR &mdash; Structural Return</h4>
            <p>Radar backscatter from Sentinel-1 penetrates cloud cover and
            works day or night &mdash; it reads surface roughness and structure,
            not color.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="card">
            <h4>Optical &mdash; Spectral Reference</h4>
            <p>Sentinel-2 multispectral imagery preserves reflectance across
            visible and infrared bands &mdash; ground truth for what things
            actually look like.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="card">
            <h4>Fusion &mdash; Cross-Channel Blend</h4>
            <p>Each modality is standardized to a common scale, then blended
            per-pixel into a single fused representation the change detector
            reads from.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="band-tag">Processing Path</div>'
        '<div class="section-title">Signal path, scene to result</div>'
        '<div class="section-underline"></div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Satellite Data -> Validate -> Preprocess -> Standardize & Fuse
        -> Change Detection -> Pretrained-Model Cross-Check -> Results
        """
    )


# ==================================================
# ANALYSIS PAGE
# ==================================================

elif page == "Analysis":

    st.markdown(
        '<div class="section-title">Analysis</div>'
        '<div class="section-underline"></div>',
        unsafe_allow_html=True
    )

    mode = st.radio(
        "Area source",
        ["Demo Area (instant)", "Custom Location (live satellite pull)"],
        horizontal=True,
    )

    use_custom = mode.startswith("Custom")

    if not use_custom:
        st.write(
            "Pick one of the pre-exported demo areas -- these run instantly, "
            "no live satellite data pull needed."
        )
        st.subheader("Demo Area")
        aoi_label = st.selectbox("Area of interest", list(DEMO_AOIS.keys()))
        aoi_name = aoi_label.split(" (")[0]
        aoi_cfg = DEMO_AOIS[aoi_label]

        st.subheader("Observation Dates")
        date1_col, date2_col = st.columns(2)
        with date1_col:
            st.text_input("Date 1", value=aoi_cfg["date_1"], disabled=True)
        with date2_col:
            st.text_input("Date 2", value=aoi_cfg["date_2"], disabled=True)

    else:
        st.warning(
            "Custom locations require a live Google Earth Engine call "
            "(usually 30s-2min, sometimes fails on a flaky connection -- "
            "the demo areas above are the reliable path)."
        )
        st.subheader("Coordinates")
        lat_col, lon_col = st.columns(2)
        with lat_col:
            latitude = st.number_input("Latitude", value=17.3850, format="%.5f")
        with lon_col:
            longitude = st.number_input("Longitude", value=78.4867, format="%.5f")

        today = datetime.date.today()
        latest_valid = today - datetime.timedelta(days=MIN_DAYS_OLD)
        default_date2 = latest_valid.replace(year=latest_valid.year - 1)
        default_date1 = latest_valid.replace(year=latest_valid.year - 5)

        st.subheader("Observation Dates")
        st.caption(
            f"Pick two dates at least {MIN_DAYS_OLD} days in the past -- "
            "Sentinel-2 imagery isn't processed and available same-day."
        )
        date1_col, date2_col = st.columns(2)
        with date1_col:
            custom_date1 = st.date_input("Date 1", value=default_date1, max_value=latest_valid)
        with date2_col:
            custom_date2 = st.date_input("Date 2", value=default_date2, max_value=latest_valid)

        aoi_name = f"custom_{latitude:.4f}_{longitude:.4f}".replace(".", "p").replace("-", "m")
        aoi_cfg = {"date_1": custom_date1.isoformat(), "date_2": custom_date2.isoformat()}

    st.markdown("---")

    if st.button(
        "Start Analysis",
        use_container_width=True
    ):
        spinner_msg = (
            "Pulling live satellite data (this can take a couple of minutes)..."
            if use_custom else
            "Running data -> fusion -> change detection pipeline..."
        )
        with st.spinner(spinner_msg):
            try:
                if use_custom:
                    import ee
                    import gee_data_collection as gdc
                    gdc.init()  # must run before constructing any ee.Geometry
                    ee_aoi = ee.Geometry.Point([longitude, latitude]).buffer(1000).bounds()
                else:
                    ee_aoi = None

                s1_1, s2_1, s1_2, s2_2 = get_training_pair(
                    ee_aoi, aoi_cfg["date_1"], aoi_cfg["date_2"], aoi_name, data_dir=DATA_DIR
                )
                fused_1 = fuse(s1_1, s2_1, data_layout="HWC")
                fused_2 = fuse(s1_2, s2_2, data_layout="HWC")
                result = compare(
                    fused_1, fused_2,
                    metadata={
                        "aoi": aoi_name, "date1": aoi_cfg["date_1"], "date2": aoi_cfg["date_2"],
                        "pixel_size": 10.0,
                    },
                    config={"enable_direction_heuristics": True},
                )
                # Real, pretrained-model-backed built/trees signal (Dynamic World),
                # separate from the generic pixel-difference detector above. Demo
                # AOIs read it from cache (no extra GEE call); custom locations
                # fetch it live, best-effort -- never let this break the main result.
                land_cover_delta = load_land_cover_delta(aoi_name, data_dir=DATA_DIR)
                if land_cover_delta is None and use_custom:
                    try:
                        land_cover_delta = get_land_cover_delta(
                            ee_aoi, aoi_cfg["date_1"], aoi_cfg["date_2"], aoi_name, out_dir=DATA_DIR
                        )
                    except Exception:
                        land_cover_delta = None
                st.session_state["land_cover_delta"] = land_cover_delta
                st.session_state["land_cover_label"] = (
                    label_from_delta(land_cover_delta) if land_cover_delta else None
                )

                st.session_state["result"] = result
                st.session_state["fused_1"] = fused_1
                st.session_state["fused_2"] = fused_2
                st.session_state["aoi_name"] = aoi_name
                st.success(
                    f"Analysis complete for {aoi_name}. "
                    "See the Fusion & Results page."
                )
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")


# ==================================================
# FUSION & RESULTS PAGE (merged)
# ==================================================

elif page == "Fusion & Results":

    st.markdown(
        '<div class="section-title">SAR-Optical Image Fusion</div>'
        '<div class="section-underline"></div>',
        unsafe_allow_html=True
    )

    result = st.session_state.get("result")
    fused_1 = st.session_state.get("fused_1")
    fused_2 = st.session_state.get("fused_2")
    aoi_name = st.session_state.get("aoi_name")

    if result is None:
        st.write(
            """
            Go to the Analysis page, pick a demo area, and click
            Start Analysis -- results will appear here.
            """
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("SAR")
            st.markdown('<div class="placeholder-box">Sentinel-1 input</div>', unsafe_allow_html=True)
        with col2:
            st.subheader("Optical")
            st.markdown('<div class="placeholder-box">Sentinel-2 input</div>', unsafe_allow_html=True)
        with col3:
            st.subheader("Fused")
            st.markdown('<div class="placeholder-box">CRCD-Net fused output</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title">Analysis Results</div>'
            '<div class="section-underline"></div>',
            unsafe_allow_html=True
        )
        rcol1, rcol2, rcol3 = st.columns(3)
        for col, label in zip((rcol1, rcol2, rcol3), ("Changed Area", "Change Percentage", "Confidence")):
            with col:
                st.markdown(
                    f'<div class="metric-card"><h4>{label}</h4>'
                    '<p style="font-size:1.5rem; font-weight:700;">-</p></div>',
                    unsafe_allow_html=True
                )
        st.markdown("---")
        st.info("Run an analysis to display change detection results.")

    else:
        st.write(f"Results for **{aoi_name}**")

        viz_dir = os.path.join(CRCD_NET_ROOT, "outputs", "dashboard_last_run")
        paths = generate_visualizations(
            fused_1, fused_2, result.difference_map, result.change_map, output_dir=viz_dir
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Fused - Date 1")
            st.image(paths["date1"])
        with col2:
            st.subheader("Fused - Date 2")
            st.image(paths["date2"])
        with col3:
            st.subheader("Change Map")
            st.image(paths["change_map"])

        st.image(paths["difference"], caption="Difference map (continuous)")

        st.markdown(
            '<div class="section-title">Analysis Results</div>'
            '<div class="section-underline"></div>',
            unsafe_allow_html=True
        )

        stats = result.statistics
        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            st.markdown(
                '<div class="metric-card"><h4>Changed Area</h4>'
                f'<p style="font-size:1.5rem; font-weight:700;">{stats.get("changed_area_km2", "-")} km&sup2;</p></div>',
                unsafe_allow_html=True
            )
        with rcol2:
            st.markdown(
                '<div class="metric-card"><h4>Change Percentage</h4>'
                f'<p style="font-size:1.5rem; font-weight:700;">{stats.get("change_percentage", "-")}%</p></div>',
                unsafe_allow_html=True
            )
        with rcol3:
            direction = result.metadata.get("direction_heuristics", {})
            label = direction.get("label", "n/a") if isinstance(direction, dict) else "n/a"
            st.markdown(
                '<div class="metric-card"><h4>Generic Signal</h4>'
                f'<p style="font-size:1.1rem; font-weight:700;">{label}</p></div>'
                '<p style="font-size:0.75rem; color:#888;">pixel-difference magnitude heuristic</p>',
                unsafe_allow_html=True
            )

        land_cover_delta = st.session_state.get("land_cover_delta")
        land_cover_label = st.session_state.get("land_cover_label")
        if land_cover_delta:
            st.markdown(
                '<p style="margin-top:0.8rem; font-weight:600;">'
                f'Pretrained-Model Signal (Dynamic World): {land_cover_label}</p>',
                unsafe_allow_html=True
            )
            lcol1, lcol2 = st.columns(2)
            with lcol1:
                st.markdown(
                    '<div class="metric-card"><h4>Built Cover Change</h4>'
                    f'<p style="font-size:1.5rem; font-weight:700;">{land_cover_delta["built_delta"]*100:+.2f}%</p></div>',
                    unsafe_allow_html=True
                )
            with lcol2:
                st.markdown(
                    '<div class="metric-card"><h4>Tree Cover Change</h4>'
                    f'<p style="font-size:1.5rem; font-weight:700;">{land_cover_delta["trees_delta"]*100:+.2f}%</p></div>',
                    unsafe_allow_html=True
                )
        else:
            st.caption(
                "No Dynamic World land-cover signal available for this run "
                "(not cached, and no live fetch succeeded)."
            )

        st.markdown("---")
        with st.expander("Full statistics"):
            st.json(stats)
            if land_cover_delta:
                st.json(land_cover_delta)