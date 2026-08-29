import os
import sys

import streamlit as st

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
# CUSTOM CSS (Light theme, black accents, gray placeholder cards)
# --------------------------------------------------

st.markdown(
    """
    <style>

    /* Force a white app background everywhere, regardless of the
       viewer's Streamlit theme (light or dark) */
    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stMain"],
    .main, .block-container {
        background-color: #ffffff !important;
    }

    /* Force default body text to dark so it is always readable
       on the white background above */
    [data-testid="stAppViewContainer"] * {
        color: #1a1a1a;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1a1a !important;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] .stRadio label {
        font-weight: 500;
    }

    /* Hero section */
    .hero {
        padding: 3rem;
        border-radius: 4px;
        background-color: #ffffff;
        text-align: left;
        margin-bottom: 2rem;
        border-bottom: 2px solid #1a1a1a;
    }

    .hero h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        color: #1a1a1a !important;
        font-weight: 800;
    }

    .hero p {
        font-size: 1.1rem;
        color: #4a4a4a !important;
    }

    .badge {
        display: inline-block;
        background-color: #1a1a1a;
        color: #ffffff !important;
        padding: 0.4rem 1rem;
        border-radius: 3px;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* Section titles */
    .section-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 2.5rem;
        margin-bottom: 0.3rem;
        color: #1a1a1a !important;
        text-align: center;
    }

    .section-underline {
        width: 50px;
        height: 3px;
        background-color: #1a1a1a;
        margin: 0 auto 2rem auto;
    }

    /* Cards */
    .card {
        padding: 1.5rem;
        border-radius: 4px;
        background-color: #d9d9d9;
        text-align: left;
        min-height: 160px;
    }

    .card h4 {
        color: #1a1a1a !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .card p {
        color: #333333 !important;
        font-size: 0.9rem;
        margin: 0;
    }

    .metric-card {
        padding: 1.5rem;
        border-radius: 4px;
        background-color: #f2f2f2;
        text-align: center;
        border: 1px solid #d9d9d9;
    }

    .metric-card h4 {
        color: #1a1a1a !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .metric-card p {
        color: #1a1a1a !important;
    }

    .placeholder-box {
        background-color: #d9d9d9;
        border-radius: 4px;
        height: 220px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #666666 !important;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    /* Buttons */
    .stButton > button {
        background-color: #1a1a1a;
        color: #ffffff !important;
        border-radius: 3px;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1.2rem;
    }

    .stButton > button:hover {
        background-color: #333333;
        color: #ffffff !important;
    }

    .stButton > button * {
        color: #ffffff !important;
    }

    /* Info / success boxes: force readable text on their own light
       background so it never inherits a mismatched theme color */
    .stAlert, [data-testid="stAlert"] {
        background-color: #eef4fb !important;
        border-radius: 4px;
    }

    .stAlert *, [data-testid="stAlert"] * {
        color: #1a1a1a !important;
    }

    /* Number input & date input fields (Latitude, Longitude,
       Date 1, Date 2): force white field background with dark,
       clearly visible text and value. Streamlit's date input does
       not always render a plain <input>, so every descendant is
       targeted, not just the input tag. */
    [data-testid="stNumberInput"],
    [data-testid="stDateInput"] {
        color: #1a1a1a !important;
    }

    [data-testid="stNumberInput"] *,
    [data-testid="stDateInput"] * {
        color: #1a1a1a !important;
        -webkit-text-fill-color: #1a1a1a !important;
    }

    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input,
    .stNumberInput input,
    .stDateInput input {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        -webkit-text-fill-color: #1a1a1a !important;
        border: 1px solid #cccccc !important;
        caret-color: #1a1a1a !important;
    }

    [data-testid="stNumberInput"] div,
    [data-testid="stDateInput"] div,
    [data-testid="stNumberInput"] div[data-baseweb="input"],
    [data-testid="stDateInput"] div[data-baseweb="input"],
    [data-testid="stNumberInput"] div[data-baseweb="base-input"],
    [data-testid="stDateInput"] div[data-baseweb="base-input"] {
        background-color: #ffffff !important;
    }

    /* Number input +/- step buttons */
    [data-testid="stNumberInput"] button {
        background-color: #f2f2f2 !important;
        color: #1a1a1a !important;
    }

    /* Date picker calendar icon and popover calendar (when opened) */
    [data-testid="stDateInput"] svg {
        fill: #1a1a1a !important;
    }

    div[data-baseweb="calendar"],
    div[data-baseweb="popover"] {
        background-color: #ffffff !important;
    }

    div[data-baseweb="calendar"] * {
        color: #1a1a1a !important;
    }

    /* Widget labels (Latitude, Longitude, Date 1, Date 2, etc.) */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {
        color: #1a1a1a !important;
    }

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

        <div class="badge">SAR-Optical Fusion Platform</div>

        <h1>CRCD-Net</h1>

        <p>
        An intelligent remote sensing platform for
        combining SAR and optical satellite imagery to detect change.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">What can CRCD-Net do?</div>'
        '<div class="section-underline"></div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="card">
            <h4>SAR Processing</h4>
            <p>Process Sentinel-1 SAR imagery
            and extract useful spatial information.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="card">
            <h4>Optical Processing</h4>
            <p>Process Sentinel-2 optical imagery
            and preserve multispectral information.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="card">
            <h4>Image Fusion</h4>
            <p>Combine SAR and optical information
            using the CRCD-Net fusion model.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">System Pipeline</div>'
        '<div class="section-underline"></div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Satellite Data -> Preprocessing -> SAR + Optical
        -> Feature Extraction -> Attention Fusion
        -> Fused Image -> Change Detection -> Results
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

        st.subheader("Observation Dates")
        date1_col, date2_col = st.columns(2)
        with date1_col:
            custom_date1 = st.date_input("Date 1")
        with date2_col:
            custom_date2 = st.date_input("Date 2")

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
                '<div class="metric-card"><h4>Signal</h4>'
                f'<p style="font-size:1.1rem; font-weight:700;">{label}</p></div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        with st.expander("Full statistics"):
            st.json(stats)