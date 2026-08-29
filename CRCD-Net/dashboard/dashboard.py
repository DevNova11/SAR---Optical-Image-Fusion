import streamlit as st

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

    st.write(
        "Select the area and dates for satellite analysis."
    )

    st.subheader("Location Coordinates")

    lat_col, lon_col = st.columns(2)

    with lat_col:
        latitude = st.number_input(
            "Latitude",
            value=17.3850
        )

    with lon_col:
        longitude = st.number_input(
            "Longitude",
            value=78.4867
        )

    st.subheader("Observation Dates")

    date1_col, date2_col = st.columns(2)

    with date1_col:
        date1 = st.date_input(
            "Date 1"
        )

    with date2_col:
        date2 = st.date_input(
            "Date 2"
        )

    st.markdown("---")

    if st.button(
        "Start Analysis",
        use_container_width=True
    ):

        st.success(
            "Analysis request created successfully!"
        )

        st.write(
            f"Location: {latitude}, {longitude}"
        )

        st.write(
            f"Dates: {date1} -> {date2}"
        )


# ==================================================
# FUSION & RESULTS PAGE (merged)
# ==================================================

elif page == "Fusion & Results":

    st.markdown(
        '<div class="section-title">SAR-Optical Image Fusion</div>'
        '<div class="section-underline"></div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This module will connect to the Person 2
        fusion implementation.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("SAR")
        st.markdown(
            '<div class="placeholder-box">Sentinel-1 input</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.subheader("Optical")
        st.markdown(
            '<div class="placeholder-box">Sentinel-2 input</div>',
            unsafe_allow_html=True
        )

    with col3:
        st.subheader("Fused")
        st.markdown(
            '<div class="placeholder-box">CRCD-Net fused output</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">Analysis Results</div>'
        '<div class="section-underline"></div>',
        unsafe_allow_html=True
    )

    rcol1, rcol2, rcol3 = st.columns(3)

    with rcol1:
        st.markdown(
            """
            <div class="metric-card">
            <h4>Changed Area</h4>
            <p style="font-size:1.5rem; font-weight:700;">-</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with rcol2:
        st.markdown(
            """
            <div class="metric-card">
            <h4>Change Percentage</h4>
            <p style="font-size:1.5rem; font-weight:700;">-</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with rcol3:
        st.markdown(
            """
            <div class="metric-card">
            <h4>Confidence</h4>
            <p style="font-size:1.5rem; font-weight:700;">-</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.info(
        "Run an analysis to display change detection results."
    )