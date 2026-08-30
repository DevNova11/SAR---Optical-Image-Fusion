"""Baseline 2-date pixel-difference analysis, kept for backward-compatibility
testing against the flagship multi-temporal provenance pipeline."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common
from common import DATA_DIR, DEMO_AOIS

import numpy as np
import streamlit as st

common.configure_page("Baseline 2-Date Analysis")

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
    from change_detection import compare
    from fusion.baseline import fuse
    from handoff import get_training_pair

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
