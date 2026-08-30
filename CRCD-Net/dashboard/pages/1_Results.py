"""
Results page: the last provenance-pipeline run's fused images, semantic /
persistence / sensor-evidence maps, metrics table, prioritized hotspot
table, and deep drill-down. Reads everything from st.session_state --
never recomputes on page switch. Run an analysis on the Home page first.
"""
from __future__ import annotations

import json
import os
import sys

# pages/*.py sit one directory deeper than Home.py, so reach dashboard/
# (where common.py lives) before importing it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common
from common import DATA_DIR

import numpy as np
import pandas as pd
import streamlit as st

common.configure_page("Results")

st.markdown('<div class="section-title">Change Provenance Results</div><div class="section-underline"></div>', unsafe_allow_html=True)

last_aoi_name = st.session_state.get("last_run_aoi_name")
if not last_aoi_name or f"prov_result_{last_aoi_name}" not in st.session_state:
    st.info(
        "No analysis has been run yet. Go to the **Home** page, pick an AOI "
        "and date range, and click **Run Full Provenance Pipeline** -- "
        "results will appear here automatically."
    )
    st.stop()

# Any AOI run this session is still cached in session_state -- default to
# the most recent one, but let the user flip back to an earlier run without
# recomputing anything.
cached_aoi_names = sorted(
    k[len("prov_result_"):] for k in st.session_state.keys()
    if k.startswith("prov_result_")
)
if len(cached_aoi_names) > 1:
    aoi_name = st.selectbox(
        "AOI Result", cached_aoi_names,
        index=cached_aoi_names.index(last_aoi_name),
    )
else:
    aoi_name = last_aoi_name

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

# Metrics table. Note: this pipeline computes change/provenance metrics
# (area, hotspots, confidence, persistence), not classic fusion-quality
# metrics -- PSNR/SSIM/RMSE/SAM aren't computed anywhere in this codebase,
# there's no reference image to compare against, so they're not shown here
# rather than being faked.
st.subheader("Metrics Table")
metrics_table = pd.DataFrame([
    {"Metric": "Changed Area (ha)", "Value": m["changed_area_hectares"]},
    {"Metric": "Change Percentage (%)", "Value": m["change_percentage"]},
    {"Metric": "Total Hotspots", "Value": m["total_hotspots"]},
    {"Metric": "Critical Priority Hotspots", "Value": m["critical_priority_hotspots"]},
    {"Metric": "High Priority Hotspots", "Value": m["high_priority_hotspots"]},
    {"Metric": "Mean Confidence (%)", "Value": round(m["mean_confidence"] * 100, 1)},
    {"Metric": "Mean Persistence (%)", "Value": round(m["mean_persistence"], 1)},
    {"Metric": "Total Observations", "Value": m["total_observations"]},
    {"Metric": "Interpolated Observations", "Value": len(interpolated_dates)},
])
st.dataframe(metrics_table, use_container_width=True, hide_index=True)

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
    from data.temporal_dataset import MultiTemporalDataset

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
