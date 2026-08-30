"""Quantitative benchmarking: Baselines vs the proposed multi-temporal
provenance system, plus a 7-stage component ablation study."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common
from common import DATA_DIR, DEMO_AOIS

import pandas as pd
import streamlit as st

common.configure_page("Research Evaluation & Ablation Suite")

st.markdown('<div class="section-title">Research Evaluation & 7-Stage Ablation Suite</div><div class="section-underline"></div>', unsafe_allow_html=True)
st.caption("Quantitative benchmarking comparing Baselines against the proposed Multi-Temporal Provenance System.")

# Custom locations you've already run on the Home page show up here too --
# ablation for them was already kicked off automatically then.
custom_options = st.session_state.get("custom_aoi_history", [])
bench_options = list(DEMO_AOIS.keys()) + custom_options
aoi_label = st.selectbox("Benchmark Target Area", bench_options, index=0)
aoi_name = aoi_label.split(" (")[0]

if custom_options:
    st.caption(
        f"Custom locations available here (already run via the Home page): "
        f"{', '.join(custom_options)}"
    )

if f"bench_{aoi_name}_error" in st.session_state:
    st.error(
        f"The linked ablation run for {aoi_name} failed: "
        f"{st.session_state[f'bench_{aoi_name}_error']}. You can retry manually below."
    )

if st.button("Execute Quantitative Benchmark & Ablation Study", use_container_width=True):
    from evaluation.benchmark_suite import BenchmarkSuite

    with st.spinner(f"Running benchmarks and 7 ablation stages for {aoi_name}..."):
        suite = BenchmarkSuite(data_dir=DATA_DIR)
        # Custom AOIs have no entry in get_default_temporal_dates()'s presets --
        # reuse the exact dates that AOI was originally run with, if we have them
        # (set when it was run from the Home page).
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
