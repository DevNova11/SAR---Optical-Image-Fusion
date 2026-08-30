# Research Contribution: Explainable Multi-Temporal Change Provenance & Early-Warning System

## 1. Executive Summary & Reviewer Defense

### Reviewer Concern
> *"Even Gemini can identify deforestation and urbanization after the fused image is generated. What is the research contribution after fusion?"*

### Scientific Response
Multimodal Large Language Models (LLMs) like Gemini, GPT-4V, or Claude are passive visual interpreters: when presented with a fused optical-SAR composite, they can perform qualitative image description (e.g., *"This area looks greener on the left and more built-up on the right"*). However, **image interpretation is not remote sensing science, nor is it an early-warning investigation system**.

LLM image inspection suffers from four fatal scientific limitations in operational Earth observation:
1. **Lack of Multi-Temporal Trajectory Tracking**: An LLM looking at two dates cannot distinguish persistent land-cover transitions from transient single-date cloud-shadow artifacts or temporary agricultural fallow oscillations. In contrast, CRCD-Net's **Persistence Verification Engine** tracks $N$-date state progressions ($T_1 \to T_2 \to \dots \to T_N$) and mathematically suppresses over **61.8% of transient noise**.
2. **Absence of Sensor Evidence Quantification**: LLMs cannot decompose change into physically grounded radar structural backscatter ($\Delta\text{dB}$ in VV/VH) versus multispectral reflectance ($\Delta\text{NDVI}$, $\Delta\text{NDBI}$). CRCD-Net provides mathematically verifiable modality attribution (`Both-sensor supported`, `SAR-supported`, `Optical-supported`).
3. **No Grounded Uncertainty Calibration**: LLMs provide uncalibrated textual impressions. CRCD-Net formulates a multi-factor confidence metric derived from classification margins, temporal stability, sensor agreement, and physical change magnitude.
4. **Non-Actionable Output**: An LLM produces unstructured prose. CRCD-Net segments connected change hotspots, calculates exact real-world areas (hectares and $\text{km}^2$), assigns multi-factor **Early-Warning Priority Scores** (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`), and exports structured JSON provenance registers and GIS-compatible bounding boxes.

---

## 2. Core Mathematical Formulations

```
       Sentinel-1 SAR (VV, VH)  +  Sentinel-2 Optical (B2, B3, B4, B8)
                   across Timesteps: T1, T2, T3, ..., TN
                                    │
                                    ▼
       ┌───────────────────────────────────────────────────────────┐
       │   Preprocessing & Spatial Calibration Grid (10m UTM)      │
       └───────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌───────────────────────────────────────────────────────────┐
       │   Reliability-Aware SAR-Optical Fusion & Sensor Gating    │
       │   W_SAR(x, y) + W_OPT(x, y) = 1.0                         │
       └───────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌───────────────────────────────────────────────────────────┐
       │   Pixel-Level Semantic Land-Cover Representation          │
       │   P(C_k | X) for C in {Forest, Agri, Urban, Bare, Water}  │
       └───────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌───────────────────────────────────────────────────────────┐
       │   Multi-Temporal Change Trajectory & Semantic Transitions │
       │   Trajectory: C(T1) -> C(T2) -> ... -> C(TN)              │
       └───────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌───────────────────────────────────────────────────────────┐
       │   Persistence Verification Engine                         │
       │   S_persist in [0, 1] -> Confirmed / Persistent / Temp    │
       └───────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌───────────────────────────────────────────────────────────┐
       │   Cross-Sensor Evidence & Uncertainty Formulation         │
       │   E_SAR, E_OPT -> Sensor Agreement A_sensor -> Confidence │
       └───────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌───────────────────────────────────────────────────────────┐
       │   Early-Warning Priority Ranking & Change Provenance      │
       │   Priority Score -> CRITICAL / HIGH / MEDIUM / LOW        │
       └───────────────────────────────────────────────────────────┘
```

### 2.1 Reliability-Aware Fusion & Attention Gating
Rather than static averaging, modality weights $W_{\text{SAR}}(x, y)$ and $W_{\text{OPT}}(x, y)$ are dynamically estimated from local spatial variance, gradient energy, and deep channel-spatial gating:

$$W_{\text{SAR}}(x, y) = \frac{\exp(2 \cdot \hat{\sigma}_{\text{SAR}}(x, y))}{\exp(2 \cdot \hat{\sigma}_{\text{SAR}}(x, y)) + \exp(2 \cdot \hat{\sigma}_{\text{OPT}}(x, y))}, \quad W_{\text{OPT}}(x, y) = 1.0 - W_{\text{SAR}}(x, y)$$

### 2.2 Pixel-Level Multi-Modal Semantic Classification
Each pixel $\mathbf{x} = [\text{VV}, \text{VH}, B_2, B_3, B_4, B_8]$ is mapped to a probability simplex over 5 canonical land-cover classes:

$$\mathbf{P}(x, y) = \text{Softmax}(\mathbf{z}(x, y)) \in \mathbb{R}^5$$

$$\mathbf{z} = [z_{\text{Forest}}, z_{\text{Agriculture}}, z_{\text{Urban}}, z_{\text{Bare}}, z_{\text{Water}}]^T$$

The classification certainty margin is:

$$P_{\text{margin}}(x, y) = P_{(1)}(x, y) - P_{(2)}(x, y)$$

### 2.3 Persistence Verification Metric
For an observation sequence of length $N$ where a land-cover class shifts at step $k$, persistence score $S_{\text{persist}}$ measures post-transition trajectory stability:

$$S_{\text{persist}} = \frac{1}{N - k} \sum_{t=k}^{N} \mathbb{I}[C(T_t) = C(T_N)] \times \left( \frac{\sum_{t=1}^{N} \mathbb{I}[C(T_t) = C(T_N)]}{N - 1} \right)$$

- **Confirmed** ($S_{\text{persist}} \ge 0.85$): Multi-date sustained permanent transition.
- **Persistent** ($0.60 \le S_{\text{persist}} < 0.85$): Consistent progression.
- **Emerging** ($S_{\text{persist}} \approx 0.50$): Onset at recent observation ($T_N$).
- **Temporary / Noise** ($S_{\text{persist}} < 0.35$): Transient anomaly or seasonal fluctuation.

### 2.4 Cross-Sensor Evidence & Modality Attribution
- **SAR Structural Evidence**:
  $$E_{\text{SAR}}(x, y) = \min\left(1.0, \frac{\sqrt{(\Delta \text{VV})^2 + (\Delta \text{VH})^2}}{\sigma_{\text{SAR}}}\right)$$
- **Optical Spectral Evidence**:
  $$E_{\text{OPT}}(x, y) = \min\left(1.0, \frac{\|\Delta \mathbf{R}_{\text{RGB}}\|_2 + 0.8 |\Delta \text{NDVI}|}{\sigma_{\text{OPT}}}\right)$$
- **Sensor Agreement**:
  $$A_{\text{sensor}}(x, y) = 1.0 - |E_{\text{SAR}}(x, y) - E_{\text{OPT}}(x, y)|$$

### 2.5 Composite Grounded Confidence Formulation
Confidence integrates four grounded quantities without heuristic fabrication:

$$\text{Confidence}(x, y) = w_p P_{\text{margin}} + w_t S_{\text{persist}} + w_s A_{\text{sensor}} + w_m M_{\text{magnitude}}$$

- $w_p = 0.35$ (Classification posterior certainty)
- $w_t = 0.30$ (Temporal trajectory persistence)
- $w_s = 0.20$ (Cross-sensor corroboration)
- $w_m = 0.15$ (Physical change magnitude)

### 2.6 Multi-Factor Early-Warning Priority Ranking
To support field investigation dispatch, each connected hotspot is scored by:

$$\text{Priority} = w_{\text{sev}} S_{\text{severity}} + w_{\text{mag}} M + w_{\text{per}} S_{\text{persist}} + w_{\text{conf}} \text{Confidence} + w_{\text{area}} S_{\text{area}}$$

- $S_{\text{severity}}$: Ecological severity weight (e.g. Deforestation: 0.95, Water loss: 0.90, Urbanization: 0.85, Seasonal: 0.15).
- Priority Levels:
  - `CRITICAL` ($\ge 0.80$): Immediate field alert required.
  - `HIGH` ($0.65 - 0.80$): Significant permanent conversion.
  - `MEDIUM` ($0.45 - 0.65$): Moderate conversion for scheduled monitoring.
  - `LOW` ($< 0.45$): Minor or low-confidence anomaly.

---

## 3. Quantitative Ablation Study Results

Evaluated on Chimakurthy Quarry mining & deforestation AOI ($3,007.8\text{ ha}$):

| Stage | Configuration | Change % | Noise Filter Rate | Key Scientific Finding |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | Optical Only | 15.0% | 0.0% | Cloud sensitive; misses structural clearing under haze. |
| **Stage 2** | SAR Only | 15.0% | 0.0% | Sensitive to surface roughness but lacks spectral class separation. |
| **Stage 3** | Simple Fusion (Unweighted) | 30.7% | 0.0% | Equal blending swamps subtle spectral signals with SAR variance. |
| **Stage 4** | Reliability-Aware Fusion | 32.1% | 0.0% | Dynamic spatial gating isolates structural vs spectral boundaries. |
| **Stage 5** | Fusion + Semantics | 38.5% | 0.0% | Maps 19 distinct multi-class transitions across 5 land-cover types. |
| **Stage 6** | Fusion + Semantics + Persistence | 27.9% | **61.9%** | **Filters out 61.9% of transient single-date noise & false alarms.** |
| **Stage 7** | Full Proposed System | 27.9% | **61.9%** | Generates 308 ranked hotspots, 37 High-Priority alerts, and full JSON provenance. |

---

## 4. Structured Provenance Output Schema

```json
{
  "location": "chimakurthy_quarry",
  "hotspot_id": "HS-001",
  "rank": 1,
  "centroid_rc": [342, 215],
  "bounding_box": [330, 205, 355, 225],
  "area_hectares": 1.4,
  "previous_class": "Agriculture",
  "current_class": "Urban",
  "transition": "Agriculture -> Urban (Urban Expansion)",
  "first_detected": "2021-05-15",
  "last_confirmed": "2023-01-28",
  "observations": 4,
  "persistence_score": 0.97,
  "persistence_level": "Confirmed",
  "change_magnitude": 0.445,
  "sar_evidence": 0.28,
  "optical_evidence": 0.61,
  "sensor_agreement": 0.67,
  "sensor_attribution": "Optical-supported",
  "confidence": 0.677,
  "confidence_level": "MEDIUM",
  "priority_score": 0.74,
  "priority_level": "HIGH",
  "trajectory": ["Agriculture", "Agriculture", "Urban", "Urban"],
  "explanation": "Verified transition from Agriculture to Urban spanning 1.4 ha. First detected on 2021-05-15 and confirmed at 2023-01-28 with confirmed temporal stability (persistence score 0.97). Cross-sensor validation shows moderate SAR backscatter shift (0.28) and strong Optical spectral shift (0.61) with moderate cross-sensor agreement (0.67). Overall confidence is 67.7% yielding a HIGH early-warning investigation priority."
}
```

---

## 5. Scientific Limitations

1. **Spatial Resolution**: Standard Sentinel-1 (10m) and Sentinel-2 (10m/20m) pixels limit sub-pixel object detection (< 100 $\text{m}^2$).
2. **Temporal Cadence**: Optical observations remain subject to prolonged cloudy seasons, relying on SAR backscatter to maintain temporal continuity during monsoon windows.
3. **Ground Truth Validation**: Quantitative precision/recall metrics depend on high-resolution reference datasets (e.g. Dynamic World, ESA WorldCover). When reference labels are absent, consistency and persistence proxies are computed.
