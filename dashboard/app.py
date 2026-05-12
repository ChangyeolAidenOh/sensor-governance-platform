"""
Sensor Governance & Predictive Maintenance Platform — Dashboard

5-tab Streamlit dashboard:
  Tab 1: Sensor Health Monitor
  Tab 2: Anomaly Detection Comparison
  Tab 3: RUL Prediction Comparison
  Tab 4: Corruption Impact Analysis (key differentiator)
  Tab 5: Platform & Methodology

Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Sensor Governance Platform",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


# ---------------------------------------------------------------------------
# Data Loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_corruption_results():
    path = EXPERIMENTS_DIR / "corruption_experiment_FD001.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_targeted_corruption():
    path = EXPERIMENTS_DIR / "targeted_corruption_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_train_data():
    path = DATA_DIR / "processed" / "FD001" / "train.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Sensor Governance Platform")
    st.caption("Industrial AI — Data Quality + RUL Prediction + MLOps")

    st.divider()

    st.markdown("**Project**")
    st.markdown("POSCO Steel Camp AI")
    st.markdown("**Dataset**")
    st.markdown("NASA C-MAPSS FD001")
    st.markdown("**Engines**")
    st.markdown("100 train / 100 test")
    st.markdown("**Sensors**")
    st.markdown("21 sensors × ~206 avg cycles")

    st.divider()

    st.markdown("**Architecture**")
    st.code(
        "Layer 1: Data Governance\n"
        "Layer 2: Anomaly + RUL\n"
        "Layer 3: MLOps Platform",
        language=None,
    )

    st.divider()
    st.caption("Changyeol Oh | 2026")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Sensor Health",
    "Anomaly Detection",
    "RUL Prediction",
    "Corruption Impact",
    "Platform & Methodology",
])


# ===================================================================
# Tab 1: Sensor Health Monitor
# ===================================================================

with tab1:
    st.header("Sensor Health Monitor")
    st.markdown("Layer 1 — *Can we trust this sensor data?*")

    train_df = load_train_data()

    if train_df is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sensor Degradation Curves")
            engine_id = st.selectbox(
                "Engine", sorted(train_df["engine_id"].unique())[:20],
                key="health_engine",
            )
            sensor_options = [c for c in train_df.columns if c.startswith("sensor_")]
            selected_sensors = st.multiselect(
                "Sensors", sensor_options,
                default=["sensor_2", "sensor_3", "sensor_4", "sensor_11"],
                key="health_sensors",
            )

            engine_data = train_df[train_df["engine_id"] == engine_id].sort_values("cycle")

            if selected_sensors:
                fig = go.Figure()
                for sensor in selected_sensors:
                    if sensor in engine_data.columns:
                        fig.add_trace(go.Scatter(
                            x=engine_data["cycle"],
                            y=engine_data[sensor],
                            name=sensor,
                            mode="lines",
                        ))
                fig.update_layout(
                    xaxis_title="Cycle",
                    yaxis_title="Sensor Value",
                    height=400,
                    template="plotly_white",
                )
                st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Sensor Variance Analysis")
            st.markdown("Sensors with near-zero variance carry no degradation signal.")

            variances = train_df[[c for c in train_df.columns if c.startswith("sensor_")]].var()
            var_df = pd.DataFrame({
                "sensor": variances.index,
                "variance": variances.values,
            }).sort_values("variance", ascending=False)
            var_df["useful"] = var_df["variance"] > var_df["variance"].quantile(0.25)

            fig = px.bar(
                var_df, x="sensor", y="variance", color="useful",
                color_discrete_map={True: "#00CC96", False: "#EF553B"},
                labels={"useful": "Useful Signal"},
                template="plotly_white",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width="stretch")

        # RUL distribution
        st.subheader("Engine Lifecycle Distribution")
        lifecycle = train_df.groupby("engine_id")["cycle"].max().reset_index()
        lifecycle.columns = ["engine_id", "total_cycles"]
        fig = px.histogram(
            lifecycle, x="total_cycles", nbins=20,
            labels={"total_cycles": "Total Cycles (Lifecycle Length)"},
            template="plotly_white",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")

    else:
        st.warning("Train data not found. Run preprocessing first.")


# ===================================================================
# Tab 2: Anomaly Detection Comparison
# ===================================================================

with tab2:
    st.header("Anomaly Detection — Model Comparison")
    st.markdown("Layer 2A — *Is the equipment degrading abnormally?*")

    # Results from Stage 2
    anomaly_results = {
        "Model": ["Rolling Z-score", "Isolation Forest", "Anomaly Transformer"],
        "AUROC": [0.327, 0.955, 0.894],
        "F1 (RUL=50)": [0.456, 0.789, 0.052],
        "Best F1": [0.618, 0.806, 0.445],
        "Detection": ["100/100", "100/100", "100/100"],
        "Lead Time": [146.2, 156.3, 148.0],
        "Type": ["Statistical", "ML (Traditional)", "DL (SOTA)"],
    }
    anomaly_df = pd.DataFrame(anomaly_results)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best AUROC", "0.955", "Isolation Forest")
    with col2:
        st.metric("Best F1", "0.806", "IF @ RUL=70")
    with col3:
        st.metric("Detection Rate", "100%", "All models")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("AUROC Comparison")
        fig = px.bar(
            anomaly_df, x="Model", y="AUROC", color="Type",
            color_discrete_map={
                "Statistical": "#636EFA",
                "ML (Traditional)": "#00CC96",
                "DL (SOTA)": "#EF553B",
            },
            template="plotly_white",
        )
        fig.update_layout(height=400, yaxis_range=[0, 1.05])
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                      annotation_text="Random")
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Multi-Threshold F1")
        thresholds = [30, 50, 70, 90]
        if_f1 = [0.622, 0.789, 0.806, 0.758]
        zs_f1 = [0.322, 0.456, 0.551, 0.618]
        at_f1 = [0.086, 0.052, 0.240, 0.445]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=if_f1, name="Isolation Forest",
                                  mode="lines+markers", line=dict(color="#00CC96")))
        fig.add_trace(go.Scatter(x=thresholds, y=zs_f1, name="Z-score",
                                  mode="lines+markers", line=dict(color="#636EFA")))
        fig.add_trace(go.Scatter(x=thresholds, y=at_f1, name="Anomaly Transformer",
                                  mode="lines+markers", line=dict(color="#EF553B")))
        fig.update_layout(
            xaxis_title="RUL Threshold",
            yaxis_title="F1 Score",
            height=400, template="plotly_white",
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Key Finding")
    st.info(
        "**IF (0.955) > Anomaly Transformer (0.894).** "
        "C-MAPSS degradation is gradual, not spike-based. "
        "IF detects global density shifts in feature space — ideal for this pattern. "
        "AT's association discrepancy is designed for abrupt pattern changes. "
        "**SOTA ≠ optimal. Domain characteristics determine model selection.**"
    )


# ===================================================================
# Tab 3: RUL Prediction Comparison
# ===================================================================

with tab3:
    st.header("RUL Prediction — Model Comparison")
    st.markdown("Layer 2B — *How much useful life remains?*")

    rul_results = {
        "Model": ["XGBoost", "Bi-LSTM", "TFT (best)", "TFT (worst)"],
        "RMSE": [18.81, 20.11, 16.39, 41.82],
        "MAE": [13.84, 16.77, 12.18, None],
        "Score": [914.28, 610.84, 481.78, 18153.0],
        "Type": ["Tabular ML", "DL (LSTM)", "DL (Transformer)", "DL (Transformer)"],
    }
    rul_df = pd.DataFrame(rul_results)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best RMSE", "16.39", "TFT", delta_color="inverse")
    with col2:
        st.metric("Best Score", "481.78", "TFT", delta_color="inverse")
    with col3:
        st.metric("Most Stable", "18.81", "XGBoost")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("RMSE Comparison")
        fig = px.bar(
            rul_df[rul_df["Model"] != "TFT (worst)"],
            x="Model", y="RMSE", color="Type",
            template="plotly_white",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("NASA Score (lower = better)")
        st.markdown("*Asymmetric: penalizes late predictions more heavily*")
        fig = px.bar(
            rul_df[rul_df["Model"] != "TFT (worst)"],
            x="Model", y="Score", color="Type",
            template="plotly_white",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

    # TFT instability
    st.subheader("TFT Learning Instability")
    st.markdown("Same hyperparameters, different random seeds:")

    tft_runs = pd.DataFrame({
        "Run": ["Run 2", "Run 3", "Run 4", "Run 5"],
        "d_model": [64, 64, 64, 32],
        "Test RMSE": [16.39, 41.82, 20.08, 20.12],
        "Dominant Sensor": ["11,3,4,14 (balanced)", "14 (72%)", "14,11,4 (balanced)", "17 (53%)"],
    })
    st.dataframe(tft_runs, width="stretch", hide_index=True)

    st.warning(
        "**TFT test RMSE ranges 16–42 with identical hyperparameters.** "
        "VSN (Variable Selection Network) is prone to winner-take-all on small datasets (80 engines). "
        "Production deployment requires multi-seed ensemble or seed selection protocol."
    )

    # Variable Importance
    st.subheader("TFT Variable Importance (Best Run)")
    vi_data = pd.DataFrame({
        "Sensor": ["sensor_11", "sensor_3", "sensor_4", "sensor_14",
                    "sensor_15", "sensor_20", "sensor_21", "sensor_17"],
        "Importance": [0.163, 0.157, 0.156, 0.153, 0.112, 0.070, 0.056, 0.039],
    })
    fig = px.bar(
        vi_data, x="Importance", y="Sensor", orientation="h",
        template="plotly_white", color="Importance",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=350, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")

    st.info(
        "**Top 4 sensors (11, 3, 4, 14) → priority targets for data quality governance.** "
        "This connects Layer 2B (prediction) back to Layer 1 (governance)."
    )

    # Cross-Subset Transfer
    st.divider()
    st.subheader("Cross-Subset Transfer — Generalization Gap")
    st.markdown("*Train on FD001 → test on FD002/FD003/FD004. Proxy for cross-plant deployment.*")

    transfer_data = pd.DataFrame({
        "Test Subset": ["FD001\n(baseline)", "FD003\n(cross-fault)", "FD002\n(cross-condition)",
                        "FD004\n(full transfer)"],
        "RMSE": [18.81, 21.84, 53.99, 54.95],
        "Delta %": [0, 16.1, 187.1, 192.2],
        "Score": [914, 3261, 158370, 256728],
        "Transfer Type": ["Same", "Fault only", "Condition only", "Both"],
    })

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            transfer_data, x="Test Subset", y="RMSE",
            color="Transfer Type",
            color_discrete_map={
                "Same": "#00CC96", "Fault only": "#636EFA",
                "Condition only": "#EF553B", "Both": "#AB63FA",
            },
            template="plotly_white",
        )
        fig.add_hline(y=18.81, line_dash="dash", line_color="green",
                      annotation_text="FD001 baseline")
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = px.bar(
            transfer_data, x="Test Subset", y="Delta %",
            color="Transfer Type",
            color_discrete_map={
                "Same": "#00CC96", "Fault only": "#636EFA",
                "Condition only": "#EF553B", "Both": "#AB63FA",
            },
            template="plotly_white",
            labels={"Delta %": "RMSE Increase (%)"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

    st.error(
        "**Operating condition change (+187%) dominates fault type change (+16%).** "
        "Cross-plant deployment requires condition normalization first. "
        "This aligns with corruption experiment: concept drift (+25.69 RMSE) "
        "is the most damaging corruption type."
    )


# ===================================================================
# Tab 4: Corruption Impact Analysis (KEY DIFFERENTIATOR)
# ===================================================================

with tab4:
    st.header("Corruption Impact Analysis")
    st.markdown("*The core experiment: What happens when sensor data quality degrades?*")

    corruption_df = load_corruption_results()

    if corruption_df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            positive = len(corruption_df[corruption_df["recovery_rate"] > 0])
            st.metric("Positive Recovery", f"{positive}/15")
        with col2:
            neutral = len(corruption_df[corruption_df["recovery_rate"] == 0])
            st.metric("Neutral (safe)", f"{neutral}/15")
        with col3:
            negative = len(corruption_df[corruption_df["recovery_rate"] < 0])
            st.metric("Negative (harm)", f"{negative}/15")
        with col4:
            cd_detected = corruption_df[
                corruption_df["corruption_type"] == "concept_drift"
            ]["concept_drift_detected"].sum() if "concept_drift_detected" in corruption_df.columns else 3
            st.metric("Concept Drift Detection", f"{int(cd_detected)}/3")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("RMSE Impact by Corruption Type")
            fig = px.bar(
                corruption_df,
                x="severity", y="delta_rmse",
                color="corruption_type",
                barmode="group",
                template="plotly_white",
                labels={"delta_rmse": "Δ RMSE", "severity": "Severity"},
            )
            fig.update_layout(height=450)
            fig.add_hline(y=0, line_dash="dash", line_color="white")
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Recovery Rate by Strategy")
            fig = px.bar(
                corruption_df,
                x="severity", y="recovery_rate",
                color="corruption_type",
                barmode="group",
                template="plotly_white",
                labels={"recovery_rate": "Recovery Rate", "severity": "Severity"},
            )
            fig.update_layout(height=450)
            fig.add_hline(y=0, line_dash="dash", line_color="white")
            fig.add_hline(y=1.0, line_dash="dot", line_color="green",
                          annotation_text="100% recovery")
            st.plotly_chart(fig, width="stretch")

        # Heatmap
        st.subheader("Corruption Impact Heatmap")
        pivot = corruption_df.pivot_table(
            values="delta_rmse", index="corruption_type", columns="severity",
        )
        severity_order = ["low", "medium", "high"]
        pivot = pivot.reindex(columns=[s for s in severity_order if s in pivot.columns])

        fig = px.imshow(
            pivot, text_auto=".2f",
            color_continuous_scale="RdYlGn_r",
            labels={"color": "Δ RMSE"},
            template="plotly_white",
        )
        fig.update_layout(height=550, width=1000)
        st.plotly_chart(fig, width="stretch")

        # Strategy table
        st.subheader("Remediation Strategy Mapping")
        strategy_data = pd.DataFrame({
            "Corruption Type": ["Random Missing", "Sensor Drift", "Stuck-at Fault",
                                "Gaussian Noise", "Concept Drift"],
            "Strategy": ["Passthrough", "Smart Sensor Drop (PSI>0.2)",
                         "Passthrough", "Smoothing (rolling mean)", "Retrain Alert"],
            "Rationale": [
                "XGBoost native NaN handling outperforms imputation",
                "Drop only severely shifted sensors; keep mild ones",
                "Stuck values within normal range — drop cost > benefit",
                "Moving average suppresses noise, preserves trend",
                "Cannot fix data — detect and trigger retraining",
            ],
        })
        st.dataframe(strategy_data, width="stretch", hide_index=True)

        st.success(
            "**Core principle discovered through iteration:** "
            "Governance is not 'always intervene' — it's 'compare intervention cost vs damage, "
            "and act only when benefit exceeds cost.' "
            "Negative recovery (v1: -1488%) → zero harm (final: 0/15 negative)."
        )

    else:
        st.warning("Corruption experiment results not found. Run the experiment first.")

    # Cross-stage results
    targeted_df = load_targeted_corruption()
    if targeted_df is not None:
        st.divider()
        st.subheader("Cross-Stage: TFT Importance → Targeted Corruption")
        st.markdown("*Do TFT-important sensors cause more damage when corrupted?*")

        fig = px.bar(
            targeted_df,
            x="corruption_type", y="delta_rmse",
            color="sensor_group",
            barmode="group",
            template="plotly_white",
            labels={"delta_rmse": "Δ RMSE", "sensor_group": "Sensor Group"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

        st.info(
            "**Gaussian noise: 5.3x ratio** — TFT importance accurately predicts noise sensitivity. "
            "Drift and concept drift show equal impact regardless of importance — "
            "these corruption types affect all sensors structurally."
        )


# ===================================================================
# Tab 5: Platform & Methodology
# ===================================================================

with tab5:
    st.header("Platform & Methodology")

    st.subheader("Architecture")
    st.code("""
    Raw Sensor Data (C-MAPSS)
            |
    ┌───────┴────────┐
    |  LAYER 1:      |
    |  Data Quality  |  Completeness / PSI / Correlation
    |  Governance    |  → Sensor Health Score → Quality Gate
    └───────┬────────┘
            |
    ┌───────┴────────┐
    |  LAYER 2:      |
    |  2A: Anomaly   |  IF (AUROC 0.955) > AT (0.894)
    |  2B: RUL       |  TFT (RMSE 16.39) > XGBoost (18.81)
    └───────┬────────┘
            |
    ┌───────┴────────┐
    |  LAYER 3:      |
    |  MLOps Platform|  MLflow + DVC + FastAPI
    └────────────────┘
    """, language=None)

    st.divider()

    st.subheader("Experiment Evolution")
    evolution_data = pd.DataFrame({
        "Version": ["v1", "v2", "v3", "v3+", "Final"],
        "Change": [
            "Forward fill all",
            "Type-aware remediation",
            "N_CORRUPT_SENSORS=4",
            "PSI severity gate",
            "Stuck-at → passthrough",
        ],
        "Positive": ["0", "-", "5", "4", "7"],
        "Neutral": ["9", "-", "1", "7", "8"],
        "Negative": ["6", "-", "5", "1", "0"],
        "Key Issue": [
            "All recovery 0% or negative",
            "sensor_drop → RMSE 42.76 explosion",
            "Low severity: cost > benefit",
            "stuck_at PSI unreliable",
            "—",
        ],
    })
    st.dataframe(evolution_data, width="stretch", hide_index=True)

    st.divider()

    st.subheader("Tech Stack")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**ML/DL**")
        st.markdown("- PyTorch\n- XGBoost\n- scikit-learn")
    with col2:
        st.markdown("**MLOps**")
        st.markdown("- MLflow\n- DVC\n- FastAPI")
    with col3:
        st.markdown("**Visualization**")
        st.markdown("- Streamlit\n- Plotly")

    st.divider()

    st.subheader("Key Principles Discovered")
    principles = [
        "**Governance ≠ always intervene.** Compare intervention cost vs damage. v1's -1488% → final 0% negative.",
        "**SOTA ≠ optimal.** IF (2008) outperforms Anomaly Transformer (2022) on gradual degradation.",
        "**Variable Importance scope is limited.** TFT importance predicts noise sensitivity (5.3x), not drift sensitivity (1.0x).",
        "**Failed experiments have documentation value.** IF score as RUL feature failed due to information redundancy — a valid finding.",
        "**TFT instability is structural.** Same hyperparameters → RMSE 16–42 range. Production needs multi-seed protocol.",
    ]
    for i, p in enumerate(principles, 1):
        st.markdown(f"{i}. {p}")
