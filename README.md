# Industrial Sensor Governance & Predictive Maintenance Platform

**센서 데이터를 신뢰할 수 없는 상황에서, 예지정비 모델의 예측을 신뢰할 수 있는가?**

대부분의 예지정비 프로젝트는 센서 데이터가 정상이라는 전제 위에서 RUL(잔여수명) 모델을 만든다. 이 프로젝트는 그 전제를 의심하는 데서 출발한다.

**Dashboard:** [sensor-governance-platform.streamlit.app](https://sensor-governance-platform.streamlit.app/)

---

## Architecture

```
Raw Sensor Data (C-MAPSS)
        |
  ┌─────┴───────┐
  | LAYER 1:    |
  | Data Quality|  Completeness / PSI / Cross-sensor Correlation
  | Governance  |  → Sensor Health Score (0-100) → Quality Gate
  └─────┬───────┘
        |
  ┌─────┴──────┐
  | LAYER 2:   |
  | 2A: Anomaly|  Isolation Forest (AUROC 0.955) > Anomaly Transformer (0.894)
  | 2B: RUL    |  TFT (RMSE 16.39) > XGBoost (18.81) > Bi-LSTM (20.11)
  └─────┬──────┘
        |
  ┌─────┴──────┐
  | LAYER 3:   |
  | ML Platform|  MLflow + DVC + FastAPI + Streamlit
  └────────────┘
```

---

## Key Differentiator: Corruption Experiment

5가지 데이터 품질 문제를 의도적으로 주입하여 RUL 예측에 미치는 영향을 정량화하고, 거버넌스 레이어의 Recovery Rate를 측정.

![Corruption Impact Heatmap](figures/tab4_heatmap.png)

| Corruption Type | Strategy | Recovery Rate | Rationale |
|---|---|---|---|
| Concept Drift | Retrain Alert | **100%** (3/3 detected) | Cannot fix data — detect and trigger retraining |
| Gaussian Noise | Smoothing | **72–79%** | Rolling mean suppresses noise, preserves trend |
| Sensor Drift | Smart Sensor Drop | **80.1%** (high only) | Drop only sensors with PSI > 0.2 |
| Random Missing | Passthrough | 0% (safe) | XGBoost native NaN handling outperforms imputation |
| Stuck-at Fault | Passthrough | 0% (safe) | Stuck values within normal range — drop cost > benefit |

**Core principle:** Governance is not "always intervene" — it's "compare intervention cost vs damage, and act only when benefit exceeds cost." v1의 -1488% 역효과 → 최종 0/15 negative.

---

## Results

### Anomaly Detection (Layer 2A)

| Model | AUROC | F1 (RUL=50) | Type |
|---|---|---|---|
| Rolling Z-score | 0.327 | 0.456 | Statistical |
| **Isolation Forest** | **0.955** | **0.789** | ML (Traditional) |
| Anomaly Transformer | 0.894 | 0.052 | DL (SOTA, ICLR 2022) |

IF > AT on gradual degradation. SOTA ≠ optimal — domain characteristics determine model selection.

### RUL Prediction (Layer 2B)

| Model | Test RMSE | NASA Score | Type |
|---|---|---|---|
| XGBoost | 18.81 | 914 | Tabular ML |
| Bi-LSTM | 20.11 | 611 | DL (LSTM) |
| **TFT (best)** | **16.39** | **482** | DL (Transformer) |

TFT Variable Selection: sensor_11, 3, 4, 14 → priority targets for governance.
TFT instability: same params → RMSE 16–42 range. Production needs multi-seed protocol.

### Cross-Subset Transfer

| Transfer | RMSE | Δ % | Mapping |
|---|---|---|---|
| FD001 → FD001 | 18.81 | baseline | Same plant |
| FD001 → FD003 | 21.84 | +16% | Cross-fault |
| FD001 → FD002 | 53.99 | **+187%** | Cross-condition |
| FD001 → FD004 | 54.95 | +192% | Full transfer |

**Operating condition change (+187%) dominates fault type change (+16%).** Cross-plant deployment requires condition normalization first.

---

## ML Platform (Layer 3)

### MLflow — Experiment Tracking

![MLflow UI](figures/MLflow_1.png)
![MLflow Runs](figures/MLflow_2.png)

Parameters, metrics, artifacts, model versioning per training run.

### FastAPI — Prediction Serving

![Swagger UI](figures/Swagger.png)

E2E pipeline: Quality Gate → RUL Prediction. Single POST request returns predicted RUL + data quality score + confidence level.

### DVC — Data & Pipeline Versioning

```bash
dvc repro  # Reproduce entire pipeline
```

Data changes → automatic dependency-aware re-execution.

---

## Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)

| Subset | Engines (train/test) | Conditions | Fault Modes |
|---|---|---|---|
| FD001 | 100 / 100 | 1 | 1 (HPC) |
| FD002 | 260 / 259 | 6 | 1 (HPC) |
| FD003 | 100 / 100 | 1 | 2 (HPC+Fan) |
| FD004 | 248 / 249 | 6 | 2 (HPC+Fan) |

Source: [NASA Open Data Portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

---

## Quick Start

### Dashboard Only

Visit [sensor-governance-platform.streamlit.app](https://sensor-governance-platform.streamlit.app/)

### Full Development Setup

```bash
git clone https://github.com/ChangyeolAidenOh/sensor-governance-platform.git
cd sensor-governance-platform
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_dev.txt
pip install -e .
```

Download C-MAPSS data:
```bash
python scripts/download_data.py
# Or manually: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
# Extract to data/raw/CMAPSSData/
```

Run experiments:
```bash
python -m core_pipeline.data.preprocess --data-dir data/raw/CMAPSSData --subset FD001
python -m core_pipeline.rul.xgboost_rul --data-dir data/raw/CMAPSSData --subset FD001
python -m experiments.run_corruption_experiment_v2
python -m core_pipeline.anomaly.isolation_forest --subset FD001
python -m experiments.run_cross_subset_transfer
```

MLflow UI:
```bash
OMP_NUM_THREADS=1 mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
# → http://127.0.0.1:5000
```

FastAPI:
```bash
uvicorn core_pipeline.platform.api:app --port 8000
# → http://127.0.0.1:8000/docs
```

Dashboard (local):
```bash
streamlit run dashboard/app.py
```

> **Apple Silicon (M2/M3) Note:** MLflow + XGBoost 동시 사용 시 `OMP_NUM_THREADS=1` prefix 필요. `os.environ["OMP_NUM_THREADS"] = "1"`을 스크립트 상단에 추가하거나 터미널에서 `OMP_NUM_THREADS=1 python ...`으로 실행.

---

## Project Structure

```
sensor-governance-platform/
|-- core_pipeline/
|   |-- data/preprocess.py             # C-MAPSS loading, RUL labels, rolling features
|   |-- governance/
|   |   |-- quality_metrics.py         # 3-dimension sensor health scoring
|   |   |-- quality_gate.py            # PASS/FLAG/BLOCK gate logic
|   |   |-- corruption.py             # 5-type corruption injection
|   |-- anomaly/
|   |   |-- isolation_forest.py        # ML baseline (AUROC 0.955)
|   |   |-- rolling_zscore.py          # Statistical baseline
|   |   |-- anomaly_transformer.py     # SOTA (ICLR 2022)
|   |-- rul/
|   |   |-- xgboost_rul.py            # Tabular baseline
|   |   |-- bilstm_rul.py             # DL baseline
|   |   |-- tft_rul.py                # SOTA with Variable Selection
|   |-- platform/
|       |-- mlflow_tracker.py          # Experiment tracking
|       |-- api.py                     # FastAPI serving
|-- experiments/                       # Experiment runners + results
|-- dashboard/app.py                   # 5-tab Streamlit dashboard
|-- dvc.yaml                           # Reproducible pipeline
```

---

## Tech Stack

Python 3.10 · PyTorch · XGBoost · scikit-learn · MLflow · DVC · FastAPI · Streamlit · Plotly


