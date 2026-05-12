# Industrial Sensor Governance & Predictive Maintenance Platform

**Can you trust your predictive maintenance model if you can't trust your sensor data?**

This project builds an end-to-end pipeline that answers this question through three layers:

```
Layer 1: Data Quality Governance    — "Can we trust this sensor data?"
Layer 2: Anomaly Detection + RUL    — "Is this equipment degrading? How much life remains?"
Layer 3: ML Platform Operations     — "Can we scale this pipeline across the enterprise?"
```

## Architecture

```
Raw Sensor Data (C-MAPSS)
        |
  [LAYER 1: Data Quality Governance]
   Completeness / Stability (PSI) / Cross-sensor Consistency
   --> Sensor Health Score (0-100) --> Quality Gate (PASS/FLAG/BLOCK)
        |
  [LAYER 2: Analytics Engine]
   2A. Anomaly Detection: Isolation Forest | USAD | Anomaly Transformer
   2B. RUL Prediction:    XGBoost | Bi-LSTM | Temporal Fusion Transformer
        |
  [LAYER 3: ML Platform Operations]
   MLflow (experiment tracking) | DVC (data versioning) | FastAPI (serving)
   --> Streamlit Dashboard (5 tabs)
```

## Key Differentiator: Corruption Experiment

Most predictive maintenance projects assume clean sensor data. This project
**systematically injects realistic data quality issues** (sensor drift, missing
data, stuck-at faults, noise, concept drift) and measures:

- **Delta RMSE**: How much does each corruption type degrade RUL prediction?
- **Recovery Rate**: How much does the governance layer recover?

This quantifies the **business value of data governance** — not as a concept,
but as a measurable improvement in prediction accuracy.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/ChangyeolAidenOh/sensor-governance-platform.git
cd sensor-governance-platform
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download C-MAPSS data
python scripts/download_data.py

# 3. Preprocess
python -m src.data.preprocess --data-dir data/raw/CMAPSSData --subset FD001

# 4. Run XGBoost baseline
python -m src.rul.xgboost_rul --data-dir data/raw/CMAPSSData --subset FD001

# 5. Run corruption experiment (core experiment)
python experiments/run_corruption_experiment.py --subset FD001
```

## Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)

| Subset | Engines (train/test) | Conditions | Fault Modes |
|--------|---------------------|------------|-------------|
| FD001  | 100 / 100           | 1          | 1 (HPC)     |
| FD002  | 260 / 259           | 6          | 1 (HPC)     |
| FD003  | 100 / 100           | 1          | 2 (HPC+Fan) |
| FD004  | 248 / 249           | 6          | 2 (HPC+Fan) |

Each engine: 3 operational settings + 21 sensor readings per cycle.

Source: [NASA Open Data Portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

## Results

*To be populated after experiments*

| Corruption Type | Severity | Baseline RMSE | Corrupted RMSE | Recovered RMSE | Recovery Rate |
|-----------------|----------|---------------|----------------|----------------|---------------|
| sensor_drift    | low      | -             | -              | -              | -             |
| ...             | ...      | ...           | ...            | ...            | ...           |

## Tech Stack

Python 3.11 · PyTorch · XGBoost · scikit-learn · MLflow · DVC · FastAPI · Streamlit · Plotly

## Project Structure

```
sensor-governance-platform/
|-- core_pipeline/
|   |-- data/preprocess.py           # C-MAPSS loading & feature engineering
|   |-- governance/
|   |   |-- quality_metrics.py       # 3-dimension health scoring
|   |   |-- quality_gate.py          # PASS/FLAG/BLOCK gate logic
|   |   |-- corruption.py            # 5-type corruption injection
|   |-- anomaly/                     # IF, USAD, Anomaly Transformer
|   |-- rul/
|   |   |-- xgboost_rul.py           # Baseline RUL model
|   |   |-- tft_rul.py               # Temporal Fusion Transformer
|   |-- platform/                    # MLflow, FastAPI, monitoring
|-- experiments/
|   |-- run_corruption_experiment.py # Core experiment runner
|-- dashboard/                       # Streamlit 5-tab dashboard
|-- notebooks/                       # EDA and analysis notebooks
```

## Author

Independent Project by Changyeol Oh
- GitHub: [ChangyeolAidenOh](https://github.com/ChangyeolAidenOh)
- M.S. Applied Mathematics and Statistics, Stony Brook University
