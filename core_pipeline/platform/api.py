"""
FastAPI Prediction Endpoint

E2E serving pipeline:
  1. Receive sensor data
  2. Quality Gate check (Layer 1)
  3. RUL prediction (Layer 2B)
  4. Return prediction + confidence + data quality score

Usage:
  uvicorn core_pipeline.platform.api:app --reload --port 8000
  # then: curl http://localhost:8000/docs for Swagger UI
"""
## Just in case
import os
os.environ["OMP_NUM_THREADS"] = "1"


import os
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core_pipeline.data.preprocess import DEGRADATION_SENSORS, SENSOR_COLS
from core_pipeline.governance.quality_metrics import compute_psi


# -------------------------------------------------------------------
# API Models
# -------------------------------------------------------------------

class SensorReading(BaseModel):
    """Single cycle sensor readings."""
    cycle: int
    sensor_values: dict[str, float] = Field(
        ..., description="Sensor name → value mapping"
    )


class PredictionRequest(BaseModel):
    """Request for RUL prediction."""
    engine_id: str
    readings: list[SensorReading] = Field(
        ..., description="Sequence of sensor readings (most recent last)",
        min_length=1,
    )


class QualityReport(BaseModel):
    """Data quality assessment."""
    status: str  # pass, flag, block
    overall_score: float
    flagged_sensors: list[str]
    message: str


class PredictionResponse(BaseModel):
    """RUL prediction result."""
    engine_id: str
    predicted_rul: float
    confidence: str  # high, medium, low
    data_quality: QualityReport
    model_version: str
    timestamp: str
    n_cycles_used: int


class HealthResponse(BaseModel):
    """API health check."""
    status: str
    model_loaded: bool
    uptime_seconds: float


# -------------------------------------------------------------------
# App
# -------------------------------------------------------------------

app = FastAPI(
    title="Sensor Governance & RUL Prediction API",
    description="E2E pipeline: Quality Gate → RUL Prediction",
    version="0.1.0",
)

# Global state
_state = {
    "model": None,
    "model_type": "xgboost",
    "model_version": "baseline_v1",
    "feature_cols": None,
    "start_time": datetime.now(),
    "reference_stats": None,  # for quality scoring
}


# -------------------------------------------------------------------
# Model Loading
# -------------------------------------------------------------------

def load_model(model_path: str = "models/xgboost_rul.pkl"):
    """Load a trained model from disk."""
    path = Path(model_path)
    if not path.exists():
        print(f"Model not found at {path}. Run training first.")
        return False

    with open(path, "rb") as f:
        saved = pickle.load(f)

    _state["model"] = saved["model"]
    _state["feature_cols"] = saved["feature_cols"]
    if "reference_stats" in saved:
        _state["reference_stats"] = saved["reference_stats"]

    print(f"Model loaded from {path}")
    return True


# -------------------------------------------------------------------
# Quality Gate (lightweight version for API)
# -------------------------------------------------------------------

def assess_quality(readings: list[SensorReading]) -> QualityReport:
    """Quick data quality assessment for incoming sensor data."""
    sensors = DEGRADATION_SENSORS
    n_readings = len(readings)

    if n_readings == 0:
        return QualityReport(
            status="block",
            overall_score=0.0,
            flagged_sensors=[],
            message="No sensor readings provided",
        )

    # Check completeness
    missing_sensors = []
    for sensor in sensors:
        present = sum(1 for r in readings if sensor in r.sensor_values)
        if present < n_readings * 0.8:
            missing_sensors.append(sensor)

    completeness = 1.0 - len(missing_sensors) / max(len(sensors), 1)

    # Check for stuck values (zero variance in last N readings)
    stuck_sensors = []
    if n_readings >= 5:
        last_5 = readings[-5:]
        for sensor in sensors:
            values = [r.sensor_values.get(sensor) for r in last_5
                      if sensor in r.sensor_values]
            if len(values) >= 3 and len(set(values)) == 1:
                stuck_sensors.append(sensor)

    # Overall score
    n_issues = len(missing_sensors) + len(stuck_sensors)
    score = max(0, 100 - n_issues * 15)

    flagged = list(set(missing_sensors + stuck_sensors))

    if score >= 70:
        status = "pass"
        message = "Data quality acceptable"
    elif score >= 30:
        status = "flag"
        message = f"Quality concerns: {len(flagged)} sensors flagged"
    else:
        status = "block"
        message = f"Poor data quality: {len(flagged)} sensors problematic"

    return QualityReport(
        status=status,
        overall_score=round(score, 1),
        flagged_sensors=flagged,
        message=message,
    )


# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------

def predict_rul(readings: list[SensorReading],
                feature_cols: list,
                model) -> tuple[float, str]:
    """Make RUL prediction from sensor readings.

    Returns:
        (predicted_rul, confidence)
    """
    # Build feature vector from last reading
    last = readings[-1]
    features = {}

    for col in feature_cols:
        if col in last.sensor_values:
            features[col] = last.sensor_values[col]
        elif col.startswith("sensor_"):
            features[col] = last.sensor_values.get(col, 0.0)
        else:
            # Rolling features — compute from readings sequence
            base_sensor = col.split("_rmean_")[0] if "_rmean_" in col else col.split("_rstd_")[0]
            window = int(col.split("_")[-1]) if "_" in col else 5

            values = [r.sensor_values.get(base_sensor, 0.0)
                      for r in readings[-window:]]

            if "_rmean_" in col:
                features[col] = np.mean(values) if values else 0.0
            elif "_rstd_" in col:
                features[col] = np.std(values) if len(values) > 1 else 0.0
            else:
                features[col] = 0.0

    # Build array in correct order
    X = np.array([[features.get(col, 0.0) for col in feature_cols]])

    # Predict
    rul = float(model.predict(X)[0])
    rul = max(0, min(125, rul))

    # Confidence based on data completeness
    n_missing = sum(1 for col in feature_cols if features.get(col, 0.0) == 0.0)
    missing_rate = n_missing / len(feature_cols)

    if missing_rate < 0.1:
        confidence = "high"
    elif missing_rate < 0.3:
        confidence = "medium"
    else:
        confidence = "low"

    return round(rul, 1), confidence


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    """API health check."""
    uptime = (datetime.now() - _state["start_time"]).total_seconds()
    return HealthResponse(
        status="healthy",
        model_loaded=_state["model"] is not None,
        uptime_seconds=round(uptime, 1),
    )


@app.post("/predict/rul", response_model=PredictionResponse)
def predict_rul_endpoint(request: PredictionRequest):
    """Predict Remaining Useful Life for an engine.

    Pipeline:
    1. Quality Gate check on sensor data
    2. RUL prediction if quality passes
    3. Return prediction + quality report
    """
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first.",
        )

    # Step 1: Quality Gate
    quality = assess_quality(request.readings)

    # Step 2: Predict (even if flagged, but note quality)
    if quality.status == "block":
        return PredictionResponse(
            engine_id=request.engine_id,
            predicted_rul=-1,
            confidence="none",
            data_quality=quality,
            model_version=_state["model_version"],
            timestamp=datetime.now().isoformat(),
            n_cycles_used=len(request.readings),
        )

    rul, confidence = predict_rul(
        request.readings, _state["feature_cols"], _state["model"],
    )

    # Downgrade confidence if quality is flagged
    if quality.status == "flag" and confidence == "high":
        confidence = "medium"

    return PredictionResponse(
        engine_id=request.engine_id,
        predicted_rul=rul,
        confidence=confidence,
        data_quality=quality,
        model_version=_state["model_version"],
        timestamp=datetime.now().isoformat(),
        n_cycles_used=len(request.readings),
    )


@app.get("/model/info")
def model_info():
    """Current model information."""
    return {
        "model_type": _state["model_type"],
        "model_version": _state["model_version"],
        "n_features": len(_state["feature_cols"]) if _state["feature_cols"] else 0,
        "status": "loaded" if _state["model"] is not None else "not_loaded",
    }


# -------------------------------------------------------------------
# Startup
# -------------------------------------------------------------------

@app.on_event("startup")
def startup():
    """Try to load model on startup."""
    model_path = os.environ.get("MODEL_PATH", "models/xgboost_rul.pkl")
    loaded = load_model(model_path)
    if not loaded:
        print("No model loaded. API will return 503 until model is available.")
