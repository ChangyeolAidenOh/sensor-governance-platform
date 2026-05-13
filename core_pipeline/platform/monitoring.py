"""
Model & Data Monitoring Module

Continuous monitoring for production deployment:
  - Data drift detection (PSI on input features)
  - Model performance tracking (prediction vs actual)
  - Sensor health degradation alerts
  - Retrain trigger logic

This is Layer 3's operational intelligence — the "운영" in
"전사 AI/ML 데이터플랫폼 구축 및 운영".
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

from core_pipeline.governance.quality_metrics import compute_psi


# ---------------------------------------------------------------------------
# Alert Definitions
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """Monitoring alert."""
    timestamp: str
    alert_type: str      # data_drift, performance_degradation, sensor_health, retrain_trigger
    severity: str        # info, warning, critical
    metric_name: str
    metric_value: float
    threshold: float
    message: str


@dataclass
class MonitoringState:
    """Tracks monitoring state across time windows."""
    alerts: list = field(default_factory=list)
    psi_history: dict = field(default_factory=dict)
    performance_history: list = field(default_factory=list)
    retrain_triggered: bool = False
    last_check: str = ""


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "psi_warning": 0.1,
    "psi_critical": 0.2,
    "sensor_health_warning": 70,
    "sensor_health_critical": 30,
    "mae_increase_pct": 20,
    "api_response_ms": 500,
    "api_error_rate": 0.01,
}


# ---------------------------------------------------------------------------
# Data Drift Monitoring
# ---------------------------------------------------------------------------

def check_data_drift(reference_df: pd.DataFrame,
                     current_df: pd.DataFrame,
                     feature_cols: list,
                     state: MonitoringState) -> list[Alert]:
    """Check PSI for each feature between reference and current data.

    PSI < 0.1: no significant drift
    0.1 <= PSI < 0.2: moderate drift (warning)
    PSI >= 0.2: significant drift (critical → retrain trigger)
    """
    alerts = []
    now = datetime.now().isoformat()
    drifted_features = []

    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref_values = reference_df[col].dropna().values
        cur_values = current_df[col].dropna().values

        if len(ref_values) < 10 or len(cur_values) < 10:
            continue

        psi = compute_psi(ref_values, cur_values)

        # Track history
        if col not in state.psi_history:
            state.psi_history[col] = []
        state.psi_history[col].append({"timestamp": now, "psi": psi})

        if psi >= THRESHOLDS["psi_critical"]:
            alerts.append(Alert(
                timestamp=now,
                alert_type="data_drift",
                severity="critical",
                metric_name=f"psi_{col}",
                metric_value=round(psi, 4),
                threshold=THRESHOLDS["psi_critical"],
                message=f"Critical drift on {col}: PSI={psi:.4f} > {THRESHOLDS['psi_critical']}",
            ))
            drifted_features.append(col)

        elif psi >= THRESHOLDS["psi_warning"]:
            alerts.append(Alert(
                timestamp=now,
                alert_type="data_drift",
                severity="warning",
                metric_name=f"psi_{col}",
                metric_value=round(psi, 4),
                threshold=THRESHOLDS["psi_warning"],
                message=f"Moderate drift on {col}: PSI={psi:.4f}",
            ))

    # Retrain trigger: if >30% of features have critical drift
    if len(drifted_features) > len(feature_cols) * 0.3:
        state.retrain_triggered = True
        alerts.append(Alert(
            timestamp=now,
            alert_type="retrain_trigger",
            severity="critical",
            metric_name="drift_feature_ratio",
            metric_value=len(drifted_features) / len(feature_cols),
            threshold=0.3,
            message=f"Retrain triggered: {len(drifted_features)}/{len(feature_cols)} features drifted",
        ))

    return alerts


# ---------------------------------------------------------------------------
# Performance Monitoring
# ---------------------------------------------------------------------------

def check_performance(predictions: np.ndarray,
                      actuals: np.ndarray,
                      baseline_mae: float,
                      state: MonitoringState) -> list[Alert]:
    """Track prediction performance and alert on degradation."""
    alerts = []
    now = datetime.now().isoformat()

    current_mae = np.mean(np.abs(predictions - actuals))
    mae_increase = (current_mae - baseline_mae) / baseline_mae * 100

    state.performance_history.append({
        "timestamp": now,
        "mae": round(current_mae, 4),
        "baseline_mae": round(baseline_mae, 4),
        "increase_pct": round(mae_increase, 2),
    })

    if mae_increase > THRESHOLDS["mae_increase_pct"]:
        alerts.append(Alert(
            timestamp=now,
            alert_type="performance_degradation",
            severity="critical",
            metric_name="mae_increase_pct",
            metric_value=round(mae_increase, 2),
            threshold=THRESHOLDS["mae_increase_pct"],
            message=f"MAE increased {mae_increase:.1f}% vs baseline ({current_mae:.2f} vs {baseline_mae:.2f})",
        ))

    return alerts


# ---------------------------------------------------------------------------
# Sensor Health Monitoring
# ---------------------------------------------------------------------------

def check_sensor_health(health_scores: dict[str, float],
                        state: MonitoringState) -> list[Alert]:
    """Alert on low sensor health scores."""
    alerts = []
    now = datetime.now().isoformat()

    avg_score = np.mean(list(health_scores.values()))

    if avg_score < THRESHOLDS["sensor_health_critical"]:
        alerts.append(Alert(
            timestamp=now,
            alert_type="sensor_health",
            severity="critical",
            metric_name="avg_sensor_health",
            metric_value=round(avg_score, 1),
            threshold=THRESHOLDS["sensor_health_critical"],
            message=f"Critical sensor health: avg={avg_score:.1f}",
        ))
    elif avg_score < THRESHOLDS["sensor_health_warning"]:
        alerts.append(Alert(
            timestamp=now,
            alert_type="sensor_health",
            severity="warning",
            metric_name="avg_sensor_health",
            metric_value=round(avg_score, 1),
            threshold=THRESHOLDS["sensor_health_warning"],
            message=f"Low sensor health: avg={avg_score:.1f}",
        ))

    # Individual sensor alerts
    for sensor, score in health_scores.items():
        if score < THRESHOLDS["sensor_health_critical"]:
            alerts.append(Alert(
                timestamp=now,
                alert_type="sensor_health",
                severity="critical",
                metric_name=f"health_{sensor}",
                metric_value=round(score, 1),
                threshold=THRESHOLDS["sensor_health_critical"],
                message=f"Sensor {sensor} health critical: {score:.1f}",
            ))

    return alerts


# ---------------------------------------------------------------------------
# Monitoring Report
# ---------------------------------------------------------------------------

def generate_monitoring_report(state: MonitoringState) -> dict:
    """Generate a summary monitoring report."""
    n_alerts = len(state.alerts)
    critical = sum(1 for a in state.alerts if a.severity == "critical")
    warnings = sum(1 for a in state.alerts if a.severity == "warning")

    drift_alerts = [a for a in state.alerts if a.alert_type == "data_drift"]
    perf_alerts = [a for a in state.alerts if a.alert_type == "performance_degradation"]

    return {
        "timestamp": datetime.now().isoformat(),
        "total_alerts": n_alerts,
        "critical": critical,
        "warnings": warnings,
        "retrain_triggered": state.retrain_triggered,
        "drift_features": len(drift_alerts),
        "performance_alerts": len(perf_alerts),
        "status": "critical" if critical > 0 else "warning" if warnings > 0 else "healthy",
    }


# ---------------------------------------------------------------------------
# Main (demo)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from core_pipeline.data.preprocess import preprocess_subset, DEGRADATION_SENSORS

    print("=== Monitoring Demo ===\n")

    data = preprocess_subset("data/raw/CMAPSSData", "FD001", 125)
    fd002 = preprocess_subset("data/raw/CMAPSSData", "FD002", 125)

    state = MonitoringState()

    # Drift check: FD001 reference vs FD002 current
    sensor_cols = [c for c in DEGRADATION_SENSORS if c in data["train"].columns]
    print("Checking data drift (FD001 → FD002)...")
    drift_alerts = check_data_drift(
        data["train"], fd002["train"], sensor_cols, state,
    )
    state.alerts.extend(drift_alerts)

    for alert in drift_alerts[:5]:
        print(f"  [{alert.severity}] {alert.message}")

    if len(drift_alerts) > 5:
        print(f"  ... and {len(drift_alerts) - 5} more alerts")

    # Report
    report = generate_monitoring_report(state)
    print(f"\n--- Report ---")
    print(f"  Status: {report['status']}")
    print(f"  Alerts: {report['total_alerts']} ({report['critical']} critical)")
    print(f"  Retrain triggered: {report['retrain_triggered']}")
