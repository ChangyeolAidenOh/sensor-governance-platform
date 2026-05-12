"""
Isolation Forest Anomaly Detection Baseline

Unsupervised anomaly detection on C-MAPSS sensor data.
Trained on early-lifecycle "healthy" data, then scores all cycles.
Anomaly = degradation acceleration phase (approaching failure).

This is Layer 2A of the architecture:
  Layer 1 (governance) answers "can we trust this sensor data?"
  Layer 2A (this) answers "is the equipment degrading abnormally?"
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc,
)
from typing import Optional

from core_pipeline.data.preprocess import DEGRADATION_SENSORS, SENSOR_COLS


# ---------------------------------------------------------------------------
# Anomaly Label Definition
# ---------------------------------------------------------------------------

def create_anomaly_labels(df: pd.DataFrame,
                          rul_threshold: int = 50) -> pd.DataFrame:
    """Define binary anomaly labels based on RUL.

    Cycles with RUL < threshold are labeled anomalous (1).
    This represents the "degradation zone" where maintenance
    action should be triggered.

    Args:
        df: DataFrame with 'rul' column
        rul_threshold: cycles remaining below which = anomaly

    Returns:
        DataFrame with 'anomaly' column added
    """
    df = df.copy()
    df["anomaly"] = (df["rul"] < rul_threshold).astype(int)
    return df


def get_healthy_data(df: pd.DataFrame,
                     healthy_fraction: float = 0.5) -> pd.DataFrame:
    """Extract early-lifecycle data as 'healthy' reference.

    Takes the first N% of each engine's life where degradation
    has not yet begun.

    Args:
        df: full training data sorted by engine_id, cycle
        healthy_fraction: fraction of each engine's life to use

    Returns:
        DataFrame containing only healthy cycles
    """
    healthy_frames = []
    for engine_id in df["engine_id"].unique():
        engine_df = df[df["engine_id"] == engine_id]
        n_healthy = int(len(engine_df) * healthy_fraction)
        healthy_frames.append(engine_df.iloc[:n_healthy])

    return pd.concat(healthy_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Feature Preparation
# ---------------------------------------------------------------------------

def prepare_anomaly_features(df: pd.DataFrame,
                             feature_cols: Optional[list] = None
                             ) -> tuple[np.ndarray, list]:
    """Extract feature matrix for anomaly detection.

    Args:
        df: DataFrame with sensor columns
        feature_cols: columns to use (default: DEGRADATION_SENSORS)

    Returns:
        (X, feature_cols)
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns
                        if c in DEGRADATION_SENSORS
                        or c.startswith(tuple(
                            f"{s}_rmean_" for s in DEGRADATION_SENSORS
                        ))
                        or c.startswith(tuple(
                            f"{s}_rstd_" for s in DEGRADATION_SENSORS
                        ))]
        # Filter to columns that actually exist
        feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0).values
    return X, feature_cols


# ---------------------------------------------------------------------------
# Model Training & Scoring
# ---------------------------------------------------------------------------

def train_isolation_forest(healthy_df: pd.DataFrame,
                           feature_cols: Optional[list] = None,
                           contamination: float = 0.05,
                           n_estimators: int = 200,
                           random_state: int = 42) -> dict:
    """Train Isolation Forest on healthy data.

    Args:
        healthy_df: early-lifecycle healthy data
        feature_cols: feature columns
        contamination: expected anomaly fraction in healthy data
        n_estimators: number of trees
        random_state: seed

    Returns:
        dict with model, feature_cols
    """
    X_healthy, feature_cols = prepare_anomaly_features(
        healthy_df, feature_cols
    )

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_healthy)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "n_healthy_samples": len(X_healthy),
    }


def score_anomalies(model: IsolationForest,
                    df: pd.DataFrame,
                    feature_cols: list) -> pd.DataFrame:
    """Score all data points with the trained model.

    Returns DataFrame with anomaly_score and anomaly_pred columns.
    anomaly_score: lower = more anomalous (Isolation Forest convention)
    anomaly_pred: -1 = anomaly, 1 = normal (sklearn convention)
    """
    X, _ = prepare_anomaly_features(df, feature_cols)

    df = df.copy()
    df["anomaly_score"] = model.decision_function(X)
    df["anomaly_pred"] = model.predict(X)
    # Convert to 0/1: anomaly=1, normal=0
    df["anomaly_pred_binary"] = (df["anomaly_pred"] == -1).astype(int)

    return df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_anomaly_detection(df: pd.DataFrame,
                               rul_threshold: int = 50) -> dict:
    """Evaluate anomaly detection against RUL-based ground truth.

    Args:
        df: DataFrame with anomaly_score, anomaly_pred_binary, and rul columns
        rul_threshold: RUL below which is true anomaly

    Returns:
        dict with evaluation metrics
    """
    df = create_anomaly_labels(df, rul_threshold)

    y_true = df["anomaly"].values
    y_pred = df["anomaly_pred_binary"].values
    # Negate score so higher = more anomalous (for AUROC)
    y_score = -df["anomaly_score"].values

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # AUROC
    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = 0.0

    # AUPRC (more informative for imbalanced data)
    try:
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_score)
        auprc = auc(rec_curve, prec_curve)
    except ValueError:
        auprc = 0.0

    # Class distribution
    n_true_anomaly = y_true.sum()
    n_total = len(y_true)
    anomaly_rate = n_true_anomaly / n_total

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "n_true_anomaly": int(n_true_anomaly),
        "n_pred_anomaly": int(y_pred.sum()),
        "n_total": n_total,
        "anomaly_rate": round(anomaly_rate, 4),
        "rul_threshold": rul_threshold,
    }


def evaluate_multiple_thresholds(df: pd.DataFrame,
                                  rul_thresholds: list = None) -> pd.DataFrame:
    """Evaluate across multiple RUL thresholds.

    Different thresholds represent different maintenance policies:
    - Low threshold (e.g., 30): detect only imminent failures
    - High threshold (e.g., 80): detect early degradation
    """
    if rul_thresholds is None:
        rul_thresholds = [30, 50, 70, 90]

    results = []
    for threshold in rul_thresholds:
        metrics = evaluate_anomaly_detection(df, threshold)
        results.append(metrics)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Per-Engine Analysis
# ---------------------------------------------------------------------------

def analyze_engine_detections(df: pd.DataFrame,
                              rul_threshold: int = 50) -> pd.DataFrame:
    """Analyze detection performance per engine.

    Reports when each engine's anomaly was first detected
    relative to actual failure.

    Returns:
        DataFrame with per-engine detection summary
    """
    df = create_anomaly_labels(df, rul_threshold)

    results = []
    for engine_id in df["engine_id"].unique():
        engine_df = df[df["engine_id"] == engine_id].sort_values("cycle")
        max_cycle = engine_df["cycle"].max()

        # True anomaly start
        true_anomaly = engine_df[engine_df["anomaly"] == 1]
        true_start = true_anomaly["cycle"].min() if len(true_anomaly) > 0 else None

        # First predicted anomaly
        pred_anomaly = engine_df[engine_df["anomaly_pred_binary"] == 1]
        pred_start = pred_anomaly["cycle"].min() if len(pred_anomaly) > 0 else None

        # Lead time: how many cycles before true anomaly was it detected?
        if pred_start is not None and true_start is not None:
            lead_time = true_start - pred_start  # positive = early detection
        else:
            lead_time = None

        results.append({
            "engine_id": engine_id,
            "total_cycles": max_cycle,
            "true_anomaly_start": true_start,
            "pred_anomaly_start": pred_start,
            "lead_time": lead_time,
            "detected": pred_start is not None and true_start is not None,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from core_pipeline.data.preprocess import preprocess_subset

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/CMAPSSData")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--max-rul", type=int, default=125)
    parser.add_argument("--rul-threshold", type=int, default=50)
    parser.add_argument("--healthy-fraction", type=float, default=0.5)
    args = parser.parse_args()

    print(f"Loading {args.subset}...")
    data = preprocess_subset(args.data_dir, args.subset, args.max_rul)
    train_df = data["train"]

    print(f"Extracting healthy data (first {args.healthy_fraction:.0%} of life)...")
    healthy = get_healthy_data(train_df, args.healthy_fraction)
    print(f"  Healthy samples: {len(healthy)} / {len(train_df)} total")

    print("Training Isolation Forest...")
    result = train_isolation_forest(healthy)
    print(f"  Features: {len(result['feature_cols'])}")

    print("Scoring all training data...")
    scored = score_anomalies(result["model"], train_df, result["feature_cols"])

    print(f"\nEvaluation (RUL threshold={args.rul_threshold}):")
    metrics = evaluate_anomaly_detection(scored, args.rul_threshold)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nMulti-threshold evaluation:")
    multi = evaluate_multiple_thresholds(scored)
    print(multi.to_string(index=False))

    print("\nPer-engine detection summary:")
    engine_summary = analyze_engine_detections(scored, args.rul_threshold)
    detected = engine_summary["detected"].sum()
    total = len(engine_summary)
    print(f"  Engines detected: {detected}/{total}")

    lead_times = engine_summary["lead_time"].dropna()
    if len(lead_times) > 0:
        print(f"  Lead time — mean: {lead_times.mean():.1f}, "
              f"median: {lead_times.median():.1f}, "
              f"min: {lead_times.min():.0f}, "
              f"max: {lead_times.max():.0f} cycles")
