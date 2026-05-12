"""
Rolling Z-score Anomaly Detection Baseline

Statistical baseline: flag data points where sensor readings
deviate beyond N standard deviations from a rolling window mean.

Simple, interpretable, no training required.
Serves as the statistical reference for SOTA comparison.
"""

import numpy as np
import pandas as pd
from typing import Optional

from core_pipeline.data.preprocess import DEGRADATION_SENSORS
from core_pipeline.anomaly.isolation_forest import (
    create_anomaly_labels,
    evaluate_anomaly_detection,
    evaluate_multiple_thresholds,
    analyze_engine_detections,
)


# ---------------------------------------------------------------------------
# Rolling Z-score Computation
# ---------------------------------------------------------------------------

def compute_rolling_zscore(df: pd.DataFrame,
                           sensor_cols: Optional[list] = None,
                           window: int = 30,
                           min_periods: int = 10) -> pd.DataFrame:
    """Compute per-sensor rolling Z-score for each engine.

    Z(t) = (x(t) - rolling_mean(t)) / rolling_std(t)

    Uses a trailing window so only past data is used (no lookahead).

    Args:
        df: DataFrame sorted by engine_id, cycle
        sensor_cols: sensors to compute Z-scores for
        window: rolling window size
        min_periods: minimum observations for valid Z-score

    Returns:
        DataFrame with z_{sensor} columns added
    """
    if sensor_cols is None:
        sensor_cols = [c for c in DEGRADATION_SENSORS if c in df.columns]

    df = df.copy()
    df = df.sort_values(["engine_id", "cycle"])

    for sensor in sensor_cols:
        grouped = df.groupby("engine_id")[sensor]

        rolling_mean = grouped.transform(
            lambda x: x.rolling(window, min_periods=min_periods).mean()
        )
        rolling_std = grouped.transform(
            lambda x: x.rolling(window, min_periods=min_periods).std()
        )

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)

        df[f"z_{sensor}"] = ((df[sensor] - rolling_mean) / rolling_std).fillna(0)

    return df


def compute_composite_zscore(df: pd.DataFrame,
                              sensor_cols: Optional[list] = None,
                              method: str = "max_abs") -> pd.DataFrame:
    """Aggregate per-sensor Z-scores into a single anomaly score.

    Args:
        df: DataFrame with z_{sensor} columns
        sensor_cols: sensors (will look for z_{sensor} columns)
        method: aggregation method
            'max_abs': maximum absolute Z-score across sensors
            'mean_abs': mean absolute Z-score
            'rms': root mean square of Z-scores

    Returns:
        DataFrame with 'anomaly_score' column added
    """
    if sensor_cols is None:
        sensor_cols = [c for c in DEGRADATION_SENSORS if c in df.columns]

    z_cols = [f"z_{s}" for s in sensor_cols if f"z_{s}" in df.columns]

    if len(z_cols) == 0:
        df["anomaly_score"] = 0.0
        return df

    df = df.copy()
    z_matrix = df[z_cols].abs().values

    if method == "max_abs":
        df["anomaly_score"] = z_matrix.max(axis=1)
    elif method == "mean_abs":
        df["anomaly_score"] = z_matrix.mean(axis=1)
    elif method == "rms":
        df["anomaly_score"] = np.sqrt((z_matrix ** 2).mean(axis=1))
    else:
        raise ValueError(f"Unknown method: {method}")

    return df


# ---------------------------------------------------------------------------
# Threshold-based Detection
# ---------------------------------------------------------------------------

def detect_anomalies_zscore(df: pd.DataFrame,
                             z_threshold: float = 3.0) -> pd.DataFrame:
    """Flag anomalies where composite Z-score exceeds threshold.

    Args:
        df: DataFrame with 'anomaly_score' column
        z_threshold: Z-score threshold for anomaly flag

    Returns:
        DataFrame with 'anomaly_pred_binary' column
    """
    df = df.copy()
    df["anomaly_pred_binary"] = (df["anomaly_score"] > z_threshold).astype(int)
    return df


def optimize_z_threshold(df: pd.DataFrame,
                          rul_threshold: int = 50,
                          z_range: tuple = (1.0, 5.0),
                          n_steps: int = 20) -> dict:
    """Find the Z-score threshold that maximizes F1.

    Args:
        df: DataFrame with 'anomaly_score' and 'rul' columns
        rul_threshold: RUL-based anomaly definition
        z_range: (min, max) Z-score thresholds to search
        n_steps: number of thresholds to try

    Returns:
        dict with best threshold and metrics
    """
    df = create_anomaly_labels(df, rul_threshold)

    best_f1 = 0
    best_threshold = z_range[0]
    best_metrics = {}

    for z_thresh in np.linspace(z_range[0], z_range[1], n_steps):
        scored = detect_anomalies_zscore(df, z_thresh)
        metrics = evaluate_anomaly_detection(scored, rul_threshold)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = z_thresh
            best_metrics = metrics

    return {
        "best_z_threshold": round(best_threshold, 2),
        "best_f1": round(best_f1, 4),
        "metrics": best_metrics,
    }


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def run_zscore_pipeline(train_df: pd.DataFrame,
                        window: int = 30,
                        z_threshold: float = 3.0,
                        method: str = "max_abs",
                        sensor_cols: Optional[list] = None) -> pd.DataFrame:
    """Run complete Z-score anomaly detection pipeline.

    Args:
        train_df: training data with sensor columns
        window: rolling window size
        z_threshold: anomaly threshold
        method: composite score aggregation method
        sensor_cols: sensors to use

    Returns:
        DataFrame with anomaly scores and predictions
    """
    df = compute_rolling_zscore(train_df, sensor_cols, window)
    df = compute_composite_zscore(df, sensor_cols, method)
    df = detect_anomalies_zscore(df, z_threshold)
    return df


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
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument("--method", default="max_abs",
                        choices=["max_abs", "mean_abs", "rms"])
    args = parser.parse_args()

    print(f"Loading {args.subset}...")
    data = preprocess_subset(args.data_dir, args.subset, args.max_rul)
    train_df = data["train"]

    print(f"Computing rolling Z-scores (window={args.window})...")
    scored = run_zscore_pipeline(
        train_df, args.window, args.z_threshold, args.method,
    )

    print(f"\nEvaluation (RUL threshold={args.rul_threshold}, "
          f"Z threshold={args.z_threshold}, method={args.method}):")
    metrics = evaluate_anomaly_detection(scored, args.rul_threshold)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nOptimizing Z-threshold...")
    opt = optimize_z_threshold(scored, args.rul_threshold)
    print(f"  Best Z-threshold: {opt['best_z_threshold']}")
    print(f"  Best F1: {opt['best_f1']}")

    print(f"\nEvaluation at optimal threshold (Z={opt['best_z_threshold']}):")
    scored_opt = detect_anomalies_zscore(scored, opt["best_z_threshold"])
    metrics_opt = evaluate_anomaly_detection(scored_opt, args.rul_threshold)
    for k, v in metrics_opt.items():
        print(f"  {k}: {v}")

    print("\nMulti-threshold evaluation (optimal Z):")
    multi = evaluate_multiple_thresholds(scored_opt)
    print(multi.to_string(index=False))

    print("\nPer-engine detection summary:")
    engine_summary = analyze_engine_detections(scored_opt, args.rul_threshold)
    detected = engine_summary["detected"].sum()
    total = len(engine_summary)
    print(f"  Engines detected: {detected}/{total}")

    lead_times = engine_summary["lead_time"].dropna()
    if len(lead_times) > 0:
        print(f"  Lead time - mean: {lead_times.mean():.1f}, "
              f"median: {lead_times.median():.1f}, "
              f"min: {lead_times.min():.0f}, "
              f"max: {lead_times.max():.0f} cycles")
