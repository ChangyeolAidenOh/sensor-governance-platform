"""
Extended Anomaly Detection Evaluation — 3-way Evaluation Framework

Point-wise F1:  standard binary classification (already in isolation_forest.py)
Range-based F1: credit for detecting any point within an anomaly segment
Event-based F1: credit for detecting entire anomaly events (segment-level)

Motivation: point-adjust F1 (used in original Anomaly Transformer paper)
has known issues — TiSAT (2022) showed Random Guess can outperform SOTA
under point-adjust. Range-based and Event-based avoid this inflation
while still accounting for time-series structure.

Usage:
    python -m core_pipeline.anomaly.extended_evaluation --subset FD001
"""

import numpy as np
import pandas as pd
from typing import Optional

from core_pipeline.data.preprocess import preprocess_subset, DEGRADATION_SENSORS
from core_pipeline.anomaly.isolation_forest import (
    get_healthy_data, train_isolation_forest, score_anomalies,
    create_anomaly_labels,
)
from core_pipeline.anomaly.rolling_zscore import run_zscore_pipeline, optimize_z_threshold


# ---------------------------------------------------------------------------
# Segment Extraction
# ---------------------------------------------------------------------------

def extract_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    """Extract contiguous segments of 1s from binary label array.

    Returns list of (start, end) inclusive index pairs.
    """
    segments = []
    in_segment = False
    start = 0

    for i, val in enumerate(labels):
        if val == 1 and not in_segment:
            start = i
            in_segment = True
        elif val == 0 and in_segment:
            segments.append((start, i - 1))
            in_segment = False

    if in_segment:
        segments.append((start, len(labels) - 1))

    return segments


# ---------------------------------------------------------------------------
# Range-based Evaluation
# ---------------------------------------------------------------------------

def range_based_f1(y_true: np.ndarray, y_pred: np.ndarray,
                   alpha: float = 0.0) -> dict:
    """Range-based precision, recall, F1.

    For each true anomaly segment, check if ANY predicted point falls
    within the segment (with optional alpha extension on both sides).

    For each predicted segment, check if it overlaps with ANY true segment.

    Args:
        y_true: binary ground truth
        y_pred: binary predictions
        alpha: fraction of segment length to extend on each side (0 = exact)

    Returns:
        dict with range_precision, range_recall, range_f1
    """
    true_segments = extract_segments(y_true)
    pred_segments = extract_segments(y_pred)

    if len(true_segments) == 0 and len(pred_segments) == 0:
        return {"range_precision": 1.0, "range_recall": 1.0, "range_f1": 1.0}
    if len(true_segments) == 0:
        return {"range_precision": 0.0, "range_recall": 1.0, "range_f1": 0.0}
    if len(pred_segments) == 0:
        return {"range_precision": 1.0, "range_recall": 0.0, "range_f1": 0.0}

    n = len(y_true)

    # Range recall: fraction of true segments detected by predictions
    detected_true = 0
    for t_start, t_end in true_segments:
        seg_len = t_end - t_start + 1
        ext = int(seg_len * alpha)
        ext_start = max(0, t_start - ext)
        ext_end = min(n - 1, t_end + ext)

        # Check if any prediction falls in extended range
        if y_pred[ext_start:ext_end + 1].sum() > 0:
            detected_true += 1

    range_recall = detected_true / len(true_segments)

    # Range precision: fraction of predicted segments that overlap with truth
    correct_pred = 0
    for p_start, p_end in pred_segments:
        for t_start, t_end in true_segments:
            seg_len = t_end - t_start + 1
            ext = int(seg_len * alpha)
            ext_start = max(0, t_start - ext)
            ext_end = min(n - 1, t_end + ext)

            # Check overlap
            if p_start <= ext_end and p_end >= ext_start:
                correct_pred += 1
                break

    range_precision = correct_pred / len(pred_segments)

    # F1
    if range_precision + range_recall > 0:
        range_f1 = 2 * range_precision * range_recall / (range_precision + range_recall)
    else:
        range_f1 = 0.0

    return {
        "range_precision": round(range_precision, 4),
        "range_recall": round(range_recall, 4),
        "range_f1": round(range_f1, 4),
        "n_true_segments": len(true_segments),
        "n_pred_segments": len(pred_segments),
    }


# ---------------------------------------------------------------------------
# Event-based Evaluation
# ---------------------------------------------------------------------------

def event_based_f1(y_true: np.ndarray, y_pred: np.ndarray,
                   min_overlap: float = 0.5) -> dict:
    """Event-based precision, recall, F1.

    A true event is "detected" if predicted anomaly points cover
    at least min_overlap fraction of the event.

    A predicted event is "correct" if it overlaps at least min_overlap
    fraction with any true event.

    Args:
        y_true: binary ground truth
        y_pred: binary predictions
        min_overlap: minimum overlap fraction for a match

    Returns:
        dict with event_precision, event_recall, event_f1
    """
    true_segments = extract_segments(y_true)
    pred_segments = extract_segments(y_pred)

    if len(true_segments) == 0 and len(pred_segments) == 0:
        return {"event_precision": 1.0, "event_recall": 1.0, "event_f1": 1.0}
    if len(true_segments) == 0:
        return {"event_precision": 0.0, "event_recall": 1.0, "event_f1": 0.0}
    if len(pred_segments) == 0:
        return {"event_precision": 1.0, "event_recall": 0.0, "event_f1": 0.0}

    # Event recall: fraction of true events sufficiently covered by predictions
    detected_events = 0
    for t_start, t_end in true_segments:
        seg_len = t_end - t_start + 1
        overlap = y_pred[t_start:t_end + 1].sum()
        if overlap / seg_len >= min_overlap:
            detected_events += 1

    event_recall = detected_events / len(true_segments)

    # Event precision: fraction of predicted events that match a true event
    correct_events = 0
    for p_start, p_end in pred_segments:
        pred_len = p_end - p_start + 1
        best_overlap = 0

        for t_start, t_end in true_segments:
            # Compute overlap
            overlap_start = max(p_start, t_start)
            overlap_end = min(p_end, t_end)
            if overlap_start <= overlap_end:
                overlap = overlap_end - overlap_start + 1
                best_overlap = max(best_overlap, overlap / pred_len)

        if best_overlap >= min_overlap:
            correct_events += 1

    event_precision = correct_events / len(pred_segments)

    # F1
    if event_precision + event_recall > 0:
        event_f1 = 2 * event_precision * event_recall / (event_precision + event_recall)
    else:
        event_f1 = 0.0

    return {
        "event_precision": round(event_precision, 4),
        "event_recall": round(event_recall, 4),
        "event_f1": round(event_f1, 4),
        "detected_events": detected_events,
        "total_true_events": len(true_segments),
        "total_pred_events": len(pred_segments),
    }


# ---------------------------------------------------------------------------
# Full 3-way Evaluation
# ---------------------------------------------------------------------------

def evaluate_3way(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  y_score: np.ndarray,
                  rul_threshold: int = 50) -> dict:
    """Run all three evaluation frameworks.

    Args:
        y_true: binary ground truth (per engine, concatenated)
        y_pred: binary predictions
        y_score: continuous anomaly scores (higher = more anomalous)
        rul_threshold: used for labeling

    Returns:
        dict with point-wise, range-based, event-based metrics
    """
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, roc_auc_score,
    )

    # Point-wise
    point = {
        "point_precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "point_recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "point_f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auroc": round(roc_auc_score(y_true, y_score), 4) if len(np.unique(y_true)) > 1 else 0.0,
    }

    # Range-based
    range_metrics = range_based_f1(y_true, y_pred, alpha=0.0)

    # Event-based
    event_metrics = event_based_f1(y_true, y_pred, min_overlap=0.5)

    return {**point, **range_metrics, **event_metrics}


# ---------------------------------------------------------------------------
# Per-Engine 3-way (proper time-series evaluation)
# ---------------------------------------------------------------------------

def evaluate_per_engine_3way(df: pd.DataFrame,
                              rul_threshold: int = 50) -> dict:
    """Evaluate per engine then average — proper time-series evaluation.

    Each engine is an independent time series. Segments and events
    are computed per engine to avoid cross-engine contamination.
    """
    df = create_anomaly_labels(df, rul_threshold)

    all_point_f1 = []
    all_range_f1 = []
    all_event_recall = []
    all_event_precision = []

    for engine_id in df["engine_id"].unique():
        engine_df = df[df["engine_id"] == engine_id].sort_values("cycle")
        y_true = engine_df["anomaly"].values
        y_pred = engine_df["anomaly_pred_binary"].values

        if y_true.sum() == 0:
            continue

        # Point F1
        from sklearn.metrics import f1_score
        pf1 = f1_score(y_true, y_pred, zero_division=0)
        all_point_f1.append(pf1)

        # Range F1
        r = range_based_f1(y_true, y_pred)
        all_range_f1.append(r["range_f1"])

        # Event F1
        e = event_based_f1(y_true, y_pred)
        all_event_recall.append(e["event_recall"])
        all_event_precision.append(e["event_precision"])

    avg_point_f1 = np.mean(all_point_f1) if all_point_f1 else 0
    avg_range_f1 = np.mean(all_range_f1) if all_range_f1 else 0
    avg_event_recall = np.mean(all_event_recall) if all_event_recall else 0
    avg_event_precision = np.mean(all_event_precision) if all_event_precision else 0

    if avg_event_precision + avg_event_recall > 0:
        avg_event_f1 = 2 * avg_event_precision * avg_event_recall / (avg_event_precision + avg_event_recall)
    else:
        avg_event_f1 = 0

    return {
        "avg_point_f1": round(avg_point_f1, 4),
        "avg_range_f1": round(avg_range_f1, 4),
        "avg_event_f1": round(avg_event_f1, 4),
        "avg_event_precision": round(avg_event_precision, 4),
        "avg_event_recall": round(avg_event_recall, 4),
        "n_engines_evaluated": len(all_point_f1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from core_pipeline.anomaly.anomaly_transformer import (
        CMAPSSWindowDataset, train_anomaly_transformer,
        score_windows, map_scores_to_cycles,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/CMAPSSData")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--max-rul", type=int, default=125)
    parser.add_argument("--rul-threshold", type=int, default=50)
    args = parser.parse_args()

    print(f"=== 3-Way Anomaly Evaluation ({args.subset}) ===\n")

    data = preprocess_subset(args.data_dir, args.subset, args.max_rul)
    train_df = data["train"]

    sensor_cols = [c for c in DEGRADATION_SENSORS if c in train_df.columns]

    # --- 1. Isolation Forest ---
    print("--- Isolation Forest ---")
    healthy = get_healthy_data(train_df, 0.5)
    if_result = train_isolation_forest(healthy)
    if_scored = score_anomalies(if_result["model"], train_df, if_result["feature_cols"])
    if_scored = create_anomaly_labels(if_scored, args.rul_threshold)

    y_true = if_scored["anomaly"].values
    y_pred = if_scored["anomaly_pred_binary"].values
    y_score = -if_scored["anomaly_score"].values

    if_3way = evaluate_3way(y_true, y_pred, y_score)
    if_engine = evaluate_per_engine_3way(if_scored, args.rul_threshold)
    print(f"  Point F1:  {if_3way['point_f1']}")
    print(f"  Range F1:  {if_3way['range_f1']}")
    print(f"  Event F1:  {if_3way['event_f1']}")
    print(f"  AUROC:     {if_3way['auroc']}")
    print(f"  Per-engine avg Point F1: {if_engine['avg_point_f1']}")
    print(f"  Per-engine avg Range F1: {if_engine['avg_range_f1']}")
    print(f"  Per-engine avg Event F1: {if_engine['avg_event_f1']}")

    # --- 2. Rolling Z-score ---
    print("\n--- Rolling Z-score ---")
    zs_scored = run_zscore_pipeline(train_df, window=30, z_threshold=3.0)
    opt = optimize_z_threshold(zs_scored, args.rul_threshold)
    from core_pipeline.anomaly.rolling_zscore import detect_anomalies_zscore
    zs_scored = detect_anomalies_zscore(zs_scored, opt["best_z_threshold"])
    zs_scored = create_anomaly_labels(zs_scored, args.rul_threshold)

    y_true_zs = zs_scored["anomaly"].values
    y_pred_zs = zs_scored["anomaly_pred_binary"].values
    y_score_zs = zs_scored["anomaly_score"].values

    zs_3way = evaluate_3way(y_true_zs, y_pred_zs, y_score_zs)
    zs_engine = evaluate_per_engine_3way(zs_scored, args.rul_threshold)
    print(f"  Point F1:  {zs_3way['point_f1']}")
    print(f"  Range F1:  {zs_3way['range_f1']}")
    print(f"  Event F1:  {zs_3way['event_f1']}")
    print(f"  AUROC:     {zs_3way['auroc']}")
    print(f"  Per-engine avg Point F1: {zs_engine['avg_point_f1']}")
    print(f"  Per-engine avg Range F1: {zs_engine['avg_range_f1']}")
    print(f"  Per-engine avg Event F1: {zs_engine['avg_event_f1']}")

    # --- Comparison ---
    print(f"\n{'='*60}")
    print("3-WAY COMPARISON")
    print(f"{'='*60}")
    print(f"\n  {'Metric':<25} {'IF':>10} {'Z-score':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Point F1':<25} {if_3way['point_f1']:>10} {zs_3way['point_f1']:>10}")
    print(f"  {'Range F1':<25} {if_3way['range_f1']:>10} {zs_3way['range_f1']:>10}")
    print(f"  {'Event F1':<25} {if_3way['event_f1']:>10} {zs_3way['event_f1']:>10}")
    print(f"  {'AUROC':<25} {if_3way['auroc']:>10} {zs_3way['auroc']:>10}")
    print(f"  {'Per-engine Point F1':<25} {if_engine['avg_point_f1']:>10} {zs_engine['avg_point_f1']:>10}")
    print(f"  {'Per-engine Range F1':<25} {if_engine['avg_range_f1']:>10} {zs_engine['avg_range_f1']:>10}")
    print(f"  {'Per-engine Event F1':<25} {if_engine['avg_event_f1']:>10} {zs_engine['avg_event_f1']:>10}")
