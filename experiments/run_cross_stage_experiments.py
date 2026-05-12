"""
Cross-Stage Integration Experiments

Experiment 1: IF anomaly score → RUL feature
  Stage 2(IF) 결과를 Stage 3(RUL)의 추가 feature로 활용.
  가설: IF score가 degradation 상태를 직접 반영하므로 RUL 예측 개선.

Experiment 2: TFT importance → targeted corruption
  Stage 3(TFT)이 중요하다고 뽑은 센서만 corruption.
  가설: 중요 센서의 corruption이 비중요 센서보다 RMSE 피해가 크다.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

from core_pipeline.data.preprocess import (
    preprocess_subset, DEGRADATION_SENSORS, SENSOR_COLS,
)
from core_pipeline.anomaly.isolation_forest import (
    get_healthy_data, train_isolation_forest, score_anomalies,
)
from core_pipeline.governance.corruption import (
    CorruptionType, Severity, apply_corruption,
)
from core_pipeline.rul.xgboost_rul import (
    train_xgboost_rul, evaluate_rul, cmapss_score,
)


# ===================================================================
# Experiment 1: IF anomaly score as RUL feature
# ===================================================================

def run_if_feature_experiment(data_dir: str = "data/raw/CMAPSSData",
                               subset: str = "FD001",
                               max_rul: int = 125) -> dict:
    """Compare XGBoost RUL with and without IF anomaly score feature."""

    print("=" * 60)
    print("Experiment 1: IF Anomaly Score as RUL Feature")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    data = preprocess_subset(data_dir, subset, max_rul)
    train_df = data["train"]
    test_df = data["test"]

    # --- Baseline: XGBoost without IF score ---
    print("\n[Baseline] XGBoost without IF score...")
    baseline = train_xgboost_rul(train_df, max_rul=max_rul)
    baseline_metrics = evaluate_rul(
        baseline["model"], test_df, data["rul_true"],
        baseline["feature_cols"], max_rul,
    )
    print(f"  RMSE: {baseline_metrics['rmse']}, Score: {baseline_metrics['score']}")

    # --- Train IF on healthy portion ---
    print("\nTraining IF on healthy data...")
    healthy = get_healthy_data(train_df, healthy_fraction=0.5)
    if_result = train_isolation_forest(healthy)

    # Score all train and test data
    print("Scoring train and test with IF...")
    train_scored = score_anomalies(if_result["model"], train_df, if_result["feature_cols"])
    test_scored = score_anomalies(if_result["model"], test_df, if_result["feature_cols"])

    # --- XGBoost with IF score as additional feature ---
    print("\n[Enhanced] XGBoost with IF score feature...")
    enhanced = train_xgboost_rul(train_scored, max_rul=max_rul)
    enhanced_metrics = evaluate_rul(
        enhanced["model"], test_scored, data["rul_true"],
        enhanced["feature_cols"], max_rul,
    )
    print(f"  RMSE: {enhanced_metrics['rmse']}, Score: {enhanced_metrics['score']}")

    # --- Comparison ---
    delta_rmse = enhanced_metrics["rmse"] - baseline_metrics["rmse"]
    delta_score = enhanced_metrics["score"] - baseline_metrics["score"]

    print(f"\n--- RESULT ---")
    print(f"  {'Metric':<10} {'Baseline':>10} {'+ IF score':>12} {'Delta':>10}")
    print(f"  {'RMSE':<10} {baseline_metrics['rmse']:>10} {enhanced_metrics['rmse']:>12} {delta_rmse:>+10.4f}")
    print(f"  {'MAE':<10} {baseline_metrics['mae']:>10} {enhanced_metrics['mae']:>12}")
    print(f"  {'Score':<10} {baseline_metrics['score']:>10} {enhanced_metrics['score']:>12} {delta_score:>+10.2f}")

    if delta_rmse < 0:
        print(f"\n  IF score feature improved RMSE by {abs(delta_rmse):.4f}")
    else:
        print(f"\n  IF score feature did not improve RMSE ({delta_rmse:+.4f})")

    return {
        "baseline_rmse": baseline_metrics["rmse"],
        "baseline_score": baseline_metrics["score"],
        "enhanced_rmse": enhanced_metrics["rmse"],
        "enhanced_score": enhanced_metrics["score"],
        "delta_rmse": round(delta_rmse, 4),
        "delta_score": round(delta_score, 2),
    }


# ===================================================================
# Experiment 2: TFT importance → targeted corruption
# ===================================================================

# TFT Run 2 (best) variable importance — top 4 vs bottom 4
TFT_TOP_SENSORS = ["sensor_11", "sensor_3", "sensor_4", "sensor_14"]
TFT_BOTTOM_SENSORS = ["sensor_8", "sensor_7", "sensor_2", "sensor_17"]


def run_targeted_corruption_experiment(data_dir: str = "data/raw/CMAPSSData",
                                        subset: str = "FD001",
                                        max_rul: int = 125,
                                        seed: int = 42) -> pd.DataFrame:
    """Compare corruption impact on TFT-important vs TFT-unimportant sensors.

    Hypothesis: corrupting top-importance sensors causes larger RMSE
    degradation than corrupting bottom-importance sensors.
    """

    print("\n" + "=" * 60)
    print("Experiment 2: TFT Importance → Targeted Corruption")
    print("=" * 60)

    print(f"\n  Top sensors (TFT importance):    {TFT_TOP_SENSORS}")
    print(f"  Bottom sensors (TFT importance): {TFT_BOTTOM_SENSORS}")

    # Load data
    print("\nLoading data...")
    data = preprocess_subset(data_dir, subset, max_rul)

    # Baseline
    print("Training clean baseline...")
    baseline = train_xgboost_rul(data["train"], max_rul=max_rul)
    baseline_metrics = evaluate_rul(
        baseline["model"], data["test"], data["rul_true"],
        baseline["feature_cols"], max_rul,
    )
    rmse_clean = baseline_metrics["rmse"]
    print(f"  Baseline RMSE: {rmse_clean}")

    # Test corruption types that showed clear impact
    test_corruptions = [
        (CorruptionType.SENSOR_DRIFT, Severity.HIGH),
        (CorruptionType.GAUSSIAN_NOISE, Severity.HIGH),
        (CorruptionType.CONCEPT_DRIFT, Severity.LOW),
    ]

    results = []

    for ctype, severity in test_corruptions:
        for group_name, sensors in [("top_4", TFT_TOP_SENSORS),
                                     ("bottom_4", TFT_BOTTOM_SENSORS)]:
            print(f"\nRunning: {ctype.value}/{severity.value} on {group_name}...")

            corruption_result = apply_corruption(
                data["train"].copy(), ctype, severity,
                target_sensors=sensors, seed=seed,
            )

            corrupt_model = train_xgboost_rul(
                corruption_result.corrupted_df, max_rul=max_rul,
            )
            corrupt_metrics = evaluate_rul(
                corrupt_model["model"], data["test"], data["rul_true"],
                corrupt_model["feature_cols"], max_rul,
            )

            delta = corrupt_metrics["rmse"] - rmse_clean

            print(f"  RMSE: {rmse_clean} -> {corrupt_metrics['rmse']} "
                  f"(delta: {delta:+.4f})")

            results.append({
                "corruption_type": ctype.value,
                "severity": severity.value,
                "sensor_group": group_name,
                "sensors": ",".join(sensors),
                "rmse_clean": rmse_clean,
                "rmse_corrupt": corrupt_metrics["rmse"],
                "delta_rmse": round(delta, 4),
                "score_clean": baseline_metrics["score"],
                "score_corrupt": corrupt_metrics["score"],
            })

    results_df = pd.DataFrame(results)

    # Comparison
    print(f"\n--- TARGETED CORRUPTION RESULTS ---\n")
    print(f"  {'Corruption':<25} {'Group':<10} {'Δ RMSE':>10} {'Impact':>10}")
    print(f"  {'-'*55}")

    for ctype, severity in test_corruptions:
        mask = (results_df["corruption_type"] == ctype.value) & \
               (results_df["severity"] == severity.value)
        subset_df = results_df[mask]

        top_delta = subset_df[subset_df["sensor_group"] == "top_4"]["delta_rmse"].values[0]
        bot_delta = subset_df[subset_df["sensor_group"] == "bottom_4"]["delta_rmse"].values[0]

        label = f"{ctype.value}/{severity.value}"
        print(f"  {label:<25} {'top_4':<10} {top_delta:>+10.4f} {'***' if abs(top_delta) > abs(bot_delta) else ''}")
        print(f"  {'':<25} {'bottom_4':<10} {bot_delta:>+10.4f} {'***' if abs(bot_delta) > abs(top_delta) else ''}")
        ratio = abs(top_delta) / abs(bot_delta) if abs(bot_delta) > 0.001 else float("inf")
        print(f"  {'':<25} {'ratio':<10} {ratio:>10.1f}x")
        print()

    return results_df


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-stage integration experiments"
    )
    parser.add_argument("--data-dir", default="data/raw/CMAPSSData")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--max-rul", type=int, default=125)
    args = parser.parse_args()

    # Experiment 1
    exp1_result = run_if_feature_experiment(
        args.data_dir, args.subset, args.max_rul,
    )

    # Experiment 2
    exp2_result = run_targeted_corruption_experiment(
        args.data_dir, args.subset, args.max_rul,
    )

    # Save
    out_dir = Path("experiments")
    out_dir.mkdir(exist_ok=True)
    exp2_result.to_csv(out_dir / "targeted_corruption_results.csv", index=False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nExp 1 — IF score as feature: RMSE delta = {exp1_result['delta_rmse']:+.4f}")
    print(f"Exp 2 — Targeted corruption results saved to experiments/")
