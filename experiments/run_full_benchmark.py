"""
Full C-MAPSS Subset Benchmark

Runs XGBoost RUL + Isolation Forest across all 4 subsets (FD001-FD004)
to demonstrate complete dataset utilization.

Each subset is independently trained and evaluated — no cross-subset transfer.
This complements the Cross-Subset Transfer experiment which tested
generalization FROM FD001 TO others.

Results table: subset × (RMSE, Score, AUROC, F1, n_engines, conditions, faults)
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

from core_pipeline.data.preprocess import preprocess_subset, DEGRADATION_SENSORS
from core_pipeline.rul.xgboost_rul import train_xgboost_rul, evaluate_rul
from core_pipeline.anomaly.isolation_forest import (
    get_healthy_data, train_isolation_forest, score_anomalies,
    evaluate_anomaly_detection, evaluate_multiple_thresholds,
    analyze_engine_detections,
)


SUBSETS = {
    "FD001": {"conditions": 1, "faults": 1, "engines_train": 100, "engines_test": 100},
    "FD002": {"conditions": 6, "faults": 1, "engines_train": 260, "engines_test": 259},
    "FD003": {"conditions": 1, "faults": 2, "engines_train": 100, "engines_test": 100},
    "FD004": {"conditions": 6, "faults": 2, "engines_train": 248, "engines_test": 249},
}


def run_full_benchmark(data_dir: str = "data/raw/CMAPSSData",
                        max_rul: int = 125) -> pd.DataFrame:
    """Run XGBoost + IF on all 4 subsets."""

    print("=" * 70)
    print("Full C-MAPSS Subset Benchmark")
    print("  XGBoost RUL + Isolation Forest × 4 Subsets")
    print("=" * 70)

    results = []

    for subset, info in SUBSETS.items():
        print(f"\n{'='*50}")
        print(f"  {subset} (conditions={info['conditions']}, faults={info['faults']})")
        print(f"{'='*50}")

        t0 = time.time()

        try:
            # Load
            data = preprocess_subset(data_dir, subset, max_rul)
            train_df = data["train"]
            n_train = len(train_df)

            # --- XGBoost RUL ---
            print(f"\n  [XGBoost] Training...")
            xgb_result = train_xgboost_rul(train_df, max_rul=max_rul)
            xgb_metrics = evaluate_rul(
                xgb_result["model"], data["test"], data["rul_true"],
                xgb_result["feature_cols"], max_rul,
            )
            print(f"    RMSE: {xgb_metrics['rmse']}, Score: {xgb_metrics['score']}")

            # --- Isolation Forest ---
            print(f"  [IF] Training...")
            healthy = get_healthy_data(train_df, healthy_fraction=0.5)
            if_result = train_isolation_forest(healthy)
            if_scored = score_anomalies(
                if_result["model"], train_df, if_result["feature_cols"],
            )
            if_metrics = evaluate_anomaly_detection(if_scored, rul_threshold=50)
            if_engines = analyze_engine_detections(if_scored, rul_threshold=50)
            detected = if_engines["detected"].sum()
            total_engines = len(if_engines)
            lead_times = if_engines["lead_time"].dropna()
            avg_lead = lead_times.mean() if len(lead_times) > 0 else 0

            print(f"    AUROC: {if_metrics['auroc']}, F1: {if_metrics['f1']}")
            print(f"    Detection: {detected}/{total_engines}")

            elapsed = time.time() - t0

            results.append({
                "subset": subset,
                "conditions": info["conditions"],
                "faults": info["faults"],
                "engines_train": info["engines_train"],
                "engines_test": info["engines_test"],
                "n_train_samples": n_train,
                "rul_rmse": xgb_metrics["rmse"],
                "rul_mae": xgb_metrics["mae"],
                "rul_score": xgb_metrics["score"],
                "if_auroc": if_metrics["auroc"],
                "if_f1": if_metrics["f1"],
                "if_precision": if_metrics["precision"],
                "if_recall": if_metrics["recall"],
                "detection_rate": f"{detected}/{total_engines}",
                "avg_lead_time": round(avg_lead, 1),
                "time_seconds": round(elapsed, 1),
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "subset": subset,
                "conditions": info["conditions"],
                "faults": info["faults"],
                "engines_train": info["engines_train"],
                "engines_test": info["engines_test"],
                "n_train_samples": 0,
                "rul_rmse": None,
                "rul_mae": None,
                "rul_score": None,
                "if_auroc": None,
                "if_f1": None,
                "if_precision": None,
                "if_recall": None,
                "detection_rate": None,
                "avg_lead_time": None,
                "time_seconds": None,
            })

    results_df = pd.DataFrame(results)

    # Summary
    print(f"\n{'='*70}")
    print("FULL BENCHMARK RESULTS")
    print(f"{'='*70}")

    print(f"\n--- RUL Prediction (XGBoost) ---\n")
    print(f"  {'Subset':<8} {'Cond':>5} {'Faults':>7} {'Engines':>8} "
          f"{'RMSE':>8} {'MAE':>8} {'Score':>10}")
    print(f"  {'-'*56}")
    for _, row in results_df.iterrows():
        rmse_str = f"{row['rul_rmse']:.2f}" if pd.notna(row['rul_rmse']) else "N/A"
        mae_str = f"{row['rul_mae']:.2f}" if pd.notna(row['rul_mae']) else "N/A"
        score_str = f"{row['rul_score']:.2f}" if pd.notna(row['rul_score']) else "N/A"
        print(f"  {row['subset']:<8} {row['conditions']:>5} {row['faults']:>7} "
              f"{row['engines_train']:>8} {rmse_str:>8} {mae_str:>8} {score_str:>10}")

    print(f"\n--- Anomaly Detection (Isolation Forest) ---\n")
    print(f"  {'Subset':<8} {'AUROC':>8} {'F1':>8} {'Precision':>10} "
          f"{'Recall':>8} {'Detection':>10} {'Lead Time':>10}")
    print(f"  {'-'*64}")
    for _, row in results_df.iterrows():
        auroc_str = f"{row['if_auroc']:.3f}" if pd.notna(row['if_auroc']) else "N/A"
        f1_str = f"{row['if_f1']:.3f}" if pd.notna(row['if_f1']) else "N/A"
        prec_str = f"{row['if_precision']:.3f}" if pd.notna(row['if_precision']) else "N/A"
        rec_str = f"{row['if_recall']:.3f}" if pd.notna(row['if_recall']) else "N/A"
        det_str = row['detection_rate'] if row['detection_rate'] else "N/A"
        lead_str = f"{row['avg_lead_time']:.1f}" if pd.notna(row['avg_lead_time']) else "N/A"
        print(f"  {row['subset']:<8} {auroc_str:>8} {f1_str:>8} {prec_str:>10} "
              f"{rec_str:>8} {det_str:>10} {lead_str:>10}")

    # Key findings
    print(f"\n--- Key Findings ---")
    if len(results_df) == 4 and results_df["rul_rmse"].notna().all():
        best_rmse = results_df.loc[results_df["rul_rmse"].idxmin()]
        worst_rmse = results_df.loc[results_df["rul_rmse"].idxmax()]
        print(f"  Best RUL (RMSE): {best_rmse['subset']} ({best_rmse['rul_rmse']:.2f})")
        print(f"  Worst RUL (RMSE): {worst_rmse['subset']} ({worst_rmse['rul_rmse']:.2f})")

        best_auroc = results_df.loc[results_df["if_auroc"].idxmax()]
        worst_auroc = results_df.loc[results_df["if_auroc"].idxmin()]
        print(f"  Best IF (AUROC): {best_auroc['subset']} ({best_auroc['if_auroc']:.3f})")
        print(f"  Worst IF (AUROC): {worst_auroc['subset']} ({worst_auroc['if_auroc']:.3f})")

        # Complexity vs performance
        simple = results_df[results_df["conditions"] == 1]["rul_rmse"].mean()
        complex_ = results_df[results_df["conditions"] == 6]["rul_rmse"].mean()
        print(f"\n  Single-condition avg RMSE: {simple:.2f}")
        print(f"  Multi-condition avg RMSE:  {complex_:.2f}")
        print(f"  Complexity gap: {complex_ - simple:+.2f} ({(complex_-simple)/simple*100:+.1f}%)")

    # Save
    out_dir = Path("experiments")
    out_dir.mkdir(exist_ok=True)
    results_df.to_csv(out_dir / "full_subset_benchmark.csv", index=False)
    print(f"\nResults saved to experiments/full_subset_benchmark.csv")

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/CMAPSSData")
    parser.add_argument("--max-rul", type=int, default=125)
    args = parser.parse_args()

    run_full_benchmark(args.data_dir, args.max_rul)
