"""
Cross-Subset Transfer Experiment

Train on FD001, evaluate on FD001/FD002/FD003/FD004 test sets.
Measures generalization gap = proxy for cross-plant deployment.

Subset mapping:
  FD001 → FD003: same conditions, different fault modes (포항 라인 A → B)
  FD001 → FD002: different conditions, same fault (포항 → 광양)
  FD001 → FD004: different conditions + different faults (최대 복잡도)
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

from core_pipeline.data.preprocess import (
    preprocess_subset, DEGRADATION_SENSORS, SENSOR_COLS,
)
from core_pipeline.rul.xgboost_rul import train_xgboost_rul, evaluate_rul, cmapss_score


SUBSETS = {
    "FD001": {"conditions": 1, "faults": 1, "description": "baseline (same)"},
    "FD002": {"conditions": 6, "faults": 1, "description": "cross-condition (1→6)"},
    "FD003": {"conditions": 1, "faults": 2, "description": "cross-fault (1→2)"},
    "FD004": {"conditions": 6, "faults": 2, "description": "cross-condition + cross-fault"},
}


def run_transfer_experiment(data_dir: str = "data/raw/CMAPSSData",
                             max_rul: int = 125) -> pd.DataFrame:
    """Train on FD001, evaluate on all subsets."""

    print("=" * 60)
    print("Cross-Subset Transfer Experiment")
    print("  Train: FD001 | Test: FD001, FD002, FD003, FD004")
    print("=" * 60)

    # Train on FD001
    print("\nLoading FD001 (training)...")
    fd001 = preprocess_subset(data_dir, "FD001", max_rul)

    print("Training XGBoost on FD001...")
    model_result = train_xgboost_rul(fd001["train"], max_rul=max_rul)
    feature_cols = model_result["feature_cols"]

    # Evaluate on FD001 (baseline)
    print("\n--- Evaluating on FD001 (baseline) ---")
    fd001_metrics = evaluate_rul(
        model_result["model"], fd001["test"], fd001["rul_true"],
        feature_cols, max_rul,
    )
    print(f"  RMSE: {fd001_metrics['rmse']}, Score: {fd001_metrics['score']}")

    results = [{
        "train_subset": "FD001",
        "test_subset": "FD001",
        "conditions_train": 1,
        "conditions_test": 1,
        "faults_train": 1,
        "faults_test": 1,
        "transfer_type": "baseline (same)",
        "rmse": fd001_metrics["rmse"],
        "mae": fd001_metrics["mae"],
        "score": fd001_metrics["score"],
    }]

    # Evaluate on FD002, FD003, FD004
    for subset in ["FD002", "FD003", "FD004"]:
        info = SUBSETS[subset]
        print(f"\n--- Evaluating on {subset} ({info['description']}) ---")

        try:
            data = preprocess_subset(data_dir, subset, max_rul)

            # Align features: test data must have same columns as training
            test_df = data["test"].copy()
            missing_cols = [c for c in feature_cols if c not in test_df.columns]
            extra_cols = [c for c in test_df.columns
                          if c not in feature_cols
                          and c not in ["engine_id", "cycle", "rul", "subset", "split"]]

            if missing_cols:
                print(f"  Missing {len(missing_cols)} features in {subset}, filling with 0")
                for col in missing_cols:
                    test_df[col] = 0.0

            metrics = evaluate_rul(
                model_result["model"], test_df, data["rul_true"],
                feature_cols, max_rul,
            )

            delta_rmse = metrics["rmse"] - fd001_metrics["rmse"]
            delta_pct = (delta_rmse / fd001_metrics["rmse"]) * 100

            print(f"  RMSE: {metrics['rmse']} (delta: {delta_rmse:+.2f}, {delta_pct:+.1f}%)")
            print(f"  Score: {metrics['score']}")

            results.append({
                "train_subset": "FD001",
                "test_subset": subset,
                "conditions_train": 1,
                "conditions_test": info["conditions"],
                "faults_train": 1,
                "faults_test": info["faults"],
                "transfer_type": info["description"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "score": metrics["score"],
            })

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "train_subset": "FD001",
                "test_subset": subset,
                "conditions_train": 1,
                "conditions_test": info["conditions"],
                "faults_train": 1,
                "faults_test": info["faults"],
                "transfer_type": info["description"],
                "rmse": None,
                "mae": None,
                "score": None,
            })

    results_df = pd.DataFrame(results)

    # Add delta columns
    baseline_rmse = fd001_metrics["rmse"]
    baseline_score = fd001_metrics["score"]
    results_df["delta_rmse"] = results_df["rmse"] - baseline_rmse
    results_df["delta_rmse_pct"] = (results_df["delta_rmse"] / baseline_rmse * 100).round(1)
    results_df["delta_score"] = results_df["score"] - baseline_score

    # Summary
    print("\n" + "=" * 60)
    print("TRANSFER RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n  Baseline (FD001): RMSE {baseline_rmse}, Score {baseline_score}\n")
    print(f"  {'Test':<8} {'Type':<35} {'RMSE':>8} {'Δ RMSE':>10} {'Δ %':>8} {'Score':>10}")
    print(f"  {'-'*80}")

    for _, row in results_df.iterrows():
        delta_str = f"{row['delta_rmse']:+.2f}" if pd.notna(row['delta_rmse']) else "N/A"
        pct_str = f"{row['delta_rmse_pct']:+.1f}%" if pd.notna(row['delta_rmse_pct']) else "N/A"
        print(f"  {row['test_subset']:<8} {row['transfer_type']:<35} "
              f"{row['rmse']:>8.2f} {delta_str:>10} {pct_str:>8} {row['score']:>10.2f}")

    # Subset mapping
    print(f"\n--- SUBSET MAPPING ---")
    print(f"  FD001→FD001: Same line, same conditions (baseline)")
    print(f"  FD001→FD003: Same conditions, new fault mode → 다른 고장 유형 대응")
    print(f"  FD001→FD002: New conditions, same fault → 다른 운전 환경 (제품 Mix)")
    print(f"  FD001→FD004: New conditions + new faults → 완전히 다른 공장")

    # Save
    out_dir = Path("experiments")
    out_dir.mkdir(exist_ok=True)
    results_df.to_csv(out_dir / "cross_subset_transfer_results.csv", index=False)
    print(f"\nResults saved to experiments/cross_subset_transfer_results.csv")

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-subset transfer experiment"
    )
    parser.add_argument("--data-dir", default="data/raw/CMAPSSData")
    parser.add_argument("--max-rul", type=int, default=125)
    args = parser.parse_args()

    run_transfer_experiment(args.data_dir, args.max_rul)
