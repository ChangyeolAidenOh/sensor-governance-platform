"""
Corruption Impact Experiment

The central experiment of this project. Quantifies:
1. How much each corruption type degrades RUL prediction
2. How much the governance layer recovers

Produces the Recovery Rate metric that justifies the governance layer.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core_pipeline.data.preprocess import (
    preprocess_subset, DEGRADATION_SENSORS, SENSOR_COLS,
)
from core_pipeline.governance.corruption import (
    CorruptionType, Severity, apply_corruption, scenarios_summary_table,
)
from core_pipeline.governance.quality_gate import QualityGate, run_governed_pipeline
from core_pipeline.rul.xgboost_rul import train_xgboost_rul, evaluate_rul


@dataclass
class ExperimentResult:
    """Result of one corruption experiment scenario."""
    corruption_type: str
    severity: str
    rmse_clean: float       # baseline (no corruption)
    rmse_corrupt: float     # after corruption, no governance
    rmse_recovered: float   # after corruption + governance
    delta_rmse: float       # rmse_corrupt - rmse_clean
    recovery_rate: float    # (rmse_corrupt - rmse_recovered) / (rmse_corrupt - rmse_clean)
    score_clean: float
    score_corrupt: float
    score_recovered: float
    params: dict
    elapsed_sec: float


def run_single_scenario(data: dict,
                        corruption_type: CorruptionType,
                        severity: Severity,
                        baseline_metrics: dict,
                        max_rul: int = 125,
                        seed: int = 42) -> ExperimentResult:
    """Run one corruption scenario and measure impact + recovery.

    Args:
        data: output of preprocess_subset
        corruption_type: type of corruption
        severity: severity level
        baseline_metrics: pre-computed clean baseline metrics
        max_rul: RUL clipping value
        seed: random seed

    Returns:
        ExperimentResult
    """
    t0 = time.time()
    train_df = data["train"].copy()

    # --- Step 1: Apply corruption ---
    corruption_result = apply_corruption(
        train_df, corruption_type, severity,
        target_sensors=DEGRADATION_SENSORS, seed=seed,
    )
    corrupted_df = corruption_result.corrupted_df

    # --- Step 2: Train on corrupted data (no governance) ---
    corrupt_model = train_xgboost_rul(corrupted_df, max_rul=max_rul)
    corrupt_metrics = evaluate_rul(
        corrupt_model["model"], data["test"], data["rul_true"],
        corrupt_model["feature_cols"], max_rul,
    )

    # --- Step 3: Apply governance layer, then train ---
    gate = QualityGate(flag_threshold=70.0, block_threshold=30.0)
    governed_df = run_governed_pipeline(
        corrupted_df, SENSOR_COLS,
        reference_size=30, window_size=20, gate=gate,
    )
    governed_model = train_xgboost_rul(governed_df, max_rul=max_rul)
    governed_metrics = evaluate_rul(
        governed_model["model"], data["test"], data["rul_true"],
        governed_model["feature_cols"], max_rul,
    )

    # --- Compute Recovery Rate ---
    rmse_clean = baseline_metrics["rmse"]
    rmse_corrupt = corrupt_metrics["rmse"]
    rmse_recovered = governed_metrics["rmse"]

    delta = rmse_corrupt - rmse_clean
    if delta > 0:
        recovery = (rmse_corrupt - rmse_recovered) / delta
    else:
        recovery = 0.0  # corruption didn't hurt

    elapsed = time.time() - t0

    return ExperimentResult(
        corruption_type=corruption_type.value,
        severity=severity.value,
        rmse_clean=rmse_clean,
        rmse_corrupt=round(rmse_corrupt, 4),
        rmse_recovered=round(rmse_recovered, 4),
        delta_rmse=round(delta, 4),
        recovery_rate=round(recovery, 4),
        score_clean=baseline_metrics["score"],
        score_corrupt=round(corrupt_metrics["score"], 2),
        score_recovered=round(governed_metrics["score"], 2),
        params=corruption_result.summary["params"],
        elapsed_sec=round(elapsed, 1),
    )


def run_full_experiment(data_dir: str = "data/raw/CMAPSSData",
                        subset: str = "FD001",
                        max_rul: int = 125,
                        output_dir: str = "experiments",
                        seed: int = 42) -> pd.DataFrame:
    """Run all 15 corruption scenarios and produce results table.

    This is the main entry point for the corruption experiment.
    """
    print(f"=== Corruption Impact Experiment ({subset}) ===\n")

    # Load and preprocess
    print("Loading data...")
    data = preprocess_subset(data_dir, subset, max_rul)

    # Train clean baseline
    print("Training clean baseline...")
    baseline_model = train_xgboost_rul(data["train"], max_rul=max_rul)
    baseline_metrics = evaluate_rul(
        baseline_model["model"], data["test"], data["rul_true"],
        baseline_model["feature_cols"], max_rul,
    )
    print(f"  Baseline RMSE: {baseline_metrics['rmse']}")
    print(f"  Baseline Score: {baseline_metrics['score']}\n")

    # Run all scenarios
    results = []
    for ctype in CorruptionType:
        for severity in Severity:
            print(f"Running: {ctype.value} / {severity.value}...")
            result = run_single_scenario(
                data, ctype, severity, baseline_metrics,
                max_rul=max_rul, seed=seed,
            )
            print(f"  RMSE: {result.rmse_clean} → {result.rmse_corrupt} "
                  f"→ {result.rmse_recovered} "
                  f"(recovery: {result.recovery_rate:.1%}, "
                  f"{result.elapsed_sec}s)")
            results.append(result)

    # Build results table
    results_df = pd.DataFrame([vars(r) for r in results])

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(out_path / f"corruption_experiment_{subset}.csv", index=False)

    with open(out_path / f"corruption_experiment_{subset}.json", "w") as f:
        json.dump({
            "subset": subset,
            "max_rul": max_rul,
            "baseline_rmse": baseline_metrics["rmse"],
            "baseline_score": baseline_metrics["score"],
            "n_scenarios": len(results),
            "results": [vars(r) for r in results],
        }, f, indent=2)

    # Print summary
    print("\n=== RESULTS SUMMARY ===\n")
    summary_cols = ["corruption_type", "severity",
                    "rmse_clean", "rmse_corrupt", "rmse_recovered",
                    "delta_rmse", "recovery_rate"]
    print(results_df[summary_cols].to_string(index=False))

    # Key findings
    print("\n=== KEY FINDINGS ===")
    worst = results_df.loc[results_df["delta_rmse"].idxmax()]
    best_recovery = results_df.loc[results_df["recovery_rate"].idxmax()]
    avg_recovery = results_df["recovery_rate"].mean()

    print(f"  Most damaging: {worst['corruption_type']} / {worst['severity']} "
          f"(+{worst['delta_rmse']:.2f} RMSE)")
    print(f"  Best recovery: {best_recovery['corruption_type']} / "
          f"{best_recovery['severity']} ({best_recovery['recovery_rate']:.1%})")
    print(f"  Average recovery rate: {avg_recovery:.1%}")

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run corruption impact experiment"
    )
    parser.add_argument("--data-dir", default="data/raw/CMAPSSData")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--max-rul", type=int, default=125)
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_full_experiment(
        args.data_dir, args.subset, args.max_rul,
        args.output_dir, args.seed,
    )
