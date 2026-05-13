"""
Uncertainty Quantification for RUL Prediction

MC Dropout: run inference multiple times with dropout enabled
to estimate prediction uncertainty (epistemic uncertainty).

High uncertainty → model is unsure → flag for human review.
Low uncertainty → model is confident → trust the prediction.

This connects to Layer 1 governance:
  - High data quality (Quality Gate PASS) → low uncertainty
  - Low data quality (Quality Gate FLAG) → high uncertainty
"""

import time
import numpy as np
import pandas as pd
import torch
from typing import Optional

from core_pipeline.data.preprocess import preprocess_subset, DEGRADATION_SENSORS
from core_pipeline.rul.bilstm_rul import (
    BiLSTM_RUL, RULSequenceDataset,
    train_bilstm_rul, evaluate_bilstm_rul,
)
from core_pipeline.rul.xgboost_rul import cmapss_score


def enable_mc_dropout(model: torch.nn.Module):
    """Enable dropout layers during inference for MC Dropout."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout_predict(model: BiLSTM_RUL,
                       test_df: pd.DataFrame,
                       sensor_cols: list,
                       seq_len: int,
                       normalizer: tuple,
                       max_rul: int = 125,
                       n_forward: int = 50,
                       device: str = "cpu") -> dict:
    """Run MC Dropout inference for uncertainty estimation.

    Args:
        model: trained BiLSTM_RUL model
        test_df: test data
        sensor_cols: feature columns
        seq_len: sequence length
        normalizer: (mean, std)
        max_rul: RUL clipping
        n_forward: number of forward passes (more = better estimate)
        device: device string

    Returns:
        dict with per-engine: mean, std, CI_lower, CI_upper
    """
    mean, std = normalizer

    model.eval()
    enable_mc_dropout(model)  # Keep dropout ON

    engine_ids = sorted(test_df["engine_id"].unique())
    all_predictions = np.zeros((len(engine_ids), n_forward))

    for eng_idx, engine_id in enumerate(engine_ids):
        engine_df = test_df[test_df["engine_id"] == engine_id].sort_values("cycle")
        values = engine_df[sensor_cols].values.astype(np.float32)

        if len(values) >= seq_len:
            seq = values[-seq_len:]
        else:
            pad = np.zeros((seq_len - len(values), len(sensor_cols)), dtype=np.float32)
            seq = np.vstack([pad, values])

        seq = (seq - mean) / std
        x = torch.FloatTensor(seq).unsqueeze(0).to(device)

        # Multiple forward passes
        with torch.no_grad():
            for i in range(n_forward):
                pred = model(x).cpu().item()
                all_predictions[eng_idx, i] = np.clip(pred, 0, max_rul)

    # Compute statistics
    pred_mean = all_predictions.mean(axis=1)
    pred_std = all_predictions.std(axis=1)
    ci_lower = np.percentile(all_predictions, 5, axis=1)
    ci_upper = np.percentile(all_predictions, 95, axis=1)

    # Confidence classification
    confidence = []
    for s in pred_std:
        if s < 5:
            confidence.append("high")
        elif s < 15:
            confidence.append("medium")
        else:
            confidence.append("low")

    results_df = pd.DataFrame({
        "engine_id": engine_ids,
        "rul_mean": np.round(pred_mean, 2),
        "rul_std": np.round(pred_std, 2),
        "ci_lower_90": np.round(ci_lower, 2),
        "ci_upper_90": np.round(ci_upper, 2),
        "ci_width": np.round(ci_upper - ci_lower, 2),
        "confidence": confidence,
    })

    return {
        "predictions": results_df,
        "all_samples": all_predictions,
        "n_forward": n_forward,
    }


def evaluate_uncertainty_quality(results_df: pd.DataFrame,
                                  rul_true: pd.Series) -> dict:
    """Evaluate uncertainty calibration.

    Good uncertainty: true RUL falls within 90% CI for ~90% of engines.
    """
    y_true = rul_true.values
    in_ci = (y_true >= results_df["ci_lower_90"].values) & \
            (y_true <= results_df["ci_upper_90"].values)

    coverage = in_ci.mean()
    avg_width = results_df["ci_width"].mean()
    avg_std = results_df["rul_std"].mean()

    # RMSE of mean predictions
    rmse = np.sqrt(np.mean((y_true - results_df["rul_mean"].values) ** 2))

    # Confidence distribution
    conf_dist = results_df["confidence"].value_counts().to_dict()

    return {
        "coverage_90ci": round(coverage, 4),
        "avg_ci_width": round(avg_width, 2),
        "avg_std": round(avg_std, 2),
        "rmse_mean": round(rmse, 4),
        "confidence_distribution": conf_dist,
        "target_coverage": 0.90,
        "calibration": "well-calibrated" if 0.85 <= coverage <= 0.95 else
                       "over-confident" if coverage < 0.85 else "under-confident",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/CMAPSSData")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--max-rul", type=int, default=125)
    parser.add_argument("--n-forward", type=int, default=50)
    parser.add_argument("--n-epochs", type=int, default=50)
    args = parser.parse_args()

    print(f"=== Uncertainty Quantification ({args.subset}) ===\n")

    print("Loading data...")
    data = preprocess_subset(args.data_dir, args.subset, args.max_rul)

    print(f"Training Bi-LSTM (epochs={args.n_epochs})...")
    result = train_bilstm_rul(
        data["train"], n_epochs=args.n_epochs, max_rul=args.max_rul,
        dropout=0.5,
    )

    print(f"\nRunning MC Dropout ({args.n_forward} forward passes)...")
    t0 = time.time()
    mc_result = mc_dropout_predict(
        result["model"], data["test"],
        result["sensor_cols"], result["seq_len"],
        result["normalizer"], args.max_rul,
        n_forward=args.n_forward, device=result["device"],
    )
    elapsed = time.time() - t0
    print(f"  MC Dropout time: {elapsed:.1f}s")

    # Evaluate
    print("\nEvaluating uncertainty quality...")
    uq_metrics = evaluate_uncertainty_quality(
        mc_result["predictions"], data["rul_true"],
    )

    print(f"\n--- Results ---")
    print(f"  RMSE (mean prediction): {uq_metrics['rmse_mean']}")
    print(f"  90% CI Coverage:        {uq_metrics['coverage_90ci']} (target: 0.90)")
    print(f"  Avg CI Width:           {uq_metrics['avg_ci_width']} cycles")
    print(f"  Avg Std:                {uq_metrics['avg_std']} cycles")
    print(f"  Calibration:            {uq_metrics['calibration']}")
    print(f"  Confidence distribution: {uq_metrics['confidence_distribution']}")

    # Per-engine summary
    pred_df = mc_result["predictions"]
    print(f"\n--- Sample Predictions (first 10 engines) ---")
    print(f"  {'Engine':>8} {'True RUL':>10} {'Pred Mean':>10} {'Std':>8} "
          f"{'CI 90%':>15} {'Confidence':>12}")
    print(f"  {'-'*65}")

    y_true = data["rul_true"].values
    for i in range(min(10, len(pred_df))):
        row = pred_df.iloc[i]
        print(f"  {int(row['engine_id']):>8} {y_true[i]:>10} {row['rul_mean']:>10.1f} "
              f"{row['rul_std']:>8.1f} [{row['ci_lower_90']:>5.1f}, {row['ci_upper_90']:>5.1f}] "
              f"{row['confidence']:>12}")

    # Save
    pred_df.to_csv("experiments/uncertainty_quantification_results.csv", index=False)
    print(f"\nResults saved to experiments/uncertainty_quantification_results.csv")
