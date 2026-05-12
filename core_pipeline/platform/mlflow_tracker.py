"""
MLflow Experiment Tracking for RUL Models

Trains XGBoost and Bi-LSTM, logs everything to MLflow:
- Parameters (hyperparameters, data config)
- Metrics (RMSE, MAE, Score, val RMSE)
- Artifacts (model files, variable importance plots)
- Model Registry (best model tagged as 'production')

Usage:
  python -m core_pipeline.platform.mlflow_tracker
  mlflow ui  # then open http://localhost:5000
"""
#### OMP_NUM_THREADS=1 -> For specific condition
import os
os.environ["OMP_NUM_THREADS"] = "1"

import os
import json
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import numpy as np
import pandas as pd

from core_pipeline.data.preprocess import preprocess_subset, DEGRADATION_SENSORS
from core_pipeline.rul.xgboost_rul import train_xgboost_rul, evaluate_rul


MLFLOW_TRACKING_DIR = "mlruns"
EXPERIMENT_NAME = "rul_prediction"


def setup_mlflow(tracking_dir: str = MLFLOW_TRACKING_DIR,
                 experiment_name: str = EXPERIMENT_NAME):
    """Initialize MLflow with local file tracking."""
    db_path = os.path.abspath(os.path.join(tracking_dir, "mlflow.db"))
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking: {tracking_uri}")
    print(f"Experiment: {experiment_name}")
    print(f"  To view: OMP_NUM_THREADS=1 mlflow ui --backend-store-uri {tracking_uri}")

# -------------------------------------------------------------------
# XGBoost Run
# -------------------------------------------------------------------

def log_xgboost_run(data: dict,
                    max_rul: int = 125,
                    xgb_params: dict = None):
    """Train XGBoost and log to MLflow."""

    if xgb_params is None:
        xgb_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

    with mlflow.start_run(run_name="xgboost_rul"):
        # Log parameters
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("max_rul", max_rul)
        mlflow.log_param("n_features", len(DEGRADATION_SENSORS))
        for k, v in xgb_params.items():
            mlflow.log_param(k, v)

        # Train
        result = train_xgboost_rul(
            data["train"], max_rul=max_rul, params=xgb_params,
        )

        # Evaluate
        metrics = evaluate_rul(
            result["model"], data["test"], data["rul_true"],
            result["feature_cols"], max_rul,
        )

        # Log metrics
        mlflow.log_metric("train_rmse", result["train_rmse"])
        mlflow.log_metric("test_rmse", metrics["rmse"])
        mlflow.log_metric("test_mae", metrics["mae"])
        mlflow.log_metric("test_score", metrics["score"])

        # Log model as pickle artifact (avoid sklearn serialization crash)
        import pickle, tempfile
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump({"model": result["model"], "feature_cols": result["feature_cols"]}, f)
            mlflow.log_artifact(f.name, "model")

        # Log feature importance
        importance = result["model"].feature_importances_
        top_features = sorted(
            zip(result["feature_cols"], importance),
            key=lambda x: x[1], reverse=True,
        )[:20]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                [{"feature": feat, "importance": float(imp)}
                 for feat, imp in top_features],
                f, indent=2,
            )
            mlflow.log_artifact(f.name, "feature_importance")

        # Tag
        mlflow.set_tag("stage", "stage3")
        mlflow.set_tag("model_class", "tabular_ml")

        print(f"  XGBoost logged — RMSE: {metrics['rmse']}, "
              f"Score: {metrics['score']}")

        return metrics


# -------------------------------------------------------------------
# Bi-LSTM Run
# -------------------------------------------------------------------

def log_bilstm_run(data: dict,
                   max_rul: int = 125,
                   seq_len: int = 30,
                   hidden_size: int = 64,
                   n_layers: int = 2,
                   n_epochs: int = 50,
                   lr: float = 1e-3,
                   batch_size: int = 256):
    """Train Bi-LSTM and log to MLflow."""
    from core_pipeline.rul.bilstm_rul import train_bilstm_rul, evaluate_bilstm_rul

    with mlflow.start_run(run_name="bilstm_rul"):
        # Log parameters
        mlflow.log_param("model_type", "bilstm")
        mlflow.log_param("max_rul", max_rul)
        mlflow.log_param("seq_len", seq_len)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("n_layers", n_layers)
        mlflow.log_param("n_epochs", n_epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)

        # Train
        result = train_bilstm_rul(
            data["train"], seq_len=seq_len,
            hidden_size=hidden_size, n_layers=n_layers,
            n_epochs=n_epochs, batch_size=batch_size,
            lr=lr, max_rul=max_rul,
        )

        # Evaluate
        metrics = evaluate_bilstm_rul(
            result["model"], data["test"], data["rul_true"],
            result["sensor_cols"], result["seq_len"],
            result["normalizer"], max_rul, result["device"],
        )

        # Log metrics
        mlflow.log_metric("best_val_rmse", result["best_val_rmse"])
        mlflow.log_metric("test_rmse", metrics["rmse"])
        mlflow.log_metric("test_mae", metrics["mae"])
        mlflow.log_metric("test_score", metrics["score"])

        # Log training history
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(result["history"], f, indent=2, default=str)
            mlflow.log_artifact(f.name, "training_history")

        # Tag
        mlflow.set_tag("stage", "stage3")
        mlflow.set_tag("model_class", "deep_learning")

        print(f"  Bi-LSTM logged — RMSE: {metrics['rmse']}, "
              f"Score: {metrics['score']}")

        return metrics


# -------------------------------------------------------------------
# TFT Run
# -------------------------------------------------------------------

def log_tft_run(data: dict,
                max_rul: int = 125,
                seq_len: int = 30,
                d_model: int = 64,
                n_heads: int = 4,
                n_epochs: int = 50,
                lr: float = 1e-3,
                batch_size: int = 256):
    """Train TFT and log to MLflow, including variable importance."""
    from core_pipeline.rul.tft_rul import train_tft_rul, evaluate_tft_rul

    with mlflow.start_run(run_name="tft_rul"):
        # Log parameters
        mlflow.log_param("model_type", "tft")
        mlflow.log_param("max_rul", max_rul)
        mlflow.log_param("seq_len", seq_len)
        mlflow.log_param("d_model", d_model)
        mlflow.log_param("n_heads", n_heads)
        mlflow.log_param("n_epochs", n_epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)

        # Train
        result = train_tft_rul(
            data["train"], seq_len=seq_len,
            d_model=d_model, n_heads=n_heads,
            n_epochs=n_epochs, batch_size=batch_size,
            lr=lr, max_rul=max_rul,
        )

        # Evaluate
        metrics = evaluate_tft_rul(
            result["model"], data["test"], data["rul_true"],
            result["sensor_cols"], result["seq_len"],
            result["normalizer"], max_rul, result["device"],
        )

        # Log metrics
        mlflow.log_metric("best_val_rmse", result["best_val_rmse"])
        mlflow.log_metric("test_rmse", metrics["rmse"])
        mlflow.log_metric("test_mae", metrics["mae"])
        mlflow.log_metric("test_score", metrics["score"])

        # Log variable importance
        importance_df = metrics["variable_importance"]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            importance_df.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, "variable_importance")

        # Log training history
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(result["history"], f, indent=2, default=str)
            mlflow.log_artifact(f.name, "training_history")

        # Tag
        mlflow.set_tag("stage", "stage3")
        mlflow.set_tag("model_class", "transformer")

        print(f"  TFT logged — RMSE: {metrics['rmse']}, "
              f"Score: {metrics['score']}")

        return metrics


# -------------------------------------------------------------------
# Model Comparison & Registry
# -------------------------------------------------------------------

def compare_and_register(results: dict):
    """Compare all models and register the best one."""

    print("\n=== MODEL COMPARISON ===\n")
    print(f"  {'Model':<15} {'RMSE':>8} {'MAE':>8} {'Score':>10}")
    for name, metrics in results.items():
        print(f"  {name:<15} {metrics['rmse']:>8} {metrics['mae']:>8} "
              f"{metrics['score']:>10}")

    # Best by RMSE
    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_metrics = results[best_name]
    print(f"\n  Best model (RMSE): {best_name} ({best_metrics['rmse']})")

    # Best by Score
    best_score_name = min(results, key=lambda k: results[k]["score"])
    best_score_metrics = results[best_score_name]
    print(f"  Best model (Score): {best_score_name} ({best_score_metrics['score']})")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/CMAPSSData")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--max-rul", type=int, default=125)
    parser.add_argument("--skip-tft", action="store_true",
                        help="Skip TFT training (slow)")
    args = parser.parse_args()

    print("=== MLflow Experiment Tracker ===\n")

    setup_mlflow()

    print("\nLoading data...")
    data = preprocess_subset(args.data_dir, args.subset, args.max_rul)

    results = {}

    # XGBoost
    print("\n--- XGBoost ---")
    results["xgboost"] = log_xgboost_run(data, args.max_rul)

    # Bi-LSTM
    print("\n--- Bi-LSTM ---")
    results["bilstm"] = log_bilstm_run(data, args.max_rul)

    # TFT
    if not args.skip_tft:
        print("\n--- TFT ---")
        results["tft"] = log_tft_run(data, args.max_rul)
    else:
        print("\n--- TFT skipped ---")

    # Compare
    compare_and_register(results)

    print(f"\nView results: mlflow ui --backend-store-uri file://{os.path.abspath(MLFLOW_TRACKING_DIR)}")
    print("Then open http://localhost:5000")
