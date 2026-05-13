"""
SHAP Analysis for XGBoost RUL Model

TreeSHAP for global and per-engine feature importance.
Connects Layer 2B (prediction) to Layer 1 (governance) by
identifying which sensors drive RUL predictions.

Complements TFT Variable Selection:
  - TFT VSN: temporal importance (per-timestep weights)
  - SHAP: causal contribution (per-prediction attribution)
"""

import numpy as np
import pandas as pd
import shap
import pickle
from pathlib import Path

from core_pipeline.data.preprocess import preprocess_subset, DEGRADATION_SENSORS
from core_pipeline.rul.xgboost_rul import train_xgboost_rul


def run_shap_analysis(data_dir: str = "data/raw/CMAPSSData",
                      subset: str = "FD001",
                      max_rul: int = 125,
                      n_background: int = 200) -> dict:
    """Run TreeSHAP analysis on XGBoost RUL model.

    Args:
        data_dir: path to C-MAPSS data
        subset: which subset
        max_rul: RUL clipping value
        n_background: number of background samples for SHAP

    Returns:
        dict with shap_values, feature_importance, top_features
    """
    print(f"=== SHAP Analysis ({subset}) ===\n")

    # Load and train
    print("Loading data...")
    data = preprocess_subset(data_dir, subset, max_rul)

    print("Training XGBoost...")
    result = train_xgboost_rul(data["train"], max_rul=max_rul)
    model = result["model"]
    feature_cols = result["feature_cols"]

    # Prepare test features (last cycle per engine)
    test_features = []
    for engine_id in sorted(data["test"]["engine_id"].unique()):
        engine_df = data["test"][data["test"]["engine_id"] == engine_id].sort_values("cycle")
        last_row = engine_df.iloc[-1]
        test_features.append(last_row[feature_cols].values)

    X_test = np.array(test_features)
    X_test_df = pd.DataFrame(X_test, columns=feature_cols)

    # TreeSHAP
    print(f"Computing TreeSHAP (test engines: {len(X_test)})...")
    # XGBoost/SHAP version compatibility: use Explainer with masker
    X_test_np = X_test_df.values.astype(np.float64)
    background = shap.maskers.Independent(X_test_np[:50])
    explainer = shap.Explainer(model.predict, background)
    explanation = explainer(X_test_np)
    shap_values = explanation.values

    # Global importance: mean |SHAP|
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    # Sensor-level aggregation (group rolling features by base sensor)
    sensor_importance = {}
    for _, row in importance_df.iterrows():
        feat = row["feature"]
        val = row["mean_abs_shap"]

        # Extract base sensor name
        base = feat
        for suffix in ["_rmean_5", "_rmean_10", "_rmean_20",
                        "_rstd_5", "_rstd_10", "_rstd_20"]:
            if feat.endswith(suffix):
                base = feat.replace(suffix, "")
                break

        if base not in sensor_importance:
            sensor_importance[base] = 0.0
        sensor_importance[base] += val

    sensor_df = pd.DataFrame([
        {"sensor": k, "total_shap": v}
        for k, v in sensor_importance.items()
    ]).sort_values("total_shap", ascending=False)

    # Normalize to percentages
    total = sensor_df["total_shap"].sum()
    sensor_df["importance_pct"] = (sensor_df["total_shap"] / total * 100).round(2)

    # Print results
    print(f"\n--- Global Feature Importance (top 15) ---\n")
    print(importance_df.head(15).to_string(index=False))

    print(f"\n--- Sensor-Level Importance ---\n")
    print(sensor_df.head(14).to_string(index=False))

    # Compare with TFT
    print(f"\n--- SHAP vs TFT Variable Importance ---\n")
    tft_importance = {
        "sensor_11": 16.3, "sensor_3": 15.7, "sensor_4": 15.6,
        "sensor_14": 15.3, "sensor_15": 11.2, "sensor_20": 7.0,
        "sensor_21": 5.6, "sensor_17": 3.9,
    }
    print(f"  {'Sensor':<12} {'SHAP %':>10} {'TFT %':>10} {'Agreement':>12}")
    print(f"  {'-'*44}")

    shap_top5 = set(sensor_df.head(5)["sensor"].values)
    tft_top5 = set(list(tft_importance.keys())[:5])
    overlap = shap_top5 & tft_top5

    for _, row in sensor_df.head(10).iterrows():
        s = row["sensor"]
        shap_pct = row["importance_pct"]
        tft_pct = tft_importance.get(s, 0)
        agree = "✓" if s in tft_top5 else ""
        print(f"  {s:<12} {shap_pct:>10.2f} {tft_pct:>10.1f} {agree:>12}")

    print(f"\n  Top-5 overlap: {len(overlap)}/5 sensors")

    # Save
    out_dir = Path("experiments")
    out_dir.mkdir(exist_ok=True)
    sensor_df.to_csv(out_dir / "shap_sensor_importance.csv", index=False)
    importance_df.to_csv(out_dir / "shap_feature_importance.csv", index=False)
    print(f"\nResults saved to experiments/shap_*.csv")

    return {
        "shap_values": shap_values,
        "feature_importance": importance_df,
        "sensor_importance": sensor_df,
        "top5_overlap_with_tft": len(overlap),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/CMAPSSData")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--max-rul", type=int, default=125)
    args = parser.parse_args()

    run_shap_analysis(args.data_dir, args.subset, args.max_rul)
