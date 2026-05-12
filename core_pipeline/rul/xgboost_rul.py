"""
XGBoost RUL Prediction Baseline

Tabular baseline for Remaining Useful Life prediction on C-MAPSS.
Serves as the reference model for corruption impact experiments.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Optional

import xgboost as xgb


def cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA scoring function for RUL prediction.

    Asymmetric: penalizes late predictions (under-estimation) more
    than early predictions (over-estimation).

    s = sum(exp(-d/13) - 1)  if d < 0  (early prediction)
        sum(exp(d/10) - 1)   if d >= 0 (late prediction)
    """
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(score))


def prepare_features(df: pd.DataFrame,
                     feature_cols: Optional[list] = None,
                     target_col: str = "rul") -> tuple:
    """Extract feature matrix and target vector.

    Args:
        df: preprocessed DataFrame with rolling features
        feature_cols: columns to use as features (auto-detect if None)
        target_col: target column name

    Returns:
        (X, y) tuple
    """
    if feature_cols is None:
        exclude = {"engine_id", "cycle", "rul", "subset", "split"}
        feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].values
    y = df[target_col].values if target_col in df.columns else None
    return X, y, feature_cols


def train_xgboost_rul(train_df: pd.DataFrame,
                      feature_cols: Optional[list] = None,
                      params: Optional[dict] = None,
                      max_rul: int = 125) -> dict:
    """Train XGBoost RUL model.

    Args:
        train_df: training data with rul column
        feature_cols: feature columns
        params: XGBoost parameters

    Returns:
        dict with model, feature_cols, metrics
    """
    if params is None:
        params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "random_state": 42,
            "n_jobs": -1,
        }

    X_train, y_train, feature_cols = prepare_features(
        train_df, feature_cols
    )

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)

    y_pred_train = model.predict(X_train)
    y_pred_train = np.clip(y_pred_train, 0, max_rul)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "train_rmse": round(train_rmse, 4),
        "train_mae": round(train_mae, 4),
    }


def evaluate_rul(model: xgb.XGBRegressor,
                 test_df: pd.DataFrame,
                 rul_true: pd.Series,
                 feature_cols: list,
                 max_rul: int = 125) -> dict:
    """Evaluate RUL model on test set.

    C-MAPSS test evaluation: predict RUL at each engine's last cycle.

    Args:
        model: trained XGBRegressor
        test_df: test data
        rul_true: ground truth RUL per engine
        feature_cols: feature columns used during training
        max_rul: clipping value

    Returns:
        dict with RMSE, MAE, Score metrics
    """
    # Get last cycle per engine
    last_idx = test_df.groupby("engine_id")["cycle"].idxmax()
    test_last = test_df.loc[last_idx]

    X_test, _, _ = prepare_features(test_last, feature_cols)
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, max_rul)

    y_true = rul_true.values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    score = cmapss_score(y_true, y_pred)

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "score": round(score, 2),
        "y_true": y_true,
        "y_pred": y_pred,
    }


if __name__ == "__main__":
    import argparse
    from core_pipeline.data.preprocess import preprocess_subset

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/CMAPSSData")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--max-rul", type=int, default=125)
    args = parser.parse_args()

    print(f"Loading {args.subset}...")
    data = preprocess_subset(args.data_dir, args.subset, args.max_rul)

    print("Training XGBoost baseline...")
    result = train_xgboost_rul(data["train"], max_rul=args.max_rul)
    print(f"  Train RMSE: {result['train_rmse']}")

    print("Evaluating...")
    metrics = evaluate_rul(
        result["model"], data["test"], data["rul_true"],
        result["feature_cols"], args.max_rul
    )
    print(f"  Test RMSE:  {metrics['rmse']}")
    print(f"  Test MAE:   {metrics['mae']}")
    print(f"  Test Score: {metrics['score']}")
