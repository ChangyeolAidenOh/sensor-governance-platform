"""
C-MAPSS Data Loading and Preprocessing

NASA Commercial Modular Aero-Propulsion System Simulation dataset.
4 subsets (FD001-FD004) with 3 operational settings + 21 sensors per engine.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Column definitions
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_FEATURE_COLS = SETTING_COLS + SENSOR_COLS
INDEX_COLS = ["engine_id", "cycle"]
COLUMN_NAMES = INDEX_COLS + ALL_FEATURE_COLS

# Sensors known to be near-constant (low variance) in FD001
# These carry minimal degradation signal
LOW_VARIANCE_SENSORS = ["sensor_1", "sensor_5", "sensor_6",
                        "sensor_10", "sensor_16", "sensor_18", "sensor_19"]

# Sensors with strong degradation signal (literature consensus)
DEGRADATION_SENSORS = ["sensor_2", "sensor_3", "sensor_4", "sensor_7",
                       "sensor_8", "sensor_9", "sensor_11", "sensor_12",
                       "sensor_13", "sensor_14", "sensor_15", "sensor_17",
                       "sensor_20", "sensor_21"]

SUBSETS = {
    "FD001": {"conditions": 1, "faults": 1, "train_engines": 100, "test_engines": 100},
    "FD002": {"conditions": 6, "faults": 1, "train_engines": 260, "test_engines": 259},
    "FD003": {"conditions": 1, "faults": 2, "train_engines": 100, "test_engines": 100},
    "FD004": {"conditions": 6, "faults": 2, "train_engines": 248, "test_engines": 249},
}


def load_subset(data_dir: str, subset: str, split: str = "train") -> pd.DataFrame:
    """Load a single C-MAPSS subset file.

    Args:
        data_dir: path to CMAPSSData directory
        subset: one of FD001, FD002, FD003, FD004
        split: 'train' or 'test'

    Returns:
        DataFrame with named columns
    """
    filename = f"{split}_{subset}.txt"
    filepath = Path(data_dir) / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"{filepath} not found. Download C-MAPSS from "
            "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data"
        )

    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)
    df["subset"] = subset
    df["split"] = split
    return df


def load_rul_labels(data_dir: str, subset: str) -> pd.Series:
    """Load ground-truth RUL values for test set."""
    filepath = Path(data_dir) / f"RUL_{subset}.txt"
    rul = pd.read_csv(filepath, sep=r"\s+", header=None, names=["rul"])
    return rul["rul"]


def compute_rul(df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """Compute piece-wise linear RUL labels for training data.

    Early cycles are clipped at max_rul because degradation has not
    yet begun (right-censoring).

    Args:
        df: training data with engine_id and cycle columns
        max_rul: clipping threshold

    Returns:
        DataFrame with 'rul' column added
    """
    df = df.copy()
    max_cycles = df.groupby("engine_id")["cycle"].transform("max")
    df["rul"] = max_cycles - df["cycle"]
    df["rul"] = df["rul"].clip(upper=max_rul)
    return df


def normalize_by_condition(df: pd.DataFrame,
                           condition_cols: Optional[list] = None) -> pd.DataFrame:
    """Normalize sensor readings within each operating condition cluster.

    For multi-condition subsets (FD002, FD004), sensors behave differently
    under different conditions. Normalizing within condition clusters
    removes this confound.

    Args:
        df: DataFrame with sensor and setting columns
        condition_cols: columns defining operating condition (default: all settings)

    Returns:
        DataFrame with normalized sensor values
    """
    if condition_cols is None:
        condition_cols = SETTING_COLS

    df = df.copy()

    # Cluster operating conditions (round settings for grouping)
    for col in condition_cols:
        df[f"{col}_bin"] = df[col].round(2)

    bin_cols = [f"{c}_bin" for c in condition_cols]

    for sensor in SENSOR_COLS:
        group_mean = df.groupby(bin_cols)[sensor].transform("mean")
        group_std = df.groupby(bin_cols)[sensor].transform("std")
        group_std = group_std.replace(0, 1)
        df[sensor] = (df[sensor] - group_mean) / group_std

    df.drop(columns=bin_cols, inplace=True)
    return df


def add_rolling_features(df: pd.DataFrame,
                         sensors: Optional[list] = None,
                         windows: tuple = (5, 10, 20)) -> pd.DataFrame:
    """Add rolling statistics as features per engine.

    Args:
        df: DataFrame sorted by engine_id and cycle
        sensors: sensor columns to compute features for
        windows: rolling window sizes

    Returns:
        DataFrame with rolling mean/std columns added
    """
    if sensors is None:
        sensors = DEGRADATION_SENSORS

    df = df.copy()
    df = df.sort_values(["engine_id", "cycle"])

    for w in windows:
        for sensor in sensors:
            grouped = df.groupby("engine_id")[sensor]
            df[f"{sensor}_rmean_{w}"] = grouped.transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            df[f"{sensor}_rstd_{w}"] = grouped.transform(
                lambda x: x.rolling(w, min_periods=1).std().fillna(0)
            )

    return df


def identify_useful_sensors(df: pd.DataFrame,
                            variance_threshold: float = 0.01) -> dict:
    """Classify sensors by their variance (degradation signal strength).

    Returns:
        dict with 'useful' and 'constant' sensor lists
    """
    variances = df[SENSOR_COLS].var()
    normalized_var = variances / variances.max()

    useful = normalized_var[normalized_var > variance_threshold].index.tolist()
    constant = normalized_var[normalized_var <= variance_threshold].index.tolist()

    return {"useful": useful, "constant": constant, "variances": normalized_var}


def preprocess_subset(data_dir: str,
                      subset: str = "FD001",
                      max_rul: int = 125,
                      add_rolling: bool = True,
                      normalize: bool = True) -> dict:
    """Full preprocessing pipeline for one subset.

    Returns:
        dict with 'train', 'test', 'rul_true' DataFrames
    """
    train = load_subset(data_dir, subset, "train")
    test = load_subset(data_dir, subset, "test")
    rul_true = load_rul_labels(data_dir, subset)

    # RUL labels for training
    train = compute_rul(train, max_rul=max_rul)

    # RUL labels for test (last cycle per engine + ground truth)
    test_last = test.groupby("engine_id")["cycle"].max().reset_index()
    test_last["rul"] = rul_true.values

    # Normalize if multi-condition
    if SUBSETS[subset]["conditions"] > 1 and normalize:
        train = normalize_by_condition(train)
        test = normalize_by_condition(test)

    # Rolling features
    if add_rolling:
        train = add_rolling_features(train)
        test = add_rolling_features(test)

    sensor_info = identify_useful_sensors(train)
    print(f"[{subset}] {len(sensor_info['useful'])} useful sensors, "
          f"{len(sensor_info['constant'])} constant sensors")

    return {
        "train": train,
        "test": test,
        "rul_true": rul_true,
        "test_last": test_last,
        "sensor_info": sensor_info,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/raw/CMAPSSData")
    parser.add_argument("--subset", type=str, default="FD001")
    parser.add_argument("--max-rul", type=int, default=125)
    parser.add_argument("--output-dir", type=str, default="data/processed")
    args = parser.parse_args()

    result = preprocess_subset(args.data_dir, args.subset, args.max_rul)

    out_dir = Path(args.output_dir) / args.subset
    out_dir.mkdir(parents=True, exist_ok=True)

    result["train"].to_parquet(out_dir / "train.parquet", index=False)
    result["test"].to_parquet(out_dir / "test.parquet", index=False)
    result["rul_true"].to_frame().to_parquet(out_dir / "rul_true.parquet", index=False)

    print(f"Saved to {out_dir}")
    print(f"  Train: {result['train'].shape}")
    print(f"  Test:  {result['test'].shape}")
