"""
core_pipeline/anomaly/if_window.py — S2-01: window-level Isolation Forest on window-statistic features, reusing the
v1 wrapper (train_isolation_forest / score_anomalies with explicit feature_cols). Score orientation: HIGHER = more
anomalous (= -decision_function), so conformal p-values are small for anomalous windows.

Fit data: training split fault-free windows (prereg §3.0). Calibration (conformal) is a separate object.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from core_pipeline.anomaly.isolation_forest import score_anomalies, train_isolation_forest
from core_pipeline.features.window_features import FEATURE_NAMES

IF_VERSION = "if_window_v1"


@dataclass
class WindowIF:
    model: object
    feature_cols: list[str]
    meta: dict

    @classmethod
    def fit(cls, X_train: pd.DataFrame, n_estimators: int = 200, contamination: float = 0.05,
            random_state: int = 20261018, meta: dict | None = None) -> "WindowIF":
        assert list(X_train.columns) == FEATURE_NAMES
        res = train_isolation_forest(X_train, feature_cols=FEATURE_NAMES, contamination=contamination,
                                     n_estimators=n_estimators, random_state=random_state)
        m = {**(meta or {}), "version": IF_VERSION, "n_train_windows": res["n_healthy_samples"],
             "n_estimators": n_estimators, "contamination": contamination, "random_state": random_state,
             "n_features": len(FEATURE_NAMES)}
        return cls(res["model"], res["feature_cols"], m)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Higher = more anomalous."""
        scored = score_anomalies(self.model, X, self.feature_cols)
        return -scored["anomaly_score"].to_numpy(dtype=np.float64)

    def save(self, dir_: Path) -> None:
        dir_ = Path(dir_); dir_.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, dir_ / "if_model.joblib")
        (dir_ / "if_meta.json").write_text(json.dumps({**self.meta, "feature_cols": self.feature_cols}, indent=2))

    @classmethod
    def load(cls, dir_: Path) -> "WindowIF":
        dir_ = Path(dir_)
        meta = json.loads((dir_ / "if_meta.json").read_text())
        return cls(joblib.load(dir_ / "if_model.joblib"), meta.pop("feature_cols"), meta)
