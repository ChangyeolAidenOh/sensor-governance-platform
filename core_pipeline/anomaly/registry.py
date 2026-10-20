"""
core_pipeline/anomaly/registry.py — one loader for the two window detectors so gate/drift calibration and dev checks
can run the SAME pipeline on either path (ST2-06 / Stage 5 comparison contract):
  models/detector_v1     → WindowIF  (window-statistic features)       ← primary
  models/detector_at_v1  → WindowAT  (raw 20×52 windows, v1 AT discrepancy mean score)  ← Should comparison
Both expose score_rows(rows_df) -> float (higher = more anomalous, oriented) and the conformal calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from core_pipeline.anomaly.conformal import ConformalCalibration
from core_pipeline.stream.window_view import PROCESS_COLS


@dataclass
class LoadedDetector:
    name: str
    version: str
    score_rows: Callable[[pd.DataFrame], float]
    conformal: ConformalCalibration
    meta: dict


def load_detector(dir_: Path, device: str = "cpu") -> LoadedDetector:
    dir_ = Path(dir_)
    conf = ConformalCalibration.load(dir_ / "conformal")
    if (dir_ / "at_meta.json").exists():
        from core_pipeline.anomaly.at_window import WindowAT
        at = WindowAT.load(dir_, device=device)
        def score_rows(rows: pd.DataFrame) -> float:
            w = rows[PROCESS_COLS].to_numpy(dtype=np.float64)[None, :, :]
            return float(at.score(w)[0])
        return LoadedDetector("at", at.meta.get("version", "at_window_v1"), score_rows, conf, at.meta)
    from core_pipeline.anomaly.if_window import WindowIF
    from core_pipeline.features.window_features import window_features, FEATURE_NAMES
    m = WindowIF.load(dir_)
    def score_rows(rows: pd.DataFrame) -> float:
        X = pd.DataFrame(window_features(rows)[None, :], columns=FEATURE_NAMES)
        return float(m.score(X)[0])
    return LoadedDetector("if", m.meta.get("version", "if_window_v1"), score_rows, conf, m.meta)
