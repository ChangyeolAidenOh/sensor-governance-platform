"""
core_pipeline/features/window_features.py — window-statistic features shared by the detector (IF) and the
diagnosis classifier (prereg §4.2): mean, std, slope, min, max × 52 sensors = 260 features. Same function for
20-sample and 40-sample (expanded) windows; only the input length differs. Deterministic, no state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core_pipeline.stream.window_view import PROCESS_COLS

STATS = ("mean", "std", "slope", "min", "max")
FEATURE_NAMES = [f"{s}__{st}" for st in STATS for s in PROCESS_COLS]


def window_features(rows: pd.DataFrame) -> np.ndarray:
    """rows: the window's samples (cycle + PROCESS_COLS), ordered by cycle. Returns a 1-D float64 vector (260,)."""
    x = rows[PROCESS_COLS].to_numpy(dtype=np.float64)
    n = x.shape[0]
    if n < 2:
        raise ValueError("window needs >= 2 samples")
    t = np.arange(n, dtype=np.float64)
    t = t - t.mean()
    slope = (t @ (x - x.mean(axis=0))) / (t @ t)
    feats = np.concatenate([x.mean(axis=0), x.std(axis=0, ddof=0), slope, x.min(axis=0), x.max(axis=0)])
    if not np.isfinite(feats).all():
        raise ValueError("non-finite feature")
    return feats


def features_frame(windows: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(np.vstack([window_features(w) for w in windows]), columns=FEATURE_NAMES)
