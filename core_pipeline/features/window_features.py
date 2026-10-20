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


def episode_window_features(x: np.ndarray, window: int, stride: int, start_window: int = 0,
                            end_window: int | None = None) -> np.ndarray:
    """Vectorised version for a whole episode: x is (n_cycles, 52) ordered by cycle. Returns (n_windows, 260) with
    rows for windows start_window..end_window (inclusive; end defaults to the last full window). Window k covers
    rows [k*stride, k*stride + window). Identical values to window_features() applied per window (tested)."""
    n = x.shape[0]
    last = (n - window) // stride
    end_window = last if end_window is None else min(end_window, last)
    idx = np.arange(start_window, end_window + 1)
    if len(idx) == 0:
        return np.empty((0, len(FEATURE_NAMES)))
    w = np.stack([x[k * stride: k * stride + window] for k in idx])          # (m, window, 52)
    t = np.arange(window, dtype=np.float64); t = t - t.mean()
    mean = w.mean(axis=1)
    slope = np.einsum("i,mij->mj", t, w - mean[:, None, :]) / (t @ t)
    feats = np.concatenate([mean, w.std(axis=1, ddof=0), slope, w.min(axis=1), w.max(axis=1)], axis=1)
    if not np.isfinite(feats).all():
        raise ValueError("non-finite feature")
    return feats


def expanded_window_features(x: np.ndarray, window: int, stride: int, ks: np.ndarray) -> np.ndarray:
    """Features of the backward-expanded window (k-1 ∪ k) = rows [(k-1)*stride, k*stride + window) for each k in ks
    (k >= 1). Same statistics over 2*window samples — what detect_diagnose(expand=2) computes at window k."""
    w = np.stack([x[(k - 1) * stride: k * stride + window] for k in ks])
    t = np.arange(w.shape[1], dtype=np.float64); t = t - t.mean()
    mean = w.mean(axis=1)
    slope = np.einsum("i,mij->mj", t, w - mean[:, None, :]) / (t @ t)
    feats = np.concatenate([mean, w.std(axis=1, ddof=0), slope, w.min(axis=1), w.max(axis=1)], axis=1)
    if not np.isfinite(feats).all():
        raise ValueError("non-finite feature")
    return feats
