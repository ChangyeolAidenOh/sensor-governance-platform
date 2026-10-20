"""
core_pipeline/anomaly/conformal.py — split conformal p-value for detection (prereg §4.1).

  p_t = (1 + #{ s_cal >= s_t }) / (n_cal + 1)      (ties count as >=; conservative)

Calibration scores come from the calibration group's FAULT-FREE windows only (prereg §3.0). Higher score = more
anomalous. Marginal validity P(p <= t) <= t holds under exchangeability of calibration and test windows (the
calibration set's own randomness included); nothing conditional is claimed. Scope on TEP: 4,800 non-overlapping
windows from 100 runs do not by themselves make window-level exchangeability hold. What is verified here is the
p-value FORMULA; the p distribution on real TEP fault-free windows is reported as a dev diagnostic (null_p_ecdf) and
is not read as a confirmation of exchangeability.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ConformalCalibration:
    cal_scores_sorted: np.ndarray        # ascending
    meta: dict

    @classmethod
    def from_scores(cls, cal_scores: np.ndarray, meta: dict | None = None) -> "ConformalCalibration":
        s = np.asarray(cal_scores, dtype=np.float64)
        if s.ndim != 1 or len(s) < 10 or not np.isfinite(s).all():
            raise ValueError("calibration scores must be a finite 1-D array with >= 10 values")
        return cls(np.sort(s), {**(meta or {}), "n_cal": int(len(s))})

    @property
    def n_cal(self) -> int:
        return len(self.cal_scores_sorted)

    def p_value(self, score: float) -> float:
        if not np.isfinite(score):
            raise ValueError("non-finite test score")
        # #{s_cal >= s} = n - (number strictly less than s)
        n_less = int(np.searchsorted(self.cal_scores_sorted, score, side="left"))
        return (1 + (self.n_cal - n_less)) / (self.n_cal + 1)

    def p_values(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float64)
        if not np.isfinite(s).all():
            raise ValueError("non-finite test score(s)")
        n_less = np.searchsorted(self.cal_scores_sorted, s, side="left")
        return (1 + (self.n_cal - n_less)) / (self.n_cal + 1)

    def save(self, path: Path) -> None:
        path = Path(path)
        np.save(path.with_suffix(".npy"), self.cal_scores_sorted)
        path.with_suffix(".json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> "ConformalCalibration":
        path = Path(path)
        return cls(np.load(path.with_suffix(".npy")), json.loads(path.with_suffix(".json").read_text()))
