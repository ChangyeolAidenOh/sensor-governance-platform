"""
core_pipeline/anomaly/zscore_l0.py — S2-02: the L0 fixed-threshold baseline B2 (prereg §4.1).

Per-sensor |z| > 3 against the CALIBRATION fault-free mean/std (external, fixed); a window alarms if any sample of
any sensor exceeds it. No gate, no conformal, no training. Not v1's rolling_zscore (that one uses an in-episode
trailing window, which is a different baseline than the pre-registered one).

Explicit choices:
  * sensors whose calibration std is 0 have no z-score; they are EXCLUDED from the test and listed in meta
    (`excluded_constant_sensors`). No separate "left its constant value" rule — that is not part of the pre-registered B2.
  * B2 runs without the gate, so non-finite inputs are rejected with ValueError rather than silently compared.
  * per-window false-alarm rate of "any of 52 sensors × 20 samples > 3σ" is far above the per-observation 3σ
    probability; B2 is the pre-registered L0 baseline as written, NOT matched to the IF/conformal false-alarm level.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core_pipeline.stream.window_view import PROCESS_COLS

Z_THRESHOLD = 3.0


@dataclass(frozen=True)
class ZScoreL0:
    mean: np.ndarray
    std: np.ndarray
    meta: dict

    @classmethod
    def fit(cls, calibration_rows: pd.DataFrame, meta: dict | None = None) -> "ZScoreL0":
        x = calibration_rows[PROCESS_COLS].to_numpy(dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("non-finite calibration rows")
        mean, std = x.mean(axis=0), x.std(axis=0, ddof=0)
        excluded = [c for c, sd in zip(PROCESS_COLS, std) if sd == 0]
        return cls(mean, std, {**(meta or {}), "n_rows": int(len(x)), "z_threshold": Z_THRESHOLD,
                               "excluded_constant_sensors": excluded})

    @property
    def active(self) -> np.ndarray:
        return self.std > 0

    def max_abs_z(self, rows: pd.DataFrame) -> float:
        x = rows[PROCESS_COLS].to_numpy(dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("non-finite input rows")
        a = self.active
        z = (x[:, a] - self.mean[a]) / self.std[a]
        return float(np.abs(z).max()) if a.any() else 0.0

    def alarm(self, rows: pd.DataFrame) -> bool:
        return self.max_abs_z(rows) > Z_THRESHOLD

    def save(self, dir_: Path) -> None:
        dir_ = Path(dir_); dir_.mkdir(parents=True, exist_ok=True)
        np.savez(dir_ / "zscore_l0.npz", mean=self.mean, std=self.std)
        (dir_ / "zscore_l0.json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, dir_: Path) -> "ZScoreL0":
        dir_ = Path(dir_); d = np.load(dir_ / "zscore_l0.npz")
        return cls(d["mean"], d["std"], json.loads((dir_ / "zscore_l0.json").read_text()))
