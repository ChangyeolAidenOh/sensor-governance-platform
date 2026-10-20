"""
core_pipeline/anomaly/at_window.py — S2-07 (prereg §4.1 "AT + conformal은 Should 비교군"): the v1 Anomaly
Transformer on the SAME 20-sample window grid, fitted on the SAME training fault-free windows and calibrated on the
SAME calibration fault-free windows as the IF detector, so the comparison is fair. IF stays the primary detector
whatever this shows (prereg §6: "AT가 IF에 밀림 → v1 결론 재확인"; the reverse never replaces IF).

Reuse: AnomalyTransformerModel, train_anomaly_transformer, score_windows, association_discrepancy — unchanged.
Changed vs v1: normalisation. v1's CMAPSSWindowDataset normalises each dataset by ITS OWN mean/std, so calibration
and evaluation windows would be normalised differently from training. Here the per-sensor mean/std are fixed from the
training fault-free windows and applied to every later window (stored with the model).

Score: per-window mean of the per-timestep association discrepancy (v1's definition; the original paper's criterion
combines reconstruction error with softmax(-discrepancy), which can point the other way). ORIENTATION is a dev-time
decision recorded once in meta ('sign': +1 = higher discrepancy more anomalous, -1 = flipped) — chosen on dev run
ids only, never on the evaluation pool.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from core_pipeline import prereg_constants as P
from core_pipeline.anomaly.anomaly_transformer import (AnomalyTransformerModel, score_windows,
                                                       train_anomaly_transformer)
from core_pipeline.stream.window_view import PROCESS_COLS

AT_VERSION = "at_window_v1"


class FixedNormWindowDataset(Dataset):
    """Windows (n, L, 52) normalised with FIXED mean/std (from training fault-free), not their own."""

    def __init__(self, windows: np.ndarray, mean: np.ndarray, std: np.ndarray):
        assert windows.ndim == 3 and windows.shape[2] == len(PROCESS_COLS)
        self.windows = ((windows - mean) / std).astype(np.float32)
        self.window_size = windows.shape[1]
        self.sensor_cols = PROCESS_COLS
        self.window_indices = [(i, 0) for i in range(len(windows))]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx])


@dataclass
class WindowAT:
    model: AnomalyTransformerModel
    mean: np.ndarray
    std: np.ndarray
    device: str
    meta: dict

    @classmethod
    def fit(cls, train_windows: np.ndarray, n_epochs: int = 30, batch_size: int = 64, d_model: int = 128,
            n_heads: int = 8, n_layers: int = 3, d_ff: int = 256, lr: float = 5e-5, seed: int = P.SEED,
            device: str = "auto", meta: dict | None = None) -> "WindowAT":
        torch.manual_seed(seed); np.random.seed(seed)
        flat = train_windows.reshape(-1, train_windows.shape[2])
        mean, std = flat.mean(axis=0), flat.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        ds = FixedNormWindowDataset(train_windows, mean, std)
        res = train_anomaly_transformer(ds, n_features=len(PROCESS_COLS), d_model=d_model, n_heads=n_heads,
                                        n_layers=n_layers, d_ff=d_ff, window_size=train_windows.shape[1],
                                        batch_size=batch_size, n_epochs=n_epochs, lr=lr, device=device)
        m = {**(meta or {}), "version": AT_VERSION, "n_train_windows": int(len(train_windows)),
             "window_size": int(train_windows.shape[1]), "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
             "d_ff": d_ff, "n_epochs": n_epochs, "batch_size": batch_size, "lr": lr, "seed": seed,
             "device": res["device"], "sign": +1, "score": "per-window mean of per-timestep association discrepancy (v1)"}
        return cls(res["model"], mean, std, res["device"], m)

    def raw_score(self, windows: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Per-window mean discrepancy, BEFORE orientation."""
        ds = FixedNormWindowDataset(windows, self.mean, self.std)
        per_t = score_windows(self.model, ds, device=self.device, batch_size=batch_size)   # (n, L)
        return per_t.mean(axis=1).astype(np.float64)

    def score(self, windows: np.ndarray) -> np.ndarray:
        """Oriented score: higher = more anomalous (sign fixed in meta)."""
        return self.meta.get("sign", 1) * self.raw_score(windows)

    def save(self, dir_: Path) -> None:
        dir_ = Path(dir_); dir_.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), dir_ / "at_model.pt")
        np.savez(dir_ / "at_norm.npz", mean=self.mean, std=self.std)
        (dir_ / "at_meta.json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, dir_: Path, device: str = "cpu") -> "WindowAT":
        dir_ = Path(dir_)
        meta = json.loads((dir_ / "at_meta.json").read_text())
        model = AnomalyTransformerModel(n_features=len(PROCESS_COLS), d_model=meta["d_model"], n_heads=meta["n_heads"],
                                        n_layers=meta["n_layers"], d_ff=meta["d_ff"], window_size=meta["window_size"])
        model.load_state_dict(torch.load(dir_ / "at_model.pt", map_location=device))
        model.to(device).eval()
        d = np.load(dir_ / "at_norm.npz")
        return cls(model, d["mean"], d["std"], device, meta)
