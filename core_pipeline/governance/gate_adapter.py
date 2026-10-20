"""
core_pipeline/governance/gate_adapter.py — S1-04: external gate reference (ST0-10 (a), prereg §3.0).

v1's run_governed_pipeline() used the first 30 rows of each episode as the gate reference. v2 injects a reference
built ONCE from the calibration group's fault-free episodes (external, fixed), so no evaluation episode is ever its
own baseline (that would read the pre-onset region = future-referencing for early windows, and leaks nothing but
also breaks causality). QualityGate.evaluate() is reused unchanged; only df_reference changes.

Reference = a deterministic ROW subsample (rows kept intact → cross-sensor correlations preserved) of the calibration
fault-free samples: N_REF_ROWS rows × 52 sensors, plus provenance (episode ids used, seed, source content hash).
PSI bins and reference correlation are then computed by quality_metrics on this frame, exactly as in v1.

Thresholds are NOT changed here (Stage 2 recalibration). This module only moves the reference outside the episode.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from core_pipeline import prereg_constants as P
from core_pipeline.governance.quality_gate import GateResult, QualityGate
from core_pipeline.stream.window_view import PROCESS_COLS, WindowView

N_REF_ROWS = 2000           # per-window cost: corr(2000×52) + 52 percentile sets ≈ ms; full 96k rows would be ~s/window
REF_VERSION = "gate_ref_v1"


@dataclass(frozen=True)
class GateReference:
    frame: pd.DataFrame          # N_REF_ROWS × PROCESS_COLS, external & fixed
    meta: dict

    @classmethod
    def from_calibration(cls, frames: Iterable[pd.DataFrame], episode_ids: list[str], seed: int = P.SEED,
                         n_rows: int = N_REF_ROWS) -> "GateReference":
        full = pd.concat([f[PROCESS_COLS] for f in frames], ignore_index=True)
        if full.isna().any().any():
            raise ValueError("calibration frames contain NaN")
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(full), size=min(n_rows, len(full)), replace=False))
        sample = full.iloc[idx].reset_index(drop=True)
        src_sha = hashlib.sha256(pd.util.hash_pandas_object(full, index=False).values.tobytes()).hexdigest()
        meta = {"version": REF_VERSION, "seed": seed, "n_source_rows": int(len(full)), "n_rows": int(len(sample)),
                "n_episodes": len(episode_ids), "episode_ids_sha256": hashlib.sha256("\n".join(sorted(episode_ids)).encode()).hexdigest(),
                "source_content_sha256": src_sha, "sensors": PROCESS_COLS}
        return cls(sample, meta)

    def save(self, path: Path) -> None:
        path = Path(path)
        self.frame.to_parquet(path, index=False)
        path.with_suffix(".json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> "GateReference":
        path = Path(path)
        return cls(pd.read_parquet(path), json.loads(path.with_suffix(".json").read_text()))

    @property
    def sha256(self) -> str:
        return self.meta["source_content_sha256"]


def evaluate_window(gate: QualityGate, view: WindowView, ref: GateReference) -> GateResult:
    """Causal gate call: current window from the view (cycles start..end only), reference external."""
    return gate.evaluate(view.current(), ref.frame, PROCESS_COLS)


def gate_status_str(result: GateResult) -> str:
    return result.status.value.upper()      # 'PASS' | 'FLAG' | 'BLOCK' as used by EpisodeState.observe
