"""
core_pipeline/stream/window_view.py — the ONLY data object a policy/skill receives for a window (S1-03).

A WindowView physically contains samples 1..end_cycle of its episode (a truncated copy) — never the full episode —
so max_sample is enforced by construction, not by convention. Label columns are stripped. Access past the bound raises.
"""
from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import pandas as pd

from core_pipeline import prereg_constants as P

PROCESS_COLS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
FORBIDDEN_COLS = {"fault_scenario", "is_fault_active", "engine_id", "run_id", "split", "observed_divergence_cycle"}


class FutureAccessError(RuntimeError):
    pass


@dataclass(frozen=True)
class WindowView:
    episode_id: UUID
    window_index: int
    start_cycle: int
    end_cycle: int
    _frame: pd.DataFrame            # cycles 1..end_cycle only, columns: cycle + PROCESS_COLS

    @classmethod
    def build(cls, episode_id: UUID, window_index: int, episode_frame: pd.DataFrame) -> "WindowView":
        start = window_index * P.STRIDE + 1
        end = start + P.WINDOW_SIZE - 1
        cols = ["cycle"] + [c for c in PROCESS_COLS if c in episode_frame.columns]
        leaked = FORBIDDEN_COLS & set(cols)
        if leaked:
            raise ValueError(f"label/identity columns must not reach a view: {leaked}")
        trunc = episode_frame.loc[episode_frame["cycle"] <= end, cols].sort_values("cycle").reset_index(drop=True).copy()
        if len(trunc) != end:
            raise ValueError(f"episode frame has {len(trunc)} cycles <= {end}, expected {end}")
        return cls(episode_id, window_index, start, end, trunc)

    @property
    def max_sample(self) -> int:
        return self.end_cycle

    def cycles(self, lo: int, hi: int) -> pd.DataFrame:
        """Samples lo..hi inclusive. hi > end_cycle is a future access and raises."""
        if hi > self.end_cycle:
            raise FutureAccessError(f"window {self.window_index}: cycle {hi} > max_sample {self.end_cycle}")
        if lo < 1:
            raise ValueError("cycles start at 1")
        return self._frame.iloc[lo - 1:hi].copy()

    def current(self) -> pd.DataFrame:
        return self.cycles(self.start_cycle, self.end_cycle)

    def expanded(self) -> pd.DataFrame | None:
        """Backward expansion t-1 ∪ t (40 samples). None at t = 0 (prereg §3.2)."""
        if self.window_index == 0:
            return None
        return self.cycles(self.start_cycle - P.STRIDE, self.end_cycle)

    def history(self) -> pd.DataFrame:
        """Everything observable so far (cycles 1..end_cycle) — for PSI/drift up to t."""
        return self.cycles(1, self.end_cycle)

    def __len__(self) -> int:
        return len(self._frame)
