"""core_pipeline/eval/b2.py — baseline B2 (L0 fixed-threshold Z-score) for the replay driver (prereg §4.1 / §4.5).

B2 reads raw window samples and the fixed calibration mean/std only: no gate, no detection state, no snapshot, no
Skill server. It therefore also runs on BLOCK windows and is untouched by E's query_more. The episode frame is built
exactly like the detect server's (_frame): engine key → processed parquet → WindowView; no lightgbm / torch import.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from uuid import UUID

import pandas as pd

from core_pipeline import prereg_constants as P
from core_pipeline.anomaly.zscore_l0 import ZScoreL0
from core_pipeline.data.tep_loader import decode_engine_id, load_processed
from core_pipeline.policy.decide_action import Decision, Signals, decide_action
from core_pipeline.skills.servers.common import engine_id_of
from core_pipeline.stream.window_view import PROCESS_COLS, WindowView


def b2_signals(alarm: bool) -> Signals:
    return Signals(gate=None, drift="ok", alarm=bool(alarm), s=0, g_AB=None, g_C=None, q_used=False, tau=None, retrieve_mode=None)


def decide_b2(alarm: bool) -> Decision:
    return decide_action(b2_signals(alarm), "B2")


class B2Runner:
    def __init__(self, dsn: str, processed_dir: str, zscore_dir: str | Path):
        self.dsn, self.processed_dir = dsn, processed_dir
        self.model = ZScoreL0.load(Path(zscore_dir))

    @lru_cache(maxsize=8)
    def _frame(self, episode_id: UUID) -> pd.DataFrame:
        eid = engine_id_of(self.dsn, episode_id)
        if eid is None:
            raise ValueError("DATA_MISSING: episode has no engine key")
        _, fault, _ = decode_engine_id(eid)
        df = load_processed(self.processed_dir, P.SPLIT, faulty=fault > 0, columns=["cycle"] + PROCESS_COLS, engine_ids=[eid])
        if len(df) < P.RUN_LENGTH:
            raise ValueError(f"DATA_MISSING: {len(df)} rows for episode (expected {P.RUN_LENGTH})")
        return df.reset_index(drop=True)

    def view(self, episode_id: UUID | str, t: int) -> WindowView:
        ep = UUID(str(episode_id))
        return WindowView.build(ep, t, self._frame(ep))

    def alarm(self, episode_id: UUID | str, t: int) -> bool:
        return bool(self.model.alarm(self.view(episode_id, t).current()))

    def max_abs_z(self, episode_id: UUID | str, t: int) -> float:
        return self.model.max_abs_z(self.view(episode_id, t).current())
