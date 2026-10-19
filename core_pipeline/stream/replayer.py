"""
core_pipeline/stream/replayer.py — Stage 1 S1-03. Replays the pre-registered evaluation episodes window by window.

Guarantees (prereg-v1 rev 3; reviewer 2026-10-19):
  * order = frozen list: evaluation_pool.run_ids[:n] in that order, fault 0..20 within each run id. No shuffle, no
    seed-driven reordering. The sha256 of the sequence actually replayed is recorded and must equal the sha256 derived
    from the frozen split file (`expected_order_sha`).
  * the policy receives (episode_id, WindowView, EpisodeState, WindowState) only — never a full frame, engine_id or label.
  * EpisodeState/WindowState lifetimes are separated; evaluator state lives in core_pipeline/eval.
  * BLOCK windows: no detect call, no p-value, m_t unchanged (EpisodeState.observe enforces).
  * time contract: wall seconds after the first 20 run ids are reported (prereg §3.1).

Sources: `DbParquetSource` (staging.episodes/episode_keys + processed parquet; loader-role DSN) for real runs,
         `ListSource` (in-memory frames) for tests. Both yield (episode_id, engine_id_hidden, frame) in frozen order;
         engine_id never leaves the source.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional
from uuid import UUID

import pandas as pd

from core_pipeline import prereg_constants as P
from core_pipeline.stream.state import EpisodeState, WindowState
from core_pipeline.stream.window_view import WindowView

# ---- order -----------------------------------------------------------------------------------------------------

def frozen_order(split_doc: dict, n: int) -> list[tuple[int, int]]:
    """[(run_id, fault)] in pre-registered replay order."""
    runs = split_doc["groups"]["evaluation_pool"]["run_ids"][:n]
    return [(r, f) for r in runs for f in range(0, P.N_FAULTS + 1)]


def order_sha(order: Iterable[tuple[int, int]]) -> str:
    h = hashlib.sha256()
    for r, f in order:
        h.update(f"{r}:{f}\n".encode())
    return h.hexdigest()


def expected_order_sha(split_path: Path, n: int) -> str:
    return order_sha(frozen_order(json.loads(Path(split_path).read_text()), n))


# ---- sources ---------------------------------------------------------------------------------------------------

@dataclass
class Episode:
    episode_id: UUID
    run_id: int                # kept inside the source/replayer for order verification only
    fault: int                 # NEVER passed to the policy
    frame: pd.DataFrame        # full episode; sliced into WindowViews


class ListSource:
    def __init__(self, episodes: list[Episode]):
        self._eps = episodes

    def __iter__(self) -> Iterator[Episode]:
        return iter(self._eps)


class DbParquetSource:
    """Reads the frozen order from staging (loader role) and each episode's rows from the processed parquet.
    group='evaluation' → eval_selected in (eval_rank, fault) order (the pre-registered replay);
    group='dev'        → dev run ids ascending × fault 0..20 (smoke/debug only; never reported)."""

    def __init__(self, dsn: str, processed_dir: str, group: str = "evaluation"):
        self.dsn, self.processed_dir, self.group = dsn, Path(processed_dir), group

    def expected_sha(self) -> str:
        import psycopg
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(self._sql())
            return order_sha((r, f) for _, _, r, f in cur.fetchall())

    def _sql(self) -> str:
        if self.group == "evaluation":
            return ("SELECT episode_id, engine_id, run_id, fault_scenario FROM staging.episodes "
                    "WHERE eval_selected ORDER BY eval_rank, fault_scenario")
        if self.group == "dev":
            return ("SELECT episode_id, engine_id, run_id, fault_scenario FROM staging.episodes "
                    "WHERE \"group\"='dev' ORDER BY run_id, fault_scenario")
        raise ValueError(self.group)

    def __iter__(self) -> Iterator[Episode]:
        import psycopg
        from core_pipeline.data.tep_loader import load_processed  # v1 contract loader
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(self._sql())
            rows = cur.fetchall()
        for episode_id, engine_id, run_id, fault in rows:
            df = load_processed(str(self.processed_dir), P.SPLIT, faulty=fault > 0,
                                columns=["cycle"] + WindowViewCols, engine_ids=[engine_id])
            yield Episode(episode_id, run_id, fault, df)


WindowViewCols = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]


# ---- replayer --------------------------------------------------------------------------------------------------

@dataclass
class ReplayRecord:
    episode_id: UUID
    window_index: int
    gate: str
    p_value: Optional[float]
    p_adj: Optional[float]
    m_t: int
    alarm: Optional[bool]
    action: str
    q_used: bool


PolicyFn = Callable[[WindowView, EpisodeState, WindowState], tuple[str, str, Optional[float]]]
"""policy(view, ep_state, win_state) -> (action, gate, p_value). The policy obtains gate/p via skills; in Stage 1
tests these are dummies. The replayer records the decision; it does not score it."""


class Replayer:
    def __init__(self, source: Iterable[Episode], n: int, expected_sha: str, alpha: float = P.ALPHA,
                 on_record: Optional[Callable[[ReplayRecord], None]] = None):
        self.source, self.n, self.expected_sha, self.alpha, self.on_record = source, n, expected_sha, alpha, on_record
        self.seen_order: list[tuple[int, int]] = []
        self.wall_first20: Optional[float] = None

    def run(self, policy: PolicyFn) -> dict:
        t0 = time.perf_counter()
        run_ids_done: list[int] = []
        for ep in self.source:
            self.seen_order.append((ep.run_id, ep.fault))
            if not run_ids_done or run_ids_done[-1] != ep.run_id:
                run_ids_done.append(ep.run_id)
                if len(run_ids_done) == 21 and self.wall_first20 is None:
                    self.wall_first20 = time.perf_counter() - t0          # prereg §3.1 time contract
            self._replay_episode(ep, policy)
        if len(run_ids_done) >= 20 and self.wall_first20 is None:
            self.wall_first20 = time.perf_counter() - t0
        sha = order_sha(self.seen_order)
        if sha != self.expected_sha:
            raise RuntimeError(f"replay order sha {sha[:12]} != frozen {self.expected_sha[:12]} — order was altered")
        return {"episodes": len(self.seen_order), "run_ids": len(run_ids_done), "order_sha256": sha,
                "wall_seconds_first20": self.wall_first20, "wall_seconds_total": time.perf_counter() - t0}

    def _replay_episode(self, ep: Episode, policy: PolicyFn) -> None:
        state = EpisodeState(ep.episode_id, self.alpha)
        for t in range(P.N_WINDOWS):
            view = WindowView.build(ep.episode_id, t, ep.frame)      # truncated copy: cycles 1..end_cycle only
            wstate = WindowState(t)                                    # q_used resets every window
            action, gate, p = policy(view, state, wstate)
            dec = state.observe(t, gate, p)                            # BLOCK → no p, m_t unchanged
            if action == "propose":
                state.record_proposal(t)
            if self.on_record:
                self.on_record(ReplayRecord(ep.episode_id, t, gate, p, dec.p_adj if dec else None, state.m_t,
                                            dec.alarm if dec else None, action, wstate.q_used))
