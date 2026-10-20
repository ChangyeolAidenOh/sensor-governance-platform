"""
core_pipeline/skills/state_store.py — S4a-01 rev 2: OWNERSHIP of episode detection state (ST4-02), safe under
concurrent requests and partial failure (reviewer 2026-10-21 point 1):
  * every observe() runs in ONE transaction holding a per-(episode, detector, contract) advisory lock, reads the
    committed prefix from mart.episode_windows, recomputes prefix-BH from those rows plus the new p, inserts, commits —
    and RETURNS THE COMMITTED ROW. No in-memory prefix: a failed INSERT advances nothing; two server instances
    serialise on the lock and the loser sees the winner's row (idempotent return), never its own computation.
  * windows must be observed in order; q_used is set once; a query is spent together with a request id so the SAME
    request may retry the expansion, a second logical query is refused.
  * expansion writes expanded_set only ('{}' = ran and empty; NULL = never ran). Detection fields are never updated.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

import psycopg

from core_pipeline import prereg_constants as P
from core_pipeline.anomaly.prefix_bh import prefix_bh_last


@dataclass(frozen=True)
class WindowRecord:
    episode_id: UUID
    window_index: int
    gate: str
    drift: str
    drift_assessed: bool
    p_value: Optional[float]
    p_adj: Optional[float]
    m_t: int
    alarm: Optional[bool]
    pred_set: list[int]
    set_size: int
    top1: Optional[int]
    q_used: bool
    query_request_id: Optional[UUID]
    expanded_set: Optional[list[int]]      # None = not expanded; [] = expanded, empty
    reused: bool


_COLS = "gate, drift, drift_assessed, p_value, p_adj, m_t, alarm, pred_set, set_size, top1, q_used, query_request_id, expanded_set"


def _row_to_record(episode_id: UUID, t: int, row, reused: bool) -> WindowRecord:
    gate, drift, da, p, padj, m_t, alarm, ps, ss, top1, q, qr, es = row
    return WindowRecord(episode_id, t, gate, drift, da, p, padj, m_t, alarm, list(ps or []), ss or 0, top1, q, qr,
                        None if es is None else list(es), reused)          # [] stays [] (reviewer point 2)


class DetectionStateStore:
    def __init__(self, dsn: str, detector_version: str, contract_sha256: str, alpha: float = P.ALPHA):
        self.dsn, self.detector_version, self.contract, self.alpha = dsn, detector_version, contract_sha256, alpha

    def _lock_key(self, episode_id: UUID) -> int:
        h = hashlib.sha256(f"{episode_id}|{self.detector_version}|{self.contract}".encode()).digest()
        return int.from_bytes(h[:8], "big", signed=True)

    def _key(self):
        return (self.detector_version, self.contract)

    def get(self, episode_id: UUID, t: int) -> Optional[WindowRecord]:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT {_COLS} FROM mart.episode_windows WHERE episode_id=%s AND window_index=%s AND detector_version=%s AND contract_sha256=%s",
                        (episode_id, t, *self._key()))
            row = cur.fetchone()
        return _row_to_record(episode_id, t, row, True) if row else None

    def tested_prefix(self, cur, episode_id: UUID) -> list[tuple[int, str, Optional[float], Optional[bool]]]:
        cur.execute("""SELECT window_index, gate, p_value, alarm FROM mart.episode_windows
                       WHERE episode_id=%s AND detector_version=%s AND contract_sha256=%s ORDER BY window_index""",
                    (episode_id, *self._key()))
        return cur.fetchall()

    def next_expected_window(self, episode_id: UUID) -> int:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            rows = self.tested_prefix(cur, episode_id)
        return (rows[-1][0] + 1) if rows else 0

    def observe(self, episode_id: UUID, t: int, gate: str, drift: str, drift_assessed: bool, p_value: Optional[float],
                pred_set: list[int], top1: Optional[int]) -> WindowRecord:
        if gate == "BLOCK" and p_value is not None:
            raise ValueError("BLOCK window must not carry a p-value")
        if gate != "BLOCK" and p_value is None:
            raise ValueError("non-BLOCK window needs a p-value")
        with psycopg.connect(self.dsn) as conn:
            with conn.transaction(), conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_xact_lock(%s)", (self._lock_key(episode_id),))   # serialise per episode
                cur.execute(f"SELECT {_COLS} FROM mart.episode_windows WHERE episode_id=%s AND window_index=%s AND detector_version=%s AND contract_sha256=%s",
                            (episode_id, t, *self._key()))
                row = cur.fetchone()
                if row:
                    return _row_to_record(episode_id, t, row, True)          # someone (maybe another instance) already observed it
                rows = self.tested_prefix(cur, episode_id)
                expected = (rows[-1][0] + 1) if rows else 0
                if t != expected:
                    raise ValueError(f"window {t} out of order for episode {episode_id}: next expected {expected}")
                if gate == "BLOCK":
                    p_adj, alarm, m_t = None, None, sum(1 for _, g, _, _ in rows if g != "BLOCK")
                else:
                    prefix = [p for _, g, p, _ in rows if g != "BLOCK"] + [float(p_value)]
                    alarm, p_adj, m_t = prefix_bh_last(prefix, self.alpha)      # prefix-BH from COMMITTED rows + this p
                cur.execute("""INSERT INTO mart.episode_windows (episode_id, window_index, detector_version, contract_sha256, gate, drift,
                                 drift_assessed, p_value, p_adj, m_t, alarm, pred_set, set_size, top1, q_used)
                               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,false)""",
                            (episode_id, t, *self._key(), gate, drift, drift_assessed, p_value, p_adj, m_t, alarm, list(pred_set), len(pred_set), top1))
            # committed → return what the DB holds
        rec = self.get(episode_id, t)
        return WindowRecord(**{**rec.__dict__, "reused": False})

    def spend_query(self, episode_id: UUID, t: int, request_id: UUID) -> bool:
        """Spend the window's single query. Returns True if this request now owns it (first call or a retry of the
        same request); False if a DIFFERENT request already spent it."""
        with psycopg.connect(self.dsn) as conn:
            with conn.transaction(), conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_xact_lock(%s)", (self._lock_key(episode_id),))
                cur.execute("""UPDATE mart.episode_windows SET q_used=true, query_request_id=%s
                               WHERE episode_id=%s AND window_index=%s AND detector_version=%s AND contract_sha256=%s
                                 AND (q_used=false OR query_request_id=%s) RETURNING query_request_id""",
                            (request_id, episode_id, t, *self._key(), request_id))
                return cur.fetchone() is not None

    def set_expanded(self, episode_id: UUID, t: int, request_id: UUID, expanded_set: list[int]) -> bool:
        """Write the expansion result once, only for the request that spent the query. Detection fields untouched."""
        with psycopg.connect(self.dsn) as conn:
            with conn.transaction(), conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_xact_lock(%s)", (self._lock_key(episode_id),))
                cur.execute("""UPDATE mart.episode_windows SET expanded_set=%s
                               WHERE episode_id=%s AND window_index=%s AND detector_version=%s AND contract_sha256=%s
                                 AND q_used=true AND query_request_id=%s AND expanded_set IS NULL""",
                            (list(expanded_set), episode_id, t, *self._key(), request_id))
                return cur.rowcount == 1

    def non_rejected_windows(self, episode_id: UUID, before: int) -> list[int]:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            rows = self.tested_prefix(cur, episode_id)
        return [w for w, g, _, a in rows if w < before and g != "BLOCK" and a is False]
