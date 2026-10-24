"""core_pipeline/eval/policy_events.py — the per-(eval run, policy, window, phase) action ledger (ST4-50 §2.3).

record_event is idempotent on the primary key: a re-insert with the same event_sha256 is a resume (no second cost
event), a different one is EVENT_MISMATCH. The ledger has no foreign keys; decision_id / decision_sha256 /
snapshot_sha256 are provenance only. Both the DB writer and an in-memory store implement the same two calls so the
driver can be exercised without a database.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Optional, Protocol
from uuid import UUID

import psycopg

from core_pipeline.policy.policy_input import sha256_json

PHASES = ("initial", "after_query")
POLICIES = ("E", "B1", "B2", "B3")


class EventMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class PolicyEvent:
    eval_run_id: str
    policy: str
    episode_id: str
    window_index: int
    phase: str
    detector_version: str
    contract_sha256: str
    alarm: bool
    action: str
    rule: int
    signals: dict
    evidence_stage: str                      # none | k5 | k10
    evidence: list
    snapshot_sha256: Optional[str] = None
    alarm_id: Optional[str] = None
    gate: Optional[str] = None
    reason: Optional[str] = None
    query_kind: Optional[str] = None
    evidence_k: Optional[int] = None
    evidence_request_id: Optional[str] = None
    decision_id: Optional[str] = None
    decision_sha256: Optional[str] = None

    def content(self) -> dict:
        """What identifies the event semantically (run-specific ids excluded)."""
        return {"policy": self.policy, "phase": self.phase, "episode_id": self.episode_id, "window_index": self.window_index,
                "alarm": self.alarm, "gate": self.gate, "action": self.action, "rule": self.rule, "query_kind": self.query_kind,
                "signals": self.signals, "evidence_stage": self.evidence_stage, "evidence": self.evidence, "snapshot_sha256": self.snapshot_sha256}

    @property
    def event_sha256(self) -> str:
        return sha256_json(self.content())

    @property
    def signals_sha256(self) -> str:
        return sha256_json(self.signals)

    def key(self) -> tuple:
        return (self.eval_run_id, self.policy, self.episode_id, int(self.window_index), self.phase)


class EventStore(Protocol):
    def record(self, ev: PolicyEvent) -> str: ...
    def load(self, eval_run_id: str, episode_id: str, t: int) -> dict[tuple[str, str], PolicyEvent]: ...


class MemoryEventStore:
    def __init__(self) -> None:
        self.rows: dict[tuple, PolicyEvent] = {}

    def record(self, ev: PolicyEvent) -> str:
        prior = self.rows.get(ev.key())
        if prior is not None:
            if prior.event_sha256 != ev.event_sha256:
                raise EventMismatch(f"EVENT_MISMATCH: {ev.key()}")
            return "resumed"
        self.rows[ev.key()] = ev
        return "recorded"

    def load(self, eval_run_id: str, episode_id: str, t: int) -> dict[tuple[str, str], PolicyEvent]:
        return {(e.policy, e.phase): e for e in self.rows.values() if e.eval_run_id == eval_run_id and e.episode_id == episode_id and e.window_index == t}


_COLS = ("eval_run_id, policy, episode_id, window_index, phase, detector_version, contract_sha256, snapshot_sha256, alarm, alarm_id, gate, action, rule, "
         "reason, query_kind, signals, signals_sha256, evidence, evidence_stage, evidence_k, evidence_request_id, decision_id, decision_sha256, event_sha256")


class DbEventStore:
    def __init__(self, dsn: str):
        self.dsn = dsn

    def record(self, ev: PolicyEvent) -> str:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(f"""INSERT INTO mart.policy_events ({_COLS})
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT (eval_run_id, policy, episode_id, window_index, phase) DO NOTHING RETURNING event_sha256""",
                        (UUID(ev.eval_run_id), ev.policy, UUID(ev.episode_id), ev.window_index, ev.phase, ev.detector_version, ev.contract_sha256,
                         ev.snapshot_sha256, ev.alarm, ev.alarm_id, ev.gate, ev.action, ev.rule, ev.reason, ev.query_kind,
                         json.dumps(ev.signals, ensure_ascii=False), ev.signals_sha256, json.dumps(ev.evidence, ensure_ascii=False), ev.evidence_stage,
                         ev.evidence_k, UUID(ev.evidence_request_id) if ev.evidence_request_id else None,
                         UUID(ev.decision_id) if ev.decision_id else None, ev.decision_sha256, ev.event_sha256))
            if cur.fetchone():
                conn.commit()
                return "recorded"
            cur.execute("""SELECT event_sha256 FROM mart.policy_events WHERE eval_run_id=%s AND policy=%s AND episode_id=%s AND window_index=%s AND phase=%s""",
                        (UUID(ev.eval_run_id), ev.policy, UUID(ev.episode_id), ev.window_index, ev.phase))
            stored = cur.fetchone()[0]
            conn.commit()
        if stored != ev.event_sha256:
            raise EventMismatch(f"EVENT_MISMATCH: {ev.key()} stored {stored[:12]} != {ev.event_sha256[:12]}")
        return "resumed"

    def load(self, eval_run_id: str, episode_id: str, t: int) -> dict[tuple[str, str], PolicyEvent]:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT {_COLS} FROM mart.policy_events WHERE eval_run_id=%s AND episode_id=%s AND window_index=%s",
                        (UUID(eval_run_id), UUID(episode_id), t))
            rows = cur.fetchall()
        out = {}
        for r in rows:
            (run, pol, ep, w, ph, det, con, ssha, alarm, aid, gate, action, rule, reason, qk, sig, _ssha, evd, stage, ek, erid, did, dsha, _esha) = r
            ev = PolicyEvent(str(run), pol, str(ep), int(w), ph, det, con, bool(alarm), action, int(rule), sig if isinstance(sig, dict) else json.loads(sig),
                             stage, evd if isinstance(evd, list) else json.loads(evd or "[]"), ssha, aid, gate, reason, qk, ek,
                             str(erid) if erid else None, str(did) if did else None, dsha)
            out[(pol, ph)] = ev
        return out

    def counts(self, eval_run_id: str) -> dict:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("""SELECT count(*) FILTER (WHERE phase='initial'),
                                  count(*) FILTER (WHERE phase='initial' AND policy='E' AND action='query_more'),
                                  count(*) FILTER (WHERE phase='after_query')
                           FROM mart.policy_events WHERE eval_run_id=%s""", (UUID(eval_run_id),))
            a, b, c = cur.fetchone()
        return {"initial": int(a), "e_query_more": int(b), "after_query": int(c)}

    def missing_after_query(self, eval_run_id: str) -> list[tuple[str, int]]:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("""SELECT i.episode_id, i.window_index FROM mart.policy_events i
                           LEFT JOIN mart.policy_events a ON a.eval_run_id=i.eval_run_id AND a.policy='E' AND a.episode_id=i.episode_id
                                AND a.window_index=i.window_index AND a.phase='after_query'
                           WHERE i.eval_run_id=%s AND i.policy='E' AND i.phase='initial' AND i.action='query_more' AND a.episode_id IS NULL
                           ORDER BY 1, 2""", (UUID(eval_run_id),))
            return [(str(e), int(w)) for e, w in cur.fetchall()]


def event_dict(ev: PolicyEvent) -> dict:
    return asdict(ev)
