"""core_pipeline/agent/plans.py — graph-side DB records that are NOT Skill state (ST4-18, ST4-19).

query_plans: when the graph spends a window's single query it records what it planned (expand / rerank / both) and
later what completed or failed, so the query state is derived from records, never guessed from NULLs. A plan missing
for a spent query (crash between spend and plan write) is recovered from the ORIGINAL spend request's thread/turn and
that turn's initial decision — never from a later decision.
history: the read-only SQL template of the `history` category (contract §3.7), bounded by the display window.
"""
from __future__ import annotations

import json
from typing import Optional
from uuid import UUID

import psycopg

from core_pipeline.agent.keys import decision_key
from core_pipeline.policy.decide_action import Signals, decide_action


def plan_write(dsn: str, query_id: str, episode_id: str, window_index: int, detector_version: str, contract_sha256: str,
               thread_id: str, turn_id: str, initial_decision_id: Optional[str], kind: str) -> bool:
    """Idempotent (query_id PK). Returns True if this call inserted the plan."""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO mart.query_plans (query_id, episode_id, window_index, detector_version, contract_sha256, thread_id, turn_id,
                                                     initial_decision_id, kind, planned_expand, planned_rerank)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (query_id) DO NOTHING""",
                    (UUID(query_id), UUID(episode_id), int(window_index), detector_version, contract_sha256, thread_id, UUID(turn_id),
                     (UUID(initial_decision_id) if initial_decision_id else None), kind, kind in ("expand", "both"), kind in ("rerank", "both")))
        inserted = cur.rowcount == 1
        conn.commit()
    return inserted


def plan_mark(dsn: str, query_id: str, op: str, status: str, reason: Optional[str]) -> None:
    assert op in ("expand", "rerank") and status in ("done", "failed", "skipped")
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(f"UPDATE mart.query_plans SET {op}_status=%s, {op}_reason=%s, updated_at=now() WHERE query_id=%s AND {op}_status IS NULL",
                    (status, reason, UUID(query_id)))
        conn.commit()


def plan_get(dsn: str, query_id: str) -> Optional[dict]:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("""SELECT thread_id, turn_id, initial_decision_id, kind, planned_expand, planned_rerank, expand_status, expand_reason,
                              rerank_status, rerank_reason FROM mart.query_plans WHERE query_id=%s""", (UUID(query_id),))
        r = cur.fetchone()
    if not r:
        return None
    return {"thread_id": r[0], "turn_id": str(r[1]), "initial_decision_id": (str(r[2]) if r[2] else None), "kind": r[3], "planned_expand": r[4],
            "planned_rerank": r[5], "expand_status": r[6], "expand_reason": r[7], "rerank_status": r[8], "rerank_reason": r[9]}


def plan_status(plan: Optional[dict], q_used: bool) -> str:
    """unused | in_progress | complete | partial | unknown (spent outside the graph or plan not yet recoverable)."""
    if not q_used:
        return "unused"
    if plan is None:
        return "unknown"
    pending, failed = [], []
    for op in ("expand", "rerank"):
        if plan[f"planned_{op}"]:
            s = plan[f"{op}_status"]
            if s is None:
                pending.append(op)
            elif s == "failed":
                failed.append(op)
    if pending:
        return "in_progress"
    return "partial" if failed else "complete"


def spend_origin(dsn: str, query_request_id: str) -> Optional[dict]:
    """The spend_query request that owns a window's query: its thread_id and run_uuid (bound ctx in agent_steps)."""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT input->'ctx'->>'thread_id', input->'ctx'->>'run_uuid', skill, status FROM mart.agent_steps WHERE request_id=%s",
                    (UUID(query_request_id),))
        r = cur.fetchone()
    return {"thread_id": r[0], "run_uuid": r[1], "skill": r[2], "status": r[3]} if r else None


def recover_plan(dsn: str, query_request_id: str, episode_id: str, window_index: int, detector_version: str, contract_sha256: str) -> Optional[dict]:
    """A query was spent but no plan exists: rebuild it from the ORIGINAL turn (graph_turns.run_uuid of the spend ctx)
    and that turn's initial decision (decision_key thread:turn:initial). Returns the plan or None if not recoverable."""
    origin = spend_origin(dsn, query_request_id)
    if not origin or not origin.get("thread_id") or not origin.get("run_uuid"):
        return None
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT turn_id FROM mart.graph_turns WHERE thread_id=%s AND run_uuid=%s", (origin["thread_id"], UUID(origin["run_uuid"])))
        r = cur.fetchone()
        if not r:
            return None
        turn_id = str(r[0])
        cur.execute("SELECT decision_id, signals FROM mart.policy_decisions WHERE decision_key=%s", (decision_key(origin["thread_id"], turn_id, "initial"),))
        d = cur.fetchone()
    if not d:
        return None
    signals = d[1] if isinstance(d[1], dict) else json.loads(d[1])
    dec = decide_action(Signals(**signals), "E")
    if dec.action != "query_more" or not dec.query_kind:
        return None
    plan_write(dsn, query_request_id, episode_id, window_index, detector_version, contract_sha256, origin["thread_id"], turn_id, str(d[0]), dec.query_kind)
    return plan_get(dsn, query_request_id)


def history(dsn: str, episode_id: str, display_window: int, detector_version: Optional[str], contract_sha256: Optional[str], alarm_id: Optional[str]) -> dict:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cond, args = ["episode_id=%s", "window_index<=%s"], [UUID(episode_id), int(display_window)]
        if detector_version:
            cond.append("detector_version=%s"); args.append(detector_version)
        if contract_sha256:
            cond.append("contract_sha256=%s"); args.append(contract_sha256)
        cur.execute(f"SELECT window_index, gate, alarm FROM mart.episode_windows WHERE {' AND '.join(cond)} ORDER BY window_index", args)
        rows = cur.fetchall()
        decisions = []
        if alarm_id:
            cur.execute("SELECT policy, action, rule, decided_at FROM mart.policy_decisions WHERE alarm_id=%s ORDER BY seq", (alarm_id,))
            decisions = [{"policy": p, "action": a, "rule": r, "at": str(t)} for p, a, r, t in cur.fetchall()]
    return {"n_windows": len(rows), "alarm_windows": [w for w, g, a in rows if a is True], "n_block": sum(1 for _, g, _ in rows if g == "BLOCK"),
            "decisions": decisions}
