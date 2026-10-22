"""demo/ledger_view.py — read-only SQL over the LLM ledger rows for the demo (ST4-28). Never writes, never locks.

Cost is reported in four buckets that are NOT summed together: settled (status <> PENDING and usage known),
pending (reservations at their estimate), usage_missing (settled at the reserved amount, cost not confirmed) and
exceeded_unresolved (count only). Per-turn rows are selected by thread_id + the stage keys of that turn
(`llm:router:<turn_id>` and `llm:narrate:<turn_id>:…`; a resume shares its parent's turn_id), never by time.
"""
from __future__ import annotations

import json

import psycopg

from core_pipeline.agent.budget import LLM_LEDGER_SKILL, budget_usd
from core_pipeline.agent.llm_ledger import SQL_BLOCKED

_USAGE_MISSING = "coalesce((output->>'usage_missing')::boolean, false)"
_EXCEEDED_OPEN = "(coalesce((output->>'reservation_exceeded')::boolean, false) AND NOT coalesce((output->>'exceeded_resolved')::boolean, false))"


def cost_buckets(dsn: str) -> dict:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(f"""SELECT
                          count(*),
                          coalesce(sum(cost_usd) FILTER (WHERE status <> 'PENDING' AND NOT {_USAGE_MISSING}), 0),
                          count(*)               FILTER (WHERE status <> 'PENDING' AND NOT {_USAGE_MISSING}),
                          coalesce(sum(cost_usd) FILTER (WHERE status = 'PENDING'), 0),
                          count(*)               FILTER (WHERE status = 'PENDING'),
                          coalesce(sum(cost_usd) FILTER (WHERE status <> 'PENDING' AND {_USAGE_MISSING}), 0),
                          count(*)               FILTER (WHERE status <> 'PENDING' AND {_USAGE_MISSING}),
                          count(*)               FILTER (WHERE status <> 'PENDING' AND {_EXCEEDED_OPEN})
                        FROM mart.agent_steps WHERE skill = %s""", (LLM_LEDGER_SKILL,))
        n, s_usd, s_n, p_usd, p_n, u_usd, u_n, x_n = cur.fetchone()
        cur.execute(SQL_BLOCKED, (LLM_LEDGER_SKILL,))
        blocked = cur.fetchone()[0]
    return {"rows": n, "settled_usd": float(s_usd), "settled_rows": s_n, "pending_usd": float(p_usd), "pending_rows": p_n,
            "usage_missing_usd": float(u_usd), "usage_missing_rows": u_n, "exceeded_unresolved_rows": x_n,
            "blocked_unresolved": blocked, "cap_usd": budget_usd()}


def _row(r) -> dict:
    inp = r[7] if isinstance(r[7], dict) else json.loads(r[7])
    out = (r[8] if isinstance(r[8], dict) else json.loads(r[8])) if r[8] else {}
    return {"request_id": str(r[0]), "tool": inp.get("tool"), "stage_key": inp.get("stage_key"), "attempt": inp.get("attempt"), "status": r[1],
            "reason_code": r[2], "reserved_usd": (inp.get("reserve") or {}).get("reserved_usd"), "cost_usd": float(r[3] or 0),
            "input_tokens": r[4], "output_tokens": r[5], "violation": r[6], "usage_missing": out.get("usage_missing"),
            "reservation_exceeded": out.get("reservation_exceeded"), "exceeded_resolved": out.get("exceeded_resolved"),
            "fallback_reason": out.get("fallback_reason"), "no_response": out.get("no_response"), "created_at": str(r[9])}


_SELECT = """SELECT request_id, status, reason_code, cost_usd, input_tokens, output_tokens, violation, input, output, created_at
             FROM mart.agent_steps WHERE skill = %s AND input->'ctx'->>'thread_id' = %s"""


def rows_for_thread(dsn: str, thread_id: str) -> list[dict]:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(_SELECT + " ORDER BY created_at", (LLM_LEDGER_SKILL, thread_id))
        return [_row(r) for r in cur.fetchall()]


def rows_for_turn(dsn: str, thread_id: str, turn_id: str) -> list[dict]:
    """Rows of one logical turn: router key equals, narrate key starts with the turn id (a modify round narrates under the
    parent turn id, so one parent turn can own several narrate rows)."""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(_SELECT + " AND (input->>'stage_key' = %s OR input->>'stage_key' LIKE %s) ORDER BY created_at",
                    (LLM_LEDGER_SKILL, thread_id, f"llm:router:{turn_id}", f"llm:narrate:{turn_id}:%"))
        return [_row(r) for r in cur.fetchall()]
