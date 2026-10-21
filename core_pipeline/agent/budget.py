"""core_pipeline/agent/budget.py — LLM spend guard (ST4-06, ST4-19).

SGP_LLM_BUDGET_USD unset → every paid call is refused. The ledger is mart.agent_steps rows with skill='llm' (the table
already has input_tokens / output_tokens / cost_usd); the guard compares the cumulative SUM(cost_usd) plus the expected
cost of the next call with the cap. S4a-04 makes no LLM call at all; S4a-05 adds reserve-before-call + settle-after
(reviewer 2026-10-21: SUM alone does not bound concurrent callers).
"""
from __future__ import annotations

import os
from typing import Optional

import psycopg

LLM_LEDGER_SKILL = "llm"


class BudgetExceeded(RuntimeError):
    pass


def budget_usd() -> Optional[float]:
    raw = os.environ.get("SGP_LLM_BUDGET_USD")
    if raw is None or not raw.strip():
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v >= 0 else None


def spent_usd(dsn: str) -> float:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT coalesce(sum(cost_usd), 0) FROM mart.agent_steps WHERE skill=%s", (LLM_LEDGER_SKILL,))
        return float(cur.fetchone()[0])


def assert_llm_allowed(dsn: Optional[str], expected_usd: float = 0.0) -> float:
    """Returns the remaining budget after the expected call, or raises BudgetExceeded."""
    cap = budget_usd()
    if cap is None:
        raise BudgetExceeded("SGP_LLM_BUDGET_USD is not set: paid LLM calls are refused (ST4-06)")
    spent = spent_usd(dsn) if dsn else 0.0
    if spent + expected_usd > cap:
        raise BudgetExceeded(f"LLM budget exceeded: spent {spent:.4f} + expected {expected_usd:.4f} > cap {cap:.4f} USD")
    return cap - spent - expected_usd
