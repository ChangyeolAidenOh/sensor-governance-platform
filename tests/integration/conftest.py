"""tests/integration/conftest.py — reapply the role grant files on the TEST DB at session start and end (ST4-24).

tests/integration/test_state_and_cmms.py drops and recreates mart tables (policy_decisions, proposal_drafts,
episode_windows), which drops their GRANTs; the demo runs as sgp_agent on the same test DB (synthetic mode), so the
grants are restored here. Superuser DSN = SGP_TEST_PG_DSN; skipped silently when it is unset or not a test database.
GRANT is additive, so reapplying is idempotent. Role files are not in replay_client.SQL_FILES by convention.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROLE_FILES = ("sql/006_roles_stage4a.sql", "sql/008_roles_llm_ledger.sql", "sql/009_roles_checkpointer.sql")


def _reapply_role_grants() -> None:
    dsn = os.environ.get("SGP_TEST_PG_DSN")
    if not dsn or "test" not in dsn:
        return
    import psycopg
    root = Path(__file__).resolve().parents[2]
    with psycopg.connect(dsn) as conn:
        for f in ROLE_FILES:
            p = root / f
            if not p.exists():
                continue
            with conn.cursor() as cur:
                cur.execute(p.read_text(encoding="utf-8"))
            conn.commit()


@pytest.fixture(scope="session", autouse=True)
def _role_grants_on_test_db():
    _reapply_role_grants()
    yield
    _reapply_role_grants()
