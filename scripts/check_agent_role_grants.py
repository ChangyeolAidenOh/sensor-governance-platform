"""Verify the privileges the Streamlit/graph role needs for the LLM ledger (ST4-21): read-only check, no GRANT is applied.

    python scripts/check_agent_role_grants.py [--role sgp_agent] [--dsn ...]   # exit 1 if anything is missing

Needs a DSN that can run has_table_privilege for another role (the project superuser DSN from SGP_PG_DSN is fine).
"""
from __future__ import annotations

import argparse
import os
import sys

import psycopg
from dotenv import load_dotenv

REQUIRED = [("table", "mart.agent_runs", "INSERT"), ("table", "mart.agent_runs", "SELECT"),
            ("table", "mart.agent_steps", "SELECT"), ("table", "mart.agent_steps", "INSERT"),
            ("sequence", "mart.agent_steps_step_id_seq", "USAGE")]
# UPDATE is granted per column (006 + 008); the ledger writes exactly these
UPDATE_COLUMNS = ("status", "reason_code", "output", "latency_ms", "violation", "input_tokens", "output_tokens", "cost_usd")


def main() -> int:
    load_dotenv(".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.environ.get("SGP_PG_DSN"))
    ap.add_argument("--role", default="sgp_agent")
    a = ap.parse_args()
    if not a.dsn:
        print("SGP_PG_DSN is required", file=sys.stderr)
        return 2
    missing = []
    with psycopg.connect(a.dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (a.role,))
        if cur.fetchone() is None:
            print(f"role {a.role} does not exist")
            return 1
        for kind, obj, priv in REQUIRED:
            fn = "has_table_privilege" if kind == "table" else "has_sequence_privilege"
            cur.execute(f"SELECT {fn}(%s, %s, %s)", (a.role, obj, priv))
            ok = bool(cur.fetchone()[0])
            print(f"{'ok     ' if ok else 'MISSING'} {priv:<7} {obj}")
            if not ok:
                missing.append((kind, obj, priv))
        for col in UPDATE_COLUMNS:
            cur.execute("SELECT has_column_privilege(%s, %s, %s, 'UPDATE')", (a.role, "mart.agent_steps", col))
            ok = bool(cur.fetchone()[0])
            print(f"{'ok     ' if ok else 'MISSING'} UPDATE  mart.agent_steps.{col}")
            if not ok:
                missing.append(("column", f"mart.agent_steps ({col})", "UPDATE"))
    if missing:
        print("\nsuggested grants (review sql/006_roles_stage4a.sql before applying):")
        for kind, obj, priv in missing:
            print(f"  GRANT {priv} ON {'SEQUENCE ' if kind == 'sequence' else ''}{obj} TO {a.role};" if kind != "column"
                  else f"  GRANT UPDATE {obj.split(' ', 1)[1]} ON {obj.split(' ', 1)[0]} TO {a.role};")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
