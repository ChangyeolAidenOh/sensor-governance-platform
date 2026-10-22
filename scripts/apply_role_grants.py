"""Reapply the role grant files to one DB in order (ST4-24). Superuser DSN. GRANT is additive, so rerunning is idempotent.

    python scripts/apply_role_grants.py --dsn "$SGP_TEST_PG_DSN"
    python scripts/apply_role_grants.py --dsn "$SGP_PG_DSN" --files sql/009_roles_checkpointer.sql

Default files: sql/006_roles_stage4a.sql, sql/008_roles_llm_ledger.sql, sql/009_roles_checkpointer.sql (002 is not included: it
creates the role; run it by hand on a fresh DB). Needed whenever a destructive integration run (ST4-09) recreated mart tables,
which drops their grants (seen on sgp_test 2026-10-22). Each file runs in its own transaction; a failing file stops the run.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

DEFAULT_FILES = ("sql/006_roles_stage4a.sql", "sql/008_roles_llm_ledger.sql", "sql/009_roles_checkpointer.sql")


def main() -> int:
    load_dotenv(".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.environ.get("SGP_PG_DSN"))
    ap.add_argument("--files", nargs="*", default=list(DEFAULT_FILES))
    a = ap.parse_args()
    a.dsn = a.dsn or os.environ.get("SGP_PG_DSN")               # an empty "--dsn $VAR" falls back to .env
    if not a.dsn:
        print("a superuser DSN is required (--dsn or SGP_PG_DSN)", file=sys.stderr)
        return 2
    with psycopg.connect(a.dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), current_user, rolsuper FROM pg_roles WHERE rolname=current_user")
            db, user, superuser = cur.fetchone()
        if not superuser:
            print(f"{user} on {db} is not a superuser; refusing", file=sys.stderr)
            return 3
        for f in a.files:
            p = Path(f)
            if not p.exists():
                print(f"missing: {f}", file=sys.stderr)
                return 4
            with conn.cursor() as cur:
                cur.execute(p.read_text(encoding="utf-8"))
            conn.commit()
            print(f"applied {f} on {db}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
