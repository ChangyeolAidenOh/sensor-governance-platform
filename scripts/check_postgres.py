"""
scripts/check_postgres.py — Stage 1 (S1-01) completion check for the v2 Postgres (compose service `postgres`, port 5436).

Phase 1 (fresh):      python scripts/check_postgres.py --dsn $DSN
                      -> connects, runs PostgresSaver.setup() (checkpointer tables initialised; this is NOT a
                         resume test — that stays in Stage 4), writes a random marker, prints it.
Phase 2 (after `docker compose restart postgres`):
                      python scripts/check_postgres.py --dsn $DSN --expect <marker>
                      -> the marker must still be there (named volume sgp_pg_data persisted).
"""
from __future__ import annotations

import argparse
import secrets
import sys

import psycopg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--expect", default=None, help="marker printed by phase 1")
    a = ap.parse_args()

    with psycopg.connect(a.dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SELECT version(), current_database()")
        ver, db = cur.fetchone()
        print("connected:", db, "|", ver.split(",")[0])
        if db != "sgp_v2":
            print("FAIL database is not sgp_v2"); return 1
        cur.execute("CREATE SCHEMA IF NOT EXISTS mart")
        cur.execute("CREATE TABLE IF NOT EXISTS mart.s1_marker (token text PRIMARY KEY, created_at timestamptz DEFAULT now())")

        if a.expect:
            cur.execute("SELECT created_at FROM mart.s1_marker WHERE token=%s", (a.expect,))
            row = cur.fetchone()
            print("ok   marker persisted across restart" if row else "FAIL marker missing after restart (volume not persisted?)")
            return 0 if row else 1

        token = secrets.token_hex(8)
        cur.execute("INSERT INTO mart.s1_marker (token) VALUES (%s)", (token,))
        print("marker written:", token)

    # checkpointer initialisation only (ST0-17 already verified interrupt/resume with these versions)
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        with PostgresSaver.from_conn_string(a.dsn) as saver:
            saver.setup()
        with psycopg.connect(a.dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'checkpoint%' ORDER BY 1")
            print("ok   PostgresSaver.setup():", [r[0] for r in cur.fetchall()])
    except ImportError:
        print("skip checkpointer (langgraph-checkpoint-postgres not importable in this env)")

    print(f"\nnext: docker compose restart postgres && python scripts/check_postgres.py --dsn $DSN --expect {token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
