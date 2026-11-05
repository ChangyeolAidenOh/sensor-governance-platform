"""scripts/public_demo_prune.py: reduce a freshly loaded mart to the public demo pool (demo-public branch).

    python scripts/public_demo_prune.py --dsn <admin DSN of the LOCAL build database> [--synthetic-preset 11 --synthetic-fault 1]

Run AFTER scripts/verify_mart.py passed on the full load (check 3b compares group sizes with the frozen split and must
see the whole grid) and BEFORE scripts/precompute_demo.py. It keeps the frozen 10 demo episodes (staging.episodes.demo_preset
1..10) plus ONE extra demo_pool episode for the labelled synthetic approval case (same fault scenario as preset 1 by
default, a different run id), which it marks with demo_preset = --synthetic-preset. Everything else is deleted in
foreign-key order: mart.window_labels -> mart.windows -> staging.episodes -> staging.episode_keys. mart.load_log and
raw.tep_files stay (load provenance). Refuses to touch a database whose name contains "eval" or equals "sgp_v2".
"""
from __future__ import annotations

import argparse
import sys

import psycopg
from psycopg.conninfo import conninfo_to_dict

KEEP_PRESETS = tuple(range(1, 11))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--synthetic-preset", type=int, default=11)
    ap.add_argument("--synthetic-fault", type=int, default=1)
    ap.add_argument("--synthetic-run-id", type=int, default=None, help="pick this run id instead of the first spare one")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dbname = conninfo_to_dict(a.dsn).get("dbname") or ""
    if "eval" in dbname or dbname == "sgp_v2":
        print(f"refusing to prune database {dbname!r}", file=sys.stderr)
        return 2
    with psycopg.connect(a.dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM staging.episodes WHERE demo_preset = ANY(%s)", (list(KEEP_PRESETS),))
        n_demo = cur.fetchone()[0]
        if n_demo != len(KEEP_PRESETS):
            print(f"expected {len(KEEP_PRESETS)} demo episodes, found {n_demo}", file=sys.stderr)
            return 2
        cur.execute("SELECT count(*) FROM staging.episodes WHERE demo_preset=%s", (a.synthetic_preset,))
        if cur.fetchone()[0] == 0:
            cur.execute("""SELECT episode_id, run_id FROM staging.episodes
                           WHERE "group"='demo_pool' AND fault_scenario=%s AND demo_preset IS NULL AND (%s::int IS NULL OR run_id=%s)
                           ORDER BY run_id LIMIT 1""", (a.synthetic_fault, a.synthetic_run_id, a.synthetic_run_id))
            r = cur.fetchone()
            if not r:
                print(f"no spare demo_pool episode with fault_scenario={a.synthetic_fault}", file=sys.stderr)
                return 2
            print(f"synthetic preset {a.synthetic_preset}: episode {r[0]} (fault {a.synthetic_fault}, run_id {r[1]})")
            if not a.dry_run:
                cur.execute("UPDATE staging.episodes SET demo_preset=%s WHERE episode_id=%s", (a.synthetic_preset, r[0]))
        keep = list(KEEP_PRESETS) + [a.synthetic_preset]
        cur.execute("SELECT count(*) FROM staging.episodes WHERE demo_preset IS NULL OR NOT (demo_preset = ANY(%s))", (keep,))
        n_drop = cur.fetchone()[0]
        print(f"episodes to delete: {n_drop}; keeping presets {keep}")
        if a.dry_run:
            conn.rollback()
            return 0
        cur.execute("CREATE TEMP TABLE drop_eps ON COMMIT DROP AS SELECT episode_id FROM staging.episodes WHERE demo_preset IS NULL OR NOT (demo_preset = ANY(%s))", (keep,))
        for sql in ("DELETE FROM mart.window_labels WHERE episode_id IN (SELECT episode_id FROM drop_eps)",
                    "DELETE FROM mart.windows WHERE episode_id IN (SELECT episode_id FROM drop_eps)",
                    "DELETE FROM staging.episodes WHERE episode_id IN (SELECT episode_id FROM drop_eps)",
                    "DELETE FROM staging.episode_keys WHERE episode_id IN (SELECT episode_id FROM drop_eps)"):
            cur.execute(sql)
            print(f"{cur.rowcount:>7}  {sql.split(' WHERE')[0]}")
        conn.commit()
    with psycopg.connect(a.dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("VACUUM (ANALYZE) mart.windows")
        cur.execute("VACUUM (ANALYZE) mart.window_labels")
        cur.execute("SELECT count(*), count(DISTINCT episode_id) FROM mart.windows")
        w, e = cur.fetchone()
    print(f"remaining: {e} episodes, {w} windows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
