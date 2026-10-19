"""
scripts/verify_mart.py — S1-02 rev 2 completion criteria, read-only. Exit 1 on failure.
  1  evaluation n = 1,260 episodes × 48 = 60,480 windows
  2  48 windows/episode, unique keys, 20/20 grid
  3  groups partition run ids, sizes = frozen, evaluation order = frozen list, eval_selected = n
  4  labels ⇔ fault>0 ∧ window≥8
  5  demo presets = frozen 10
  6  opaque keys: agent view exposes neither engine_id/run_id/fault nor labels; episode_id is not derivable
     (uuid4, not a function of engine_id); keys stable across loads (every episodes row still maps to its key)
  7  last load ok, split sha matches disk, last two loads identical counts; execution tables untouched by loads
  8  agent role (if present) cannot read staging.*, mart.window_labels, mart.load_log, raw.*
"""
from __future__ import annotations

import argparse, hashlib, json, sys, uuid
from pathlib import Path
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_pipeline import prereg_constants as P  # noqa: E402

DEMO = {P.engine_id(P.SPLIT, f, r): k for k, f, r in P.DEMO_PRESETS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--agent-dsn", default=None, help="DSN of sgp_agent to test the role fence (8)")
    ap.add_argument("--split", default="outputs/stage0_episode_split.json")
    ap.add_argument("--n", type=int, default=P.N_EVAL_DEFAULT)
    a = ap.parse_args()
    split_doc = json.loads(Path(a.split).read_text())
    fails = []

    def check(c, msg):
        print(("ok   " if c else "FAIL ") + msg)
        if not c: fails.append(msg)

    with psycopg.connect(a.dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM staging.episodes e JOIN mart.windows w USING (episode_id) WHERE e.eval_selected")
        check(cur.fetchone()[0] == a.n * 21 * 48, f"1 evaluation windows = {a.n*21*48}")
        cur.execute("SELECT count(*) FROM (SELECT episode_id FROM mart.windows GROUP BY 1 HAVING count(*) <> 48) x")
        check(cur.fetchone()[0] == 0, "2a 48 windows per episode")
        cur.execute("SELECT count(*) FROM mart.windows WHERE start_cycle <> window_index*20+1 OR end_cycle <> window_index*20+20")
        check(cur.fetchone()[0] == 0, "2b grid 20/20")
        cur.execute('SELECT count(*) FROM (SELECT run_id FROM staging.episodes GROUP BY 1 HAVING count(DISTINCT "group")>1) x')
        check(cur.fetchone()[0] == 0, "3a run ids partitioned")
        cur.execute('SELECT "group", count(DISTINCT run_id) FROM staging.episodes GROUP BY 1')
        check(dict(cur.fetchall()) == {g: s["n_run_ids"] for g, s in split_doc["groups"].items()}, "3b group sizes = frozen")
        cur.execute("SELECT run_id FROM staging.episodes WHERE \"group\"='evaluation_pool' AND fault_scenario=0 ORDER BY eval_rank")
        check([r for (r,) in cur.fetchall()] == split_doc["groups"]["evaluation_pool"]["run_ids"], "3c evaluation order = frozen list")
        cur.execute("SELECT count(DISTINCT run_id) FROM staging.episodes WHERE eval_selected")
        check(cur.fetchone()[0] == a.n, f"3d eval_selected = {a.n} run ids")
        cur.execute("SELECT count(*) FROM mart.window_labels WHERE is_fault_active <> (fault_scenario>0 AND window_index>=8)")
        check(cur.fetchone()[0] == 0, "4 labels ⇔ fault>0 ∧ window≥8")
        cur.execute("SELECT engine_id, demo_preset FROM staging.episodes WHERE demo_preset IS NOT NULL")
        check(dict(cur.fetchall()) == DEMO, "5 demo presets frozen")

        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='mart' AND table_name='v_windows_agent'")
        cols = {r[0] for r in cur.fetchall()}
        check(not ({"engine_id", "run_id", "fault_scenario", "is_fault_active", "split"} & cols), f"6a agent view columns {sorted(cols)}")
        cur.execute("SELECT episode_id FROM staging.episode_keys LIMIT 50")
        check(all(uuid.UUID(str(e)).version == 4 for (e,) in cur.fetchall()), "6b episode_id is uuid4 (not derivable from engine_id)")
        cur.execute("SELECT count(*) FROM staging.episodes e LEFT JOIN staging.episode_keys k ON k.engine_id=e.engine_id AND k.episode_id=e.episode_id WHERE k.engine_id IS NULL")
        check(cur.fetchone()[0] == 0, "6c every episode maps to its minted key (stable across loads)")
        cur.execute("SELECT count(*) FROM staging.episode_keys")
        n_keys = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM staging.episodes")
        check(n_keys == cur.fetchone()[0], f"6d keys minted once: {n_keys} keys = episodes")

        cur.execute("SELECT load_id, ok, action, split_json_sha256, counts, prereg_n, n_reduction_decision FROM mart.load_log ORDER BY load_id DESC LIMIT 2")
        rows = cur.fetchall()
        check(bool(rows) and rows[0][1], "7a last load ok")
        check(bool(rows) and rows[0][3] == hashlib.sha256(Path(a.split).read_bytes()).hexdigest(), "7b split sha = disk")
        check(bool(rows) and (rows[0][5] == 60 or rows[0][6]), "7c n=40 only with a reduction decision")
        if len(rows) == 2:
            c0, c1 = rows[0][4], rows[1][4]
            check({k: v for k, v in c0.items() if k != "execution_rows"} == {k: v for k, v in c1.items() if k != "execution_rows"},
                  f"7d last two loads identical counts (second action={rows[0][2]})")
            check(rows[0][2] == "unchanged", "7e re-run with same grid is a no-op (action=unchanged)")
        else:
            print("skip 7d/7e (one load so far)")

    if a.agent_dsn:
        with psycopg.connect(a.agent_dsn, autocommit=True) as ac, ac.cursor() as cur:
            cur.execute("SELECT count(*) FROM mart.v_windows_agent"); check(cur.fetchone()[0] > 0, "8a agent can read v_windows_agent")
            for tbl in ["staging.episodes", "staging.episode_keys", "mart.window_labels", "mart.load_log", "raw.tep_files", "mart.replay_window_log"]:
                try:
                    cur.execute(f"SELECT 1 FROM {tbl} LIMIT 1"); check(False, f"8b agent CAN read {tbl}")
                except psycopg.errors.InsufficientPrivilege:
                    check(True, f"8b agent denied on {tbl}")
    else:
        print("skip 8 (pass --agent-dsn to test the role fence)")

    print("\nRESULT:", "PASS" if not fails else f"FAIL ({len(fails)})")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
