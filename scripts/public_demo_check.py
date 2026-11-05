"""scripts/public_demo_check.py: self-check of a public demo database before (local) and after (hosted) the move (demo-public).

    python scripts/public_demo_check.py --dsn <admin DSN> --agent-dsn <sgp_agent DSN> [--manifest outputs/demo_presets_public.json]
        [--questions outputs/public_demo_questions.json] [--skill-dsn <sgp_skill DSN>]

Checks (exit 1 on any failure):
  1 manifest: schema, public flag, every synthetic preset is kind=synthetic, no truth fields
  2 every preset has 48 stored windows under the manifest's detector / contract
  3 alarm and synthetic presets: mart.alarms row, stored scoped k=5 retrieve, stored k=10 when the window's query is spent
  4 every preset: the evidence question's open-mode retrieve is stored (exact sentence, k=5, role engineer)
  5 no turn is left registered / running / waiting_approval (priming must close what it opened)
  6 admin DSN: staging.episodes holds exactly the manifest's episodes, each with its demo_preset
  7 agent DSN (and skill DSN if given): can read mart.v_windows_agent, cannot read staging.*, mart.window_labels, mart.load_log
  8 database size (informational)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import psycopg

SKILL = "skills_retrieve"
FENCED = ("staging.episodes", "staging.episode_keys", "mart.window_labels", "mart.load_log", "raw.tep_files")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True, help="admin DSN")
    ap.add_argument("--agent-dsn", default=None)
    ap.add_argument("--skill-dsn", default=None)
    ap.add_argument("--manifest", default="outputs/demo_presets_public.json")
    ap.add_argument("--questions", default="outputs/public_demo_questions.json")
    a = ap.parse_args()
    fails: list[str] = []

    def check(ok: bool, msg: str) -> None:
        print(("ok   " if ok else "FAIL ") + msg)
        if not ok:
            fails.append(msg)

    m = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    qs = json.loads(Path(a.questions).read_text(encoding="utf-8"))["questions"]
    evidence_q = next((q["text"].strip() for q in qs if q["id"] == "evidence"), None)
    syn = set(m.get("synthetic_presets") or [])
    check(m.get("schema") == "demo_presets/1" and m.get("public") is True, "1a manifest schema demo_presets/1 and public=true")
    check(all((int(e["preset"]) in syn) == (e.get("kind") == "synthetic") for e in m["presets"]), "1b synthetic presets are exactly the kind=synthetic entries")
    check(not any(k in e for e in m["presets"] for k in ("fault_scenario", "run_id", "reveal")), "1c no truth fields in the manifest")
    check(bool(evidence_q), "1d evidence question present in the question set")

    with psycopg.connect(a.dsn) as conn, conn.cursor() as cur:
        for e in m["presets"]:
            p, ep, v = int(e["preset"]), e["episode_id"], e["versions"]
            det, sha = v.get("detector"), v.get("contract_sha256")
            cur.execute("SELECT count(*) FROM mart.episode_windows WHERE episode_id=%s AND detector_version=%s AND contract_sha256=%s", (ep, det, sha))
            check(cur.fetchone()[0] == 48, f"2 preset {p}: 48 stored windows")
            if e.get("kind") in ("alarm", "synthetic"):
                aid = e["alarm_id"]
                cur.execute("SELECT count(*) FROM mart.alarms WHERE alarm_id=%s", (aid,))
                check(cur.fetchone()[0] == 1, f"3a preset {p}: alarm row {aid[:12]}")
                cur.execute("""SELECT count(*) FROM mart.agent_steps WHERE skill=%s AND status='OK' AND input->'args'->>'alarm_id'=%s
                               AND input->'args'->>'mode'='scoped' AND (input->'args'->>'k')::int=5 AND input->'ctx'->>'caller_role'='engineer'
                               AND output->'trace'->'versions'->>'contract_sha256'=%s""", (SKILL, aid, sha))
                check(cur.fetchone()[0] >= 1, f"3b preset {p}: stored scoped k=5 retrieve")
                cur.execute("SELECT q_used FROM mart.episode_windows WHERE episode_id=%s AND window_index=%s AND detector_version=%s AND contract_sha256=%s",
                            (ep, int(e["display_window"]), det, sha))
                r = cur.fetchone()
                if r and r[0]:
                    cur.execute("""SELECT count(*) FROM mart.agent_steps WHERE skill=%s AND status='OK' AND input->'args'->>'alarm_id'=%s
                                   AND input->'args'->>'mode'='scoped' AND (input->'args'->>'k')::int=10 AND input->'ctx'->>'caller_role'='engineer'
                                   AND output->'trace'->'versions'->>'contract_sha256'=%s""", (SKILL, aid, sha))
                    check(cur.fetchone()[0] >= 1, f"3c preset {p}: query spent and stored k=10 retrieve")
                else:
                    print(f"info preset {p}: query not spent (no k=10 needed)")
            if evidence_q:
                aid = e.get("alarm_id") if e.get("kind") in ("alarm", "synthetic") else None
                cur.execute("""SELECT count(*) FROM mart.agent_steps WHERE skill=%s AND status='OK' AND input->'args'->>'mode'='open'
                               AND input->'args'->>'query'=%s AND (input->'args'->>'k')::int=5 AND input->'ctx'->>'caller_role'='engineer'
                               AND coalesce(input->'args'->>'alarm_id','')=%s AND output->'trace'->'versions'->>'contract_sha256'=%s""",
                            (SKILL, evidence_q, aid or "", sha))
                check(cur.fetchone()[0] >= 1, f"4 preset {p}: stored open-mode retrieve for the evidence question")
        cur.execute("SELECT count(*) FROM mart.graph_turns WHERE status IN ('registered','running','waiting_approval')")
        check(cur.fetchone()[0] == 0, "5 no open turn left")
        try:
            cur.execute("SELECT episode_id::text, demo_preset FROM staging.episodes ORDER BY demo_preset")
            rows = cur.fetchall()
            want = {str(e["episode_id"]): int(e["preset"]) for e in m["presets"]}
            check({r[0]: r[1] for r in rows} == want, f"6 staging.episodes = manifest episodes ({len(rows)} rows)")
        except psycopg.errors.InsufficientPrivilege:
            conn.rollback()
            print("skip 6 (DSN is not admin)")
        cur.execute("SELECT pg_size_pretty(pg_database_size(current_database())), current_database()")
        sz, db = cur.fetchone()
        print(f"8 database {db}: {sz}")

    for label, dsn in (("agent", a.agent_dsn), ("skill", a.skill_dsn)):
        if not dsn:
            continue
        with psycopg.connect(dsn, autocommit=True) as c2, c2.cursor() as cur:
            cur.execute("SELECT current_user")
            user = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM mart.v_windows_agent")
            check(cur.fetchone()[0] > 0, f"7a {label} ({user}) can read mart.v_windows_agent")
            for tbl in FENCED:
                try:
                    cur.execute(f"SELECT 1 FROM {tbl} LIMIT 1")
                    check(False, f"7b {label} ({user}) CAN read {tbl}")
                except psycopg.errors.InsufficientPrivilege:
                    check(True, f"7b {label} ({user}) cannot read {tbl}")
    print(f"{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
