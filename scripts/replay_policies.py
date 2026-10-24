"""scripts/replay_policies.py — Stage 4b replay driver CLI (ST4-50 S4b-02). Test / evaluation databases only; $0, no LLM.

  python scripts/replay_policies.py start   --dsn-env SGP_TEST_PG_DSN --presets 8 [--t-max 47] [--commit <sha>]
  python scripts/replay_policies.py resume  --dsn-env SGP_TEST_PG_DSN --run <uuid> [--commit <sha>]
  python scripts/replay_policies.py status  --dsn-env SGP_TEST_PG_DSN --run <uuid>
  python scripts/replay_policies.py archive --dsn-env SGP_TEST_PG_DSN --run <uuid>
  python scripts/replay_policies.py reset   --dsn-env SGP_TEST_PG_DSN --presets 8      (clean_episode of the listed presets; refused while a run is active)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg

from core_pipeline.eval.archive import archive_run
from core_pipeline.eval.eval_runs import (RUN_SPACE_KEY, DriverLease, RunStateError, assert_replay_dsn, completeness, finish_run, get_run,
                                          resume_run, set_status, start_run)
from core_pipeline.eval.replay_driver import ReplayDriver, RunInfo, assert_fastmcp_update_check_off, assert_no_llm_env, make_db_deps
from core_pipeline.skills.client import SyncSkillClient
from core_pipeline.skills.replay_client import clean_episode, episode_for_preset


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def meta():
    m = json.loads(Path("outputs/model_meta/detector_v1.json").read_text())
    tau = json.loads(Path(os.environ.get("SGP_TAU_PATH", "models/retrieval_v1/tau.json")).read_text())
    expected = {"contract_sha256": m["input_contract_sha256"], "detector": m["version"], "tau": tau["tau"],
                "fault_map_sha256": tau.get("fault_map_sha256"), "k_primary": 5}
    return m["version"], m["input_contract_sha256"], expected


def episodes_of(dsn: str, presets: list[int]) -> list[str]:
    return [str(episode_for_preset(dsn, p)) for p in presets]


def drive(dsn: str, run, t_max: int) -> dict:
    detector, contract, expected = meta()
    info = RunInfo(run.eval_run_id, detector, contract, run.caller_role)
    timing = {"episodes": []}
    with DriverLease(dsn), SyncSkillClient(env={"SGP_PG_DSN": dsn}) as skills:
        deps = make_db_deps(dsn, skills, info, os.environ.get("SGP_PROCESSED_DIR", "data/processed/TEP"),
                            os.environ.get("SGP_DETECTOR_DIR", "models/detector_v1"), expected_k5_versions=run.expected_k5_versions)
        driver = ReplayDriver(deps, info)
        try:
            for ep in run.planned_episodes:
                t0 = time.perf_counter()
                rows = driver.run_episode(ep, t_max)
                dt = time.perf_counter() - t0
                timing["episodes"].append({"episode_id": ep, "seconds": round(dt, 1)})
                acts = {p: sum(1 for r in rows if r.get(p) == "propose") for p in ("B1", "B2", "B3")}
                acts["E"] = sum(1 for r in rows if r.get("E_after", r.get("E_initial")) == "propose")
                print(f"episode {ep[:8]} windows {len(rows)} propose {acts} {dt:.1f}s")
        except BaseException as e:
            set_status(dsn, run.eval_run_id, "incomplete", timing)
            print(f"stopped: {type(e).__name__}: {e}")
            raise
    report = finish_run(dsn, run, timing)
    print(json.dumps(report, ensure_ascii=False))
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["start", "resume", "status", "archive", "reset"])
    ap.add_argument("--dsn-env", default="SGP_TEST_PG_DSN")
    ap.add_argument("--presets", type=int, nargs="*", default=[])
    ap.add_argument("--run")
    ap.add_argument("--t-max", type=int, default=47)
    ap.add_argument("--commit")
    a = ap.parse_args()
    dsn = os.environ.get(a.dsn_env)
    if not dsn:
        print(f"{a.dsn_env} is not set")
        return 2
    assert_replay_dsn(dsn)
    commit = a.commit or git_commit()
    if a.cmd == "start":
        assert_no_llm_env()
        assert_fastmcp_update_check_off()
        if not a.presets:
            print("--presets required")
            return 2
        detector, contract, expected = meta()
        run = start_run(dsn, episodes_of(dsn, a.presets), commit, detector, contract, expected, prereg_tag="prereg-v1")
        print(f"eval_run {run.eval_run_id} started on {len(run.planned_episodes)} episodes")
        drive(dsn, run, a.t_max)
        return 0
    if a.cmd == "resume":
        assert_no_llm_env()
        assert_fastmcp_update_check_off()
        run = resume_run(dsn, a.run, commit)
        print(f"eval_run {run.eval_run_id} resumed")
        drive(dsn, run, a.t_max)
        return 0
    if a.cmd == "status":
        run = get_run(dsn, a.run)
        print(json.dumps({"status": run.status, **completeness(dsn, run)}, ensure_ascii=False))
        return 0
    if a.cmd == "archive":
        print(json.dumps(archive_run(dsn, a.run), ensure_ascii=False))
        return 0
    if a.cmd == "reset":
        with psycopg.connect(dsn) as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (RUN_SPACE_KEY,))
            cur.execute("SELECT eval_run_id, status FROM mart.eval_runs WHERE status IN ('running','incomplete')")
            row = cur.fetchone()
            if row:
                raise RunStateError(f"run {row[0]} is {row[1]}: archive it before resetting episodes")
            for ep in episodes_of(dsn, a.presets):
                clean_episode(dsn, ep)
                print(f"cleaned {ep[:8]}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
