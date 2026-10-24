"""core_pipeline/eval/eval_runs.py — evaluation-run registry and the run-space execution right (ST4-50 §2.4).

Serial execution on a dedicated test / evaluation database is enforced, not assumed:
  * at most one active run (status running | incomplete) per database — partial UNIQUE index in sql/010; a
    concurrent start loses with UniqueViolation → RunActive;
  * start / resume / finish / reset all run under the same run-space advisory lock (RUN_SPACE_KEY) in one transaction;
  * a driver holds a session-level advisory lock (DRIVER_KEY) for its lifetime, so two drivers of the same run cannot
    run at once (RunBusy);
  * a new run requires the planned episodes to carry NO Stage 4a execution rows (dirty state is refused, never reused).
Only test or evaluation databases are accepted (db name containing 'test' or 'eval'); sgp_v2 is refused here —
applying schema v8 to the project DB stays a separate decision.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse
from uuid import UUID

import psycopg

from core_pipeline.policy.policy_input import sha256_json

RUN_SPACE_KEY = 4_200_000_001
DRIVER_KEY = 4_200_000_002
N_WINDOWS = 48
N_POLICIES = 4


class RunActive(RuntimeError):
    pass


class RunBusy(RuntimeError):
    pass


class RunStateError(RuntimeError):
    pass


class DsnNotAllowed(RuntimeError):
    pass


def db_name(dsn: str) -> str:
    return (urlparse(dsn).path or "").lstrip("/").split("?")[0] if "://" in dsn else dsn


def assert_replay_dsn(dsn: str) -> None:
    name = db_name(dsn).lower()
    if "test" not in name and "eval" not in name:
        raise DsnNotAllowed(f"replay runs only on a test or evaluation database, got {name!r}")


@dataclass(frozen=True)
class EvalRun:
    eval_run_id: str
    planned_episodes: list[str]
    detector_version: str
    contract_sha256: str
    expected_k5_versions: dict
    caller_role: str
    status: str
    code_commit: str

    @property
    def expected_initial_rows(self) -> int:
        return len(self.planned_episodes) * N_WINDOWS * N_POLICIES


def _row_to_run(row) -> EvalRun:
    rid, planned, det, con, ver, role, status, commit = row
    planned = planned if isinstance(planned, list) else json.loads(planned)
    ver = ver if isinstance(ver, dict) else json.loads(ver)
    return EvalRun(str(rid), [str(e) for e in planned], det, con, ver, role, status, commit)


def stage4a_row_counts(cur, episodes: list[UUID]) -> dict:
    cur.execute("SELECT count(*) FROM mart.episode_windows WHERE episode_id = ANY(%s)", (episodes,))
    w = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM mart.gate_checks WHERE episode_id = ANY(%s)", (episodes,))
    g = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM mart.policy_input_snapshots WHERE episode_id = ANY(%s)", (episodes,))
    s = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM mart.agent_steps WHERE input->'args'->>'episode_id' = ANY(%s::text[])", ([str(e) for e in episodes],))
    a = cur.fetchone()[0]
    return {"episode_windows": int(w), "gate_checks": int(g), "policy_input_snapshots": int(s), "agent_steps": int(a)}


def start_run(dsn: str, planned_episodes: list[UUID | str], code_commit: str, detector_version: str, contract_sha256: str,
              expected_k5_versions: dict, caller_role: str = "engineer", seed: Optional[int] = None, prereg_tag: Optional[str] = None,
              n_run_ids: Optional[int] = None, eval_run_id: Optional[UUID] = None) -> EvalRun:
    assert_replay_dsn(dsn)
    eps = [UUID(str(e)) for e in planned_episodes]
    rid = eval_run_id or uuid.uuid4()
    with psycopg.connect(dsn) as conn:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (RUN_SPACE_KEY,))
            cur.execute("SELECT eval_run_id, status FROM mart.eval_runs WHERE status IN ('running','incomplete')")
            active = cur.fetchone()
            if active:
                raise RunActive(f"run {active[0]} is {active[1]}; archive or resume it first")
            counts = stage4a_row_counts(cur, eps)
            dirty = {k: v for k, v in counts.items() if v}
            if dirty:
                raise RunStateError(f"planned episodes carry Stage 4a execution rows {dirty}; reset the evaluation database first")
            planned = [str(e) for e in eps]
            try:
                cur.execute("""INSERT INTO mart.eval_runs (eval_run_id, code_commit, prereg_tag, seed, planned_episodes, planned_sha256, n_run_ids,
                                 detector_version, contract_sha256, expected_k5_versions, caller_role, status)
                               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'running')""",
                            (rid, code_commit, prereg_tag, seed, json.dumps(planned), sha256_json(planned), n_run_ids, detector_version, contract_sha256,
                             json.dumps(expected_k5_versions, default=str), caller_role))
            except psycopg.errors.UniqueViolation as e:
                raise RunActive("another run became active concurrently") from e
    return get_run(dsn, rid)


def get_run(dsn: str, eval_run_id: UUID | str) -> EvalRun:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("""SELECT eval_run_id, planned_episodes, detector_version, contract_sha256, expected_k5_versions, caller_role, status, code_commit
                       FROM mart.eval_runs WHERE eval_run_id=%s""", (UUID(str(eval_run_id)),))
        row = cur.fetchone()
    if not row:
        raise RunStateError(f"unknown eval run {eval_run_id}")
    return _row_to_run(row)


def resume_run(dsn: str, eval_run_id: UUID | str, code_commit: str) -> EvalRun:
    assert_replay_dsn(dsn)
    with psycopg.connect(dsn) as conn:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (RUN_SPACE_KEY,))
            cur.execute("SELECT status, code_commit FROM mart.eval_runs WHERE eval_run_id=%s FOR UPDATE", (UUID(str(eval_run_id)),))
            row = cur.fetchone()
            if not row:
                raise RunStateError(f"unknown eval run {eval_run_id}")
            status, commit = row
            if status not in ("running", "incomplete"):
                raise RunStateError(f"run {eval_run_id} is {status}: not resumable")
            if commit != code_commit:
                raise RunStateError(f"run {eval_run_id} was started on commit {commit}, now {code_commit}: not the same evaluation")
            cur.execute("UPDATE mart.eval_runs SET status='running', updated_at=now() WHERE eval_run_id=%s", (UUID(str(eval_run_id)),))
    return get_run(dsn, eval_run_id)


def set_status(dsn: str, eval_run_id: UUID | str, status: str, timing: Optional[dict] = None) -> None:
    with psycopg.connect(dsn) as conn:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (RUN_SPACE_KEY,))
            cur.execute("UPDATE mart.eval_runs SET status=%s, timing=coalesce(%s::jsonb, timing), updated_at=now() WHERE eval_run_id=%s",
                        (status, json.dumps(timing) if timing is not None else None, UUID(str(eval_run_id))))
            if cur.rowcount != 1:
                raise RunStateError(f"unknown eval run {eval_run_id}")


def completeness(dsn: str, run: EvalRun) -> dict:
    """Complete ⇔ every planned (episode, window, policy) has its initial row AND every E initial query_more has its
    after_query row. A window that never ran is incomplete, exactly like a missing after_query."""
    from core_pipeline.eval.policy_events import DbEventStore
    store = DbEventStore(dsn)
    c = store.counts(run.eval_run_id)
    missing_after = store.missing_after_query(run.eval_run_id)
    missing_initial = run.expected_initial_rows - c["initial"]
    return {"expected_initial": run.expected_initial_rows, "initial": c["initial"], "missing_initial": missing_initial,
            "e_query_more": c["e_query_more"], "after_query": c["after_query"], "missing_after_query": missing_after,
            "complete": missing_initial == 0 and not missing_after}


def finish_run(dsn: str, run: EvalRun, timing: Optional[dict] = None) -> dict:
    report = completeness(dsn, run)
    set_status(dsn, run.eval_run_id, "complete" if report["complete"] else "incomplete", timing)
    return report


class DriverLease:
    """Session-level advisory lock held for the driver's lifetime (released when the connection closes)."""

    def __init__(self, dsn: str):
        self.dsn = dsn
        self.conn: Optional[psycopg.Connection] = None

    def __enter__(self) -> "DriverLease":
        self.conn = psycopg.connect(self.dsn)
        with self.conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (DRIVER_KEY,))
            ok = cur.fetchone()[0]
        self.conn.commit()
        if not ok:
            self.conn.close()
            raise RunBusy("another replay driver holds the execution right on this database")
        return self

    def __exit__(self, *exc) -> None:
        if self.conn is not None:
            try:
                with self.conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_unlock(%s)", (DRIVER_KEY,))
                self.conn.commit()
            finally:
                self.conn.close()
