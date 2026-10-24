"""tests/integration/test_eval_runs_db.py — ST4-50 criteria 5 / 7 on the TEST DB: run-space execution right (one active run,
concurrent start → exactly one wins, incomplete blocks a new run but allows resume), dirty-state refusal, ledger
idempotency / EVENT_MISMATCH, completeness = planned × 48 × 4 initial rows, driver lease, DSN guard. No Skill servers.
Skipped without SGP_TEST_PG_DSN; skipped if another (foreign) run is active on that DB."""
from __future__ import annotations

import os
import threading
import uuid

import psycopg
import pytest

from core_pipeline.eval.archive import archive_run
from core_pipeline.eval.eval_runs import (DriverLease, DsnNotAllowed, RunActive, RunBusy, RunStateError, assert_replay_dsn, completeness,
                                          finish_run, get_run, resume_run, set_status, start_run)
from core_pipeline.eval.policy_events import DbEventStore, EventMismatch, PolicyEvent
from core_pipeline.skills.replay_client import clean_episode, episode_for_preset
from core_pipeline.skills.servers.common import is_test_dsn

DSN = os.environ.get("SGP_TEST_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="SGP_TEST_PG_DSN not set")
COMMIT = "test-s4b02"
DET, CON = "detector_v1", "c" * 64
EXPECTED = {"tau": 0.5, "k_primary": 5}


def _cleanup():
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM mart.policy_events WHERE eval_run_id IN (SELECT eval_run_id FROM mart.eval_runs WHERE code_commit=%s)", (COMMIT,))
        cur.execute("DELETE FROM mart.eval_runs WHERE code_commit=%s", (COMMIT,))
        conn.commit()


@pytest.fixture(scope="module")
def ep():
    assert is_test_dsn(DSN)
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT eval_run_id, status FROM mart.eval_runs WHERE status IN ('running','incomplete') AND code_commit<>%s", (COMMIT,))
        foreign = cur.fetchone()
    if foreign:
        pytest.skip(f"foreign active run {foreign[0]} ({foreign[1]}) on the test DB")
    _cleanup()
    e = episode_for_preset(DSN, 9)
    clean_episode(DSN, e)
    yield str(e)
    _cleanup()
    clean_episode(DSN, e)


def _start(ep, **kw):
    return start_run(DSN, [ep], COMMIT, DET, CON, EXPECTED, **kw)


def _event(run_id, ep, policy, phase, t, action, **over):
    base = dict(eval_run_id=run_id, policy=policy, episode_id=ep, window_index=t, phase=phase, detector_version=DET, contract_sha256=CON,
                alarm=action in ("propose", "hold", "query_more"), action=action, rule=4 if action == "propose" else 2, signals={"gate": "PASS"},
                evidence_stage="none", evidence=[], query_kind="rerank" if action == "query_more" else None)
    base.update(over)
    return PolicyEvent(**base)


def test_dsn_guard():
    with pytest.raises(DsnNotAllowed):
        assert_replay_dsn("postgresql://u:p@localhost:5436/sgp_v2")
    assert_replay_dsn(DSN)


def test_start_refuses_dirty_episode(ep):
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO mart.episode_windows (episode_id, window_index, detector_version, contract_sha256, gate, drift, drift_assessed,
                         p_value, p_adj, m_t, alarm, pred_set, set_size, top1, q_used) VALUES (%s,0,%s,%s,'PASS','ok',true,0.5,0.5,1,false,'{}',0,NULL,false)""",
                    (ep, DET, CON))
        conn.commit()
    try:
        with pytest.raises(RunStateError):
            _start(ep)
    finally:
        clean_episode(DSN, uuid.UUID(ep))
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM mart.eval_runs WHERE code_commit=%s", (COMMIT,))
        assert cur.fetchone()[0] == 0


def test_concurrent_start_exactly_one_wins(ep):
    results = []
    barrier = threading.Barrier(2)

    def go():
        barrier.wait()
        try:
            results.append(("ok", _start(ep).eval_run_id))
        except RunActive as e:
            results.append(("active", str(e)))
    ts = [threading.Thread(target=go) for _ in range(2)]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert sorted(r[0] for r in results) == ["active", "ok"]
    rid = next(r[1] for r in results if r[0] == "ok")
    set_status(DSN, rid, "incomplete")
    archive_run(DSN, rid)
    assert get_run(DSN, rid).status == "archived"


def test_incomplete_blocks_new_run_but_allows_resume(ep):
    run = _start(ep)
    set_status(DSN, run.eval_run_id, "incomplete")
    with pytest.raises(RunActive):
        _start(ep)
    with pytest.raises(RunStateError):
        resume_run(DSN, run.eval_run_id, "another-commit")
    assert resume_run(DSN, run.eval_run_id, COMMIT).status == "running"
    with pytest.raises(RunStateError):
        archive_run(DSN, run.eval_run_id)                               # running runs are not archivable
    set_status(DSN, run.eval_run_id, "incomplete")
    archive_run(DSN, run.eval_run_id)
    with pytest.raises(RunStateError):
        resume_run(DSN, run.eval_run_id, COMMIT)                        # archived: not resumable


def test_ledger_idempotent_mismatch_and_completeness(ep):
    run = _start(ep)
    store = DbEventStore(DSN)
    e1 = _event(run.eval_run_id, ep, "B1", "initial", 0, "propose")
    e2 = _event(run.eval_run_id, ep, "E", "initial", 0, "query_more")
    assert store.record(e1) == "recorded" and store.record(e2) == "recorded"
    assert store.record(e1) == "resumed"
    with pytest.raises(EventMismatch):
        store.record(PolicyEvent(**{**e1.__dict__, "action": "hold", "rule": 6}))
    loaded = store.load(run.eval_run_id, ep, 0)
    assert set(loaded) == {("B1", "initial"), ("E", "initial")} and loaded[("E", "initial")].query_kind == "rerank"
    rep = completeness(DSN, run)
    assert rep["expected_initial"] == 48 * 4 and rep["initial"] == 2 and rep["missing_initial"] == 48 * 4 - 2
    assert rep["missing_after_query"] == [(ep, 0)] and rep["complete"] is False
    assert finish_run(DSN, run)["complete"] is False and get_run(DSN, run.eval_run_id).status == "incomplete"
    with pytest.raises(psycopg.errors.CheckViolation):
        store.record(_event(run.eval_run_id, ep, "B1", "after_query", 0, "propose"))    # after_query is E-only (schema CHECK)
    out = archive_run(DSN, run.eval_run_id)
    assert out["events"] == 2 and get_run(DSN, run.eval_run_id).status == "archived"
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM mart.policy_events WHERE eval_run_id=%s", (uuid.UUID(run.eval_run_id),))
        assert cur.fetchone()[0] == 2                                   # archive exports, never deletes


def test_driver_lease_is_exclusive(ep):
    with DriverLease(DSN):
        with pytest.raises(RunBusy):
            with DriverLease(DSN):
                pass
    with DriverLease(DSN):
        pass
