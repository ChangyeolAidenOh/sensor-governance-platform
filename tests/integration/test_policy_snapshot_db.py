"""tests/integration/test_policy_snapshot_db.py — ST4-50 criteria 3 / 3' / 4 / 5(part) on the TEST DB with SYNTHETIC rows
(no Skill servers, no models): episode_windows / alarms / agent_steps rows are inserted directly for preset 9's episode,
then freeze_policy_input / load_policy_input are exercised, including the real advisory-lock contention with the
detect server's DetectionStateStore.spend_query. Everything is cleaned up with clean_episode. Skipped without
SGP_TEST_PG_DSN (and refuses a non-test DSN)."""
from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from uuid import UUID

import psycopg
import pytest

from core_pipeline.policy.initial_snapshot import SnapshotError, alarm_id_of, freeze_policy_input, load_policy_input, lock_key
from core_pipeline.skills.replay_client import clean_episode, episode_for_preset
from core_pipeline.skills.servers.common import is_test_dsn
from core_pipeline.skills.state_store import DetectionStateStore

DSN = os.environ.get("SGP_TEST_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="SGP_TEST_PG_DSN not set")
ROLE = "engineer"
VERSIONS = {"embedder": "bge-m3", "reranker": "bge-reranker-v2-m3", "tau": 0.5, "fault_map_sha256": "f" * 64, "k_primary": 5, "k_query_more": 10}


def _meta():
    m = json.loads(Path("outputs/model_meta/detector_v1.json").read_text())
    return m["version"], m["input_contract_sha256"]


def _insert_window(cur, ep, t, detector, contract, alarm, pred_set, gate="PASS", q_used=False):
    cur.execute("""INSERT INTO mart.episode_windows (episode_id, window_index, detector_version, contract_sha256, gate, drift, drift_assessed,
                     p_value, p_adj, m_t, alarm, pred_set, set_size, top1, q_used)
                   VALUES (%s,%s,%s,%s,%s,'ok',true,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (ep, t, detector, contract, gate, None if gate == "BLOCK" else 0.01, None if gate == "BLOCK" else 0.02, t + 1,
                 None if gate == "BLOCK" else alarm, list(pred_set), len(pred_set), pred_set[0] if pred_set else None, q_used))
    if alarm and gate != "BLOCK":
        cur.execute("""INSERT INTO mart.alarms (alarm_id, episode_id, window_index, gate, drift, p_value, p_adj, m_t, alarm, detector)
                       VALUES (%s,%s,%s,%s,'ok',0.01,0.02,%s,true,%s) ON CONFLICT DO NOTHING""",
                    (alarm_id_of(ep, t, detector), ep, t, gate, t + 1, detector))


def _insert_k5(cur, aid, detector, contract, fault_set, run_uuid, status="OK"):
    rid = uuid.uuid4()
    bound = {"ctx": {"request_id": str(rid), "thread_id": "snapshot-test", "run_uuid": str(run_uuid), "caller_role": ROLE, "mode": "replay"},
             "skill": "skills_retrieve", "tool": "retrieve", "contract_version": "1.0",
             "args": {"query": "q", "mode": "scoped", "fault_set": list(fault_set), "k": 5, "alarm_id": aid, "query_id": None}}
    out = {"status": status, "payload": {"clauses": [], "k": 5, "mode": "scoped", "scope_size": 1, "n_candidates": 1, "alarm_id": aid, "reused": False},
           "grounding": {"mode": "scoped", "tau": 0.5, "g_AB": 0.7, "g_C": 0.2, "ev_AB": True, "ev_C": False,
                         "snapshot": [{"clause_key": "P-132-2013:0:4.8", "grade": "A", "score": 0.7}]},
           "state": None, "reason": None, "trace": {"versions": {**VERSIONS, "contract_sha256": contract, "detector": detector}}}
    cur.execute("SELECT coalesce(max(seq),0)+1 FROM mart.agent_steps WHERE run_uuid=%s", (run_uuid,))
    seq = cur.fetchone()[0]
    cur.execute("""INSERT INTO mart.agent_steps (run_uuid, request_id, seq, skill, status, input, output)
                   VALUES (%s,%s,%s,'skills_retrieve',%s,%s,%s)""",
                (run_uuid, rid, seq, status, json.dumps(bound), json.dumps(out)))
    return rid


@pytest.fixture(scope="module")
def world():
    assert is_test_dsn(DSN), "refusing to run destructive snapshot tests outside a test DB"
    detector, contract = _meta()
    ep = episode_for_preset(DSN, 9)
    aid_expected = {**VERSIONS, "contract_sha256": contract, "detector": detector}

    def reset():
        with psycopg.connect(DSN) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM mart.policy_input_snapshots WHERE episode_id=%s", (ep,))
            conn.commit()
        clean_episode(DSN, ep)

    reset()
    run_uuid = uuid.uuid4()
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO mart.agent_runs (run_uuid, thread_id, mode) VALUES (%s,'snapshot-test','replay') ON CONFLICT DO NOTHING", (run_uuid,))
        _insert_window(cur, ep, 0, detector, contract, False, [])                       # no alarm
        _insert_window(cur, ep, 1, detector, contract, True, [1])                       # alarm + first k=5 on pred_set
        _insert_k5(cur, alarm_id_of(ep, 1, detector), detector, contract, [1], run_uuid)
        _insert_window(cur, ep, 2, detector, contract, False, [], q_used=True)          # already spent before any freeze
        _insert_window(cur, ep, 3, detector, contract, True, [2])                       # alarm, no k=5 envelope
        _insert_window(cur, ep, 4, detector, contract, True, [3])                       # alarm, k=5 computed on another set
        _insert_k5(cur, alarm_id_of(ep, 4, detector), detector, contract, [3, 4], run_uuid)
        _insert_window(cur, ep, 5, detector, contract, False, [])                       # never frozen
        _insert_window(cur, ep, 6, detector, contract, False, [])                       # contention: spend wins
        _insert_window(cur, ep, 7, detector, contract, False, [])                       # contention: freeze wins
        _insert_window(cur, ep, 8, detector, contract, False, [], gate="BLOCK")         # BLOCK window
        _insert_window(cur, ep, 9, detector, contract, True, [5])                       # alarm, k=5 refused row only
        _insert_k5(cur, alarm_id_of(ep, 9, detector), detector, contract, [5], run_uuid, status="REFUSED")
        conn.commit()
    yield {"ep": ep, "detector": detector, "contract": contract, "expected": aid_expected}
    reset()


def _freeze(w, t, **kw):
    return freeze_policy_input(DSN, w["ep"], t, w["detector"], w["contract"], ROLE, **kw)


def _load(w, t, **kw):
    return load_policy_input(DSN, w["ep"], t, w["detector"], w["contract"], ROLE, **kw)


def test_lock_key_equals_detection_state_store(world):
    store = DetectionStateStore(DSN, world["detector"], world["contract"])
    assert lock_key(world["ep"], world["detector"], world["contract"]) == store._lock_key(world["ep"])


def test_freeze_and_load_no_alarm_and_alarm_windows(world):
    s0 = _freeze(world, 0)
    assert s0.alarm is False and s0.k5 is None and s0.snapshot_id and s0.content_sha256 == s0.content_hash()
    assert _load(world, 0) == s0
    s1 = _freeze(world, 1, expected_k5_versions=world["expected"])
    assert s1.alarm is True and s1.k5 is not None and s1.k5.fault_set == [1] and s1.k5.g_ab == 0.7 and s1.pred_set == [1]
    assert _load(world, 1, expected_k5_versions=world["expected"]) == s1
    s8 = _freeze(world, 8)
    assert s8.gate == "BLOCK" and s8.alarm is None and s8.pred_set == [] and s8.k5 is None


def test_freeze_is_idempotent_and_survives_the_query_being_spent(world):
    s1 = _load(world, 1)
    store = DetectionStateStore(DSN, world["detector"], world["contract"])
    assert store.spend_query(world["ep"], 1, uuid.uuid4()) is True
    again = _freeze(world, 1)                                     # existing snapshot: current q_used is not consulted
    assert again.snapshot_id == s1.snapshot_id and again.content_sha256 == s1.content_sha256 and again == s1
    assert store.set_expanded(world["ep"], 1, _query_id(world["ep"], 1), [1, 2]) is True
    assert _freeze(world, 1) == s1 and _load(world, 1) == s1


def _query_id(ep, t):
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT query_request_id FROM mart.episode_windows WHERE episode_id=%s AND window_index=%s", (ep, t))
        return cur.fetchone()[0]


def test_freeze_refuses_a_window_whose_query_was_spent_before_any_freeze(world):
    with pytest.raises(SnapshotError) as e:
        _freeze(world, 2)
    assert e.value.code == "SNAPSHOT_TOO_LATE"


def test_freeze_refuses_missing_or_mismatched_k5(world):
    with pytest.raises(SnapshotError) as e:
        _freeze(world, 3)
    assert e.value.code == "SNAPSHOT_K5_MISSING"
    with pytest.raises(SnapshotError) as e:
        _freeze(world, 4)
    assert e.value.code == "SNAPSHOT_K5_MISMATCH"
    with pytest.raises(SnapshotError) as e:
        _freeze(world, 9)                                          # only a REFUSED envelope exists
    assert e.value.code == "SNAPSHOT_K5_MISSING"


def test_load_is_explicit_about_missing_stale_and_corrupted_snapshots(world):
    with pytest.raises(SnapshotError) as e:
        _load(world, 5)
    assert e.value.code == "SNAPSHOT_MISSING"
    with pytest.raises(SnapshotError) as e:
        _load(world, 1, expected_k5_versions={**world["expected"], "tau": 0.6})
    assert e.value.code == "SNAPSHOT_STALE"
    with pytest.raises(SnapshotError) as e:
        _freeze(world, 1, expected_k5_versions={**world["expected"], "reranker": "other"})
    assert e.value.code == "SNAPSHOT_STALE"
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:                              # simulate a corrupted source row
        cur.execute("UPDATE mart.episode_windows SET m_t=m_t+1 WHERE episode_id=%s AND window_index=0", (world["ep"],))
        conn.commit()
    try:
        with pytest.raises(SnapshotError) as e:
            _load(world, 0)
        assert e.value.code == "SNAPSHOT_MISMATCH"
    finally:
        with psycopg.connect(DSN) as conn, conn.cursor() as cur:
            cur.execute("UPDATE mart.episode_windows SET m_t=m_t-1 WHERE episode_id=%s AND window_index=0", (world["ep"],))
            conn.commit()
    assert _load(world, 0).window_index == 0


def _contend(world, t, holder_spends: bool):
    """A holder connection takes the SAME advisory lock, a worker thread calls freeze (blocks), the holder then spends
    the query (or not) and commits. Only two outcomes are admissible: (spend first → TOO_LATE) or (freeze → snapshot)."""
    key = lock_key(world["ep"], world["detector"], world["contract"])
    result: dict = {}

    def worker():
        try:
            result["snap"] = _freeze(world, t)
        except SnapshotError as e:
            result["error"] = e.code

    with psycopg.connect(DSN) as holder:
        with holder.transaction(), holder.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (key,))
            th = threading.Thread(target=worker)
            th.start()
            th.join(timeout=2.0)
            assert th.is_alive(), "freeze must block while the lock is held"
            if holder_spends:
                cur.execute("""UPDATE mart.episode_windows SET q_used=true, query_request_id=%s
                               WHERE episode_id=%s AND window_index=%s AND detector_version=%s AND contract_sha256=%s AND q_used=false""",
                            (uuid.uuid4(), world["ep"], t, world["detector"], world["contract"]))
                assert cur.rowcount == 1
        th.join(timeout=30.0)
    assert not th.is_alive()
    return result


def test_freeze_vs_spend_contention(world):
    lost = _contend(world, 6, holder_spends=True)
    assert lost == {"error": "SNAPSHOT_TOO_LATE"}
    won = _contend(world, 7, holder_spends=False)
    assert "snap" in won and won["snap"].window_index == 7
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM mart.policy_input_snapshots WHERE episode_id=%s AND window_index IN (6,7)", (world["ep"],))
        assert cur.fetchone()[0] == 1                                 # never a snapshot of a consumed window


def test_snapshot_fk_blocks_mixing_contracts(world):
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            cur.execute("""INSERT INTO mart.policy_input_snapshots (episode_id, window_index, detector_version, contract_sha256, caller_role,
                             gate, drift, drift_assessed, m_t, alarm, pred_set, set_size, content_sha256)
                           VALUES (%s, 0, %s, %s, 'engineer', 'PASS', 'ok', true, 1, false, '{}', 0, 'x')""",
                        (world["ep"], world["detector"], "0" * 64))
        conn.rollback()
