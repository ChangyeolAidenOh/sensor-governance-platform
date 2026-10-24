"""tests/unit/test_initial_snapshot.py — ST4-50 criterion 4 (snapshot validation), the pure parts: validate_k5,
verify_against_source, content hash, alarm_id formula. Lock-key equality with DetectionStateStore and the DB paths
are in tests/integration/test_policy_snapshot_db.py."""
from __future__ import annotations

import pytest

from core_pipeline.policy.initial_snapshot import SnapshotError, alarm_id_of, validate_k5, verify_against_source
from core_pipeline.policy.policy_input import K5_VERSION_KEYS, PolicyInput, versions_mismatch
from core_pipeline.skills.servers.common import alarm_id_of as server_alarm_id_of

CONTRACT = "c" * 64
DETECTOR = "detector_v1"
EP = "11111111-1111-1111-1111-111111111111"
AID = f"{EP}:8:{DETECTOR}"
RID = "22222222-2222-2222-2222-222222222222"
VERSIONS = {"contract_sha256": CONTRACT, "detector": DETECTOR, "embedder": "bge-m3", "reranker": "bge-reranker-v2-m3", "tau": 0.5,
            "fault_map_sha256": "f" * 64, "k_primary": 5, "k_query_more": 10}


def _bound(fault_set=(1,), k=5, mode="scoped", alarm_id=AID, contract="1.0"):
    return {"ctx": {"request_id": RID, "thread_id": "t", "run_uuid": RID, "caller_role": "engineer", "mode": "replay"},
            "skill": "skills_retrieve", "tool": "retrieve", "contract_version": contract,
            "args": {"query": "q", "mode": mode, "fault_set": list(fault_set), "k": k, "alarm_id": alarm_id, "query_id": None}}


def _out(status="OK", mode="scoped", alarm_id=AID, versions=VERSIONS, snapshot=None):
    return {"status": status, "payload": {"clauses": [], "k": 5, "mode": mode, "alarm_id": alarm_id, "reused": False},
            "grounding": {"mode": mode, "tau": 0.5, "g_AB": 0.7, "g_C": 0.2, "ev_AB": True, "ev_C": False,
                          "snapshot": [{"clause_key": "P-132-2013:0:4.8", "grade": "A", "score": 0.7}] if snapshot is None else snapshot},
            "state": None, "reason": None, "trace": {"versions": dict(versions)}}


def _pi(**over):
    base = dict(episode_id=EP, window_index=8, detector_version=DETECTOR, contract_sha256=CONTRACT, caller_role="engineer", gate="PASS",
                drift="ok", drift_assessed=True, p_value=0.01, p_adj=0.02, m_t=9, alarm=True, alarm_id=AID, pred_set=[1], set_size=1, top1=1,
                k5=validate_k5(RID, _bound(), _out(), [1], AID, CONTRACT))
    base.update(over)
    pi = PolicyInput(**base)
    return PolicyInput(**{**pi.__dict__, "content_sha256": pi.content_hash()})


def _win(pi: PolicyInput, **over):
    d = {"gate": pi.gate, "drift": pi.drift, "drift_assessed": pi.drift_assessed, "p_value": pi.p_value, "p_adj": pi.p_adj, "m_t": pi.m_t,
         "alarm": pi.alarm, "pred_set": list(pi.pred_set), "set_size": pi.set_size, "top1": pi.top1, "q_used": True, "expanded_set": [1, 2]}
    d.update(over)
    return tuple(d.values())


def test_alarm_id_matches_server_formula():
    assert alarm_id_of(EP, 8, DETECTOR) == server_alarm_id_of(EP, 8, DETECTOR) == AID


def test_validate_k5_accepts_first_observation_envelope():
    k5 = validate_k5(RID, _bound(), _out(), [1], AID, CONTRACT)
    assert k5.fault_set == [1] and k5.g_ab == 0.7 and k5.ev_ab is True and k5.snapshot[0]["grade"] == "A" and k5.versions["tau"] == 0.5


@pytest.mark.parametrize("bound, out, pred_set", [
    (_bound(fault_set=(1, 2)), _out(), [1]),                      # computed on the expanded set, not the first pred_set
    (_bound(k=10), _out(), [1]),                                  # k=10 is the re-search, never the first observation
    (_bound(mode="open"), _out(mode="open"), [1]),                # open mode never grounds
    (_bound(alarm_id="x"), _out(), [1]),                          # another alarm
    (_bound(), _out(alarm_id="x"), [1]),
    (_bound(), _out(versions={**VERSIONS, "contract_sha256": "d" * 64}), [1]),   # another input contract
    (_bound(contract="0.9"), _out(), [1]),
    (_bound(), _out(status="REFUSED"), [1]),
    (_bound(), _out(snapshot="nope"), [1]),
])
def test_validate_k5_refuses_mixed_or_later_envelopes(bound, out, pred_set):
    with pytest.raises(SnapshotError) as e:
        validate_k5(RID, bound, out, pred_set, AID, CONTRACT)
    assert e.value.code == "SNAPSHOT_K5_MISMATCH"


def test_verify_against_source_ignores_current_query_state_but_not_detection_columns():
    pi = _pi()
    verify_against_source(pi, _win(pi), None)                                      # q_used=true / expanded present: irrelevant
    with pytest.raises(SnapshotError) as e:
        verify_against_source(pi, _win(pi, m_t=10), None)
    assert e.value.code == "SNAPSHOT_MISMATCH"
    with pytest.raises(SnapshotError) as e:
        verify_against_source(PolicyInput(**{**pi.__dict__, "content_sha256": "0" * 64}), _win(pi), None)
    assert e.value.code == "SNAPSHOT_MISMATCH"
    with pytest.raises(SnapshotError) as e:
        verify_against_source(pi, None, None)
    assert e.value.code == "SNAPSHOT_SOURCE_MISSING"


def test_verify_against_source_checks_expected_retrieval_versions_for_alarm_windows_only():
    pi = _pi()
    verify_against_source(pi, _win(pi), VERSIONS)
    with pytest.raises(SnapshotError) as e:
        verify_against_source(pi, _win(pi), {**VERSIONS, "tau": 0.6})
    assert e.value.code == "SNAPSHOT_STALE"
    assert versions_mismatch(VERSIONS, {**VERSIONS, "reranker": "other", "k_primary": 7}) == ["reranker", "k_primary"]
    assert set(versions_mismatch({}, VERSIONS)) == set(K5_VERSION_KEYS)
    no_alarm = _pi(alarm=False, alarm_id=None, k5=None)
    verify_against_source(no_alarm, _win(no_alarm), {**VERSIONS, "tau": 0.6})       # nothing to compare


def test_content_hash_is_stable_and_sensitive():
    a, b = _pi(), _pi()
    assert a.content_sha256 == b.content_sha256
    assert _pi(top1=2).content_sha256 != a.content_sha256
    assert _pi(caller_role="restricted").content_sha256 != a.content_sha256
