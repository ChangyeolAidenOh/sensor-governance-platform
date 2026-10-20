"""tests/unit/test_schemas.py — Skill contract v1.0 JSON Schemas: envelope invariants and tool inputs (no DB, no models)."""
from __future__ import annotations

import uuid

import pytest

from core_pipeline.skills.schemas import validate_envelope, validate_input, validate_payload, validate_result

U = lambda: str(uuid.uuid4())
CTX = {"request_id": U(), "thread_id": "t1", "run_uuid": U(), "caller_role": "engineer"}
TRACE = {"skill": "skills_detect", "contract_version": "1.0", "latency_ms": 3}


def _state(**kw):
    s = {"episode_id": U(), "window_index": 3, "detector_version": "detector_v1", "contract_sha256": "c", "gate": "PASS", "drift": "ok",
         "drift_assessed": False, "p_value": 0.2, "p_adj": 0.3, "m_t": 3, "alarm": False, "pred_set": [], "set_size": 0, "top1": None,
         "q_used": False, "expanded_set": None, "reused": False}
    s.update(kw)
    return s


def test_ok_needs_null_reason_and_refused_needs_reason():
    assert validate_envelope({"status": "OK", "payload": None, "grounding": None, "state": None, "reason": None, "trace": TRACE}) == []
    assert validate_envelope({"status": "REFUSED", "payload": None, "grounding": None, "state": None, "reason": None, "trace": TRACE})
    assert validate_envelope({"status": "REFUSED", "payload": None, "grounding": None, "state": None, "reason": {"code": "GATE_BLOCK", "message": "x"}, "trace": TRACE}) == []
    assert validate_envelope({"status": "REFUSED", "payload": None, "grounding": None, "state": None, "reason": {"code": "NOPE", "message": "x"}, "trace": TRACE})


def test_status_vocabulary_is_three_valued():
    for bad in ("UNDECIDABLE", "NO_EVIDENCE"):
        assert validate_envelope({"status": bad, "payload": None, "grounding": None, "state": None, "reason": None, "trace": TRACE})


def test_block_state_has_no_p_and_no_alarm():
    assert validate_envelope({"status": "OK", "payload": None, "grounding": None, "state": _state(gate="BLOCK", p_value=None, p_adj=None, alarm=None), "reason": None, "trace": TRACE}) == []
    assert validate_envelope({"status": "OK", "payload": None, "grounding": None, "state": _state(gate="BLOCK", p_value=0.1, p_adj=None, alarm=None), "reason": None, "trace": TRACE})
    assert validate_envelope({"status": "OK", "payload": None, "grounding": None, "state": _state(gate="PASS", p_value=None), "reason": None, "trace": TRACE})


def test_grounding_block_shape():
    g = {"mode": "scoped", "tau": 0.5, "g_AB": 0.7, "g_C": 0.4, "ev_AB": True, "ev_C": False}
    assert validate_envelope({"status": "OK", "payload": None, "grounding": g, "state": None, "reason": None, "trace": TRACE}) == []
    assert validate_envelope({"status": "OK", "payload": None, "grounding": {**g, "mode": "hybrid"}, "state": None, "reason": None, "trace": TRACE})


def test_tool_inputs():
    assert validate_input("gate_check", {"ctx": CTX, "episode_id": U(), "window_index": 47}) == []
    assert validate_input("gate_check", {"ctx": CTX, "episode_id": U(), "window_index": 48})
    assert validate_input("detect_diagnose", {"ctx": CTX, "episode_id": U(), "window_index": 0, "expand": 3})
    assert validate_input("retrieve", {"ctx": CTX, "query": "냉각수", "mode": "scoped", "fault_set": [4, 11], "k": 10}) == []
    assert validate_input("retrieve", {"ctx": CTX, "query": "냉각수", "mode": "scoped", "fault_set": [21], "k": 5})
    assert validate_input("retrieve", {"ctx": CTX, "query": "냉각수", "mode": "scoped", "fault_set": [4], "k": 7})
    assert validate_input("work_order", {"ctx": {**CTX, "caller_role": "admin"}, "proposal_id": U()}, "status")
    d = {"proposal_id": U(), "draft_version": 1, "content_sha256": "a" * 64, "form": {}}
    assert validate_input("work_order", {"ctx": CTX, "draft": d, "approval_token": None}, "issue") == []
    assert validate_input("work_order", {"ctx": CTX, "alarm_id": "a", "episode_id": U(), "window_index": 8, "policy": "B1", "decision_id": U(),
                                         "action_items": [{"action_type": "inspect", "item": "x", "clause_key": "k"}]}, "draft")
    assert validate_input("work_order", {"ctx": CTX, "alarm_id": "a", "episode_id": U(), "window_index": 8, "policy": "E", "decision_id": U(),
                                         "action_items": [{"action_type": "shutdown", "item": "x", "clause_key": "k"}]}, "draft")


def test_payload_validation_runs_only_on_ok():
    env = {"status": "REFUSED", "payload": None, "grounding": None, "state": None, "reason": {"code": "NO_APPROVAL", "message": "m"}, "trace": TRACE}
    assert validate_result("work_order", env, "issue_payload") == []
    env_ok = {"status": "OK", "payload": {"work_order_id": "w", "cmms_ref": "r", "idempotent_replay": False}, "grounding": None, "state": None, "reason": None, "trace": TRACE}
    assert validate_result("work_order", env_ok, "issue_payload") == []
    assert validate_payload("work_order", {"work_order_id": "w"}, "issue_payload")
