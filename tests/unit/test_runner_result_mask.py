"""ST4-38: compose_result() masks approval_token in the returned result; the input State is not mutated (no DB, no graph run)."""
from __future__ import annotations

import copy
import json

from core_pipeline.agent.runner import RESULT_FIELDS, compose_result, masked_approval

TOKEN = "11111111-2222-3333-4444-555555555555"


def _values(token=TOKEN, with_key=True):
    pl = {"decision": "approve", "proposal_id": "p1", "draft_version": 1}
    if with_key:
        pl["approval_token"] = token
    return {"category": "action", "decision": {"action": "propose", "rule": 4}, "trace": [{"tool": "work_order_approve", "step": "approve:v1", "status": "OK"}],
            "answer": "승인 결과: approve", "approval": {"status": "OK", "payload": pl, "reason": None, "state": None, "grounding": None, "trace": {"replayed": False}},
            "work_order": {"status": "OK", "payload": {"work_order_id": "w1", "cmms_ref": "MOCK"}},
            "draft": {"status": "OK", "payload": {"proposal_id": "p1", "draft_version": 1, "content_sha256": "abc", "form": {"x": 1}}, "reason": None, "extra": "dropped"}}


def test_result_has_no_token_but_an_issued_flag_and_other_fields_unchanged():
    v = _values()
    out = compose_result(v, "t1")
    assert '"approval_token":' not in json.dumps(out) and TOKEN not in json.dumps(out)
    assert "approval_token" not in out["approval"]["payload"]
    assert out["approval"]["payload"]["approval_token_issued"] is True
    assert out["approval"]["payload"]["decision"] == "approve" and out["approval"]["status"] == "OK"
    assert out["status"] == "done" and out["turn_id"] == "t1" and out["category"] == "action"
    assert out["work_order"] == v["work_order"] and out["trace"] == v["trace"] and out["answer"] == v["answer"]
    assert out["draft"] == {"status": "OK", "payload": v["draft"]["payload"], "reason": None}        # draft summary unchanged by ST4-38
    assert set(out) == {"status", "turn_id", "draft"} | ({k for k in RESULT_FIELDS if k in v})


def test_input_state_keeps_the_token_deep_copy_not_shallow():
    v = _values()
    before = copy.deepcopy(v)
    out = compose_result(v, "t1")
    assert v == before                                                        # State untouched: approval_resume still issues with it
    assert v["approval"]["payload"]["approval_token"] == TOKEN
    out["approval"]["payload"]["decision"] = "changed"
    assert v["approval"]["payload"]["decision"] == "approve"                   # no aliasing between result and State


def test_reject_and_missing_token_cases():
    out = compose_result(_values(token=None), "t1")
    assert out["approval"]["payload"]["approval_token_issued"] is False and "approval_token" not in out["approval"]["payload"]
    out = compose_result(_values(with_key=False), "t1")
    assert "approval_token_issued" not in out["approval"]["payload"] and "approval_token" not in out["approval"]["payload"]
    assert masked_approval(None) is None and masked_approval("REFUSED") == "REFUSED"
    v = _values(); v["approval"] = {"status": "REFUSED", "payload": None, "reason": {"code": "APPROVER_MISMATCH"}}
    assert compose_result(v, "t1")["approval"] == v["approval"]
