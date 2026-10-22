"""S4a-05 unit tests: FakeLLMClient + MemoryLedger only (no DB, no network, $0). ST4-20..23."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from core_pipeline.agent import llm_client as C
from core_pipeline.agent.llm_client import FakeLLMClient, LLMOutcomeUnknown, LLMRequest, LLMRequestRejected, LLMResponse, NetworkBlocked, network_blocked
from core_pipeline.agent.llm_ledger import MemoryLedger, request_id
from core_pipeline.agent.llm_narrator import LLMNarrator, check_prose, fact_sheet, stage_key
from core_pipeline.agent.llm_router import ROUTER_SCHEMA, LLMRouter, merge, parse_validate
from core_pipeline.agent.narrator import TemplateNarrator
from core_pipeline.agent.router import CATEGORIES, RouteResult, RuleRouter

THREAD, RUN = "thread-1", "11111111-1111-1111-1111-111111111111"


def base_state(**over) -> dict:
    st = {"thread_id": THREAD, "run_uuid": RUN, "turn_id": "turn-1", "display_window": 8, "kind": "alarm", "alarm_id": "ep:8:detector_v1",
          "draft_version": 0, "modify_round": 0, "category": None, "route": None}
    st.update(over)
    return st


def llm_json(**over) -> str:
    d = {"category": "alarm", "user_claims": [], "ref_window": None, "ref_faults": [], "ref_proposal": None, "rationale": "x"}
    d.update(over)
    return json.dumps(d)


# ----------------------------------------------------------------------------------------------------- client / estimate
def test_request_rejects_disallowed_model_and_estimates_conservatively():
    with pytest.raises(ValueError):
        LLMRequest(model="claude-sonnet-5-5", system="s", user="u", max_tokens=10, timeout_s=1)
    req = LLMRequest(model="claude-haiku-5-5", system="시스템", user="사용자 메시지", max_tokens=1000, timeout_s=1)
    n_bytes = len(("시스템" + "사용자 메시지").encode("utf-8"))
    assert req.estimated_input_tokens() == n_bytes + C.RESERVE_FRAMING_TOKENS
    assert req.reserved_usd() == pytest.approx(C.cost_usd("claude-haiku-5-5", n_bytes + 64, 1000))
    rec = req.reserve_record()
    assert rec["formula_version"] == C.RESERVE_FORMULA_VERSION and rec["price_table_version"] == C.PRICE_TABLE_VERSION
    assert "system_sha256" in req.bound_input() and req.bound_input() != LLMRequest(model="claude-haiku-5-5", system="s2", user="사용자 메시지", max_tokens=1000, timeout_s=1).bound_input()


def test_anthropic_client_binds_no_tools_static():
    src = inspect.getsource(C.AnthropicClient)
    assert ".bind_tools(" not in src and ".with_structured_output(" not in src and "tools=" not in src
    assert "max_retries=0" in src


# ----------------------------------------------------------------------------------------------------- ledger (memory)
def test_ledger_reserve_settle_replay_mismatch_and_budget_unset():
    req = LLMRequest(model="claude-haiku-5-5", system="s", user="u", max_tokens=100, timeout_s=1)
    led = MemoryLedger(cap=None)
    assert led.reserve(thread_id=THREAD, run_uuid=RUN, stage_key="k", attempt=1, tool="router", req=req).status == "budget_refused"
    led = MemoryLedger(cap=1.0)
    r1 = led.reserve(thread_id=THREAD, run_uuid=RUN, stage_key="k", attempt=1, tool="router", req=req)
    assert r1.status == "owned" and r1.request_id == request_id(THREAD, RUN, "k", 1)
    assert led.reserve(thread_id=THREAD, run_uuid=RUN, stage_key="k", attempt=1, tool="router", req=req).status == "in_progress"
    resp = LLMResponse(text="{}", input_tokens=10, output_tokens=5, latency_ms=1, model=req.model)
    out = led.settle(r1.request_id, resp, req=req, output={"merged": {"category": "alarm"}})
    assert out["cost_usd"] == C.cost_usd(req.model, 10, 5) and out["reservation_exceeded"] is False
    assert out["calibration"]["estimated_input_tokens"] == req.estimated_input_tokens() and out["calibration"]["actual_input_tokens"] == 10
    assert led.reserve(thread_id=THREAD, run_uuid=RUN, stage_key="k", attempt=1, tool="router", req=req).status == "replay"
    req2 = LLMRequest(model="claude-haiku-5-5", system="s", user="DIFFERENT", max_tokens=100, timeout_s=1)
    assert led.reserve(thread_id=THREAD, run_uuid=RUN, stage_key="k", attempt=1, tool="router", req=req2).status == "mismatch"


def test_ledger_concurrent_reservations_bounded_by_cap():
    req = LLMRequest(model="claude-haiku-5-5", system="s", user="u", max_tokens=100, timeout_s=1)
    led = MemoryLedger(cap=req.reserved_usd() * 1.5)
    assert led.reserve(thread_id=THREAD, run_uuid=RUN, stage_key="a", attempt=1, tool="router", req=req).status == "owned"
    assert led.reserve(thread_id=THREAD, run_uuid=RUN, stage_key="b", attempt=1, tool="router", req=req).status == "budget_refused"


def test_ledger_reservation_exceeded_blocks_even_with_budget_headroom():
    """The cap is far from exhausted; the ACTUAL cost exceeded its own reservation → next reserve blocked until resolved."""
    req = LLMRequest(model="claude-haiku-5-5", system="s", user="u", max_tokens=10, timeout_s=1)
    led = MemoryLedger(cap=100.0)
    r1 = led.reserve(thread_id=THREAD, run_uuid=RUN, stage_key="a", attempt=1, tool="router", req=req)
    big = LLMResponse(text="x", input_tokens=10_000, output_tokens=10_000, latency_ms=1, model=req.model)   # actual > reserved
    out = led.settle(r1.request_id, big, req=req, output={})
    assert out["reservation_exceeded"] is True and out["exceeded_resolved"] is False
    assert out["cost_usd"] == C.cost_usd(req.model, 10_000, 10_000) > r1.reserved_usd      # settled at the actual amount
    assert led.spent_usd() < 100.0
    assert led.reserve(thread_id=THREAD, run_uuid=RUN, stage_key="b", attempt=1, tool="router", req=req).status == "exceeded_unresolved"
    with pytest.raises(ValueError):
        led.resolve_exceeded(r1.request_id, evidence="", by="t")
    led.resolve_exceeded(r1.request_id, evidence="price table and formula re-checked", by="t")
    assert led.reserve(thread_id=THREAD, run_uuid=RUN, stage_key="b", attempt=1, tool="router", req=req).status == "owned"


def test_ledger_unknown_usage_blocks_like_exceeded():
    req = LLMRequest(model="claude-haiku-5-5", system="s", user="u", max_tokens=10, timeout_s=1)
    led = MemoryLedger(cap=100.0)
    r1 = led.reserve(thread_id=THREAD, run_uuid=RUN, stage_key="a", attempt=1, tool="router", req=req)
    out = led.settle(r1.request_id, LLMResponse(text="x", input_tokens=0, output_tokens=0, latency_ms=1, model=req.model, usage_known=False), req=req, output={})
    assert out["usage_missing"] is True and out["cost_usd"] == r1.reserved_usd and out["reservation_exceeded"] is False
    assert led.reserve(thread_id=THREAD, run_uuid=RUN, stage_key="b", attempt=1, tool="router", req=req).status == "exceeded_unresolved"
    with pytest.raises(RuntimeError):
        led.resolve_exceeded(r1.request_id, evidence="just a string", by="t")           # evidence alone cannot clear usage_missing
    with pytest.raises(ValueError):
        led.settle_usage(r1.request_id, -1, 5, evidence="e", by="t")
    out = led.settle_usage(r1.request_id, 5, 5, evidence="vendor console: 5/5", by="t")   # real usage below the reservation
    assert out["usage_missing"] is False and out["reservation_exceeded"] is False and out["cost_usd"] == C.cost_usd(req.model, 5, 5)
    assert led.reserve(thread_id=THREAD, run_uuid=RUN, stage_key="b", attempt=1, tool="router", req=req).status == "owned"


def test_ledger_usage_recorded_above_reservation_is_a_separate_exceeded_event():
    req = LLMRequest(model="claude-haiku-5-5", system="s", user="u", max_tokens=10, timeout_s=1)
    led = MemoryLedger(cap=100.0)
    r1 = led.reserve(thread_id=THREAD, run_uuid=RUN, stage_key="a", attempt=1, tool="router", req=req)
    led.settle(r1.request_id, LLMResponse(text="x", input_tokens=0, output_tokens=0, latency_ms=1, model=req.model, usage_known=False), req=req, output={})
    out = led.settle_usage(r1.request_id, 50_000, 50_000, evidence="vendor console", by="t")
    assert out["usage_missing"] is False and out["reservation_exceeded"] is True and out["exceeded_resolved"] is False
    assert led.reserve(thread_id=THREAD, run_uuid=RUN, stage_key="b", attempt=1, tool="router", req=req).status == "exceeded_unresolved"
    led.resolve_exceeded(r1.request_id, evidence="formula v1 margin re-checked", by="t")
    assert led.reserve(thread_id=THREAD, run_uuid=RUN, stage_key="b", attempt=1, tool="router", req=req).status == "owned"


# ----------------------------------------------------------------------------------------------------- router
def test_rule_decided_turn_makes_no_call_and_no_row():
    fake, led = FakeLLMClient([llm_json(category="evidence")]), MemoryLedger(cap=1.0)
    rr = LLMRouter(fake, led).route("작업지시 초안 만들어줘", base_state())
    assert rr.category == "action" and rr.matched == "action"
    assert fake.calls == [] and led.rows == {}


def test_default_turn_calls_llm_and_merges_with_rule_extraction():
    fake, led = FakeLLMClient([llm_json(category="diagnosis", user_claims=[3], ref_faults=[3, 4], ref_window=5,
                                        ref_proposal="AAAAAAAA-1111-2222-3333-444444444444")]), MemoryLedger(cap=1.0)
    msg = "창 7 기준으로 보면 3번이 맞는 것 같은데 상태 어때?"
    rule = RuleRouter().route(msg, base_state())
    assert rule.matched == "default" and rule.ref_window == 7
    rr = LLMRouter(fake, led).route(msg, base_state())
    assert rr.category == "diagnosis" and rr.matched == "llm"
    assert rr.ref_window == 7                                  # rule reference takes precedence over the LLM's 5
    assert rr.user_claims == [3] and rr.ref_faults == sorted(set(rule.ref_faults) | {3, 4})
    assert rr.ref_proposal == "aaaaaaaa-1111-2222-3333-444444444444"
    assert len(fake.calls) == 1 and "tools" not in fake.calls[0].bound_input()
    row = next(iter(led.rows.values()))
    assert row.status == "OK" and row.output["merged"]["category"] == "diagnosis" and row.output["rule_extract"]["matched"] == "default"
    assert row.output["llm_parsed"]["ref_faults"] == [3, 4] and row.output["validation_errors"] == []


def test_invalid_output_retries_once_then_falls_back_preserving_rule_result():
    fake, led = FakeLLMClient(["not json", json.dumps({"category": "nope"})]), MemoryLedger(cap=1.0)
    msg = "창 7은 어때"
    rule = RuleRouter().route(msg, base_state())
    rr = LLMRouter(fake, led).route(msg, base_state())
    assert len(fake.calls) == 2 and len(led.rows) == 2
    assert rr == RouteResult(rule.category, rule.user_claims, rule.ref_window, rule.ref_faults, rule.ref_proposal, "llm_fallback:invalid_output")
    assert all(r.violation and r.reason_code == "LLM_OUTPUT_INVALID" and r.status == "OK" for r in led.rows.values())   # received → settled
    assert led.spent_usd() > 0


def test_no_response_keeps_pending_and_resend_does_not_recall():
    fake, led = FakeLLMClient([LLMOutcomeUnknown("timeout"), llm_json()]), MemoryLedger(cap=1.0)
    router = LLMRouter(fake, led)
    rr = router.route("상태?", base_state())
    assert rr.matched == "llm_outcome_unknown" and len(fake.calls) == 1
    row = next(iter(led.rows.values()))
    assert row.status == "PENDING" and row.cost_usd > 0 and "no_response" in (row.output or {})
    rr2 = router.route("상태?", base_state())                 # same turn resent: PENDING seen → fallback, no second call
    assert rr2.matched == "llm_refused:in_progress" and len(fake.calls) == 1


def test_rejected_request_settles_zero_and_falls_back():
    fake, led = FakeLLMClient([LLMRequestRejected("400")]), MemoryLedger(cap=1.0)
    rr = LLMRouter(fake, led).route("상태?", base_state())
    assert rr.matched == "llm_fallback:rejected"
    row = next(iter(led.rows.values()))
    assert row.status == "ERROR" and row.cost_usd == 0.0 and row.reason_code == "LLM_REQUEST_REJECTED"


def test_replay_returns_same_route_without_call():
    fake, led = FakeLLMClient([llm_json(category="history")]), MemoryLedger(cap=1.0)
    router = LLMRouter(fake, led)
    a = router.route("지금은?", base_state())
    b = router.route("지금은?", base_state())
    assert a == b and a.category == "history" and len(fake.calls) == 1 and len(led.rows) == 1


def test_budget_unset_refuses_without_call():
    fake, led = FakeLLMClient([llm_json()]), MemoryLedger(cap=None)
    rr = LLMRouter(fake, led).route("상태?", base_state())
    assert rr.matched == "llm_refused:budget_refused" and fake.calls == [] and led.rows == {}


def test_parse_validate_and_schema():
    assert parse_validate("```json\n" + llm_json() + "\n```")[1] == []
    assert parse_validate(llm_json(category="action", extra=1))[1]           # additionalProperties
    assert parse_validate("")[1] == ["no JSON object in the response"]
    assert ROUTER_SCHEMA["properties"]["category"]["enum"] == list(CATEGORIES)


def test_merge_keeps_rule_proposal_and_drops_bad_uuid():
    rule = RouteResult("alarm", [], 3, [], "11111111-1111-1111-1111-111111111111", "default")
    m = merge(rule, {"category": "evidence", "ref_proposal": "not-a-uuid", "ref_window": 9})
    assert m.ref_proposal == rule.ref_proposal and m.ref_window == 3 and m.category == "evidence"
    assert merge(RouteResult("alarm", [], None, [], None, "default"), {"category": "alarm", "ref_proposal": "zzz"}).ref_proposal is None
    assert merge(RouteResult("alarm", [], None, [], None, "default"), {"category": "alarm", "ref_faults": [3, 99, 0]}).ref_faults == [3]


# ----------------------------------------------------------------------------------------------------- narrator
def action_state(matched: str, **over) -> dict:
    retr = {"status": "OK", "payload": {"clauses": [{"clause_key": "P-116-2012:main:5.2.2", "grade": "A", "score": 0.73, "title": "경보 설정",
                                                      "text": "경보 설정값은 공정 안전 한계 이내여야 한다 (원문 42자)."}]},
            "grounding": {"mode": "scoped", "tau": 0.5, "g_AB": 0.73, "g_C": 0.1, "ev_AB": True}}
    det = {"status": "OK", "payload": {}, "state": {"gate": "PASS", "drift": "ok", "p_value": 0.0123, "p_adj": 0.031, "alarm": True, "m_t": 3,
                                                   "pred_set": [1], "top1": 1, "alarm_id": "ep:8:detector_v1"}}
    st = base_state(category="action", route={"category": "action", "matched": matched}, detect=det, retrieve=retr,
                    decision={"action": "hold", "rule": 3, "reason": "drift critical", "decision_id": "d1"})
    st.update(over)
    return st


def test_fact_sheet_excludes_clause_text_and_fixed_block_is_kept():
    st = action_state("llm")
    fixed = TemplateNarrator().narrate(st)
    assert "원문 42자" in fixed
    sheet = fact_sheet(st)
    assert "원문 42자" not in sheet and "P-116-2012:main:5.2.2" in sheet and "0.0123" in sheet and "규칙 3" in sheet


def test_narrator_fixed_categories_make_no_call():
    fake, led = FakeLLMClient(["x"] * 4), MemoryLedger(cap=1.0)
    n = LLMNarrator(fake, led, mode="llm_all")
    for st in (base_state(category="out_of_scope", route={"matched": "oos"}), base_state(category="methodology", route={"matched": "method"}),
               base_state(category="history", route={"matched": "history"}, history={"n_windows": 9, "alarm_windows": [8], "n_block": 0, "decisions": []}),
               base_state(category="action", route={"matched": "llm"}, refusal={"code": "DEMO_LIMIT", "message": ""})):
        n.narrate(st)
    assert fake.calls == [] and led.rows == {}


def test_narrator_mode_llm_only_on_llm_routed_turns_and_llm_all_on_rule_turns():
    fake, led = FakeLLMClient(["창 8에서 알람이 있고 p_adj 0.031로 보류 결정입니다. 규칙 3 적용입니다."] * 2), MemoryLedger(cap=1.0)
    fixed = TemplateNarrator().narrate(action_state("action"))
    out = LLMNarrator(fake, led, mode="llm").narrate(action_state("action"))
    assert out == fixed and fake.calls == []                                    # deterministic turn stays $0
    out = LLMNarrator(fake, led, mode="llm").narrate(action_state("llm"))
    assert out.startswith(fixed) and "요약(LLM 생성):" in out and len(fake.calls) == 1
    assert "원문 42자" not in fake.calls[0].user                               # clause text never sent
    out = LLMNarrator(fake, led, mode="llm_all").narrate(action_state("action", turn_id="turn-2"))
    assert "요약(LLM 생성):" in out and len(fake.calls) == 2


def test_narrator_drops_prose_with_fabricated_number_or_clause_key():
    fixed = TemplateNarrator().narrate(action_state("llm"))
    fake, led = FakeLLMClient(["알람 확률은 0.42이며 보류입니다.", "P-999-2012:main:1.1 조항이 근거입니다."]), MemoryLedger(cap=1.0)
    out = LLMNarrator(fake, led).narrate(action_state("llm"))
    assert out.startswith(fixed) and "요약(LLM 생성)" not in out and "보조 검사 위반" in out
    assert len(fake.calls) == 2 and all(r.violation and r.status == "OK" for r in led.rows.values())
    assert led.spent_usd() > 0                                                  # received → settled even though rejected


def test_check_prose_rules():
    sheet = "창 8: p=0.0123, p_adj=0.031, m_t=3\n[P-116-2012:main:5.2.2] (A, score 0.730) 경보 설정"
    assert check_prose("창 8의 p_adj 0.031, 조항 P-116-2012:main:5.2.2 참조.", sheet) == []
    assert any("numbers" in e for e in check_prose("p_adj 0.013", sheet))      # transposed digits are a NEW number
    assert any("clause key" in e for e in check_prose("P-116-2012:main:9.9 참조", sheet))
    assert check_prose("", sheet)


def test_narrator_stage_key_distinguishes_outcomes_and_replays_same_stage():
    s0 = action_state("llm")
    s_ap_ok = action_state("llm", draft_version=1, approval_input={"decision": "approve"}, approval={"status": "OK", "payload": {"decision": "approve"}},
                           work_order={"status": "OK", "payload": {"work_order_id": "WO-1", "cmms_ref": "c"}})
    s_ap_rej = action_state("llm", draft_version=1, approval_input={"decision": "reject"}, approval={"status": "OK", "payload": {"decision": "reject"}})
    s_wo_ref = action_state("llm", draft_version=1, approval_input={"decision": "approve"}, approval={"status": "OK", "payload": {}},
                            work_order={"status": "REFUSED", "reason": {"code": "TOKEN_USED"}})
    keys = {stage_key(s) for s in (s0, s_ap_ok, s_ap_rej, s_wo_ref)}
    assert len(keys) == 4 and stage_key(s_ap_ok).endswith("ap=approve/OK:wo=WO-1") and "wo=REFUSED:TOKEN_USED" in stage_key(s_wo_ref)
    fake, led = FakeLLMClient(["보류 결정, 규칙 3, p_adj 0.031."]), MemoryLedger(cap=1.0)
    n = LLMNarrator(fake, led)
    a, b = n.narrate(s0), n.narrate(s0)
    assert a == b and len(fake.calls) == 1 and len(led.rows) == 1          # same stage → replay, no second call


# ----------------------------------------------------------------------------------------------------- review 2026-10-21 counterexamples
def test_router_replay_restores_attempt2_success_without_a_call():
    """attempt 1 invalid → attempt 2 success → same stage re-executed → identical result, zero new calls (L1)."""
    fake, led = FakeLLMClient(["not json", llm_json(category="action")]), MemoryLedger(cap=1.0)
    first = LLMRouter(fake, led).route("지금은?", base_state())
    assert first.category == "action" and first.matched == "llm" and len(fake.calls) == 2
    again = LLMRouter(FakeLLMClient([]), led).route("지금은?", base_state())   # no scripted response: any call would fail
    assert again == first and len(led.rows) == 2


def test_router_replay_of_two_stored_failures_falls_back_without_a_call():
    fake, led = FakeLLMClient(["x", "y"]), MemoryLedger(cap=1.0)
    first = LLMRouter(fake, led).route("지금은?", base_state())
    again = LLMRouter(FakeLLMClient([]), led).route("지금은?", base_state())
    assert first == again and again.matched == "llm_fallback:invalid_output" and len(led.rows) == 2


def test_narrator_replay_restores_attempt2_success_without_a_call():
    good = "보류 결정이며 규칙 3이 적용되었고 p_adj는 0.031입니다."
    fake, led = FakeLLMClient(["확률 0.42입니다.", good]), MemoryLedger(cap=1.0)
    first = LLMNarrator(fake, led).narrate(action_state("llm"))
    assert good in first and len(fake.calls) == 2
    again = LLMNarrator(FakeLLMClient([]), led).narrate(action_state("llm"))
    assert again == first and len(led.rows) == 2


def test_fact_sheet_never_contains_multiline_clause_text():
    body = "첫 줄 원문 알파\n둘째 줄 원문 베타\n셋째 줄 원문 감마"
    st = action_state("llm")
    st["retrieve"]["payload"]["clauses"][0]["text"] = body
    st["retrieve"]["payload"]["clauses"].append({"clause_key": "P-129-2012:main:4.3", "grade": "B", "score": 0.61, "title": "경보 관리",
                                                 "text": "넷째 줄 원문 델타\n다섯째 줄 원문 엡실론"})
    fixed = TemplateNarrator().narrate(st)
    assert "둘째 줄 원문 베타" in fixed and "다섯째 줄 원문 엡실론" in fixed          # the screen text keeps the verbatim block
    sheet = fact_sheet(st)
    for frag in ("알파", "베타", "감마", "델타", "엡실론", "원문"):
        assert frag not in sheet
    assert "P-116-2012:main:5.2.2" in sheet and "P-129-2012:main:4.3" in sheet and "0.0123" in sheet and "규칙 3" in sheet
    fake, led = FakeLLMClient(["보류 결정, 규칙 3, p_adj 0.031."]), MemoryLedger(cap=1.0)
    LLMNarrator(fake, led).narrate(st)
    assert "원문" not in fake.calls[0].user and "원문" not in fake.calls[0].system


def test_fact_sheet_covers_evidence_and_no_alarm_paths():
    st = action_state("llm", category="evidence")
    sheet = fact_sheet(st)
    assert "open 모드" in sheet and "[P-116-2012:main:5.2.2]" in sheet and "원문 42자" not in sheet
    st = action_state("llm", category="alarm")
    st["detect"]["state"]["alarm"] = False
    assert "알람이 없습니다" in fact_sheet(st)


def test_network_guard_blocks_sockets():
    import socket
    with network_blocked():
        with pytest.raises(NetworkBlocked):
            socket.create_connection(("127.0.0.1", 9))
        with pytest.raises(NetworkBlocked):
            socket.socket().connect(("127.0.0.1", 9))
    assert socket.create_connection is not None                                # restored


def test_http_errors_are_unknown_outcome_unless_documented():
    class Err(Exception):
        status_code = 429
    assert C.AnthropicClient._classify(Err()) is LLMOutcomeUnknown
    assert C.AnthropicClient._classify(TimeoutError()) is LLMOutcomeUnknown
    assert C.REJECTED_NO_CHARGE_STATUSES == frozenset()                        # filled only after the docs check


# ----------------------------------------------------------------------------------------------------- commit 3: price tiers, guard, max_attempts
HI_IN, HI_OUT = 0.50, 2.50
LO_IN, LO_OUT = 0.10, 0.50


def test_price_tier_switches_input_and_output_rates_at_the_boundary():
    assert C.cost_usd("claude-haiku-5-5", 100_000, 1_000) == pytest.approx(100_000 * LO_IN / 1e6 + 1_000 * LO_OUT / 1e6, abs=1e-6)
    assert C.cost_usd("claude-haiku-5-5", 100_001, 1_000) == pytest.approx(100_001 * HI_IN / 1e6 + 1_000 * HI_OUT / 1e6, abs=1e-6)
    t = C.PRICE_TABLES["claude-haiku-5-5"]
    assert C.tier_for(t, 100_000) == 0 and C.tier_for(t, 100_001) == 1 and t["version"] == C.PRICE_TABLE_VERSION


def test_input_guard_refuses_large_estimates_only():
    LLMRequest(model="claude-haiku-5-5", system="s", user="x" * (C.INPUT_GUARD_TOKENS - C.RESERVE_FRAMING_TOKENS - 1), max_tokens=10, timeout_s=1)
    with pytest.raises(ValueError):
        LLMRequest(model="claude-haiku-5-5", system="s", user="x" * C.INPUT_GUARD_TOKENS, max_tokens=10, timeout_s=1)


def test_reserve_record_preserves_the_full_price_table():
    req = LLMRequest(model="claude-haiku-5-5", system="s", user="u", max_tokens=10, timeout_s=1)
    rec = req.reserve_record()
    assert rec["price_table"]["tiers"] == [[LO_IN, LO_OUT], [HI_IN, HI_OUT]] and rec["price_table"]["boundary_input_tokens"] == 100_000
    assert rec["prices_per_mtok"] == [LO_IN, LO_OUT]                           # tier used by the estimate (low)


def test_settlement_prices_actual_usage_at_the_higher_tier_when_the_estimate_was_low():
    req = LLMRequest(model="claude-haiku-5-5", system="s", user="u", max_tokens=10, timeout_s=1)
    led = MemoryLedger(cap=100.0)
    r = led.reserve(thread_id=THREAD, run_uuid=RUN, stage_key="a", attempt=1, tool="router", req=req)
    out = led.settle(r.request_id, LLMResponse(text="x", input_tokens=100_001, output_tokens=10, latency_ms=1, model=req.model), req=req, output={})
    assert out["cost_usd"] == pytest.approx(100_001 * HI_IN / 1e6 + 10 * HI_OUT / 1e6, abs=1e-6)
    assert out["calibration"]["price_tier_estimated"] == 0 and out["calibration"]["price_tier_actual"] == 1
    assert out["reservation_exceeded"] is True


def test_settle_usage_uses_the_stored_table_and_tier():
    req = LLMRequest(model="claude-haiku-5-5", system="s", user="u", max_tokens=10, timeout_s=1)
    led = MemoryLedger(cap=100.0)
    r = led.reserve(thread_id=THREAD, run_uuid=RUN, stage_key="a", attempt=1, tool="router", req=req)
    led.settle(r.request_id, LLMResponse(text="x", input_tokens=0, output_tokens=0, latency_ms=1, model=req.model, usage_known=False), req=req, output={})
    led.rows[r.request_id].input["reserve"]["price_table"]["tiers"] = [[1.0, 2.0], [3.0, 4.0]]   # row's own table, not the global one
    out = led.settle_usage(r.request_id, 100_000, 1_000_000, evidence="console", by="t")
    assert out["cost_usd"] == pytest.approx(100_000 * 1.0 / 1e6 + 1_000_000 * 2.0 / 1e6, abs=1e-6)
    r2 = led.reserve(thread_id=THREAD, run_uuid=RUN, stage_key="b", attempt=1, tool="router", req=req)
    assert r2.status == "exceeded_unresolved"                                   # 2.1 USD > the reservation


def test_legacy_row_with_only_a_price_pair_settles_at_that_pair():
    from core_pipeline.agent.llm_ledger import _stored_cost
    inp = {"reserve": {"prices_per_mtok": [1.0, 2.0], "price_table_version": "ST0-22/2026-10"}}
    assert _stored_cost(inp, 1_000_000, 1_000_000) == pytest.approx(3.0)
    assert _stored_cost(inp, 5_000_000, 0) == pytest.approx(5.0)             # no tier boundary in a legacy row


def test_max_attempts_one_falls_back_without_a_second_call():
    fake, led = FakeLLMClient(["not json", llm_json()]), MemoryLedger(cap=1.0)
    rr = LLMRouter(fake, led, max_attempts=1).route("지금은?", base_state())
    assert rr.matched == "llm_fallback:invalid_output" and len(fake.calls) == 1 and len(led.rows) == 1
    fake, led = FakeLLMClient(["확률 0.42입니다.", "보류."]), MemoryLedger(cap=1.0)
    out = LLMNarrator(fake, led, max_attempts=1).narrate(action_state("llm"))
    assert "보조 검사 위반" in out and len(fake.calls) == 1 and len(led.rows) == 1
    for bad in (0, 3):
        with pytest.raises(ValueError):
            LLMRouter(fake, led, max_attempts=bad)
        with pytest.raises(ValueError):
            LLMNarrator(fake, led, max_attempts=bad)
