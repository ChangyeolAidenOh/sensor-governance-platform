"""S4a-07 unit tests: evals/chat_dev_set.json + core_pipeline/agent/chat_eval.py — no DB, no servers, no network, $0."""
from __future__ import annotations

import json
import uuid

import pytest

from core_pipeline.agent import chat_eval as CE
from core_pipeline.agent import keys as K
from core_pipeline.agent.narrator import METHODOLOGY_KO, TemplateNarrator
from core_pipeline.agent.router import RouteResult, RuleRouter

DS = CE.load_dev_set("evals/chat_dev_set.json")
ITEMS = {it["id"]: it for it in DS["items"]}
TID, RUN, DW = "s-thread", str(uuid.uuid4()), 8


# ------------------------------------------------------------------------------------------------- dev set
def test_dev_set_is_valid_and_fixed():
    assert CE.validate_dev_set(DS) == []
    assert DS["_meta"]["revision"] == "rev1" and len(DS["items"]) == 10
    cats = [it["expected"]["category"] for it in DS["items"]]
    assert set(cats) == set(CE.CATEGORIES) and cats.count("action") == 3 and cats.count("history") == 2
    assert {it["context"]["preset"] for it in DS["items"]} == {1, 2, 7, 8}
    assert ITEMS["chat01"]["context"]["preset"] == 2 and ITEMS["chat07"]["context"]["preset"] == 8
    assert ITEMS["chat08"]["context"]["preset"] == 1 and ITEMS["chat09"]["context"]["preset"] == 7
    assert ITEMS["chat08"]["context"]["requires"]["claim_in_set"] is False and ITEMS["chat09"]["context"]["requires"]["claim_in_set"] is True


def test_validate_catches_bad_items():
    bad = json.loads(json.dumps(DS))
    bad["items"][0]["expected"]["category"] = "nope"
    bad["items"][1]["context"]["preset"] = 3
    bad["items"][2]["contract"]["skill_calls"].append("work_order_issue")
    bad["items"][3]["contract"]["skill_calls"].append("gate:t{window}")
    errs = CE.validate_dev_set(bad)
    assert any("bad category" in e for e in errs) and any("preset not designated" in e for e in errs)
    assert any("human-only" in e for e in errs) and any("unknown placeholder" in e for e in errs)


def test_expected_values_are_intent_not_rule_output():
    """The set is fixed by intent; the rule router's own output on chat05 differs (not reworded) — documented, not asserted."""
    rr = RuleRouter().route(ITEMS["chat05"]["question"], {})
    assert ITEMS["chat05"]["expected"]["category"] == "methodology"
    assert rr.category in CE.CATEGORIES                                        # whatever it is, the set does not bend to it


# ------------------------------------------------------------------------------------------------- step keys
def test_request_map_resolves_every_graph_step():
    m = CE.request_map(TID, RUN, DW)
    for step in (K.k_gate(8), K.k_detect(8, 1, "initial"), K.k_detect(8, 2, "initial"), K.k_detect(8, 1, "after_query"), K.k_spend(8),
                 K.k_retrieve("scoped", 5), K.k_retrieve("scoped", 10), K.k_retrieve("open", 5), K.k_draft(1), K.k_approve(2), K.k_issue(3)):
        assert m[K.request_id(TID, RUN, step)] == step
    assert K.request_id(TID, RUN, K.k_gate(9)) not in m


def test_match_skill_steps_optional_and_extra():
    exp = ["gate:t8", "detect:t8:x1:initial", "retrieve:scoped:k5", "spend:t8", "detect:t8:x2:initial?", "detect:t8:x1:after_query", "retrieve:scoped:k10", "draft:v1?"]
    assert CE.match_skill_steps(exp, ["gate:t8", "detect:t8:x1:initial", "retrieve:scoped:k5", "spend:t8", "detect:t8:x1:after_query", "retrieve:scoped:k10"])[0]
    assert CE.match_skill_steps(exp, ["gate:t8", "detect:t8:x1:initial", "retrieve:scoped:k5", "spend:t8", "detect:t8:x2:initial", "detect:t8:x1:after_query", "retrieve:scoped:k10", "draft:v1"])[0]
    ok, why = CE.match_skill_steps(["gate:t8", "detect:t8:x1:initial"], ["gate:t8", "detect:t8:x1:initial", "retrieve:open:k5"])
    assert not ok and "extra" in why
    ok, why = CE.match_skill_steps(["gate:t8", "detect:t8:x1:initial"], ["gate:t8"])
    assert not ok and "detect" in why


# ------------------------------------------------------------------------------------------------- wrappers
class FakeSkills:
    def __init__(self):
        self.calls = []

    def gate_check(self, ctx, episode_id, window_index):
        self.calls.append(("gate_check", ctx["request_id"]))
        return {"status": "OK", "payload": {"gate": "PASS", "drift": "ok", "secret_thing": "x"}, "state": {"window_index": window_index, "gate": "PASS"}, "grounding": None, "reason": None, "trace": {"replayed": False}}

    def retrieve(self, ctx, query, mode, fault_set, k=5, alarm_id=None, query_id=None):
        self.calls.append(("retrieve", mode, list(fault_set), k))
        return {"status": "OK", "payload": {"reused": False, "query": query, "clauses": [{"clause_key": "P-1:0:1", "grade": "A", "score": 0.3, "title": "t", "text": "SECRET BODY\nline2"}]},
                "state": None, "grounding": {"mode": mode, "tau": 0.5, "g_AB": 0.3, "ev_AB": False, "snapshot": [{"clause_key": "P-1:0:1", "grade": "A", "score": 0.3}]}, "reason": None, "trace": {"replayed": True}}

    def work_order_approve(self, ctx, draft, approver, decision="approve", modifications=None):
        return {"status": "OK", "payload": {"decision": decision, "approval_token": "TOKEN-123"}, "state": None, "grounding": None, "reason": None, "trace": {}}

    def not_a_tool(self):
        return "passthrough"


def _ctx(step):
    return K.ctx_for(TID, RUN, step)


def test_recording_skills_delegates_and_records_allow_listed_fields():
    rec = CE.Recorder()
    rec.begin("x", TID, DW)
    fake = FakeSkills()
    rs = CE.RecordingSkills(fake, rec)
    env = rs.gate_check(_ctx(K.k_gate(8)), "ep", 8)
    assert env["payload"]["secret_thing"] == "x"                               # the graph still gets the full envelope
    env2 = rs.retrieve(_ctx(K.k_retrieve("scoped", 5)), "q", "scoped", [1], 5, alarm_id="a")
    assert env2["payload"]["clauses"][0]["text"].startswith("SECRET")
    rs.work_order_approve(_ctx(K.k_approve(1)), {"proposal_id": "p", "draft_version": 1, "content_sha256": "abc", "form": {"big": 1}}, "aiden", "approve")
    assert rs.not_a_tool() == "passthrough"
    s = rec.items["x"]["skills"]
    assert [x["step"] for x in s] == [K.k_gate(8), K.k_retrieve("scoped", 5), K.k_approve(1)]
    assert s[0]["payload"] == {"gate": "PASS", "drift": "ok"}
    assert s[1]["args"]["fault_set"] == [1] and s[1]["args"]["mode"] == "scoped" and s[1]["replayed"] is True
    assert s[1]["payload"]["clauses"] == [{"clause_key": "P-1:0:1", "grade": "A", "score": 0.3, "title": "t"}]
    assert s[2]["args"]["draft"] == {"proposal_id": "p", "draft_version": 1, "content_sha256": "abc"}
    dumped = json.dumps(s, ensure_ascii=False)
    assert "SECRET" not in dumped and "TOKEN-123" not in dumped and "big" not in dumped


def test_recording_router_and_narrator():
    rec = CE.Recorder()
    rec.begin("x", TID, DW)
    rr = CE.RecordingRouter(RuleRouter(), rec).route("결함 14로 진단됐어. 작업지시 제안해 줘.", {"turn_id": "t1", "display_window": 8, "kind": "alarm", "alarm_id": "a"})
    assert isinstance(rr, RouteResult) and rr.category == "action"
    assert rec.items["x"]["route"]["result"]["user_claims"] == [14] and rec.items["x"]["route"]["stored_alarm"] is True
    state = {"category": "action", "display_window": 8, "user_claims": [14], "messages": [{"role": "user", "content": "hidden"}],
             "detect": {"status": "OK", "state": {"alarm": True, "pred_set": [1], "top1": 1, "gate": "PASS", "drift": "critical", "p_value": 0.001, "p_adj": 0.002, "m_t": 9}, "payload": {}, "trace": {}},
             "retrieve": {"status": "OK", "payload": {"clauses": [{"clause_key": "P-1:0:1", "grade": "A", "score": 0.3, "title": "t", "text": "BODY"}]}, "grounding": {"mode": "scoped", "tau": 0.5, "g_AB": 0.3, "g_C": 0.0, "ev_AB": False}},
             "approval": {"status": "OK", "payload": {"decision": "approve", "approval_token": "TOKEN-9"}},
             "decision": {"action": "hold", "rule": 3, "reason": "drift"}}
    text = CE.RecordingNarrator(TemplateNarrator(), rec).narrate(state)
    assert text == TemplateNarrator().narrate(state)                           # transparent
    snap = rec.items["x"]["narration"]["snapshot"]
    dumped = json.dumps(snap, ensure_ascii=False)
    assert "messages" not in snap and "hidden" not in dumped and "BODY" not in dumped and "TOKEN-9" not in dumped
    assert snap["approval"]["payload"] == {"decision": "approve"} and snap["retrieve"]["payload"]["clauses"][0]["clause_key"] == "P-1:0:1"


def test_recording_decision_delegates_records_and_reraises():
    rec = CE.Recorder()
    rec.begin("x", TID, DW)
    did = uuid.uuid4()
    calls = []
    def inner(dsn, alarm_id, policy, signals, action, rule, evidence=None, decision_key=None):
        calls.append((dsn, alarm_id, policy, action, rule, decision_key))
        if action == "boom":
            raise ValueError("DECISION_MISMATCH: x")
        return did
    w = CE.recording_decision(inner, rec)
    assert w("dsn", "a", "E", {"s": 1}, "hold", 3, evidence=[{"clause_key": "k", "grade": "A", "score": 0.9}], decision_key=K.decision_key(TID, "t1", "initial")) == did
    with pytest.raises(ValueError):
        w("dsn", "a", "E", {}, "boom", 9, decision_key=K.decision_key(TID, "t1", "after_query"))
    d = rec.items["x"]["decisions"]
    assert d[0]["phase"] == "initial" and d[0]["decision_id"] == str(did) and d[0]["evidence_n"] == 1
    assert d[1]["decision_id"] is None and "DECISION_MISMATCH" in d[1]["error"]
    assert calls[0][0] == "dsn" and calls[0][5].endswith(":initial")


# ------------------------------------------------------------------------------------------------- narration checks
def test_split_answer_and_template_expectations_follow_narrator_branches():
    fixed, prose, note = CE.split_answer("창 8: x\n정책 결정(E): 보류\n\n요약(LLM 생성): 산문입니다.")
    assert fixed.endswith("보류") and prose == "산문입니다." and note is None
    fixed, prose, note = CE.split_answer("창 8: x\n\n(LLM 요약 생략: 보조 검사 위반)")
    assert prose is None and note.startswith("(LLM 요약")
    snap = {"category": "action", "decision": {"action": "hold"}, "query": None, "retrieve": {"status": "OK"}, "retrieve_k10": {"status": "REFUSED"},
            "draft": {"status": "REFUSED"}, "approval": None, "work_order": {"status": "REFUSED"}, "user_claims": [14], "detect": {"state": {"alarm": True}}}
    e = CE.template_line_expectations(snap)
    assert e["정책 결정(E):"] and not e["query 상태:"] and e["초안 거부:"] and not e[" 생성(sha "]
    assert e["발행 거부:"] and not e["작업지시 발행:"]                           # a refused issue prints 발행 거부, never 발행
    assert e["검색 mode="] is False                                             # narrator uses `retrieve_k10 or retrieve`: a failed k10 hides the block
    assert e["사용자 주장 결함 "] and not e["이 창에서는 prefix-BH 알람이 없습니다."]
    assert CE.template_line_expectations({"category": "methodology"}) == {} and CE.template_line_expectations({"refusal": {"code": "DEMO_LIMIT"}}) == {}


def test_fixed_responses_history_parse_windows_and_flags():
    assert CE.fixed_response_expected("methodology", {}) == METHODOLOGY_KO
    assert "권한 밖" in CE.fixed_response_expected("out_of_scope", {})
    assert "데모 범위 밖" in CE.fixed_response_expected("refusal", {"refusal": {"code": "DEMO_LIMIT", "message": "창 20은 선택 창 8 이후입니다"}})
    h = {"n_windows": 9, "alarm_windows": [8], "n_block": 0, "decisions": []}
    line = TemplateNarrator().narrate({"category": "history", "history": h, "display_window": 8, "alarm_id": "ep:8:detector_v1"})
    p = CE.parse_history_line(line)
    assert p["n_windows"] == 9 and p["alarm_windows"] == [8] and p["n_block"] == 0 and p["alarm_id"] == "ep:8:detector_v1"
    assert CE.window_numbers(line) == [8] and CE.window_numbers("관측 창 9개, 알람 창 [8], 창 20까지") == [20]
    assert CE.review_flags("알람이 없지만 정상이나 안전을 보장하지는 않습니다.", ["정상", "안전", "이상 없"]) == ["정상", "안전"]


# ------------------------------------------------------------------------------------------------- scoring taxonomy
def _skill(tool, step, status="OK", args=None, state=None, grounding=None, payload=None, reason=None):
    return {"tool": tool, "step": step, "status": status, "args": args or {}, "state": state, "grounding": grounding, "payload": payload or {}, "reason": reason}


def _route(category, matched="action", claims=(), faults=(), window=None):
    return {"result": {"category": category, "user_claims": list(claims), "ref_faults": list(faults), "ref_window": window, "ref_proposal": None, "matched": matched}}


def _action_obs(**over):
    """A chat08-like observation that passes: hold rule 3 on a drift-critical window, claim [14] outside the set [1]."""
    det_state = {"alarm": True, "pred_set": [1], "expanded_set": None, "gate": "PASS", "drift": "critical", "p_value": 0.001, "p_adj": 0.002, "m_t": 9, "top1": 1}
    skills = [_skill("gate_check", "gate:t8", state={"gate": "PASS"}), _skill("detect_diagnose", "detect:t8:x1:initial", state=det_state),
              _skill("retrieve", "retrieve:scoped:k5", args={"mode": "scoped", "fault_set": [1], "k": 5}, grounding={"g_AB": 0.03, "g_C": 0.0, "ev_AB": False},
                     payload={"clauses": [{"clause_key": "P-1:0:1", "grade": "A", "score": 0.03}, {"clause_key": "P-1:0:1", "grade": "A", "score": 0.02}]})]
    decisions = [{"phase": "initial", "action": "hold", "rule": 3, "decision_id": "d1", "evidence": [{"clause_key": "P-1:0:1"}, {"clause_key": "P-1:0:1"}]}]
    snap = {"category": "action", "display_window": 8, "user_claims": [14], "decision": {"action": "hold", "rule": 3, "reason": "drift critical"},
            "detect": {"status": "OK", "state": det_state}, "retrieve": {"status": "OK", "payload": {"clauses": []}, "grounding": {"mode": "scoped", "tau": 0.5, "g_AB": 0.03, "g_C": 0.0, "ev_AB": False}},
            "query": None, "draft": None, "approval": None, "work_order": None}
    answer = "창 8: gate=PASS, drift=critical, p=0.001, p_adj=0.002, alarm=True, m_t=9\n진단 집합(후보): [1] (크기 1, top1=1)\n사용자 주장 결함 [14] vs 탐지 집합 [1]: [14]은(는) 집합에 없으므로 그 주장을 근거로 제안하지 않습니다.\n정책 결정(E): 보류 — 규칙 3 (drift critical)\n검색 mode=scoped, τ=0.5, g_AB=0.03, g_C=0, ev_AB=False\n(조항 없음)"
    trace = [{"tool": "gate_check", "step": "gate:t8", "status": "OK", "reason": None}, {"tool": "detect_diagnose", "step": "detect:t8:x1:initial", "status": "OK", "reason": None},
             {"tool": "retrieve", "step": "retrieve:scoped:k5", "status": "OK", "reason": None}, {"tool": "decide_action", "step": "decide:initial", "status": "hold", "reason": "rule 3"}]
    obs = {"id": "chat08", "status": "done", "dw": 8, "context": {"prep": "ok", "contaminated": False}, "route": _route("action", claims=[14], faults=[14]),
           "skills": skills, "decisions": decisions, "narration": {"snapshot": snap, "answer": answer}, "views_roundtrip": True,
           "result": {"status": "done", "refusal": None, "trace": trace, "query": None}, "db": {"decision_rows": 1, "approvals": 0, "work_orders": 0}}
    obs.update(over)
    return obs


def _checks(sc, area="contract"):
    return {c["name"]: c["ok"] for c in sc[area]["checks"]}


def test_score_chat08_like_observation_passes_everything():
    sc = CE.score_item(ITEMS["chat08"], _action_obs())
    assert sc["router"]["ok"] and all(sc["extraction"][f]["ok"] for f in ("user_claims", "ref_faults", "ref_window", "ref_proposal"))
    assert sc["contract"]["ok"] is True, _checks(sc)
    assert sc["narration"]["ok"] is True
    # views round-trip unobserved → narration incomplete, not a pass
    sc_u = CE.score_item(ITEMS["chat08"], _action_obs(views_roundtrip=None))
    assert sc_u["narration"]["ok"] == "incomplete" and sc_u["narration"]["unobserved"] == ["views_roundtrip_lossless"]
    sc_p = CE.score_item(ITEMS["chat08"], _action_obs(views_roundtrip=True, views_roundtrip_source={"posthoc": True, "views_sha16": "b19c2b3c5ef63a94"}))
    assert sc_p["narration"]["ok"] is True and _checks(sc_p, "narration")["views_roundtrip_lossless"] is True
    assert next(c for c in sc_p["narration"]["checks"] if c["name"] == "views_roundtrip_lossless")["detail"]["posthoc"] is True
    assert CE.summarize([sc_u], "baseline")["narration"]["auto_pass"] == "0/1"
    assert sc["contract"]["failure_kind"] is None
    assert sc["narration"]["ok"], _checks(sc, "narration")
    assert sc["narration"]["flags"] == {"fixed": [], "prose": []}
    assert sc["observations"]["retrieve"][0]["n_dup"] == 1 and sc["observations"]["decision_snapshots"][0]["n_dup"] == 1


def test_prep_failure_is_not_scored_and_contamination_is_kept():
    sc = CE.score_item(ITEMS["chat08"], {"id": "chat08", "status": "not_executed", "dw": 8, "context": {"prep": "failed", "reasons": ["drift ok != critical"]}})
    assert sc["scored"] is False and "router" not in sc
    sc2 = CE.score_item(ITEMS["chat08"], _action_obs(context={"prep": "ok", "contaminated": True, "contamination": [{"when": "before_turn"}]}))
    assert sc2["scored"] and sc2["context"]["contaminated"] and sc2["contract"]["ok"]


def test_misrouted_non_action_item_gets_router_failure_kind():
    """chat05-like: a methodology question routed to diagnosis (gate+detect ran). failure_kind must say 'router', not null."""
    obs = {"id": "chat05", "status": "done", "dw": 8, "context": {"prep": "ok"}, "route": _route("diagnosis", matched="diagnosis"),
           "skills": [_skill("gate_check", "gate:t8"), _skill("detect_diagnose", "detect:t8:x1:initial", state={"alarm": True, "pred_set": [1]})], "decisions": [],
           "narration": {"snapshot": {"category": "diagnosis", "display_window": 8, "detect": {"status": "OK", "state": {"alarm": True, "pred_set": [1]}}}, "answer": "창 8: …\n진단 집합(후보): [1] (크기 1, top1=1)"},
           "views_roundtrip": True, "result": {"status": "done", "refusal": None, "trace": []}, "db": {}}
    sc = CE.score_item(ITEMS["chat05"], obs)
    assert sc["router"]["ok"] is False and sc["contract"]["ok"] is False and sc["contract"]["failure_kind"] == "router"
    assert sc["narration"]["ok"] is False
    s = CE.summarize([sc], "baseline")
    assert s["contract"]["failure_kinds"]["router"] == ["chat05"] and s["contract"]["failed_items"] == ["chat05"] and "chat05" in s["narration"]["failed_items"]
    # a contract failure with the right category and no decision expected → 'contract'
    obs2 = dict(obs, route=_route("methodology", matched="method"))
    sc2 = CE.score_item(ITEMS["chat05"], obs2)
    assert sc2["router"]["ok"] and sc2["contract"]["failure_kind"] == "contract"


def test_missing_decision_is_classified_by_observation_not_as_prep_failure():
    # category wrong → router failure (no decision because the path was never entered)
    obs = _action_obs(route=_route("diagnosis", matched="diagnosis", claims=[14], faults=[14]), decisions=[], skills=_action_obs()["skills"][:2])
    obs["result"]["trace"] = obs["result"]["trace"][:2]
    sc = CE.score_item(ITEMS["chat08"], obs)
    assert sc["scored"] and sc["router"]["ok"] is False and sc["contract"]["failure_kind"] == "router"
    # category right, detect envelope not OK → skill_error
    obs = _action_obs(decisions=[])
    obs["skills"][1]["status"] = "REFUSED"
    sc = CE.score_item(ITEMS["chat08"], obs)
    assert sc["contract"]["failure_kind"] == "skill_error"
    # category right, Skills OK, no decision recorded → contract
    sc = CE.score_item(ITEMS["chat08"], _action_obs(decisions=[]))
    assert sc["contract"]["failure_kind"] == "contract"
    # decision present but different from the assertion → contract, other scores untouched
    sc = CE.score_item(ITEMS["chat08"], _action_obs(decisions=[{"phase": "initial", "action": "query_more", "rule": 5, "decision_id": "d1", "evidence": []}]))
    assert sc["contract"]["failure_kind"] == "contract" and _checks(sc)["initial_decision_asserted"] is False and sc["router"]["ok"]


def test_waiting_approval_turn_scored_without_narration_and_with_db_counts():
    base = _action_obs()
    skills = base["skills"] + [_skill("spend_query", "spend:t8", payload={"query_id": "q"}), _skill("detect_diagnose", "detect:t8:x1:after_query", state=dict(base["skills"][1]["state"], q_used=True)),
                               _skill("retrieve", "retrieve:scoped:k10", args={"mode": "scoped", "fault_set": [1], "k": 10}, grounding={"g_AB": 0.9, "ev_AB": True}, payload={"clauses": []}),
                               _skill("work_order_draft", "draft:v1", payload={"proposal_id": "p1", "draft_version": 1})]
    decisions = [{"phase": "initial", "action": "query_more", "rule": 5, "decision_id": "d1", "evidence": []}, {"phase": "after_query", "action": "propose", "rule": 4, "decision_id": "d2", "evidence": []}]
    obs = _action_obs(status="waiting_approval", skills=skills, decisions=decisions, narration=None, result={"status": "waiting_approval"},
                      interrupt={"proposal_id": "p1", "draft_version": 1, "content_sha256": "abc"},
                      db={"decision_rows": 2, "approvals": 0, "work_orders": 0, "plan": {"owned": True, "status": "complete"}})
    sc = CE.score_item(ITEMS["chat07"], obs)
    ch = _checks(sc)
    assert sc["contract"]["ok"], ch
    assert ch["query"] is True and ch["decision_after_observed"] is True and ch["interrupt_payload"] is True
    assert sc["narration"] == {"applicable": False, "note": "N/A — 승인 대기"}
    # unobserved values stay unobserved — the item is INCOMPLETE, never a pass (reviewer counter-example)
    obs2 = dict(obs, db={"decision_rows": None, "approvals": None, "work_orders": None, "plan": None})
    sc2 = CE.score_item(ITEMS["chat07"], obs2)
    assert set(sc2["contract"]["unobserved"]) >= {"decision_rows_db", "approvals_zero", "work_orders_zero", "query"}
    assert sc2["contract"]["ok"] == "incomplete" and sc2["contract"]["failure_kind"] is None
    s = CE.summarize([sc, sc2], "baseline")
    assert s["contract"]["pass"] == "1/2" and list(s["contract"]["incomplete"]) == ["chat07"] and s["contract"]["failed_items"] == []
    # one real failure + unobserved → False (failure wins)
    obs3 = dict(obs2, db={"decision_rows": None, "approvals": 1, "work_orders": None, "plan": None})
    assert CE.score_item(ITEMS["chat07"], obs3)["contract"]["ok"] is False


def test_state_line_contrast_fails_on_wrong_template_line_and_review_flags_do_not_fail():
    obs = _action_obs()
    obs["narration"]["answer"] += "\n작업지시 발행: wo-1 (cmms_ref MOCK)"                   # printed without a work_order envelope
    sc = CE.score_item(ITEMS["chat08"], obs)
    assert sc["narration"]["ok"] is False and _checks(sc, "narration")["state_lines_auto"] is False
    obs = _action_obs()
    obs["narration"]["answer"] += "\n\n요약(LLM 생성): 승인이나 발행은 없고 보류입니다."
    sc = CE.score_item(ITEMS["chat08"], obs)
    assert sc["narration"]["ok"] is True and sc["narration"]["flags"]["prose"] == ["발행", "승인"] and sc["narration"]["prose_present"]


def test_fixed_response_and_history_items():
    obs = {"id": "chat05", "status": "done", "dw": 8, "context": {"prep": "ok"}, "route": _route("methodology", matched="method"), "skills": [], "decisions": [],
           "narration": {"snapshot": {"category": "methodology"}, "answer": METHODOLOGY_KO}, "views_roundtrip": True, "result": {"status": "done", "refusal": None, "trace": []}, "db": {}}
    sc = CE.score_item(ITEMS["chat05"], obs)
    assert sc["contract"]["ok"] and sc["narration"]["ok"] and sc["narration"]["flags"]["fixed"] == []
    h = {"n_windows": 9, "alarm_windows": [8], "n_block": 0, "decisions": []}
    snap = {"category": "history", "history": h, "display_window": 8, "alarm_id": "ep:8:detector_v1"}
    ans = TemplateNarrator().narrate(snap)
    obs = {"id": "chat04", "status": "done", "dw": 8, "context": {"prep": "ok"}, "route": _route("history", matched="history"), "skills": [], "decisions": [],
           "narration": {"snapshot": snap, "answer": ans}, "views_roundtrip": True, "result": {"status": "done", "refusal": None, "trace": []}, "db": {"decision_rows": 0}}
    sc = CE.score_item(ITEMS["chat04"], obs)
    assert sc["narration"]["ok"], _checks(sc, "narration")
    obs["db"]["decision_rows"] = 1
    assert _checks(CE.score_item(ITEMS["chat04"], obs), "narration")["history_decisions_match_db"] is False
    # DEMO_LIMIT refusal item
    ref = {"code": "DEMO_LIMIT", "message": "창 20은 선택 창 8 이후입니다"}
    snap = {"category": "history", "refusal": ref, "display_window": 8}
    obs = {"id": "chat10", "status": "done", "dw": 8, "context": {"prep": "ok"}, "route": _route("history", matched="history", window=20), "skills": [], "decisions": [],
           "narration": {"snapshot": snap, "answer": TemplateNarrator().narrate(snap)}, "views_roundtrip": True, "result": {"status": "done", "refusal": ref, "trace": []}, "db": {}}
    sc = CE.score_item(ITEMS["chat10"], obs)
    assert sc["contract"]["ok"] and sc["narration"]["ok"] and sc["extraction"]["ref_window"]["ok"]


def test_summary_metrics_separate_llm_called_valid_not_sent():
    scored = [CE.score_item(ITEMS["chat08"], _action_obs(route=_route("action", matched=m, claims=[14], faults=[14]))) for m in
              ("action", "default", "llm", "llm", "llm_fallback:invalid_output", "llm_refused:budget_refused", "llm_outcome_unknown")]
    scored[3]["router"]["ok"] = False                                           # one valid-but-wrong classification
    scored.append(CE.score_item(ITEMS["chat08"], {"id": "chat08", "status": "not_executed", "dw": 8, "context": {"prep": "failed", "reasons": ["x"]}}))
    scored.append(CE.score_item(ITEMS["chat08"], _action_obs(context={"prep": "ok", "contaminated": True})))
    s = CE.summarize(scored, "paid")
    llm = s["router"]["llm"]
    assert llm["source"] == "matched" and llm["n_called"] == 4 and llm["n_not_sent"] == 1 and llm["n_rule"] == 3
    assert llm["valid_rate"] == 0.5 and llm["acc_called"] == 0.25 and llm["acc_valid_conditional"] == 0.5   # fallback items never count as LLM-correct
    assert s["router"]["default_count"] == 1 and s["n_scored"] == 8 and s["context_prep_failed"] == ["chat08"] and s["context_contaminated"] == ["chat08"]
    assert s["router"]["accuracy_all"] == "7/8"                                 # the correct fallback results DO count here
    # counting-client records take precedence over matched strings
    for sc_, calls in zip(scored[:7], ([], [{"tool": "router"}], [{"tool": "router"}, {"tool": "narrate"}], [{"tool": "router"}], [{"tool": "router"}], [], [{"tool": "router"}])):
        sc_["llm_calls"] = calls
    s2 = CE.summarize(scored, "paid")["router"]["llm"]
    assert s2["source"] == "counting_client" and s2["n_called"] == 5 and s2["n_http_calls"] == 6 and s2["n_http_router"] == 5 and s2["n_http_narrate"] == 1
    assert s2["acc_called"] == 0.2 and s2["acc_valid_conditional"] == 0.5


# ------------------------------------------------------------------------------------------------- backup filter / DSN guard
def test_strip_secret_columns_and_dsn_guard():
    row = {"proposal_id": "p", "approval_token": "T", "owner_secret_sha256": "s", "token_used": True, "decision": "approve"}
    assert CE.strip_secret_columns(row) == {"proposal_id": "p", "decision": "approve"}
    assert CE.dbname_of("postgresql://u:pw@localhost:5999/main_test?sslmode=disable") == "main_test"
    assert CE.dbname_of("host=localhost port=5999 dbname=main_v2 user=u") == "main_v2"
    CE.assert_test_dsn("postgresql://u:pw@localhost:5999/main_test", "postgresql://u:pw@localhost:5999/main_v2")
    for bad in ("postgresql://u:pw@localhost:5999/main_v2", None):
        with pytest.raises(SystemExit):
            CE.assert_test_dsn(bad, None)
    with pytest.raises(SystemExit):
        CE.assert_test_dsn("postgresql://u:pw@localhost:5999/main_test", "postgresql://u:pw@localhost:5999/main_test")


# ------------------------------------------------------------------------------------------------- judgment / refined result
def test_judgment_status_distinguishes_sheet_from_completion_and_catches_missing_criteria():
    scored = [CE.score_item(ITEMS["chat08"], _action_obs()), CE.score_item(ITEMS["chat08"], _action_obs(status="waiting_approval", narration=None, result={"status": "waiting_approval"}))]
    scored[1]["id"] = "chat07"
    req = CE.required_criteria(DS, "chat08")
    assert len(req) == 5 and req[:4] == DS["_meta"]["human_criteria_common"] and req[4].startswith("core:")
    st = CE.judgment_status(None, scored, DS)
    assert st["sheet"] is False and st["complete"] is False and st["n_applicable"] == 1 and st["n_unjudged_items"] == 1 and st["n_unjudged_criteria"] == 5
    full = {c: "pass" for c in req}
    j = {"judge": "aiden", "items": {"chat08": {"applicable": True, "verdicts": dict(full)}, "chat07": {"applicable": False}}}
    assert CE.judgment_status(j, scored, DS)["complete"] is True
    # reviewer counter-example: one required criterion deleted from the sheet → NOT complete, exactly one unjudged criterion
    del j["items"]["chat08"]["verdicts"][req[1]]
    st = CE.judgment_status(j, scored, DS)
    assert st["complete"] is False and st["n_unjudged_items"] == 1 and st["n_unjudged_criteria"] == 1 and st["per_criterion"][req[1]]["unjudged"] == 1
    # a null verdict counts the same; a stray extra key is ignored
    j["items"]["chat08"]["verdicts"][req[1]] = None
    j["items"]["chat08"]["verdicts"]["not-a-criterion"] = "pass"
    st = CE.judgment_status(j, scored, DS)
    assert st["complete"] is False and st["n_unjudged_criteria"] == 1 and "not-a-criterion" not in st["per_criterion"]
    j["items"]["chat08"]["verdicts"][req[1]] = "fail"
    st = CE.judgment_status(j, scored, DS)
    assert st["complete"] and st["failed_items"] == {"chat08": [req[1]]} and st["per_criterion"][req[1]] == {"pass": 0, "fail": 1, "unjudged": 0}


def test_refined_result_has_allow_listed_fields_only():
    obs = _action_obs(thread_id="s-1", turn_id="t-1", preset=1, episode_id="ep", alarm_id="ep:8:detector_v1", elapsed_s=1.2)
    obs["skills"][2]["payload"]["clauses"].append({"clause_key": "P-9", "grade": "C", "score": 0.1, "title": "t"})
    sc = CE.score_item(ITEMS["chat08"], obs)
    ctx = {"dbname": "main_test", "project_manifest_sha16": "abc", "code_commit": "786bbc4",
           "presets": {"1": {"detector": "detector_v1", "contract": "808601", "prep": "ok", "entry": {"preset": 1, "display_window": 8}}}}
    j = {"judge": "aiden", "basis": "reviewer draft, adopted by aiden", "readability_scope": "text only", "readability_observation": "4 items",
         "items": {"chat08": {"applicable": True, "verdicts": {"x": "pass"}, "note": "ok", "answer": "SHOULD NOT LEAK VIA answer field of sheet"}}}
    out = CE.refined_result(DS, ctx, [obs], [sc], j, {"run_id": "b1", "mode": "baseline", "dev_set_sha16": "dddd", "summary": {"n_items": 1}})
    it = out["items"][0]
    assert it["question"] == ITEMS["chat08"]["question"] and it["answer"] == obs["narration"]["answer"] and it["human"] == {"verdicts": {"x": "pass"}, "note": "ok"}
    assert it["context"]["detector"] == "detector_v1" and it["skill_steps"] == ["gate:t8", "detect:t8:x1:initial", "retrieve:scoped:k5"]
    dumped = json.dumps(out, ensure_ascii=False)
    assert "SHOULD NOT LEAK" not in dumped and "text" not in it["scores"]["observations"]["retrieve"][0] and out["schema"] == "chat_dev_result/1"
    assert out["judgment"] == {"judge": "aiden", "basis": "reviewer draft, adopted by aiden", "readability_scope": "text only", "readability_observation": "4 items"}
