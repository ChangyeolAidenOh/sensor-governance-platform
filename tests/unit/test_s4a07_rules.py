"""ST4-41~44 (S4a-07 follow-up): router methodology intent, claim extraction counter-examples, narrator flow lines,
views excerpt-chunk label, chat_eval template table. No DB."""
from __future__ import annotations

import pytest

from core_pipeline.agent import chat_eval as CE
from core_pipeline.agent.narrator import ACTION_KO, TemplateNarrator
from core_pipeline.agent.router import RuleRouter
from demo.views import clause_label


# ------------------------------------------------------------------------------------------------- ST4-42
@pytest.mark.parametrize("msg,cat", [
    ("진단 집합에 하나만 남으면 그 결함이 90% 확실하다는 뜻이야?", "methodology"),       # chat05 intent
    ("p_adj가 작다는 게 무슨 의미야?", "methodology"),
    ("알람이 났다는 뜻이야?", "methodology"),
    ("밸브 개도 90%인데 현재 상태는?", "alarm"),                                          # '%' is never a signal on its own
    ("어떤 결함이야", "diagnosis"),
    ("이 알람은 어떻게 판단한 거야?", "methodology"),                                      # existing rule unchanged
    ("결함 14로 진단됐으니 작업지시 내줘", "action"),                                       # ACTION still precedes METHOD
    ("결함 14일 가능성도 있어? 현재 진단 후보와 비교해 줘.", "diagnosis"),                 # chat02 unchanged
])
def test_methodology_intent_rule(msg, cat):
    assert RuleRouter().route(msg, {}).category == cat


# ------------------------------------------------------------------------------------------------- ST4-43
@pytest.mark.parametrize("msg,claims", [
    ("결함 14가 확실하니 다시 검사하지 말고 바로 작업지시를 발행해 줘.", [14]),            # chat09 (assertion)
    ("결함 14로 진단됐어. 탐지 결과를 먼저 확인하고 작업지시를 제안해 줘.", [14]),          # chat08 (unchanged)
    ("결함 14로 진단됐어. 작업지시를 제안해 줄래?", [14]),                                  # a '?' elsewhere does not erase a claim
    ("결함 14가 맞으니 조치해 줘", [14]),
    ("결함 14가 확실하다", [14]), ("결함 14가 확실해", [14]), ("결함 14 맞아", [14]),
    ("결함 14가 확실하고 명백하니 조치해 줘", [14]),
    ("결함 14번으로 확인됐어", [14]), ("결함 14라고 진단됐어", [14]), ("이건 결함 7이다", [7]),
    ("결함 14가 확실해?", []), ("결함 14가 확실하니?", []), ("결함 14는 맞아?", []), ("결함 14가 맞지?", []),   # questions
    ("결함 14가 확실하다고 볼 수 있어?", []), ("결함 14가 맞다고 단정할 수 없어", []),                     # quotative / hedge
    ("결함 14라고?", []), ("결함 14라고 보는 게 맞아?", []), ("결함 14이다?", []),
    ("결함 14가 확실하지 않아", []),                                                                      # negation
    ("결함 14일 가능성도 있어? 현재 진단 후보와 비교해 줘.", []),                                          # chat02 (possibility)
    ("결함 3이 의심되는데 어떤 결함이야", []),                                                             # existing test case
])
def test_claim_extraction_counterexamples(msg, claims):
    r = RuleRouter().route(msg, {})
    assert r.user_claims == claims
    assert r.ref_faults == sorted({int(x) for x in __import__("re").findall(r"(?:결함|fault)\s*(\d{1,2})", msg) if 1 <= int(x) <= 20})


# ------------------------------------------------------------------------------------------------- ST4-44
def _action_state(after: bool):
    st = {"category": "action", "display_window": 8, "user_claims": [],
          "detect": {"status": "OK", "state": {"gate": "PASS", "drift": "ok", "p_value": 0.008, "p_adj": 0.07, "alarm": True, "m_t": 9, "pred_set": [15], "top1": 15, "expanded_set": None}},
          "decision": {"action": "query_more", "rule": 5, "reason": "closed query available"},
          "retrieve": {"status": "OK", "grounding": {"mode": "scoped", "tau": 0.5, "g_AB": 0.02, "g_C": 0.0, "ev_AB": False}, "payload": {"clauses": []}}}
    if after:
        st["decision_after"] = {"action": "hold", "rule": 6, "reason": "no A/B grounding"}
        st["query"] = {"status": "complete", "expand_status": None, "rerank_status": "done"}
    return st


def test_narrator_shows_initial_then_query_then_redecision():
    out = TemplateNarrator().narrate(_action_state(after=True)).splitlines()
    i0 = next(i for i, l in enumerate(out) if l.startswith("초기 결정(E): "))
    iq = next(i for i, l in enumerate(out) if l.startswith("query 상태: "))
    i1 = next(i for i, l in enumerate(out) if l.startswith("정책 결정(E): "))
    assert i0 < iq < i1
    assert out[i0] == f"초기 결정(E): {ACTION_KO['query_more']} — 규칙 5 (closed query available)"
    assert out[i1] == f"정책 결정(E): {ACTION_KO['hold']} — 규칙 6 (no A/B grounding)"      # label and wording unchanged
    assert sum(1 for l in out if l.startswith("정책 결정(E): ")) == 1


def test_narrator_without_redecision_is_unchanged():
    out = TemplateNarrator().narrate(_action_state(after=False))
    assert "초기 결정(E):" not in out and "query 상태:" not in out
    assert f"정책 결정(E): {ACTION_KO['query_more']} — 규칙 5 (closed query available)" in out


def test_chat_eval_table_expects_initial_line_only_with_redecision():
    snap = {"category": "action", "decision": {"action": "query_more"}, "decision_after": {"action": "hold"}, "query": {"status": "complete"},
            "retrieve": {"status": "OK"}, "detect": {"state": {"alarm": True}}}
    e = CE.template_line_expectations(snap)
    assert e["초기 결정(E):"] and e["정책 결정(E):"] and e["query 상태:"]
    snap2 = {"category": "action", "decision": {"action": "hold"}, "retrieve": {"status": "OK"}, "detect": {"state": {"alarm": True}}}
    e2 = CE.template_line_expectations(snap2)
    assert not e2["초기 결정(E):"] and e2["정책 결정(E):"]


# ------------------------------------------------------------------------------------------------- ST4-41
def test_clause_label_marks_retrieved_excerpt_chunks_never_i_of_n():
    one = clause_label({"clause_key": "P-132-2013:0:4.6", "grade": "A", "score": 0.4, "title": "[P-132-2013:0:4.6] (A, score 0.400) 점검", "n_items": 1})
    two = clause_label({"clause_key": "P-129-2013:0:5.2.2", "grade": "A", "score": 0.026, "title": "[P-129-2013:0:5.2.2] (A, score 0.026) 점검방법", "n_items": 2})
    assert one.endswith("검색된 발췌 청크 1건") and two.endswith("검색된 발췌 청크 2건")
    assert "1/2" not in two and "#0" not in two and "최대 청크 점수" in two and "청크 2개" not in two
    assert clause_label({"clause_key": None, "title": "weird", "lines": []}) == "weird"          # no n_items → no excerpt count


# ------------------------------------------------------------------------------------------------- ST4-47
from core_pipeline.agent.narrator import METHODOLOGY_KO, METHOD_DIRECT_KO, method_topics
from demo import views as V

CHAT05 = "진단 집합에 하나만 남으면 그 결함이 90% 확실하다는 뜻이야?"


def test_methodology_direct_answer_precedes_block_for_chat05():
    out = TemplateNarrator().narrate({"category": "methodology", "message": CHAT05})
    direct = METHOD_DIRECT_KO["singleton_coverage"]
    assert out.startswith(direct) and out.endswith(METHODOLOGY_KO) and out.index(direct) < out.index("탐지: ")
    assert "90% 확실하다는 뜻이 아닙니다" in out and "후보가 하나인 경우만의 정확도와도 다릅니다" in out


@pytest.mark.parametrize("msg", [
    "p-value가 결함 확률이야?",            # coverage/certainty words without a set context
    "후보 집합이 뭐야?",                    # set context without coverage/certainty words
    "p-value가 뭐야",
    "이 알람은 어떻게 판단한 거야?",
    None,
])
def test_methodology_block_only_when_topic_not_matched(msg):
    assert method_topics(msg) == []
    assert TemplateNarrator().narrate({"category": "methodology", "message": msg}) == METHODOLOGY_KO


def test_methodology_topic_requires_both_contexts():
    assert method_topics(CHAT05) == ["singleton_coverage"]
    assert method_topics("후보 집합의 커버리지가 뭐야?") == ["singleton_coverage"]
    assert method_topics("진단 집합이 한 개면 확실한 거야?") == ["singleton_coverage"]


def test_methodology_block_is_spelled_out():
    for sym in ("¬", "∧", "1≤s≤3", "휴리스틱"):
        assert sym not in METHODOLOGY_KO, sym
    assert "FDR(거짓 발견률) 제어를 보증하지 않습니다" in METHODOLOGY_KO and "현재 창까지의 정보만 사용하는 알람 규칙" in METHODOLOGY_KO
    assert len(METHODOLOGY_KO.splitlines()) == 5 and all(l.startswith(("탐지: ", "진단: ", "근거: ", "정책 E", "작업지시: ")) for l in METHODOLOGY_KO.splitlines())


def test_chat_eval_fixed_response_follows_question_format_only():
    assert CE.fixed_response_expected("methodology", {}) == METHODOLOGY_KO                     # no question → block only (unchanged call sites)
    assert CE.fixed_response_expected("methodology", {}, CHAT05) == TemplateNarrator().narrate({"category": "methodology", "message": CHAT05})
    assert CE.fixed_response_expected("methodology", {}, CHAT05) != METHODOLOGY_KO


def test_views_shows_every_methodology_line():
    ans = TemplateNarrator().narrate({"category": "methodology", "message": CHAT05})
    sec = V.turn_sections(V.result_view({"status": "done", "turn_id": "t", "category": "methodology", "answer": ans}))
    shown = [sec["state_line"]] + sec["main_lines"]
    assert sec["hidden_lines"] == [] and shown == [l for l in ans.splitlines() if l.strip()]
    assert V.answer_from_blocks(V.answer_blocks(ans)) == ans


# ------------------------------------------------------------------------------------------------- ST4-48
def _diag_state(refs, claims=(), pred_set=(1,), expanded=None, status="OK", category="diagnosis", after=None):
    st = {"category": category, "display_window": 8, "user_claims": list(claims),
          "route": {"category": category, "user_claims": list(claims), "ref_faults": list(refs), "ref_window": None, "ref_proposal": None, "matched": "diagnosis"},
          "detect": {"status": status, "state": {"gate": "PASS", "drift": "critical", "p_value": 0.001, "p_adj": 0.002, "alarm": True, "m_t": 9,
                                                 "pred_set": (None if pred_set is None else list(pred_set)), "top1": 1, "expanded_set": expanded}}}
    if after is not None:
        st["detect_after"] = {"status": "OK", "state": {**st["detect"]["state"], "pred_set": list(after), "expanded_set": None}}
    return st


OUTSIDE_14 = "언급한 결함 [14]은(는) 현재 후보 집합 [1]에 포함되지 않았습니다. 이는 이번 집합 구성 기준에서 선택되지 않았다는 뜻이며, 해당 결함의 가능성이 0이라는 뜻은 아닙니다."


def test_refs_contrast_outside_is_not_exclusion():
    out = TemplateNarrator().narrate(_diag_state([14]))                           # chat02 state: ref 14, no claim, set [1]
    assert OUTSIDE_14 in out and "진단 집합(후보): [1]" in out and "사용자 주장 결함" not in out
    assert out.index("진단 집합(후보)") < out.index("언급한 결함")
    for w in ("배제", "아니다", "발행"):
        assert w not in out


def test_refs_contrast_inside_is_candidate_not_confirmation():
    out = TemplateNarrator().narrate(_diag_state([1]))
    assert "언급한 결함 [1]은(는) 현재 후보 집합 [1]에 포함됩니다(집합 크기 1). 포함은 후보라는 뜻이며 확정 진단을 뜻하지 않습니다." in out


def test_refs_contrast_mixed_prints_two_lines_inside_first():
    out = TemplateNarrator().narrate(_diag_state([1, 14])).splitlines()
    i = next(k for k, l in enumerate(out) if l.startswith("언급한 결함 [1]은(는)"))
    o = next(k for k, l in enumerate(out) if l.startswith("언급한 결함 [14]은(는)"))
    assert i < o and sum(1 for l in out if l.startswith("언급한 결함 ")) == 2


def test_refs_contrast_unobserved_set_is_not_empty_set():
    out = TemplateNarrator().narrate(_diag_state([14], pred_set=None))                # OK envelope, set field None
    assert "언급한 결함 [14]: 진단 집합이 관측되지 않아 대조할 수 없습니다." in out and "포함되지 않았습니다" not in out and "비어 있어" not in out
    out = TemplateNarrator().narrate(_diag_state([14], status="REFUSED"))              # non-OK envelope
    assert "관측되지 않아 대조할 수 없습니다" in out and "포함되지 않았습니다" not in out


def test_refs_contrast_empty_set():
    out = TemplateNarrator().narrate(_diag_state([14], pred_set=()))
    assert "언급한 결함 [14]: 현재 후보 집합이 비어 있어 포함된 후보가 없습니다. 모든 결함의 가능성이 0이라는 뜻은 아닙니다." in out
    assert "포함되지 않았습니다" not in out


def test_refs_contrast_prefers_detect_after_and_expanded_set():
    out = TemplateNarrator().narrate(_diag_state([14], pred_set=(1,), expanded=(1, 14)))
    assert "언급한 결함 [14]은(는) 현재 후보 집합 [1, 14]에 포함됩니다(집합 크기 2)" in out
    out = TemplateNarrator().narrate(_diag_state([14], pred_set=(1,), after=(14,)))
    assert "언급한 결함 [14]은(는) 현재 후보 집합 [14]에 포함됩니다(집합 크기 1)" in out


def test_refs_contrast_skips_asserted_faults_and_other_categories():
    out = TemplateNarrator().narrate(_diag_state([14], claims=[14]))                  # chat09-like assertion → claims line only
    assert "언급한 결함" not in out and "사용자 주장 결함 [14] vs 탐지 집합 [1]:" in out
    for cat in ("alarm", "evidence", "action"):
        st = _diag_state([14], category=cat)
        if cat == "action":
            st["decision"] = {"action": "hold", "rule": 3, "reason": "model_trust_low"}
        assert "언급한 결함" not in TemplateNarrator().narrate(st), cat


def test_chat_eval_table_and_views_for_refs_line():
    base = {"category": "diagnosis", "user_claims": [], "route": {"ref_faults": [14]}, "detect": {"state": {"alarm": True}}}
    assert CE.template_line_expectations(base)["언급한 결함 "] is True
    assert CE.template_line_expectations({**base, "user_claims": [14]})["언급한 결함 "] is False
    assert CE.template_line_expectations({**base, "route": {"ref_faults": []}})["언급한 결함 "] is False
    assert CE.template_line_expectations({**base, "category": "alarm"})["언급한 결함 "] is False
    ans = TemplateNarrator().narrate(_diag_state([14]))
    sec = V.turn_sections(V.result_view({"status": "done", "turn_id": "t", "category": "diagnosis", "answer": ans}))
    assert OUTSIDE_14 in sec["main_lines"] and sec["hidden_lines"] == []
