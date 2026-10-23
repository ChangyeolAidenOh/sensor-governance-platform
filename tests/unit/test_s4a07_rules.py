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
