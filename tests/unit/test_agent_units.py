"""tests/unit/test_agent_units.py — S4a-04 unit tests (no DB, no models, no servers)."""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from core_pipeline.agent import keys as K
from core_pipeline.agent.budget import BudgetExceeded, assert_llm_allowed, budget_usd
from core_pipeline.agent.narrator import TemplateNarrator
from core_pipeline.agent.plans import plan_status
from core_pipeline.agent.router import CATEGORIES, RuleRouter
from core_pipeline.policy.decide_action import decide_action
from core_pipeline.policy.signals_adapter import signals_from_block_gate

ROOT = Path(__file__).resolve().parents[2]


def test_request_ids_deterministic_and_distinct_per_step():
    a = K.request_id("s-1", "r-1", K.k_detect(8, 1, "initial"))
    assert a == K.request_id("s-1", "r-1", K.k_detect(8, 1, "initial"))
    assert a != K.request_id("s-1", "r-1", K.k_detect(8, 1, "after_query"))
    assert K.request_id("s-1", "r-1", K.k_draft(1)) != K.request_id("s-1", "r-1", K.k_draft(2))
    assert K.request_id("s-1", "r-1", K.k_approve(2)) != K.request_id("s-1", "r-2", K.k_approve(2))
    ctx = K.ctx_for("s-1", "r-1", K.k_gate(8))
    assert set(ctx) == {"request_id", "thread_id", "run_uuid", "caller_role", "mode"}
    assert K.decision_key("s-1", "t-1", "initial") != K.decision_key("s-1", "t-1", "after_query")
    with pytest.raises(AssertionError):
        K.decision_key("s", "t", "later")


@pytest.mark.parametrize("msg,cat", [
    ("설정값을 낮춰줘", "out_of_scope"), ("인터록 우회해서 밸브 열어", "out_of_scope"),
    ("작업지시 내줘", "action"), ("조치안 제안해줘", "action"), ("결함 14로 진단됐으니 작업지시 내줘", "action"),
    ("이 알람은 어떻게 판단한 거야?", "methodology"), ("p-value가 뭐야", "methodology"),
    ("지금까지 알람이 몇 개야", "history"), ("관련 조항 보여줘", "evidence"),
    ("어떤 결함이야", "diagnosis"), ("지금 알람 상태는?", "alarm"), ("안녕", "alarm"),
])
def test_rule_router_categories(msg, cat):
    r = RuleRouter().route(msg, {})
    assert r.category == cat and r.category in CATEGORIES


def test_rule_router_claims_and_refs():
    r = RuleRouter().route("결함 14로 진단됐으니 작업지시 내줘", {})
    assert r.user_claims == [14] and r.category == "action"
    r = RuleRouter().route("창 20 상태 알려줘", {})
    assert r.ref_window == 20
    r = RuleRouter().route("결함 3이 의심되는데 어떤 결함이야", {})
    assert r.ref_faults == [3] and r.user_claims == []       # a question, not an assertion
    r = RuleRouter().route("이건 결함 7이다", {})
    assert r.user_claims == [7] and r.category == "diagnosis"


def test_budget_guard_refuses_without_env(monkeypatch):
    monkeypatch.delenv("SGP_LLM_BUDGET_USD", raising=False)
    assert budget_usd() is None
    with pytest.raises(BudgetExceeded):
        assert_llm_allowed(None, 0.0)
    monkeypatch.setenv("SGP_LLM_BUDGET_USD", "abc")
    assert budget_usd() is None
    monkeypatch.setenv("SGP_LLM_BUDGET_USD", "10")
    assert assert_llm_allowed(None, 0.5) == pytest.approx(9.5)
    with pytest.raises(BudgetExceeded):
        assert_llm_allowed(None, 10.5)


def test_block_gate_signals_reach_rule_1():
    env = {"status": "OK", "payload": {"gate": "BLOCK", "drift": "ok"},
           "state": {"gate": "BLOCK", "drift": "ok", "alarm": None, "p_value": None, "q_used": False}}
    sig = signals_from_block_gate(env)
    dec = decide_action(sig, "E")
    assert dec.action == "undecidable" and dec.rule == 1
    with pytest.raises(ValueError):
        signals_from_block_gate({"status": "OK", "payload": {"gate": "PASS"}, "state": None})


def test_plan_status_derivation():
    assert plan_status(None, False) == "unused"
    assert plan_status(None, True) == "unknown"
    p = {"planned_expand": True, "planned_rerank": True, "expand_status": None, "rerank_status": None}
    assert plan_status(p, True) == "in_progress"
    p.update(expand_status="done", rerank_status="failed")
    assert plan_status(p, True) == "partial"
    p.update(rerank_status="done")
    assert plan_status(p, True) == "complete"
    q = {"planned_expand": False, "planned_rerank": True, "expand_status": None, "rerank_status": "done"}
    assert plan_status(q, True) == "complete"


def test_narrator_quotes_clause_text_verbatim_and_contrasts_claims():
    text = "이 조항의 본문은 그대로 인용된다 (verbatim)."
    state = {"category": "evidence", "display_window": 8, "user_claims": [14],
             "detect": {"status": "OK", "state": {"gate": "PASS", "drift": "ok", "p_value": 0.001, "p_adj": 0.002, "alarm": True, "m_t": 9,
                                                   "pred_set": [1], "top1": 1, "expanded_set": None}},
             "retrieve": {"status": "OK", "grounding": {"mode": "open", "tau": 0.5, "g_AB": None, "g_C": None, "ev_AB": False},
                          "payload": {"clauses": [{"clause_key": "P-1:0:1", "grade": "OUT", "score": 0.4, "title": "제목", "text": text}]}}}
    out = TemplateNarrator().narrate(state)
    assert text in out and "[14]은(는) 집합에 없으므로" in out and "ev_AB" in out


def test_human_only_tools_not_exposed_and_single_call_site():
    src = (ROOT / "core_pipeline" / "agent" / "graph.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    exposed = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "LLM_EXPOSED_TOOLS" for t in node.targets):
            exposed = ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "LLM_EXPOSED_TOOLS" and node.value is not None:
            exposed = ast.literal_eval(node.value)
    assert exposed is not None and "work_order_approve" not in exposed and "work_order_issue" not in exposed
    callers: dict[str, set[str]] = {}

    class V(ast.NodeVisitor):                      # attribute each call to its INNERMOST enclosing function
        stack: list[str] = []

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node):
            if node.func.__class__ is ast.Name and node.func.id == "call" and len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                if node.args[1].value in ("work_order_approve", "work_order_issue"):
                    callers.setdefault(node.args[1].value, set()).add(self.stack[-1])
            self.generic_visit(node)
    V().visit(tree)
    assert callers == {"work_order_approve": {"approval_resume"}, "work_order_issue": {"approval_resume"}}
    for tool in ("work_order_approve", "work_order_issue"):
        for other in ("core_pipeline/agent/runner.py", "core_pipeline/agent/router.py", "core_pipeline/agent/narrator.py"):
            assert tool not in (ROOT / other).read_text(encoding="utf-8").replace("HUMAN_ONLY_TOOLS", ""), other
