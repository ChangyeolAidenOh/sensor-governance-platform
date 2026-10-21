"""core_pipeline/agent/graph.py — the S4a-04 LangGraph (0-D spike structure + SkillClient; ST4-15..ST4-19).

    START → load_context → router ─┬─ out_of_scope / methodology ──────────────────────────────────────────► narrate
                                   ├─ history ──► history ─────────────────────────────────────────────────► narrate
                                   └─ alarm / diagnosis / evidence / action ──► gate ──┬─ BLOCK ─► (action: decide_initial | else narrate)
                                                                                      └─► detect ─┬─ alarm/diagnosis ─────────────► narrate
                                                                                                  ├─ evidence ─► retrieve_open ─► narrate
                                                                                                  └─ action ─► retrieve_scoped ─► decide_initial
      decide_initial ─┬─ propose ──► draft ─┬─ OK ─► approval_wait(interrupt) ─► approval_resume ─┬─ modify(≤2) ─► draft
                      │                     └─ REFUSED ──────────────────────────────────────────┴───────────────► narrate
                      ├─ query_more ─► query_more ─► decide_after ─┬─ propose ─► draft
                      │                                            └─ else ──────────────────────────────────────► narrate
                      └─ hold / no_alarm / undecidable ─────────────────────────────────────────────────────────► narrate

Rules enforced here, not by prompts:
  * the LLM calls NO tool (ST4-15): every Skill call below takes episode_id / window_index / alarm_id / fault_set /
    proposal_id from the State; LLM_EXPOSED_TOOLS is the (empty) list a future LLM node may use — approve/issue are
    asserted absent from it by tests/unit (static), and approval_resume is their only call site.
  * every call carries request_id = uuid5(thread, run_uuid, step_key) (keys.py): node re-execution replays.
  * decide = signals_adapter → decide_action(E) → record_decision(evidence = grounding.snapshot, decision_key) —
    one decision row per (thread, turn, phase ∈ {initial, after_query}); the graph never hard-codes an action.
  * query_more: spend_query once per window (shared across sessions — DB authoritative), the same query_id for
    expand=2 and k=10, the latest detect state re-read AFTER the query, decide_action re-run on what is available.
  * the approval boundary: approval_wait has no side effect; approval_resume re-verifies the session secret against
    the DB and passes the DB owner as approver — never a value from the resume payload.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from core_pipeline.agent import keys as K
from core_pipeline.agent.narrator import Narrator, TemplateNarrator
from core_pipeline.agent.plans import history as _history_sql
from core_pipeline.agent.plans import plan_get, plan_mark, plan_status, plan_write, recover_plan
from core_pipeline.agent.router import Router, RuleRouter
from core_pipeline.agent.session import current_auth, verify_secret
from core_pipeline.agent.state import AgentState
from core_pipeline.agent.turns import TurnStore
from core_pipeline.policy.decide_action import Decision, Signals, decide_action
from core_pipeline.policy.signals_adapter import signals_dict, signals_from_block_gate, signals_from_envelopes
from core_pipeline.skills.replay_client import scoped_query
from core_pipeline.skills.work_order import record_decision

log = logging.getLogger("sgp.agent")

LLM_EXPOSED_TOOLS: tuple[str, ...] = ()        # ST4-15: the LLM is a router/writer; it calls no Skill tool
HUMAN_ONLY_TOOLS = ("work_order_approve", "work_order_issue")
POLICY = "E"
PER_TURN_RESET = ("category", "route", "refusal", "gate", "detect", "retrieve", "decision", "query", "detect_x", "detect_after",
                  "retrieve_k10", "decision_after", "draft", "approval_input", "approval", "work_order", "history", "answer")


@dataclass
class GraphDeps:
    dsn: str
    skills: Any                                  # SyncSkillClient (or a test double with the same methods)
    turns: TurnStore
    manifest: dict
    router: Router = field(default_factory=RuleRouter)
    narrator: Narrator = field(default_factory=TemplateNarrator)
    policy: Callable[[Signals], Decision] = lambda sig: decide_action(sig, POLICY)
    record_decision: Callable[..., Any] = record_decision
    fault_map_path: str = "outputs/fault_to_guidance_map.json"
    caller_role: str = "engineer"
    max_modify_rounds: int = 2


def _code(env: Optional[dict]) -> Optional[str]:
    return ((env or {}).get("reason") or {}).get("code")


def _refuse(code: str, message: str = "") -> dict:
    return {"refusal": {"code": code, "message": message}}


def build_graph(deps: GraphDeps, checkpointer=None):
    def guard(state: AgentState) -> None:
        """Lease check at every node boundary (ST4-16): a lagging executor stops here instead of writing further."""
        auth = current_auth()
        if auth.thread_id != state["thread_id"] or auth.turn_id != state["turn_id"]:
            raise PermissionError("session context does not match the graph state")
        deps.turns.touch(state["thread_id"], auth.lease_turn, auth.executor)

    def call(state: AgentState, tool: str, step_key: str, *args, **kw) -> tuple[dict, dict]:
        ctx = K.ctx_for(state["thread_id"], state["run_uuid"], step_key, state.get("caller_role") or deps.caller_role, "chat")
        env = getattr(deps.skills, tool)(ctx, *args, **kw)
        entry = {"tool": tool, "step": step_key, "status": env.get("status"), "reason": _code(env),
                 "replayed": bool((env.get("trace") or {}).get("replayed"))}
        return env, entry

    def detect_set(env: dict) -> list[int]:
        st = env["state"]
        return list(st["expanded_set"] if st.get("expanded_set") is not None else st["pred_set"])

    # ------------------------------------------------------------------------------------------------- nodes
    def load_context(state: AgentState) -> dict:
        auth = current_auth()
        row = verify_secret(deps.dsn, state["thread_id"], auth.secret)        # DB check again inside the graph
        from core_pipeline.agent.manifest import entry_for_preset
        e = entry_for_preset(deps.manifest, row.preset)
        if row.episode_id and str(row.episode_id) != str(e["episode_id"]):
            raise ValueError("thread episode does not match the demo manifest")
        upd: dict = {k: None for k in PER_TURN_RESET}
        upd.update({"owner": row.owner, "preset": row.preset, "episode_id": str(e["episode_id"]), "display_window": int(e["display_window"]),
                    "alarm_id": e.get("alarm_id"), "kind": e["kind"], "caller_role": state.get("caller_role") or deps.caller_role,
                    "turn_refs_discarded": 0, "turn_refs_note": None, "draft_version": 0, "modify_round": 0, "trace": [],
                    "messages": [{"role": "user", "content": state["message"], "turn_id": state["turn_id"]}]})
        guard(state)
        return upd

    def router(state: AgentState) -> dict:
        guard(state)
        rr = deps.router.route(state["message"], dict(state))
        upd: dict = {"category": rr.category, "route": asdict(rr), "user_claims": rr.user_claims}
        discarded, notes = 0, []
        if rr.ref_window is not None and rr.ref_window != state["display_window"]:
            if rr.ref_window > state["display_window"]:
                upd.update(_refuse("DEMO_LIMIT", f"창 {rr.ref_window}은 선택 창 {state['display_window']} 이후입니다"))
                return upd
            discarded += 1
            notes.append(f"창 {rr.ref_window}")
        cur_prop = ((state.get("draft") or {}).get("payload") or {}).get("proposal_id")
        if rr.ref_proposal and rr.ref_proposal != cur_prop:
            discarded += 1
            notes.append(f"proposal {rr.ref_proposal[:8]}…")
        upd.update({"turn_refs_discarded": discarded, "turn_refs_note": ", ".join(notes) or None, "refs_discarded": discarded})
        return upd

    def gate(state: AgentState) -> dict:
        guard(state)
        env, tr = call(state, "gate_check", K.k_gate(state["display_window"]), state["episode_id"], state["display_window"])
        upd = {"gate": env, "trace": state["trace"] + [tr]}
        if env["status"] != "OK":
            upd.update(_refuse(_code(env) or "INTERNAL", (env.get("reason") or {}).get("message", "")))
        return upd

    def detect(state: AgentState) -> dict:
        guard(state)
        t = state["display_window"]
        env, tr = call(state, "detect_diagnose", K.k_detect(t, 1, "initial"), state["episode_id"], t)
        upd = {"detect": env, "trace": state["trace"] + [tr]}
        if env["status"] != "OK":
            upd.update(_refuse(_code(env) or "INTERNAL", (env.get("reason") or {}).get("message", "")))
        return upd

    def retrieve_scoped(state: AgentState) -> dict:
        guard(state)
        det = state["detect"]
        fs = detect_set(det)
        env, tr = call(state, "retrieve", K.k_retrieve("scoped", 5), scoped_query(fs, deps.fault_map_path), "scoped", fs, 5,
                       alarm_id=det["state"].get("alarm_id") or det["payload"].get("alarm_id"))
        upd = {"retrieve": env, "trace": state["trace"] + [tr]}
        if env["status"] != "OK":
            upd.update(_refuse(_code(env) or "INTERNAL", (env.get("reason") or {}).get("message", "")))
            return upd
        if det["state"].get("q_used"):
            # the window's query was already spent (this or another session): the stored k=10 re-search, if any, is the
            # scoped result in force for this window — re-asking must reach the same decision (L1 goal), so use it
            rk, tr2 = call(state, "retrieve", K.k_retrieve("scoped", 10), scoped_query(fs, deps.fault_map_path), "scoped", fs, 10,
                           alarm_id=det["state"].get("alarm_id") or det["payload"].get("alarm_id"), query_id=None)
            upd["retrieve_k10"] = rk
            upd["trace"] = upd["trace"] + [tr2]
            qrid = det["state"].get("query_request_id")
            plan = plan_get(deps.dsn, str(qrid)) if qrid else None
            if plan is None and qrid:
                plan = recover_plan(deps.dsn, str(qrid), state["episode_id"], state["display_window"], det["state"]["detector_version"], det["state"]["contract_sha256"])
            upd["query"] = {"owned": False, "query_id": (str(qrid) if qrid else None), "kind": (plan or {}).get("kind"),
                            "planned_expand": (plan or {}).get("planned_expand"), "planned_rerank": (plan or {}).get("planned_rerank"),
                            "expand_status": (plan or {}).get("expand_status"), "rerank_status": (plan or {}).get("rerank_status"),
                            "k10_reused": (rk["status"] == "OK" and bool((rk.get("payload") or {}).get("reused"))),
                            "status": plan_status(plan, True) + ":reused"}
        return upd

    def retrieve_open(state: AgentState) -> dict:
        guard(state)
        det = state["detect"]
        alarm_id = det["state"].get("alarm_id") if det["state"].get("alarm") else None
        env, tr = call(state, "retrieve", K.k_retrieve("open", 5), state["message"], "open", [], 5, alarm_id=alarm_id)
        return {"retrieve": env, "trace": state["trace"] + [tr]}

    def _decide(state: AgentState, phase: str) -> dict:
        guard(state)
        det = (state.get("detect_after") or state.get("detect")) if phase == "after_query" else state.get("detect")
        retr = state.get("retrieve_k10") if (state.get("retrieve_k10") or {}).get("status") == "OK" else state.get("retrieve")
        if det is None or det.get("status") != "OK":
            if (state.get("gate") or {}).get("payload", {}).get("gate") == "BLOCK":
                sig = signals_from_block_gate(state["gate"])
            else:
                raise RuntimeError("decide without a detect envelope")
        else:
            sig = signals_from_envelopes(det, retr if (retr or {}).get("status") == "OK" else None)
        dec = deps.policy(sig)
        snapshot = list(((retr or {}).get("grounding") or {}).get("snapshot") or []) if (retr or {}).get("status") == "OK" and ((retr or {}).get("grounding") or {}).get("mode") == "scoped" else []
        tau = ((retr or {}).get("grounding") or {}).get("tau")
        rec = {"phase": phase, "action": dec.action, "rule": dec.rule, "reason": dec.reason, "query_kind": dec.query_kind,
               "signals": signals_dict(sig), "evidence_snapshot": snapshot, "tau": tau, "decision_id": None}
        alarm_id = (det or {}).get("state", {}).get("alarm_id") if det else None
        if sig.alarm and alarm_id:
            did = deps.record_decision(deps.dsn, alarm_id, POLICY, signals_dict(sig), dec.action, dec.rule, evidence=snapshot,
                                       decision_key=K.decision_key(state["thread_id"], state["turn_id"], phase))
            rec["decision_id"] = str(did)
        return {("decision" if phase == "initial" else "decision_after"): rec,
                "trace": state["trace"] + [{"tool": "decide_action", "step": f"decide:{phase}", "status": dec.action, "reason": f"rule {dec.rule}", "replayed": False}]}

    def decide_initial(state: AgentState) -> dict:
        return _decide(state, "initial")

    def decide_after(state: AgentState) -> dict:
        return _decide(state, "after_query")

    def query_more(state: AgentState) -> dict:
        guard(state)
        t, ep, det, dec = state["display_window"], state["episode_id"], state["detect"], state["decision"]
        kind = dec.get("query_kind") or "both"
        planned_expand, planned_rerank = kind in ("expand", "both"), kind in ("rerank", "both")
        detector, contract = det["state"]["detector_version"], det["state"]["contract_sha256"]
        trace = list(state["trace"])
        q: dict = {"kind": kind, "planned_expand": planned_expand, "planned_rerank": planned_rerank, "owned": False, "query_id": None,
                   "expand_status": None, "rerank_status": None, "status": None}
        upd: dict = {}
        sp, tr = call(state, "spend_query", K.k_spend(t), ep, t)
        trace.append(tr)
        if sp["status"] == "OK":
            q["owned"], q["query_id"] = True, sp["payload"]["query_id"]
            plan_write(deps.dsn, q["query_id"], ep, t, detector, contract, state["thread_id"], state["turn_id"], dec.get("decision_id"), kind)
            expand_allowed = bool(sp["payload"].get("expand_allowed"))
        elif _code(sp) == "QUERY_USED":
            expand_allowed = False                                      # not ours: never expand/spend under another request
        else:
            return {"query": {**q, "status": "refused:" + str(_code(sp))}, "trace": trace, **_refuse(_code(sp) or "INTERNAL", (sp.get("reason") or {}).get("message", ""))}
        # expansion (owned only)
        if q["owned"] and planned_expand:
            if t == 0 or not expand_allowed:
                plan_mark(deps.dsn, q["query_id"], "expand", "skipped", "NO_EXPANSION")
                q["expand_status"] = "skipped"
            else:
                dx, tr = call(state, "detect_diagnose", K.k_detect(t, 2, "initial"), ep, t, expand=2, query_id=q["query_id"])
                trace.append(tr)
                upd["detect_x"] = dx
                q["expand_status"] = "done" if dx["status"] == "OK" else "failed"
                plan_mark(deps.dsn, q["query_id"], "expand", q["expand_status"], None if dx["status"] == "OK" else _code(dx))
        # the latest committed state AFTER the query (distinct step key: never the pre-query envelope)
        da, tr = call(state, "detect_diagnose", K.k_detect(t, 1, "after_query"), ep, t)
        trace.append(tr)
        if da["status"] != "OK":
            return {"query": {**q, "status": "refused:" + str(_code(da))}, "trace": trace, **_refuse(_code(da) or "INTERNAL")}
        upd["detect_after"] = da
        qrid = da["state"].get("query_request_id")
        if not q["owned"] and qrid:
            q["query_id"] = str(qrid)
            if plan_get(deps.dsn, q["query_id"]) is None:
                recover_plan(deps.dsn, q["query_id"], ep, t, detector, contract)
        # re-search (same query_id; a second session gets the stored envelope by role, or QUERY_REQUIRED)
        if planned_rerank:
            fs = detect_set(da)
            rk, tr = call(state, "retrieve", K.k_retrieve("scoped", 10), scoped_query(fs, deps.fault_map_path), "scoped", fs, 10,
                          alarm_id=da["state"].get("alarm_id"), query_id=(q["query_id"] if q["owned"] else None))
            trace.append(tr)
            upd["retrieve_k10"] = rk
            if q["owned"]:
                q["rerank_status"] = "done" if rk["status"] == "OK" else "failed"
                plan_mark(deps.dsn, q["query_id"], "rerank", q["rerank_status"], None if rk["status"] == "OK" else _code(rk))
            else:
                q["rerank_status"] = ("reused" if rk["status"] == "OK" else "unavailable:" + str(_code(rk)))
        plan = plan_get(deps.dsn, q["query_id"]) if q["query_id"] else None
        q["status"] = plan_status(plan, bool(da["state"].get("q_used"))) + ("" if q["owned"] else ":other")
        upd["query"] = q
        upd["trace"] = trace
        return upd

    def draft(state: AgentState) -> dict:
        guard(state)
        dec = state.get("decision_after") or state["decision"]
        retr = state.get("retrieve_k10") if (state.get("retrieve_k10") or {}).get("status") == "OK" else state.get("retrieve")
        det = state.get("detect_after") or state["detect"]
        titles = {c["clause_key"]: c.get("title") or "" for c in ((retr or {}).get("payload") or {}).get("clauses") or []}
        tau = dec.get("tau")
        snap = dec.get("evidence_snapshot") or []
        ab = []
        for e in snap:
            if e["grade"] in ("A", "B") and tau is not None and e["score"] >= tau and e["clause_key"] not in ab:
                ab.append(e["clause_key"])
        mods = ((state.get("approval_input") or {}).get("modifications") or {}) if state.get("modify_round", 0) > 0 else {}
        if mods.get("action_items"):
            items = mods["action_items"]
        else:
            cite = ab if ab else [snap[0]["clause_key"]] if snap else []
            items = [{"action_type": "inspect", "item": f"{titles.get(k, k)} 관련 계측·제어 루프 점검", "clause_key": k} for k in cite]
        version = int(state.get("draft_version") or 0) + 1
        base = ((state.get("draft") or {}).get("payload") or {}).get("proposal_id") if state.get("modify_round", 0) > 0 else None
        if not items:
            env = {"status": "REFUSED", "payload": None, "grounding": None, "state": None,
                   "reason": {"code": "NO_EVIDENCE", "message": "no scoped snapshot to cite"}, "trace": {"skill": "graph", "contract_version": "1.0"}}
            tr = {"tool": "work_order_draft", "step": K.k_draft(version), "status": "REFUSED", "reason": "NO_EVIDENCE", "replayed": False}
        else:
            env, tr = call(state, "work_order_draft", K.k_draft(version), alarm_id=det["state"]["alarm_id"], episode_id=state["episode_id"],
                           window_index=state["display_window"], policy=POLICY, decision_id=dec["decision_id"], action_items=items,
                           base_proposal_id=base)
        return {"draft": env, "draft_version": version, "trace": state["trace"] + [tr]}

    def approval_wait(state: AgentState) -> dict:
        """No side effect here (ST4-03): interrupt() re-runs this node on resume."""
        d = state["draft"]["payload"]
        ans = interrupt({"question": "작업지시 초안을 승인하시겠습니까? (approve / modify / reject)", "proposal_id": d["proposal_id"],
                         "draft_version": d["draft_version"], "content_sha256": d["content_sha256"], "form": d["form"],
                         "turn_id": state["turn_id"], "alarm_id": state.get("alarm_id")})
        return {"approval_input": ans if isinstance(ans, dict) else {"decision": str(ans)}}

    def approval_resume(state: AgentState) -> dict:
        """The ONLY call site of work_order_approve / work_order_issue (HUMAN_ONLY_TOOLS). Approver = DB owner."""
        guard(state)
        auth = current_auth()
        row = verify_secret(deps.dsn, state["thread_id"], auth.secret)       # re-verify against the DB before any token
        ai = state.get("approval_input") or {}
        decision = str(ai.get("decision", "")).strip().lower()
        if decision not in ("approve", "modify", "reject"):
            return _refuse("BAD_REQUEST", f"approval decision must be approve/modify/reject, got {decision!r}")
        v = int(state["draft_version"])
        d = state["draft"]["payload"]
        ap, tr = call(state, "work_order_approve", K.k_approve(v), d, row.owner, decision, (ai.get("modifications") or None))
        upd: dict = {"approval": ap, "trace": state["trace"] + [tr]}
        if decision == "approve" and ap["status"] == "OK" and ap["payload"].get("approval_token"):
            iss, tr2 = call(state, "work_order_issue", K.k_issue(v), d, ap["payload"]["approval_token"])
            upd["work_order"] = iss
            upd["trace"] = upd["trace"] + [tr2]
        if decision == "modify":
            upd["modify_round"] = int(state.get("modify_round") or 0) + 1
        return upd

    def history(state: AgentState) -> dict:
        guard(state)
        vers = ((state.get("detect") or {}).get("trace") or {}).get("versions") or {}
        h = _history_sql(deps.dsn, state["episode_id"], state["display_window"], vers.get("detector"), vers.get("contract_sha256"), state.get("alarm_id"))
        return {"history": h}

    def narrate(state: AgentState) -> dict:
        guard(state)
        text = deps.narrator.narrate(dict(state))
        return {"answer": text, "messages": [{"role": "assistant", "content": text, "turn_id": state["turn_id"]}]}

    # ------------------------------------------------------------------------------------------------- routing
    def after_router(s: AgentState) -> str:
        if s.get("refusal"):
            return "narrate"
        c = s.get("category")
        if c in ("out_of_scope", "methodology"):
            return "narrate"
        if c == "history":
            return "history"
        return "gate"

    def after_gate(s: AgentState) -> str:
        if s.get("refusal"):
            return "narrate"
        if (s["gate"].get("payload") or {}).get("gate") == "BLOCK":
            return "decide_initial" if s.get("category") == "action" else "narrate"
        return "detect"

    def after_detect(s: AgentState) -> str:
        if s.get("refusal"):
            return "narrate"
        c = s.get("category")
        if c == "evidence":
            return "retrieve_open"
        if c == "action":
            return "retrieve_scoped" if (s["detect"]["state"].get("alarm") is True) else "decide_initial"
        return "narrate"

    def after_retrieve_scoped(s: AgentState) -> str:
        return "narrate" if s.get("refusal") else "decide_initial"

    def after_decide_initial(s: AgentState) -> str:
        a = (s.get("decision") or {}).get("action")
        return "draft" if a == "propose" else ("query_more" if a == "query_more" else "narrate")

    def after_query_more(s: AgentState) -> str:
        return "narrate" if s.get("refusal") else "decide_after"

    def after_decide_after(s: AgentState) -> str:
        return "draft" if (s.get("decision_after") or {}).get("action") == "propose" else "narrate"

    def after_draft(s: AgentState) -> str:
        return "approval_wait" if (s.get("draft") or {}).get("status") == "OK" else "narrate"

    def after_resume(s: AgentState) -> str:
        ai = s.get("approval_input") or {}
        ok = (s.get("approval") or {}).get("status") == "OK"
        if str(ai.get("decision", "")).lower() == "modify" and ok and int(s.get("modify_round") or 0) <= deps.max_modify_rounds and not s.get("refusal"):
            return "draft"
        return "narrate"

    g = StateGraph(AgentState)
    for name, fn in [("load_context", load_context), ("router", router), ("gate", gate), ("detect", detect), ("retrieve_scoped", retrieve_scoped),
                     ("retrieve_open", retrieve_open), ("decide_initial", decide_initial), ("query_more", query_more), ("decide_after", decide_after),
                     ("draft", draft), ("approval_wait", approval_wait), ("approval_resume", approval_resume), ("history", history), ("narrate", narrate)]:
        g.add_node(name, fn)
    g.add_edge(START, "load_context")
    g.add_edge("load_context", "router")
    g.add_conditional_edges("router", after_router, {"narrate": "narrate", "history": "history", "gate": "gate"})
    g.add_conditional_edges("gate", after_gate, {"narrate": "narrate", "decide_initial": "decide_initial", "detect": "detect"})
    g.add_conditional_edges("detect", after_detect, {"narrate": "narrate", "retrieve_open": "retrieve_open", "retrieve_scoped": "retrieve_scoped", "decide_initial": "decide_initial"})
    g.add_edge("retrieve_open", "narrate")
    g.add_conditional_edges("retrieve_scoped", after_retrieve_scoped, {"narrate": "narrate", "decide_initial": "decide_initial"})
    g.add_conditional_edges("decide_initial", after_decide_initial, {"draft": "draft", "query_more": "query_more", "narrate": "narrate"})
    g.add_conditional_edges("query_more", after_query_more, {"narrate": "narrate", "decide_after": "decide_after"})
    g.add_conditional_edges("decide_after", after_decide_after, {"draft": "draft", "narrate": "narrate"})
    g.add_conditional_edges("draft", after_draft, {"approval_wait": "approval_wait", "narrate": "narrate"})
    g.add_edge("approval_wait", "approval_resume")
    g.add_conditional_edges("approval_resume", after_resume, {"draft": "draft", "narrate": "narrate"})
    g.add_edge("history", "narrate")
    g.add_edge("narrate", END)
    return g.compile(checkpointer=checkpointer)
