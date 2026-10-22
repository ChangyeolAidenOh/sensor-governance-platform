"""core_pipeline/agent/llm_router.py — hybrid router behind the Router protocol (ST4-22).

RuleRouter first; the LLM classifies ONLY when the rule result is the default fallback (matched == "default"). On every
failure outcome (refused reservation, unknown outcome, invalid output after one retry) the RULE result is returned
unchanged except for `matched`, so the deterministically extracted window / fault / proposal references and claims are
never lost. Re-execution of the same stage (lease recovery) walks the stored attempts in order: a stored success is
restored, a stored retryable failure moves on to the next attempt's stored state, a stored final failure falls back —
so a turn that succeeded on attempt 2 restores that result with zero new calls (L1). The LLM output is JSON text validated against ROUTER_SCHEMA — no tool use, no structured-output helper.
Rule and LLM extractions, the merged result and the validation / fallback reason are stored in the ledger row's output
(agent_steps.output), not in RouteResult: the graph and RouteResult are unchanged.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import re
from typing import Optional

import jsonschema

from core_pipeline.agent.llm_client import DEFAULT_MODEL, ROUTER_MAX_TOKENS, ROUTER_TIMEOUT_S, LLMClient, LLMOutcomeUnknown, \
    LLMRequest, LLMRequestRejected
from core_pipeline.agent.llm_ledger import LlmLedger
from core_pipeline.agent.router import CATEGORIES, RouteResult, RuleRouter

log = logging.getLogger("sgp.agent.llm")

TOOL = "router"
MAX_ATTEMPTS = 2                                   # one retry, only after a RECEIVED but invalid response
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)

ROUTER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["category"],
    "properties": {
        "category": {"enum": list(CATEGORIES)},
        "user_claims": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 20}, "maxItems": 20},
        "ref_window": {"type": ["integer", "null"], "minimum": 0, "maximum": 999},
        "ref_faults": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 20}, "maxItems": 20},
        "ref_proposal": {"type": ["string", "null"], "maxLength": 64},
        "rationale": {"type": "string", "maxLength": 400},
    },
}

SYSTEM_PROMPT = """You classify ONE user message sent to a process-reliability assistant for a chemical plant (TEP-like).
You do not answer the message and you call no tools. Output exactly one JSON object and nothing else (no markdown fence).

Categories (choose one):
- alarm: asks about the current window's alarm / detection status (p-value, drift, whether there is an alarm).
- diagnosis: asks which fault / cause it is, or asserts a fault as the diagnosis.
- evidence: asks for guideline clauses, documents, KOSHA guides, grounds, regulations.
- history: asks about previous windows, trends, counts, cumulative results, what happened so far.
- action: asks for a work order, a proposal, an action plan, an approval, something to be done or issued.
- methodology: asks how detection / diagnosis / thresholds / conformal / prefix-BH / tau are computed.
- out_of_scope: asks to change setpoints, bypass interlocks, open/close valves, emergency stop, shutdown, or anything
  outside reading results and proposing work orders.

Fields:
- user_claims: fault numbers (1..20) the user ASSERTS as the diagnosis (statements, not questions). Empty if none.
- ref_window: a window number the user refers to, else null.
- ref_faults: fault numbers (1..20) the user mentions (asserted or not). Empty if none.
- ref_proposal: a proposal / draft UUID the user refers to, else null.
- rationale: one short sentence.

Schema: {"category": string, "user_claims": [int], "ref_window": int|null, "ref_faults": [int], "ref_proposal": string|null, "rationale": string}"""


def _context_line(state: dict) -> str:
    draft = ((state.get("draft") or {}).get("payload") or {}).get("proposal_id")
    return (f"[state] display_window={state.get('display_window')} kind={state.get('kind')} "
            f"stored_alarm={'yes' if state.get('alarm_id') else 'no'} pending_draft={'yes' if draft else 'no'}")


def build_request(message: str, state: dict, model: str = DEFAULT_MODEL) -> LLMRequest:
    user = f"{_context_line(state)}\n[message]\n{message or ''}"
    return LLMRequest(model=model, system=SYSTEM_PROMPT, user=user, max_tokens=ROUTER_MAX_TOKENS, timeout_s=ROUTER_TIMEOUT_S)


def extract_json(text: str) -> Optional[dict]:
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_validate(text: str) -> tuple[Optional[dict], list[str]]:
    obj = extract_json(text)
    if obj is None:
        return None, ["no JSON object in the response"]
    errs = [e.message for e in jsonschema.Draft202012Validator(ROUTER_SCHEMA).iter_errors(obj)]
    return (obj if not errs else None), errs


def _ints(xs) -> list[int]:
    return sorted({int(x) for x in (xs or []) if isinstance(x, int) and 1 <= int(x) <= 20})


def merge(rule: RouteResult, llm: dict) -> RouteResult:
    """Category from the LLM; claims / faults = rule ∪ LLM (1..20); rule window and proposal references take precedence."""
    prop = rule.ref_proposal
    cand = llm.get("ref_proposal")
    if prop is None and isinstance(cand, str) and _UUID.match(cand.strip()):
        prop = cand.strip().lower()
    win = rule.ref_window if rule.ref_window is not None else (int(llm["ref_window"]) if isinstance(llm.get("ref_window"), int) else None)
    return RouteResult(category=llm["category"],
                       user_claims=sorted(set(rule.user_claims) | set(_ints(llm.get("user_claims")))),
                       ref_window=win,
                       ref_faults=sorted(set(rule.ref_faults) | set(_ints(llm.get("ref_faults")))),
                       ref_proposal=prop, matched="llm")


def _from_merged(d: dict) -> RouteResult:
    return RouteResult(category=d["category"], user_claims=list(d.get("user_claims") or []), ref_window=d.get("ref_window"),
                       ref_faults=list(d.get("ref_faults") or []), ref_proposal=d.get("ref_proposal"), matched="llm")


class LLMRouter:
    def __init__(self, client: LLMClient, ledger: LlmLedger, rule: Optional[RuleRouter] = None, model: str = DEFAULT_MODEL) -> None:
        self.client, self.ledger, self.rule, self.model = client, ledger, rule or RuleRouter(), model

    @staticmethod
    def stage_key(state: dict) -> str:
        return f"llm:router:{state.get('turn_id')}"

    def route(self, message: str, state: dict) -> RouteResult:
        rr = self.rule.route(message, state)
        if rr.matched != "default":
            return rr                                           # deterministic turn: no call, no ledger row, $0
        req = build_request(message, state, self.model)
        rule_extract = dataclasses.asdict(rr)
        thread, run = str(state.get("thread_id")), str(state.get("run_uuid"))
        for attempt in range(1, MAX_ATTEMPTS + 1):
            res = self.ledger.reserve(thread_id=thread, run_uuid=run, stage_key=self.stage_key(state), attempt=attempt, tool=TOOL, req=req)
            if res.status == "replay":
                out = res.output or {}
                if out.get("merged"):
                    return _from_merged(out["merged"])                   # stored success: restored, no call
                if out.get("fallback_reason") == "invalid_output":
                    continue                                             # stored retryable failure: look at the next attempt
                return dataclasses.replace(rr, matched="llm_fallback:" + str(out.get("fallback_reason") or "replayed").split(":")[0])
            if res.status != "owned":
                log.warning("llm router reservation refused (%s): %s", res.status, res.reason)
                return dataclasses.replace(rr, matched=f"llm_refused:{res.status}")
            try:
                resp = self.client.complete(req)
            except LLMOutcomeUnknown as e:
                self.ledger.note_no_response(res.request_id, str(e))        # PENDING stays: billing unknown, no re-call
                return dataclasses.replace(rr, matched="llm_outcome_unknown")
            except LLMRequestRejected as e:
                self.ledger.settle(res.request_id, None, req=req, output={"tool": TOOL, "rule_extract": rule_extract,
                                   "fallback_reason": f"rejected: {e}"}, violation=False)
                return dataclasses.replace(rr, matched="llm_fallback:rejected")
            parsed, errs = parse_validate(resp.text)
            merged = merge(rr, parsed) if parsed else None
            out = {"tool": TOOL, "attempt": attempt, "rule_extract": rule_extract, "llm_parsed": parsed, "validation_errors": errs,
                   "merged": (dataclasses.asdict(merged) if merged else None),
                   "fallback_reason": (None if merged else "invalid_output")}
            self.ledger.settle(res.request_id, resp, req=req, output=out, violation=bool(errs),
                               reason_code=("LLM_OUTPUT_INVALID" if errs else None))
            if merged:
                return merged
            log.warning("llm router output invalid (attempt %d): %s", attempt, errs)
        return dataclasses.replace(rr, matched="llm_fallback:invalid_output")
