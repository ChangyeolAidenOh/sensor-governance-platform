"""core_pipeline/agent/llm_narrator.py — LLM narrator behind the Narrator protocol (ST4-23).

Answer = ① the deterministic TemplateNarrator text (window numbers, decision label + rule, query / draft / approval /
work-order status, clause blocks quoted verbatim from the retrieve payload) + ② "요약(LLM 생성)": a short prose summary
written by the LLM from a FACT SHEET built from the structured State with an allow-list of fields (the same helpers
that render ①, plus clause key / grade / score / title — never `clauses[].text`, however many lines it has). The LLM
never sees clause text, never decides, and its paragraph is appended after the fixed block — it cannot replace a
number or a decision. Re-execution of the same narration stage restores a stored attempt-2 success (zero new calls).

Auxiliary checks on ② (fabrication filters, NOT a semantic guarantee): every numeric literal in ② must occur in the
fact sheet; every clause-key-like token (P-…) in ② must occur in the fact sheet; ② must be non-empty prose. A failed
check drops ② (the fixed block is still returned) and settles the ledger row with violation=true.

Modes: "llm" — the LLM writes ② only on turns the LLM router classified (route.matched == "llm"), so turns handled by
the deterministic router stay at $0; "llm_all" — ② on every turn of an allowed category (explicit cost option).
Fixed in every mode (no call): refusals, out_of_scope, methodology, history.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from core_pipeline.agent.llm_client import DEFAULT_MODEL, NARRATE_MAX_TOKENS, NARRATE_TIMEOUT_S, LLMClient, LLMOutcomeUnknown, \
    LLMRequest, LLMRequestRejected
from core_pipeline.agent.llm_ledger import LlmLedger
from core_pipeline.agent.narrator import ACTION_KO, REASON_KO, TemplateNarrator, _claims_contrast, _fmt_p, _window_summary

log = logging.getLogger("sgp.agent.llm")

TOOL = "narrate"
MODES = ("llm", "llm_all")
LLM_CATEGORIES = ("alarm", "diagnosis", "evidence", "action")
MAX_ATTEMPTS = 2
PROSE_HEADER = "요약(LLM 생성):"
OMITTED_NOTE = "(LLM 요약 생략: 보조 검사 위반)"
_NUM = re.compile(r"(?<![\d.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\d.])")      # particles may follow a number (0.42이며)
_CLAUSE_TOKEN = re.compile(r"P-\d+[^\s\]\),。]*")

SYSTEM_PROMPT = """You write a short Korean summary (3 to 6 plain sentences, no headings, no markdown, no lists) of a FACT SHEET
produced by a process-reliability assistant, for a plant engineer. Hard rules:
1. Use only facts in the sheet. Never introduce a number that is not written in the sheet (no percentages, counts or
   thresholds of your own); copy numbers exactly as written.
2. Never quote, paraphrase or invent guideline clause text. You may mention a clause only by the key exactly as written
   in the sheet (e.g. [P-116-2012:…]).
3. Do not state or imply an approval, a work order, a proposal or a decision that the sheet does not state.
4. Do not change the decision: if the sheet says 보류 it is 보류.
5. Say that user-asserted faults are not used as input when the sheet contrasts them.
Output the summary text only."""


def _clause_meta(env: Optional[dict], limit: int = 5) -> str:
    if not env or env.get("status") != "OK":
        return ""
    g = env.get("grounding") or {}
    lines = [f"검색 mode={g.get('mode')}, τ={g.get('tau')}, g_AB={_fmt_p(g.get('g_AB'))}, g_C={_fmt_p(g.get('g_C'))}, ev_AB={g.get('ev_AB')}"]
    for c in ((env.get("payload") or {}).get("clauses") or [])[:limit]:
        lines.append(f"[{c.get('clause_key')}] ({c.get('grade')}, score {float(c.get('score') or 0):.3f}) {c.get('title') or ''}")
    return "\n".join(lines)


def fact_sheet(state: dict) -> str:
    """Allow-listed facts from the structured State. Clause bodies (`clauses[].text`) are never read."""
    cat = state.get("category")
    parts: list[str] = [f"범주: {cat}"]
    if state.get("turn_refs_discarded"):
        parts.append(f"무시한 참조: {state.get('turn_refs_note')} (선택 창 기준으로 답함)")
    ws = _window_summary(state)
    if ws:
        parts.append(ws)
    cc = _claims_contrast(state)
    if cc:
        parts.append(cc)
    if cat == "evidence":
        parts.append(_clause_meta(state.get("retrieve")))
        parts.append("open 모드 결과는 참고용이며 제안의 근거(ev_AB)가 될 수 없습니다.")
    if cat in ("alarm", "diagnosis") and not (((state.get("detect") or {}).get("state") or {}).get("alarm")):
        parts.append("이 창에서는 prefix-BH 알람이 없습니다.")
    if cat == "action":
        dec = state.get("decision_after") or state.get("decision") or {}
        if dec:
            parts.append(f"정책 결정(E): {ACTION_KO.get(dec.get('action'), dec.get('action'))} — 규칙 {dec.get('rule')} ({dec.get('reason')})")
        q = state.get("query") or {}
        if q:
            parts.append(f"query 상태: {q.get('status')} (expand={q.get('expand_status')}, rerank={q.get('rerank_status')})")
        retr = state.get("retrieve_k10") or state.get("retrieve")
        if retr:
            parts.append(_clause_meta(retr))
        d = state.get("draft") or {}
        if d.get("status") == "OK":
            pl = d.get("payload") or {}
            parts.append(f"초안 {pl.get('proposal_id')} v{pl.get('draft_version')} 생성")
        elif d:
            parts.append("초안 거부: " + REASON_KO.get((d.get("reason") or {}).get("code"), str((d.get("reason") or {}).get("code", ""))))
        ap = state.get("approval") or {}
        if ap:
            parts.append(f"승인 결과: {(ap.get('payload') or {}).get('decision')}" if ap.get("status") == "OK"
                         else "승인 거부: " + REASON_KO.get((ap.get("reason") or {}).get("code"), str((ap.get("reason") or {}).get("code"))))
        wo = state.get("work_order") or {}
        if wo:
            parts.append(f"작업지시 발행: {(wo.get('payload') or {}).get('work_order_id')}" if wo.get("status") == "OK"
                         else "발행 거부: " + REASON_KO.get((wo.get("reason") or {}).get("code"), str((wo.get("reason") or {}).get("code"))))
        if dec.get("action") == "no_alarm":
            parts.append("선택 가능한 알람이 없어 조치안 경로로 진입하지 않습니다.")
    return "\n".join(p for p in parts if p)


def numbers_of(text: str) -> set[str]:
    out = set()
    for m in _NUM.findall(text or ""):
        try:
            out.add(repr(float(m)))
        except ValueError:
            pass
    return out


def check_prose(prose: str, sheet: str) -> list[str]:
    errs: list[str] = []
    p = (prose or "").strip()
    if len(p) < 10:
        errs.append("empty or too short")
    extra = numbers_of(p) - numbers_of(sheet)
    if extra:
        errs.append("numbers not in the fact sheet: " + ", ".join(sorted(extra)))
    for tok in set(_CLAUSE_TOKEN.findall(p)):
        if tok not in sheet:
            errs.append(f"clause key not in the fact sheet: {tok}")
    if p.lstrip().startswith(("#", "-", "*", "|")):
        errs.append("markdown/list formatting")
    return errs


def stage_key(state: dict) -> str:
    """Logical narration stage: turn + draft version + modify round + approval outcome + work-order outcome."""
    ap = state.get("approval") or {}
    ai = state.get("approval_input") or {}
    ap_part = "-" if not ap else f"{str(ai.get('decision', '?')).lower()}/{ap.get('status')}" + (
        f":{((ap.get('reason') or {}).get('code'))}" if ap.get("status") != "OK" else "")
    wo = state.get("work_order") or {}
    wo_part = "-" if not wo else (str((wo.get("payload") or {}).get("work_order_id")) if wo.get("status") == "OK"
                                   else f"REFUSED:{((wo.get('reason') or {}).get('code'))}")
    return (f"llm:narrate:{state.get('turn_id')}:d{int(state.get('draft_version') or 0)}:m{int(state.get('modify_round') or 0)}"
            f":ap={ap_part}:wo={wo_part}")


def build_request(sheet: str, model: str = DEFAULT_MODEL) -> LLMRequest:
    return LLMRequest(model=model, system=SYSTEM_PROMPT, user="[FACT SHEET]\n" + sheet, max_tokens=NARRATE_MAX_TOKENS, timeout_s=NARRATE_TIMEOUT_S)


class LLMNarrator:
    def __init__(self, client: LLMClient, ledger: LlmLedger, template: Optional[TemplateNarrator] = None, mode: str = "llm",
                 model: str = DEFAULT_MODEL) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.client, self.ledger, self.template, self.mode, self.model = client, ledger, template or TemplateNarrator(), mode, model

    def wants_llm(self, state: dict) -> bool:
        if state.get("refusal") or state.get("category") not in LLM_CATEGORIES:
            return False
        if self.mode == "llm_all":
            return True
        return (state.get("route") or {}).get("matched") == "llm"

    def narrate(self, state: dict) -> str:
        fixed = self.template.narrate(state)
        if not self.wants_llm(state):
            return fixed
        sheet = fact_sheet(state)
        req = build_request(sheet, self.model)
        thread, run, key = str(state.get("thread_id")), str(state.get("run_uuid")), stage_key(state)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            res = self.ledger.reserve(thread_id=thread, run_uuid=run, stage_key=key, attempt=attempt, tool=TOOL, req=req)
            if res.status == "replay":
                out = res.output or {}
                if out.get("prose"):
                    return self._compose(fixed, out["prose"], replayed=True)      # stored success: restored, no call
                if out.get("fallback_reason") == "check_failed":
                    continue                                                       # stored retryable failure: next attempt
                return self._compose(fixed, None, note="(LLM 요약 없음: 요청 거부)")
            if res.status != "owned":
                log.warning("llm narrate reservation refused (%s): %s", res.status, res.reason)
                return self._compose(fixed, None, note=f"(LLM 요약 없음: {res.status})")
            try:
                resp = self.client.complete(req)
            except LLMOutcomeUnknown as e:
                self.ledger.note_no_response(res.request_id, str(e))
                return self._compose(fixed, None, note="(LLM 요약 없음: 결과 미확정)")
            except LLMRequestRejected as e:
                self.ledger.settle(res.request_id, None, req=req, output={"tool": TOOL, "fallback_reason": f"rejected: {e}"})
                return self._compose(fixed, None, note="(LLM 요약 없음: 요청 거부)")
            prose = (resp.text or "").strip()
            errs = check_prose(prose, sheet)
            out = {"tool": TOOL, "attempt": attempt, "stage_key": key, "prose": (prose if not errs else None), "rejected_prose": (prose if errs else None),
                   "check_errors": errs, "fallback_reason": ("check_failed" if errs else None)}
            self.ledger.settle(res.request_id, resp, req=req, output=out, violation=bool(errs), reason_code=("LLM_OUTPUT_INVALID" if errs else None))
            if not errs:
                return self._compose(fixed, prose)
            log.warning("llm narrate check failed (attempt %d): %s", attempt, errs)
        return self._compose(fixed, None, note=OMITTED_NOTE)

    @staticmethod
    def _compose(fixed: str, prose: Optional[str], note: Optional[str] = None, replayed: bool = False) -> str:
        if prose:
            return f"{fixed}\n\n{PROSE_HEADER} {prose}"
        return f"{fixed}\n\n{note or OMITTED_NOTE}"
