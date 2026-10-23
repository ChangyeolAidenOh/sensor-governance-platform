"""core_pipeline/agent/router.py — category routing (ST4-15).

The LLM never calls tools. A Router only classifies the user message into one of the 7 pre-registered categories and
extracts (a) user_claims — fault numbers the user ASSERTS as a diagnosis (contract §1.5 / §3.2: stored in state, never a
Skill input) and (b) refs — windows / faults / proposals the user REFERS to. The graph compares refs with its own state:
a mismatching ref is discarded and counted (refs_discarded, the new report metric; the pre-registered "LLM tool-argument
discard count" is structurally N/A under ST4-15). S4a-04 ships only the deterministic RuleRouter ($0); S4a-05 adds the
claude-haiku-5-5 router behind the same Protocol (hybrid: rules first, LLM for the rest).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

CATEGORIES = ("alarm", "diagnosis", "evidence", "history", "action", "methodology", "out_of_scope")


@dataclass(frozen=True)
class RouteResult:
    category: str
    user_claims: list[int] = field(default_factory=list)
    ref_window: Optional[int] = None
    ref_faults: list[int] = field(default_factory=list)
    ref_proposal: Optional[str] = None
    matched: str = ""                    # which rule fired (for the trace)


class Router(Protocol):
    def route(self, message: str, state: dict) -> RouteResult: ...


_OOS = re.compile(r"(설정값|set\s*point|setpoint|인터록|interlock|bypass|우회|긴급\s*정지|비상\s*정지|emergency\s*stop|shutdown|밸브를?\s*(열|닫)|open the valve|close the valve)", re.I)
_ACTION = re.compile(r"(작업\s*지시|작업지시|조치\s*안|조치안|제안|승인|work\s*order|propose|proposal|조치해|조치를|처리해|발행)", re.I)
_EVIDENCE = re.compile(r"(조항|근거|지침|clause|guide|kosha|문서|규정|어떤 지침)", re.I)
_HISTORY = re.compile(r"(이력|지난\s*창|이전\s*창|지금까지|몇\s*개|몇\s*번|history|전체\s*창|창\s*별|추이|누적)", re.I)
_METHOD = re.compile(r"(어떻게\s*계산|어떻게\s*판단|방법론|방법이|conformal|prefix[- ]?bh|p[- ]?value가\s*뭐|τ|tau|임계|기준이\s*뭐|원리|알고리즘|methodology|how (is|are|do you) (compute|decide|calculate)"
                     r"|(?:다는|라는|란|는\s*게|라는\s*게)\s*(?:뜻|의미)|무슨\s*(?:뜻|의미)|(?:뜻|의미)(?:이야|인가|인지|야|인데|이지))", re.I)
# ST4-42: the last three alternatives detect a question about what a statement MEANS (interpretation / methodology);
# '%' and numbers are never a signal on their own. Precedence is unchanged (METHOD was already evaluated before DIAG).
_DIAG = re.compile(r"(결함|진단|fault|원인|어떤\s*고장|무슨\s*문제|diagnos)", re.I)
# ST4-43: a claim is an ASSERTION that fault N is the diagnosis. An assertive ending counts only when it is not followed by
# a question mark, a quotative/conjunctive 고 (…다고 볼/단정/생각…) or a negation (…지 않); questions and hedges are not claims.
_NOT_ASSERTED = r"(?!\s*[?？]|\s*고(?:\b|[\s,.?？]|\s*(?:볼|보|생각|단정|가정|할|하|해|말|치|여|여기))|\s*지\s*않)"
_CLAIM = re.compile(
    r"(?:결함|fault)\s*(\d{1,2})\s*(?:번|호)?\s*(?:"
    r"(?:으로|로)\s*(?:진단|확인|판정|밝혀)|이다" + _NOT_ASSERTED + r"|입니다" + _NOT_ASSERTED + r"|임\b" + _NOT_ASSERTED +
    r"|(?:이|가|은|는)?\s*확실(?:하니까|하니|하다|하므로|해서|하고|합니다|함|해)" + _NOT_ASSERTED +
    r"|(?:이|가|은|는)?\s*맞(?:으니까|으니|다|아서|으므로|습니다|음|아)" + _NOT_ASSERTED +
    r"|(?:이)?라고(?!\s*(?:[?？]|보|생각|볼|가정|치|해도|하면|할|말|단정))"
    r"|is the fault|was diagnosed|diagnosed as)", re.I)
_FAULT_REF = re.compile(r"(?:결함|fault)\s*(\d{1,2})", re.I)
_WINDOW_REF = re.compile(r"(?:창|window|윈도우)\s*(\d{1,2})", re.I)
_PROPOSAL_REF = re.compile(r"(?:proposal|제안서|초안)\s*[#:]?\s*([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I)


class RuleRouter:
    """Deterministic keyword router. Order = precedence (first match wins); default = alarm."""

    def route(self, message: str, state: dict) -> RouteResult:
        m = message or ""
        claims = sorted({int(x) for x in _CLAIM.findall(m) if 1 <= int(x) <= 20})
        faults = sorted({int(x) for x in _FAULT_REF.findall(m) if 1 <= int(x) <= 20})
        w = _WINDOW_REF.search(m)
        p = _PROPOSAL_REF.search(m)
        ref_window = int(w.group(1)) if w else None
        ref_proposal = p.group(1) if p else None
        if _OOS.search(m):
            cat, rule = "out_of_scope", "oos"
        elif _ACTION.search(m):
            cat, rule = "action", "action"
        elif _METHOD.search(m):
            cat, rule = "methodology", "method"
        elif _HISTORY.search(m):
            cat, rule = "history", "history"
        elif _EVIDENCE.search(m):
            cat, rule = "evidence", "evidence"
        elif _DIAG.search(m) or claims:
            cat, rule = "diagnosis", "diagnosis"
        else:
            cat, rule = "alarm", "default"
        return RouteResult(cat, claims, ref_window, faults, ref_proposal, rule)
