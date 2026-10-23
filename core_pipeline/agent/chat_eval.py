"""core_pipeline/agent/chat_eval.py — S4a-07 chat dev-set evaluation (ST4-32..35): loader, recording wrappers, scoring.

Nothing here touches the graph, the runner or the checkpointer. Observation comes from the four GraphDeps injection
points only — router, narrator, skills, record_decision — each wrapped by a pass-through recorder (delegate unchanged,
record allow-listed fields). The scoring functions are pure (no DB) so tests/unit covers them; scripts/chat_dev_run.py
does the DB reads and feeds `score_item()` an observation dict.

Failure taxonomy (reviewer 2026-10-22/23, ST4-33):
  context_prep_failed   — pre-run state/version/evidence mismatch (reset verification), scores N/A
  context_contaminated  — a state change during the run not explained by this run's own turns; kept, excluded from
                          mode comparison, counted in the summary
  router / skill_error / contract / error — failures of the evaluated system, classified from what was actually
                          observed (category, Skill envelopes, decision records), never guessed
Narration checks are two layers: ① template-shape / state-contrast (hard; a template line is present iff narrator.py's
own branch condition holds, so a correct "발행 거부:" is never a failure) and ② review flags (words that only mark
the answer for human review; occurrence itself is not a failure). Values the harness could not observe are reported
as "unobserved" — never estimated into a pass.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from core_pipeline.agent import keys as K
from core_pipeline.agent.narrator import METHODOLOGY_KO, REASON_KO, Narrator, TemplateNarrator
from core_pipeline.agent.router import CATEGORIES, RouteResult, Router

DEV_SET_PATH = "evals/chat_dev_set.json"
SCHEMA = "chat_dev_set/1"
ALLOWED_PRESETS = (1, 2, 7, 8)
HUMAN_ONLY_TOOLS = ("work_order_approve", "work_order_issue")
PROSE_HEADER = "요약(LLM 생성):"
OMITTED_PREFIX = "(LLM 요약"
LLM_SENT = ("llm", "llm_fallback:invalid_output", "llm_outcome_unknown")        # an HTTP request went out
LLM_NOT_SENT_PREFIXES = ("llm_refused:", "llm_fallback:rejected")             # reservation refused / client refused before sending

# positional parameter names of the SkillClient methods (client.py) — used to bind *args for the record
SKILL_PARAMS = {
    "gate_check": ("episode_id", "window_index"),
    "detect_diagnose": ("episode_id", "window_index", "expand", "query_id"),
    "spend_query": ("episode_id", "window_index"),
    "retrieve": ("query", "mode", "fault_set", "k", "alarm_id", "query_id"),
    "work_order_draft": (),
    "work_order_approve": ("draft", "approver", "decision", "modifications"),
    "work_order_issue": ("draft", "approval_token"),
    "work_order_status": ("proposal_id",),
}
# payload keys kept per tool (everything else dropped; retrieve clauses keep key/grade/score/title, never text)
PAYLOAD_KEEP = {
    "gate_check": ("gate", "drift"),
    "detect_diagnose": ("alarm_id",),
    "spend_query": ("query_id", "expand_allowed"),
    "retrieve": ("reused", "query"),
    "work_order_draft": ("proposal_id", "draft_version", "content_sha256"),
    "work_order_approve": ("decision",),
    "work_order_issue": ("work_order_id", "cmms_ref", "idempotent_replay"),
    "work_order_status": ("proposal_id", "status"),
}
SECRET_COLUMN_MARKERS = ("token", "secret")


# ------------------------------------------------------------------------------------------------- dev set
def load_dev_set(path: str = DEV_SET_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_dev_set(ds: dict) -> list[str]:
    errs: list[str] = []
    meta, items = ds.get("_meta") or {}, ds.get("items") or []
    if meta.get("schema") != SCHEMA:
        errs.append(f"schema must be {SCHEMA}")
    if len(items) != int(meta.get("n", -1)):
        errs.append("n != len(items)")
    ids = [it.get("id") for it in items]
    if len(set(ids)) != len(ids):
        errs.append("duplicate ids")
    for it in items:
        i = it.get("id", "?")
        exp = it.get("expected") or {}
        if exp.get("category") not in CATEGORIES:
            errs.append(f"{i}: bad category")
        for f in ("user_claims", "ref_faults"):
            if not isinstance(exp.get(f), list) or any(not (isinstance(x, int) and 1 <= x <= 20) for x in exp.get(f, [])):
                errs.append(f"{i}: {f} must be a list of 1..20")
        ctx = it.get("context") or {}
        if ctx.get("preset") not in ALLOWED_PRESETS:
            errs.append(f"{i}: preset not designated")
        con = it.get("contract") or {}
        if not con.get("status_in"):
            errs.append(f"{i}: status_in missing")
        for s in con.get("skill_calls", []):
            for ph in re.findall(r"\{(\w+)\}", s):
                if ph != "dw":
                    errs.append(f"{i}: unknown placeholder {ph}")
        if any(t in con.get("skill_calls", []) for t in HUMAN_ONLY_TOOLS):
            errs.append(f"{i}: human-only tool in required calls")
        nar = it.get("narration") or {}
        if nar.get("fixed_response") not in (None, "methodology", "out_of_scope", "refusal"):
            errs.append(f"{i}: bad fixed_response")
    return errs


def resolve(spec: str, dw: int) -> str:
    return spec.replace("{dw}", str(int(dw)))


# ------------------------------------------------------------------------------------------------- step keys
def step_candidates(dw: int, max_version: int = 3) -> list[str]:
    t = int(dw)
    out = [K.k_gate(t), K.k_spend(t), K.k_retrieve("scoped", 5), K.k_retrieve("scoped", 10), K.k_retrieve("open", 5)]
    for x in (1, 2):
        for ph in ("initial", "after_query"):
            out.append(K.k_detect(t, x, ph))
    for v in range(1, max_version + 1):
        out += [K.k_draft(v), K.k_approve(v), K.k_issue(v)]
    return out


def request_map(thread_id: str, run_uuid: str, dw: int) -> dict[str, str]:
    """request_id -> step key for every step the graph can issue on this thread/run (keys.request_id is deterministic)."""
    return {K.request_id(thread_id, str(run_uuid), s): s for s in step_candidates(dw)}


# ------------------------------------------------------------------------------------------------- views
def strip_secret_columns(row: dict) -> dict:
    """Backup / raw-record rows keep allow-listed columns only: any column whose name carries token/secret is dropped."""
    return {k: v for k, v in (row or {}).items() if not any(m in str(k).lower() for m in SECRET_COLUMN_MARKERS)}


def clauses_meta(payload: Optional[dict]) -> list[dict]:
    return [{"clause_key": c.get("clause_key"), "grade": c.get("grade"), "score": c.get("score"), "title": c.get("title")}
            for c in ((payload or {}).get("clauses") or [])]


def envelope_view(tool: str, env: dict) -> dict:
    env = env or {}
    pl = env.get("payload") or {}
    keep = {k: pl.get(k) for k in PAYLOAD_KEEP.get(tool, ()) if k in pl}
    if tool == "retrieve":
        keep["clauses"] = clauses_meta(pl)
    if tool == "work_order_draft" and isinstance(pl.get("form"), dict):
        f = pl["form"]
        keep["form"] = {k: f.get(k) for k in ("alarm_id", "window_index", "policy", "decision_id", "evidence_keys", "action_items") if k in f}
    return {"status": env.get("status"), "reason": copy.deepcopy(env.get("reason")), "state": copy.deepcopy(env.get("state")),
            "grounding": copy.deepcopy(env.get("grounding")), "payload": keep,
            "replayed": bool((env.get("trace") or {}).get("replayed"))}


def args_view(tool: str, args: tuple, kw: dict) -> dict:
    names = SKILL_PARAMS.get(tool, ())
    d = dict(zip(names, args))
    d.update(kw)
    out: dict = {}
    for k, v in d.items():
        if k in ("draft",):
            v = {kk: (v or {}).get(kk) for kk in ("proposal_id", "draft_version", "content_sha256")} if isinstance(v, dict) else v
        if k == "approval_token":
            v = "<present>" if v else None
        if k == "modifications":
            v = {"action_items": [str(x.get("item"))[:80] for x in ((v or {}).get("action_items") or [])]} if isinstance(v, dict) else v
        out[k] = copy.deepcopy(v)
    return out


def state_snapshot(state: dict) -> dict:
    """Allow-listed view of the graph State at narrate time. No messages, no secrets, no clause text, no token."""
    s = state or {}
    def env(name: str) -> Optional[dict]:
        e = s.get(name)
        if not e:
            return None
        tool = {"gate": "gate_check", "detect": "detect_diagnose", "detect_after": "detect_diagnose", "detect_x": "detect_diagnose",
                "retrieve": "retrieve", "retrieve_k10": "retrieve", "draft": "work_order_draft", "approval": "work_order_approve",
                "work_order": "work_order_issue"}[name]
        return envelope_view(tool, e)
    out = {k: copy.deepcopy(s.get(k)) for k in ("category", "route", "user_claims", "refusal", "turn_refs_discarded", "turn_refs_note",
                                                "display_window", "kind", "episode_id", "alarm_id", "turn_id", "thread_id", "run_uuid",
                                                "draft_version", "modify_round", "decision", "decision_after", "query", "history", "trace")}
    for name in ("gate", "detect", "detect_after", "detect_x", "retrieve", "retrieve_k10", "draft", "approval", "work_order"):
        out[name] = env(name)
    return out


# ------------------------------------------------------------------------------------------------- recorder
class Recorder:
    """One per run. `begin(item_id, thread_id)` opens the current item; wrappers append to it (serial execution)."""

    def __init__(self) -> None:
        self.items: dict[str, dict] = {}
        self.current: Optional[str] = None
        self.request_maps: dict[tuple[str, str], dict[str, str]] = {}
        self._dw: dict[str, int] = {}

    def begin(self, item_id: str, thread_id: str, dw: int) -> None:
        self.current = item_id
        self._dw[thread_id] = int(dw)
        self.items[item_id] = {"thread_id": thread_id, "route": None, "skills": [], "decisions": [], "narration": None}

    def cur(self) -> dict:
        if self.current is None:
            raise RuntimeError("Recorder: no item begun")
        return self.items[self.current]

    def step_of(self, ctx: dict) -> Optional[str]:
        thread, run = str(ctx.get("thread_id")), str(ctx.get("run_uuid"))
        key = (thread, run)
        if key not in self.request_maps:
            dw = self._dw.get(thread)
            self.request_maps[key] = request_map(thread, run, dw) if dw is not None else {}
        return self.request_maps[key].get(str(ctx.get("request_id")))


class RecordingRouter:
    def __init__(self, inner: Router, rec: Recorder) -> None:
        self.inner, self.rec = inner, rec

    def route(self, message: str, state: dict) -> RouteResult:
        rr = self.inner.route(message, state)
        self.rec.cur()["route"] = {"turn_id": state.get("turn_id"), "message": message, "display_window": state.get("display_window"),
                                   "kind": state.get("kind"), "stored_alarm": bool(state.get("alarm_id")), "result": asdict(rr)}
        return rr


class RecordingNarrator:
    def __init__(self, inner: Narrator, rec: Recorder) -> None:
        self.inner, self.rec = inner, rec

    def narrate(self, state: dict) -> str:
        text = self.inner.narrate(state)
        self.rec.cur()["narration"] = {"snapshot": state_snapshot(state), "answer": text}
        return text


class RecordingSkills:
    """Pass-through around the SyncSkillClient (same __getattr__ delegation as the test double GroundingOverride, but
    nothing is rewritten): every Skill call is recorded in order with its resolved step key, bound arguments and an
    allow-listed envelope view."""

    def __init__(self, inner: Any, rec: Recorder) -> None:
        self._inner, self._rec = inner, rec

    def __getattr__(self, name: str):
        fn = getattr(self._inner, name)
        if name not in SKILL_PARAMS:
            return fn

        def call(ctx: dict, *args, **kw):
            env = fn(ctx, *args, **kw)
            rec = self._rec.cur()
            rec["skills"].append({"seq": len(rec["skills"]) + 1, "tool": name, "request_id": ctx.get("request_id"),
                                  "step": self._rec.step_of(ctx), "args": args_view(name, args, kw), **envelope_view(name, env)})
            return env
        return call


def recording_decision(inner: Callable[..., Any], rec: Recorder) -> Callable[..., Any]:
    """Wraps core_pipeline.skills.work_order.record_decision (same signature; delegates, then records)."""
    def record_decision(dsn: str, alarm_id: str, policy: str, signals: dict, action: str, rule: int, evidence=None, decision_key=None):
        try:
            did = inner(dsn, alarm_id, policy, signals, action, rule, evidence=evidence, decision_key=decision_key)
        except Exception as e:                                               # recorded, then re-raised unchanged
            rec.cur()["decisions"].append({"alarm_id": alarm_id, "policy": policy, "action": action, "rule": rule, "decision_key": decision_key,
                                           "evidence_n": len(evidence or []), "decision_id": None, "error": f"{type(e).__name__}: {e}"[:300]})
            raise
        rec.cur()["decisions"].append({"alarm_id": alarm_id, "policy": policy, "action": action, "rule": rule, "decision_key": decision_key,
                                       "phase": (decision_key or "").rsplit(":", 1)[-1] or None, "evidence_n": len(evidence or []),
                                       "signals": copy.deepcopy(signals), "evidence": copy.deepcopy(evidence or []), "decision_id": str(did)})
        return did
    return record_decision


# ------------------------------------------------------------------------------------------------- narration checks
def split_answer(answer: Optional[str]) -> tuple[str, Optional[str], Optional[str]]:
    """(fixed template part, LLM prose or None, omission note or None)."""
    a = answer or ""
    i = a.find("\n\n" + PROSE_HEADER)
    if i >= 0:
        return a[:i], a[i + 2 + len(PROSE_HEADER):].strip(), None
    j = a.find("\n\n" + OMITTED_PREFIX)
    if j >= 0:
        return a[:j], None, a[j + 2:].strip()
    return a, None, None


def _ok(env: Optional[dict]) -> bool:
    return bool(env) and env.get("status") == "OK"


def template_line_expectations(snap: dict) -> dict[str, bool]:
    """Template line -> whether narrator.py's own branch emits it for this State (status / decision / payload based)."""
    cat = snap.get("category")
    dec = snap.get("decision_after") or snap.get("decision") or {}
    det = ((snap.get("detect") or {}).get("state")) or {}
    draft, ap, wo = snap.get("draft"), snap.get("approval"), snap.get("work_order")
    retr = snap.get("retrieve_k10") or snap.get("retrieve")                 # narrator.py: `retrieve_k10 or retrieve`, then status check inside
    exp = {
        "요청한 참조 ": bool(snap.get("turn_refs_discarded")),
        "사용자 주장 결함 ": bool(snap.get("user_claims")) and cat not in ("out_of_scope", "methodology", "history"),
        "이 창에서는 prefix-BH 알람이 없습니다.": cat in ("alarm", "diagnosis") and not det.get("alarm"),
        "정책 결정(E):": cat == "action" and bool(dec),
        "초기 결정(E):": cat == "action" and bool(snap.get("decision_after")) and bool(snap.get("decision")),
        "query 상태:": cat == "action" and bool(snap.get("query")),
        "검색 mode=": (cat == "action" and _ok(retr)) or (cat == "evidence" and _ok(snap.get("retrieve"))),
        "open 모드 결과는 참고용": cat == "evidence",
        " 생성(sha ": cat == "action" and _ok(draft),
        "초안 거부:": cat == "action" and bool(draft) and not _ok(draft),
        "승인 결과:": cat == "action" and _ok(ap),
        "승인 거부:": cat == "action" and bool(ap) and not _ok(ap),
        "작업지시 발행:": cat == "action" and _ok(wo),
        "발행 거부:": cat == "action" and bool(wo) and not _ok(wo),
        "선택 가능한 알람이 없어": cat == "action" and dec.get("action") == "no_alarm",
    }
    if snap.get("refusal") or cat in ("out_of_scope", "methodology", "history"):
        return {}                                                            # fixed responses: compared by equality instead
    return exp


def fixed_response_expected(kind: str, snap: dict) -> Optional[str]:
    if kind == "methodology":
        return METHODOLOGY_KO
    if kind == "out_of_scope":
        return TemplateNarrator().narrate({"category": "out_of_scope"})
    if kind == "refusal":
        return TemplateNarrator().narrate({"refusal": snap.get("refusal") or {}}) if snap.get("refusal") else None
    return None


_HIST = re.compile(r"관측 창 (\d+)개, 알람 창 (\[[^\]]*\]), BLOCK 창 (\d+)개, 현재 alarm_id=(\S+)\. 정책 결정 이력: (.*)$", re.S)
_WIN = re.compile(r"창 (\d+)(?!개)")


def parse_history_line(fixed: str) -> Optional[dict]:
    m = _HIST.search(fixed or "")
    if not m:
        return None
    try:
        wins = json.loads(m.group(2))
    except json.JSONDecodeError:
        wins = None
    return {"n_windows": int(m.group(1)), "alarm_windows": wins, "n_block": int(m.group(3)), "alarm_id": m.group(4), "decisions_text": m.group(5)}


def window_numbers(text: str) -> list[int]:
    return [int(x) for x in _WIN.findall(text or "")]


def review_flags(text: Optional[str], flags: list[str]) -> list[str]:
    return [f for f in flags if f and f in (text or "")]


# ------------------------------------------------------------------------------------------------- scoring
def match_skill_steps(expected: list[str], actual: list[Optional[str]]) -> tuple[bool, str]:
    """Ordered match; an expected entry ending with '?' is optional. Extra actual steps are a mismatch."""
    i = 0
    for spec in expected:
        opt = spec.endswith("?")
        s = spec[:-1] if opt else spec
        if i < len(actual) and actual[i] == s:
            i += 1
        elif not opt:
            return False, f"expected {s} at position {i}, got {actual[i] if i < len(actual) else 'nothing'}"
    if i != len(actual):
        return False, f"unexpected extra calls {actual[i:]}"
    return True, "ok"


def _check(name: str, ok: Optional[bool], detail: Any = None) -> dict:
    return {"name": name, "ok": ("unobserved" if ok is None else bool(ok)), "detail": detail}


def verdict(checks: list[dict]):
    """True only when EVERY check is True; False when any check is False; "incomplete" when nothing failed but something
    required was not observed (never counted as a pass)."""
    if any(c["ok"] is False for c in checks):
        return False
    if any(c["ok"] == "unobserved" for c in checks):
        return "incomplete"
    return True


def _detect_set(env: dict) -> Optional[list]:
    st = (env or {}).get("state") or {}
    if not st:
        return None
    return list(st["expanded_set"] if st.get("expanded_set") is not None else (st.get("pred_set") or []))


def classify_missing_decision(exp_cat: str, route: Optional[dict], skills: list[dict], decisions: list[dict], error: Optional[str]) -> str:
    """Why is there no initial decision although one was expected? Observed causes only."""
    cat = ((route or {}).get("result") or {}).get("category")
    if cat != exp_cat:
        return "router"
    if error:
        return "error"
    if any(d.get("error") for d in decisions):
        return "error"
    if any(s.get("status") != "OK" for s in skills if s["tool"] in ("gate_check", "detect_diagnose", "retrieve")):
        return "skill_error"
    return "contract"


def score_item(item: dict, obs: dict) -> dict:
    exp, con, nar = item["expected"], item["contract"], item["narration"]
    dw = obs.get("dw")
    ctx = obs.get("context") or {}
    out: dict = {"id": item["id"], "status": obs.get("status"), "context": ctx, "error": obs.get("error")}
    if ctx.get("prep") == "failed":
        out["scored"] = False
        out["note"] = "context_prep_failed: scores N/A"
        return out
    out["scored"] = True
    route = obs.get("route")
    rr = (route or {}).get("result") or {}
    skills = obs.get("skills") or []
    decisions = obs.get("decisions") or []
    db = obs.get("db") or {}
    res = obs.get("result") or {}
    status = obs.get("status")
    # ---- B router
    out["router"] = {"expected": exp["category"], "actual": rr.get("category"), "matched": rr.get("matched"), "ok": rr.get("category") == exp["category"]}
    # ---- C extraction
    ex = {}
    for f in ("user_claims", "ref_faults"):
        ex[f] = {"expected": sorted(exp[f]), "actual": sorted(rr.get(f) or []), "ok": sorted(rr.get(f) or []) == sorted(exp[f])}
    for f in ("ref_window", "ref_proposal"):
        ex[f] = {"expected": exp[f], "actual": rr.get(f), "ok": rr.get(f) == exp[f]}
    ex["ledger"] = obs.get("router_ledger")                                   # rule_extract / llm_parsed / merged / raw text (paid runs)
    out["extraction"] = ex
    # ---- D contract
    checks: list[dict] = [_check("status", status in con["status_in"], status)]
    checks.append(_check("refusal", ((res.get("refusal") or {}).get("code")) == con.get("refusal"), (res.get("refusal") or {}).get("code")))
    steps = [s.get("step") for s in skills]
    ok, why = match_skill_steps([resolve(s, dw) for s in con.get("skill_calls", [])], steps)
    checks.append(_check("skill_calls", ok, {"actual": steps, "why": why, "unresolved": [s["tool"] for s in skills if s.get("step") is None]}))
    bad_tools = sorted({s["tool"] for s in skills if s["tool"] in con.get("forbidden_tools", [])})
    checks.append(_check("forbidden_tools", not bad_tools, bad_tools))
    bad_steps = sorted({s for s in steps if s in [resolve(x, dw) for x in con.get("forbidden_steps", [])]})
    checks.append(_check("forbidden_steps", not bad_steps, bad_steps))
    human_calls = [s["tool"] for s in skills if s["tool"] in HUMAN_ONLY_TOOLS]
    checks.append(_check("no_human_only_tools", not human_calls, human_calls))
    # trace (done turns) vs recorded Skill calls — Skill entries only, decide entries vs decision records
    trace = res.get("trace")
    if isinstance(trace, list):
        tr_skill = [(t.get("tool"), t.get("step"), t.get("status"), t.get("reason")) for t in trace if t.get("tool") in SKILL_PARAMS]
        rec_skill = [(s["tool"], s.get("step"), s.get("status"), (s.get("reason") or {}).get("code")) for s in skills]
        checks.append(_check("trace_skill_entries_match_records", tr_skill == rec_skill, {"trace": tr_skill, "records": rec_skill}))
        tr_dec = [(t.get("step"), t.get("status")) for t in trace if t.get("tool") == "decide_action"]
        rec_dec = [(f"decide:{d.get('phase')}", d.get("action")) for d in decisions if d.get("decision_id")]
        if tr_dec or rec_dec:
            checks.append(_check("trace_decide_entries_match_records", tr_dec == rec_dec, {"trace": tr_dec, "records": rec_dec}))
    # decisions
    initial = next((d for d in decisions if d.get("phase") == "initial" and d.get("decision_id")), None)
    eid = item["context"].get("expected_initial_decision")
    failure_kind = None
    if con.get("decision_expected"):
        if initial is None:
            failure_kind = classify_missing_decision(exp["category"], route, skills, decisions, obs.get("error"))
            checks.append(_check("initial_decision_present", False, {"failure_kind": failure_kind}))
        else:
            checks.append(_check("initial_decision_present", True, {"action": initial["action"], "rule": initial["rule"]}))
            if eid:
                same = initial["action"] == eid["action"] and int(initial["rule"]) == int(eid["rule"])
                checks.append(_check("initial_decision_asserted", same, {"expected": eid, "actual": {"action": initial["action"], "rule": initial["rule"]}}))
                if not same:
                    failure_kind = "contract"
    else:
        checks.append(_check("no_decision", not any(d.get("decision_id") for d in decisions), [d.get("action") for d in decisions]))
    if con.get("decision_rows") is not None:
        checks.append(_check("decision_rows_db", None if db.get("decision_rows") is None else db["decision_rows"] == con["decision_rows"], db.get("decision_rows")))
    if con.get("fault_set_equals_detect"):
        last_set, mism = None, []
        for s in skills:
            if s["tool"] == "detect_diagnose" and s.get("status") == "OK":
                last_set = _detect_set(s)
            if s["tool"] == "retrieve" and (s.get("args") or {}).get("mode") == "scoped":
                fs = sorted(int(x) for x in ((s.get("args") or {}).get("fault_set") or []))
                if last_set is None or fs != sorted(last_set):
                    mism.append({"step": s.get("step"), "fault_set": fs, "detect_set": last_set})
        n_scoped = sum(1 for s in skills if s["tool"] == "retrieve" and (s.get("args") or {}).get("mode") == "scoped")
        checks.append(_check("fault_set_equals_detect", (not mism) if n_scoped else None, mism))
        claims = set(exp["user_claims"])
        leaked = [s.get("step") for s in skills if s["tool"] == "retrieve" and claims and set((s.get("args") or {}).get("fault_set") or []) == claims
                  and (last_set is None or sorted(claims) != sorted(last_set))]
        checks.append(_check("claims_not_skill_input", not leaked, leaked))
    if con.get("approvals_zero"):
        a, w = db.get("approvals"), db.get("work_orders")
        checks.append(_check("approvals_zero", None if a is None else a == 0, a))
        checks.append(_check("work_orders_zero", None if w is None else w == 0, w))
    if con.get("query"):
        q = res.get("query") if status == "done" else None
        src = "result"
        if q is None and db.get("plan") is not None:
            q, src = {"owned": db["plan"].get("owned"), "status": db["plan"].get("status")}, "query_plans"
        if q is None:
            checks.append(_check("query", None, "unobserved"))
        else:
            qs = str(q.get("status") or "").split(":")[0]
            checks.append(_check("query", bool(q.get("owned")) == bool(con["query"]["owned"]) and qs in con["query"]["status_in"], {"source": src, "query": q}))
    if con.get("decision_after_in"):
        after = next((d for d in decisions if d.get("phase") == "after_query" and d.get("decision_id")), None)
        checks.append(_check("decision_after_observed", None if after is None else after["action"] in con["decision_after_in"],
                             {"action": after and after["action"], "rule": after and after["rule"]}))
    if status == "waiting_approval":
        intr = obs.get("interrupt") or {}
        checks.append(_check("interrupt_payload", bool(intr.get("proposal_id")) and bool(intr.get("draft_version")), {k: intr.get(k) for k in ("proposal_id", "draft_version", "content_sha256")}))
    contract_ok = verdict(checks)
    if not out["router"]["ok"]:
        failure_kind = "router"                                               # a misrouted item is a router failure whatever else followed
    elif failure_kind is None and contract_ok is False:
        failure_kind = "error" if (status in ("failed", "error") or obs.get("error")) else "contract"
    out["contract"] = {"checks": checks, "ok": contract_ok, "unobserved": [c["name"] for c in checks if c["ok"] == "unobserved"],
                       "failure_kind": failure_kind}
    out["llm_calls"] = obs.get("llm_calls")                                   # paid runs: HTTP calls made during this item (counting client), else None
    # ---- E narration
    narr = obs.get("narration")
    if status != "done" or not narr:
        out["narration"] = {"applicable": False, "note": "N/A — 승인 대기" if status == "waiting_approval" else f"N/A — status {status}"}
    else:
        snap, answer = narr["snapshot"], narr["answer"]
        fixed, prose, note = split_answer(answer)
        nchecks: list[dict] = []
        fr = nar.get("fixed_response")
        if fr:
            expct = fixed_response_expected(fr, snap)
            nchecks.append(_check("fixed_response_equal", None if expct is None else fixed == expct, {"kind": fr}))
        for line in nar.get("template_lines", []):
            nchecks.append(_check(f"template_line:{line}", line in fixed))
        if nar.get("state_lines") == "auto":
            mism = [{"line": line.strip(), "expected_present": should, "present": (line in fixed)}
                    for line, should in template_line_expectations(snap).items() if (line in fixed) != should]
            nchecks.append(_check("state_lines_auto", not mism, mism))
        if nar.get("structured") == "history":
            h = snap.get("history") or {}
            parsed = parse_history_line(fixed)
            rendered = TemplateNarrator().narrate({"category": "history", "history": h, "display_window": snap.get("display_window"), "alarm_id": snap.get("alarm_id")})
            nchecks.append(_check("history_render_equal", fixed == rendered))
            nchecks.append(_check("history_parsed_equals_state", bool(parsed) and parsed["n_windows"] == h.get("n_windows") and parsed["alarm_windows"] == h.get("alarm_windows")
                                  and parsed["n_block"] == h.get("n_block"), parsed))
            nchecks.append(_check("history_n_windows_is_dw_plus_1", h.get("n_windows") == int(dw) + 1, h.get("n_windows")))
            dbd = db.get("decision_rows")
            nchecks.append(_check("history_decisions_match_db", None if dbd is None else len(h.get("decisions") or []) == dbd, {"state": len(h.get("decisions") or []), "db": dbd}))
        scan = answer
        if (snap.get("refusal") or {}).get("message"):                        # a refusal echoes the user's own reference (e.g. 창 20); not a leak
            scan = scan.replace(str(snap["refusal"]["message"]), "")
        wins = [w for w in window_numbers(scan) if w > int(dw)]
        nchecks.append(_check("no_window_after_display", not wins, wins))
        views = obs.get("views_roundtrip")                                     # harness fills: True/False/None(import unavailable); rescore may fill post-hoc
        nchecks.append(_check("views_roundtrip_lossless", views, obs.get("views_roundtrip_source")))
        out["narration"] = {"applicable": True, "checks": nchecks, "ok": verdict(nchecks), "unobserved": [c["name"] for c in nchecks if c["ok"] == "unobserved"],
                            "flags": {"fixed": review_flags(fixed, nar.get("review_flags", [])), "prose": review_flags(prose, nar.get("review_flags", []))},
                            "prose_present": prose is not None, "omitted_note": note, "llm": obs.get("narrate_ledger")}
    # ---- observations for the change candidates (not scores)
    out["observations"] = observations(skills, decisions)
    return out


def observations(skills: list[dict], decisions: list[dict]) -> dict:
    retr = []
    for s in skills:
        if s["tool"] != "retrieve" or s.get("status") != "OK":
            continue
        keys = [c.get("clause_key") for c in (s.get("payload") or {}).get("clauses") or []]
        g = s.get("grounding") or {}
        retr.append({"step": s.get("step"), "mode": (s.get("args") or {}).get("mode"), "k": (s.get("args") or {}).get("k"), "query": (s.get("args") or {}).get("query"),
                     "n_chunks": len(keys), "n_unique_keys": len(set(keys)), "n_dup": len(keys) - len(set(keys)),
                     "g_AB": g.get("g_AB"), "g_C": g.get("g_C"), "ev_AB": g.get("ev_AB"), "reused": (s.get("payload") or {}).get("reused"),
                     "grades": sorted((c.get("grade") or "?") for c in (s.get("payload") or {}).get("clauses") or [])})
    snaps = []
    for d in decisions:
        ev = d.get("evidence") or []
        keys = [e.get("clause_key") for e in ev]
        snaps.append({"phase": d.get("phase"), "n_items": len(keys), "n_unique_keys": len(set(keys)), "n_dup": len(keys) - len(set(keys))})
    return {"retrieve": retr, "decision_snapshots": snaps}


def summarize(scored: list[dict], mode: str) -> dict:
    n = len(scored)
    prep_failed = [s["id"] for s in scored if not s.get("scored")]
    contaminated = [s["id"] for s in scored if (s.get("context") or {}).get("contaminated")]
    errors = [s["id"] for s in scored if s.get("status") in ("failed", "error") or s.get("error")]
    sc = [s for s in scored if s.get("scored")]
    routed = [s for s in sc if s.get("router")]
    matched = [str((s["router"] or {}).get("matched") or "") for s in routed]
    counters = any(s.get("llm_calls") is not None for s in routed)
    if counters:                                                              # paid run: the counting client's record is the source
        sent = [s for s in routed if any(c.get("tool") == "router" for c in (s.get("llm_calls") or []))]
        n_http = sum(len(s.get("llm_calls") or []) for s in routed)
        n_http_router = sum(1 for s in routed for c in (s.get("llm_calls") or []) if c.get("tool") == "router")
        n_http_narrate = sum(1 for s in routed for c in (s.get("llm_calls") or []) if c.get("tool") == "narrate")
    else:
        sent = [s for s in routed if s["router"]["matched"] in LLM_SENT]
        n_http = n_http_router = n_http_narrate = None
    valid = [s for s in sent if s["router"]["matched"] == "llm"]
    not_sent = [s for s in routed if any(str(s["router"]["matched"]).startswith(p) for p in LLM_NOT_SENT_PREFIXES)]
    def frac(a, b):
        return None if not b else round(a / b, 3)
    summary = {
        "mode": mode, "n_items": n, "n_scored": len(sc), "context_prep_failed": prep_failed, "context_contaminated": contaminated, "errors": errors,
        "router": {"accuracy_all": f"{sum(1 for s in routed if s['router']['ok'])}/{len(routed)}",
                   "default_count": sum(1 for m in matched if m == "default"),
                   "matched_counts": {m: matched.count(m) for m in sorted(set(matched))},
                   "llm": {"definitions": "n_called = items with a router HTTP request (counting-client record when present, else matched ∈ llm / llm_fallback:invalid_output / "
                                          "llm_outcome_unknown); valid_rate = matched=='llm' ÷ n_called; acc_called = LLM-valid AND correct ÷ n_called (a fallback that "
                                          "kept a correct rule result counts in accuracy_all only, never as an LLM correct); acc_valid_conditional = LLM-valid AND correct ÷ "
                                          "valid responses; n_not_sent = refused reservation / client refusal before sending",
                           "source": "counting_client" if counters else "matched",
                           "n_called": len(sent), "n_not_sent": len(not_sent), "n_rule": len(routed) - len(sent) - len(not_sent),
                           "n_http_calls": n_http, "n_http_router": n_http_router, "n_http_narrate": n_http_narrate,
                           "valid_rate": frac(len(valid), len(sent)),
                           "acc_called": frac(sum(1 for s in valid if s["router"]["ok"]), len(sent)),
                           "acc_valid_conditional": frac(sum(1 for s in valid if s["router"]["ok"]), len(valid))}},
        "extraction": {f: f"{sum(1 for s in sc if (s.get('extraction') or {}).get(f, {}).get('ok'))}/{len(sc)}" for f in ("user_claims", "ref_faults", "ref_window", "ref_proposal")},
        "contract": {"pass": f"{sum(1 for s in sc if (s.get('contract') or {}).get('ok') is True)}/{len(sc)}",
                     "incomplete": {s["id"]: s["contract"]["unobserved"] for s in sc if (s.get("contract") or {}).get("ok") == "incomplete"},
                     "failure_kinds": {k: [s["id"] for s in sc if (s.get("contract") or {}).get("failure_kind") == k] for k in ("router", "skill_error", "contract", "error")},
                     "failed_items": [s["id"] for s in sc if (s.get("contract") or {}).get("ok") is False],
                     "unobserved": {s["id"]: s["contract"]["unobserved"] for s in sc if (s.get("contract") or {}).get("unobserved")}},
        "narration": {"applicable": [s["id"] for s in sc if (s.get("narration") or {}).get("applicable")],
                      "auto_pass": f"{sum(1 for s in sc if (s.get('narration') or {}).get('applicable') and s['narration'].get('ok') is True)}/{sum(1 for s in sc if (s.get('narration') or {}).get('applicable'))}",
                      "incomplete": {s["id"]: s["narration"]["unobserved"] for s in sc if (s.get("narration") or {}).get("applicable") and s["narration"].get("ok") == "incomplete"},
                      "flagged": {s["id"]: s["narration"]["flags"] for s in sc if (s.get("narration") or {}).get("applicable") and (s["narration"]["flags"]["fixed"] or s["narration"]["flags"]["prose"])},
                      "prose_present": [s["id"] for s in sc if (s.get("narration") or {}).get("prose_present")],
                      "failed_items": {s["id"]: [c["name"] for c in s["narration"]["checks"] if c["ok"] is False]
                                       for s in sc if (s.get("narration") or {}).get("applicable") and s["narration"].get("ok") is False},
                      "failed_due_to_router": [s["id"] for s in sc if (s.get("narration") or {}).get("applicable") and s["narration"].get("ok") is False and not s["router"]["ok"]]},
        "note": "development observations on 10 items; not a performance claim. contaminated items are kept but excluded from mode comparison.",
    }
    return summary


# ------------------------------------------------------------------------------------------------- DSN guard
_DBNAME_URL = re.compile(r"^[a-z]+://[^/]*/([^?\s]+)")
_DBNAME_KV = re.compile(r"(?:^|\s)dbname=([^\s]+)")


def dbname_of(dsn: str) -> Optional[str]:
    m = _DBNAME_URL.match(dsn or "")
    if m:
        return m.group(1)
    m = _DBNAME_KV.search(dsn or "")
    return m.group(1) if m else None


def assert_test_dsn(dsn: Optional[str], project_dsn: Optional[str] = None) -> None:
    """ST4-09: the harness runs only against a DSN whose dbname contains 'test' and that differs from the project DSN."""
    if not dsn:
        raise SystemExit("a test DSN is required (SGP_TEST_PG_DSN)")
    name = dbname_of(dsn)
    if not name or "test" not in name:
        raise SystemExit("refusing: the DSN must name a database containing 'test'")
    if project_dsn and project_dsn == dsn:
        raise SystemExit("refusing: the test DSN equals SGP_PG_DSN")


# ------------------------------------------------------------------------------------------------- human judgment / publish
def required_criteria(ds: dict, item_id: str) -> list[str]:
    """The criteria a judgment must cover: the dev set's common criteria + the item's own — the dev set is the source, not the sheet."""
    common = list((ds.get("_meta") or {}).get("human_criteria_common") or [])
    it = next((x for x in ds.get("items") or [] if x.get("id") == item_id), None)
    return common + list((it or {}).get("human") or [])


def judgment_status(judgment: Optional[dict], scored: list[dict], ds: dict) -> dict:
    """Sheet existence is not completion: complete only when, for every applicable item, EVERY required criterion (from the
    dev set) carries a pass/fail verdict — a criterion missing from the sheet counts as unjudged."""
    applicable = [s["id"] for s in scored if s.get("scored") and (s.get("narration") or {}).get("applicable")]
    if not judgment:
        return {"sheet": False, "complete": False, "n_applicable": len(applicable), "n_unjudged_items": len(applicable),
                "n_unjudged_criteria": sum(len(required_criteria(ds, i)) for i in applicable), "per_criterion": {}, "failed_items": {}, "judge": None}
    per, fails, unjudged_items, unjudged_criteria = {}, {}, [], 0
    for iid in applicable:
        v = (judgment.get("items") or {}).get(iid) or {}
        verdicts = v.get("verdicts") or {}
        required = required_criteria(ds, iid)
        missing = [c for c in required if verdicts.get(c) not in ("pass", "fail")]
        if missing:
            unjudged_items.append(iid)
            unjudged_criteria += len(missing)
        for c in required:
            x = verdicts.get(c)
            per.setdefault(c, {"pass": 0, "fail": 0, "unjudged": 0})
            per[c]["pass" if x == "pass" else "fail" if x == "fail" else "unjudged"] += 1
            if x == "fail":
                fails.setdefault(iid, []).append(c)
    return {"sheet": True, "complete": not unjudged_items, "n_applicable": len(applicable), "n_unjudged_items": len(unjudged_items),
            "unjudged_items": unjudged_items, "n_unjudged_criteria": unjudged_criteria, "per_criterion": per, "failed_items": fails,
            "judge": judgment.get("judge")}


def refined_result(ds: dict, context: dict, obs_list: list[dict], scored: list[dict], judgment: Optional[dict], run_meta: dict) -> dict:
    """The tracked result file: allow-listed fields joined explicitly from the dev set, the raw observations, the scores and
    the judgment sheet. No secrets, no tokens, no clause text, no messages."""
    items = {it["id"]: it for it in ds["items"]}
    obs_by = {o["id"]: o for o in obs_list}
    j_items = (judgment or {}).get("items") or {}
    out_items = []
    for sc in scored:
        it, o = items[sc["id"]], obs_by.get(sc["id"], {})
        pc = (context.get("presets") or {}).get(str((o.get("preset") or it["context"]["preset"]))) or {}
        narr = o.get("narration") or {}
        out_items.append({
            "id": sc["id"], "question": it["question"], "expected": it["expected"], "context": {"preset": it["context"]["preset"], "display_window": o.get("dw"),
                                                                                              "episode_id": o.get("episode_id"), "alarm_id": o.get("alarm_id"),
                                                                                              "detector": pc.get("detector"), "contract_sha256": pc.get("contract"),
                                                                                              "prep": (o.get("context") or {}).get("prep"), "contaminated": (o.get("context") or {}).get("contaminated")},
            "status": o.get("status"), "thread_id": o.get("thread_id"), "turn_id": o.get("turn_id"), "elapsed_s": o.get("elapsed_s"), "error": o.get("error"),
            "route": (o.get("route") or {}).get("result"), "skill_steps": [x.get("step") or x.get("tool") for x in (o.get("skills") or [])],
            "decisions": [{k: d.get(k) for k in ("phase", "action", "rule", "decision_id")} for d in (o.get("decisions") or [])],
            "query": (o.get("result") or {}).get("query"), "interrupt": o.get("interrupt"), "answer": narr.get("answer"),
            "scores": {k: sc.get(k) for k in ("router", "extraction", "contract", "narration", "observations")},
            "llm_calls": o.get("llm_calls"), "ledger": [{k: r.get(k) for k in ("tool", "stage_key", "status", "reason_code", "cost_usd", "input_tokens", "output_tokens",
                                                                              "violation", "usage_missing", "reservation_exceeded", "fallback_reason", "calibration")}
                                                       for r in (o.get("ledger") or [])] or None,
            "human": {"verdicts": (j_items.get(sc["id"]) or {}).get("verdicts"), "note": (j_items.get(sc["id"]) or {}).get("note")} if (j_items.get(sc["id"]) or {}).get("applicable") else None,
        })
    jm = judgment or {}
    return {"schema": "chat_dev_result/1", "run": run_meta, "dev_set": {"revision": (ds.get("_meta") or {}).get("revision"), "sha16": run_meta.get("dev_set_sha16"),
                                                                     "n": len(items)},
            "judgment": {k: jm.get(k) for k in ("judge", "basis", "readability_scope", "readability_observation")} if judgment else None,
            "context": {"dbname": context.get("dbname"), "project_manifest_sha16": context.get("project_manifest_sha16"), "code_commit_at_reset": context.get("code_commit"),
                        "presets": {p: {"detector": v.get("detector"), "contract_sha256": v.get("contract"), "prep": v.get("prep"), "entry": v.get("entry")}
                                    for p, v in (context.get("presets") or {}).items()}},
            "summary": run_meta.get("summary"), "items": out_items}
