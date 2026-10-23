"""demo/views.py — pure view-model functions for the Streamlit demo (ST4-26, ST4-27). No Streamlit, no DB.

Everything the UI renders goes through an ALLOW-LIST here: the manifest entry (preset / display_window / kind only),
the turn result (`runner.compose_result` output, which can carry the whole `approval` envelope including the one-shot
approval_token — F12), and the interrupt payload (proposal id, version, draft hash, form; never a token or secret).
The functions never copy unknown keys, so a new field in an envelope does not reach the screen until it is added here.
"""
from __future__ import annotations

import re
from typing import Any, Optional

VISIBLE_PRESET_FIELDS = ("preset", "display_window", "kind")
# fields of outputs/demo_presets.json that must NOT be shown (post-display-window information or precompute-time
# predictions; the screen shows only this session's server envelopes)
HIDDEN_PRESET_FIELDS = ("alarm_windows", "pred_set", "top1", "note", "retrieve_k5", "decide_preview", "versions", "alarm_id", "episode_id")
APPROVAL_DECISIONS = ("approve", "modify", "reject")        # the graph accepts exactly these (schema also lists override)
FORM_FIELDS = ("alarm_id", "window_index", "policy", "decision_id", "p_adj", "pred_set", "evidence_keys")
TRACE_FIELDS = ("tool", "step", "status", "reason", "replayed")
KIND_LABELS = {"alarm": "알람", "no_alarm": "알람 없음"}


def visible_preset(entry: dict) -> dict:
    return {k: entry.get(k) for k in VISIBLE_PRESET_FIELDS}


def visible_presets(manifest: dict) -> list[dict]:
    return sorted((visible_preset(e) for e in manifest.get("presets", [])), key=lambda e: int(e["preset"]))


def preset_label(entry: dict) -> str:
    return f"프리셋 {entry['preset']} — 창 {entry['display_window']} · {KIND_LABELS.get(entry.get('kind'), entry.get('kind'))}"


def _reason(env: Optional[dict]) -> dict:
    r = (env or {}).get("reason") or {}
    return {"code": r.get("code"), "message": r.get("message")}


def _decision_view(d: Optional[dict]) -> Optional[dict]:
    if not d:
        return None
    snap = [{"clause_key": e.get("clause_key"), "grade": e.get("grade"), "score": e.get("score")} for e in (d.get("evidence_snapshot") or [])]
    return {"action": d.get("action"), "rule": d.get("rule"), "reason": d.get("reason"), "query_kind": d.get("query_kind"),
            "decision_id": d.get("decision_id"), "tau": d.get("tau"), "n_snapshot": len(snap), "evidence_snapshot": snap}


def _query_view(q: Optional[dict]) -> Optional[dict]:
    if not q:
        return None
    return {k: q.get(k) for k in ("status", "kind", "owned", "query_id", "expand_status", "rerank_status", "k10_reused")}


def _draft_view(d: Optional[dict]) -> Optional[dict]:
    if not d:
        return None
    p = d.get("payload") or {}
    return {"status": d.get("status"), "reason": d.get("reason") if isinstance(d.get("reason"), (str, type(None))) else _reason(d),
            "proposal_id": p.get("proposal_id"), "draft_version": p.get("draft_version"), "content_sha256": p.get("content_sha256")}


def _approval_view(a: Optional[dict]) -> Optional[dict]:
    """status / reason / decision only — the payload also carries approval_token, which is never copied (F12)."""
    if not a:
        return None
    return {"status": a.get("status"), "reason": _reason(a), "decision": ((a.get("payload") or {}).get("decision"))}


def _work_order_view(w: Optional[dict]) -> Optional[dict]:
    if not w:
        return None
    p = w.get("payload") or {}
    return {"status": w.get("status"), "reason": _reason(w), "work_order_id": p.get("work_order_id"), "cmms_ref": p.get("cmms_ref"),
            "idempotent_replay": p.get("idempotent_replay")}


def trace_rows(trace: Optional[list]) -> list[dict]:
    return [{k: t.get(k) for k in TRACE_FIELDS} for t in (trace or []) if isinstance(t, dict)]


def approval_form_model(interrupt: dict) -> dict:
    """Allow-listed model of an approval_wait interrupt payload (graph.py approval_wait)."""
    form = interrupt.get("form") or {}
    items = [{"action_type": i.get("action_type"), "item": i.get("item"), "clause_key": i.get("clause_key")} for i in (form.get("action_items") or [])]
    return {"question": interrupt.get("question"), "proposal_id": interrupt.get("proposal_id"), "draft_version": interrupt.get("draft_version"),
            "content_sha256": interrupt.get("content_sha256"), "turn_id": interrupt.get("turn_id"),
            "form": {**{k: form.get(k) for k in FORM_FIELDS}, "action_items": items}}


def modifications_from_items(items: list[dict], edited_texts: list[str]) -> dict:
    """ST4-27: only the item wording changes; action_type and clause_key are copied from the presented draft."""
    if len(items) != len(edited_texts):
        raise ValueError("edited texts must match the draft items one to one")
    out = []
    for it, text in zip(items, edited_texts):
        text = (text or "").strip()
        if not text:
            raise ValueError("an action item text is empty")
        out.append({"action_type": it["action_type"], "item": text, "clause_key": it["clause_key"]})
    return {"action_items": out}


def result_view(result: dict) -> dict:
    """Screen model of a runner result; unknown keys and envelope payloads are dropped."""
    r = result or {}
    out: dict[str, Any] = {"status": r.get("status"), "turn_id": r.get("turn_id"), "category": r.get("category"), "answer": r.get("answer"),
                           "refusal": ({"code": (r.get("refusal") or {}).get("code"), "message": (r.get("refusal") or {}).get("message")} if r.get("refusal") else None),
                           "decision": _decision_view(r.get("decision")), "decision_after": _decision_view(r.get("decision_after")),
                           "query": _query_view(r.get("query")), "draft": _draft_view(r.get("draft")), "approval": _approval_view(r.get("approval")),
                           "work_order": _work_order_view(r.get("work_order")), "trace": trace_rows(r.get("trace")),
                           "turn_refs_discarded": r.get("turn_refs_discarded"), "error": r.get("error")}
    if r.get("status") == "waiting_approval" and isinstance(r.get("interrupt"), dict):
        out["interrupt"] = approval_form_model(r["interrupt"])
    return out


_CLAUSE_HEADER = re.compile(r"^\[P-[^\]]+\] \(")
_CLAUSE_TITLE = re.compile(r"^\[(?P<key>P-[^\]]+)\] \((?P<grade>[ABC]), score (?P<score>[0-9.]+)\)\s*(?P<title>.*)$")
# template lines the main view shows (narrator.py parts; everything else stays in the raw answer only)
_MAIN_PREFIXES = ("창 ", "진단 집합", "정책 결정", "검색 ", "query 상태", "초안 ", "승인 ", "작업지시 발행", "발행 거부", "open 모드", "이 창에서는",
                  "선택 가능한 알람", "요청한 참조", "(조항 없음)", "요약(LLM 생성)", "(LLM 요약",
                  "탐지: ", "진단: ", "근거: ", "정책 E", "작업지시: ",                           # ST4-47 methodology block lines
                  "언급한 결함 ")                                                                  # ST4-48 referred-fault contrast
_MAIN_SUFFIXES = ("집합에 포함됩니다.", "근거로 제안하지 않습니다.")      # the user-claim comparison line has a variable start


def is_template_line(line: str) -> bool:
    return line.startswith(_MAIN_PREFIXES) or line.endswith(_MAIN_SUFFIXES)


def parse_clause_title(line: Optional[str]) -> dict:
    """`[P-132-2013:0:4.4] (A, score 0.035) 연속식 반응기` → key / grade / narrator score / title; unknown → raw."""
    m = _CLAUSE_TITLE.match(line or "")
    if not m:
        return {"key": None, "grade": None, "narrator_score": None, "title": line}
    return {"key": m.group("key"), "grade": m.group("grade"), "narrator_score": float(m.group("score")), "title": m.group("title").strip()}


def short_key(key: Optional[str]) -> str:
    """P-132-2013:0:4.4 → P-132 §4.4 (display only; the full key stays in the row)."""
    if not key:
        return ""
    m = re.match(r"^(P-\d+)-\d{4}:\d+:(.+)$", key)
    return f"{m.group(1)} §{m.group(2)}" if m else key


INJECTED_GRADES = ("A", "B")      # demo/synthetic.GroundingOverride rewrites A/B scores only


def clause_label(c: dict, synthetic: bool = False) -> str:
    """Expander label: title · short key · grade · representative score (max over the clause's chunks) · criterion.
    `synthetic` is the explicit demo mode flag (runtime cfg), never inferred from score differences: in synthetic mode an
    A/B clause shows "합성 주입 <snapshot max> / 검색 <narrator max>", other grades and the real demo show the snapshot score."""
    t = parse_clause_title(c.get("title"))
    parts = [t["title"] or c.get("clause_key") or "조항", short_key(c.get("clause_key") or t["key"])]
    grade = c.get("grade") or t["grade"]
    if grade:
        parts.append(f"{grade}등급")
    sc = c.get("score")
    ns = c.get("narrator_score", t["narrator_score"])
    multi = c.get("n_items", 1) > 1
    rep = "최대 청크 점수 " if multi else "score "
    if isinstance(sc, (int, float)):
        if synthetic and grade in INJECTED_GRADES:
            parts.append(f"합성 주입 {sc:.3f} / 검색 {ns:.3f}" if ns is not None else f"합성 주입 {sc:.3f}")
        else:
            parts.append(f"{rep}{sc:.3f}")
    elif ns is not None:
        parts.append(f"검색 점수 {ns:.3f}")
    if c.get("meets") is not None:
        parts.append("기준 충족" if c["meets"] else "기준 미충족")
    if "n_items" in c:                                                   # ST4-41: clause groups built by turn_sections carry n_items; an excerpt count, never "i of n"
        parts.append(f"검색된 발췌 청크 {c['n_items']}건")
    return " · ".join(p for p in parts if p)


def headline(view: dict) -> dict:
    """Outcome-first headline (reviewer 2026-10-22 point 3): issued work order > rejected/refused approval > draft
    refused > policy decision. Returns {"title", "kind"}; None when the turn has none of these."""
    wo, ap, dr = view.get("work_order") or {}, view.get("approval") or {}, view.get("draft") or {}
    ver = dr.get("draft_version")
    vtxt = f" · 승인된 초안 v{ver}" if ver else ""
    if wo.get("status") == "OK" and wo.get("work_order_id"):
        return {"kind": "issued", "title": f"모의(mock CMMS) 작업지시 발행 완료{vtxt} · {wo['work_order_id']}"}
    if wo and wo.get("status") != "OK":
        return {"kind": "issue_refused", "title": f"작업지시 발행 거부 — {(wo.get('reason') or {}).get('code')}"}
    if ap.get("status") == "OK" and ap.get("decision") == "reject":
        return {"kind": "rejected", "title": f"초안{(' v' + str(ver)) if ver else ''} 거부 — 작업지시 없음"}
    if ap and ap.get("status") != "OK":
        return {"kind": "approval_refused", "title": f"승인 처리 거부 — {(ap.get('reason') or {}).get('code')}"}
    if dr and dr.get("status") and dr.get("status") != "OK":
        return {"kind": "draft_refused", "title": f"초안 거부 — {dr.get('reason') if isinstance(dr.get('reason'), str) else (dr.get('reason') or {}).get('code')}"}
    dec = view.get("decision_after") or view.get("decision")
    if dec:
        return {"kind": "decision", "title": f"결정: {ACTION_LABELS.get(dec.get('action'), dec.get('action'))} ({dec.get('action')}) · 규칙 {dec.get('rule')}"}
    return None
# lines of TemplateNarrator (core_pipeline/agent/narrator.py, read 2026-10-22) that can follow the last clause body —
# they END a clause body; any other line after the `> ` first line is a body continuation line
_TAIL_PREFIXES = ("초안 ", "승인 ", "작업지시 발행", "발행 거부", "open 모드", "요약(LLM 생성)", "(LLM 요약")


def answer_blocks(answer: Optional[str]) -> list[dict]:
    """Display-only split of the narrator answer into blocks. The TemplateNarrator prints each clause as a header line
    `[P-…] (grade, score) title`, then the body's FIRST line prefixed with `> ` and the remaining body lines without a
    prefix (observed on the real output 2026-10-22). Blocks: {"type": "text", "lines": [...]} and
    {"type": "clause", "title": <header line>, "lines": [first line without `> `, continuation lines...]}.
    A clause body ends at the next header line or at a template tail line (_TAIL_PREFIXES). Nothing is reworded:
    answer_from_blocks() restores the exact text (tested)."""
    blocks: list[dict] = []
    in_clause = False
    for line in (answer or "").split("\n"):
        if _CLAUSE_HEADER.match(line):
            blocks.append({"type": "clause", "title": line, "lines": [], "first_quoted": False})
            in_clause = True
            continue
        if in_clause and blocks[-1]["type"] == "clause":
            b = blocks[-1]
            if not b["lines"] and line.startswith("> "):
                b["lines"].append(line[2:]); b["first_quoted"] = True
                continue
            if b["lines"] and not line.startswith(_TAIL_PREFIXES):
                b["lines"].append(line)
                continue
            in_clause = False
        if blocks and blocks[-1]["type"] == "text":
            blocks[-1]["lines"].append(line)
        else:
            blocks.append({"type": "text", "lines": [line]})
    return blocks


def answer_from_blocks(blocks: list[dict]) -> str:
    """Inverse of answer_blocks (used by the tests to prove the display split loses nothing)."""
    out: list[str] = []
    for b in blocks:
        if b["type"] == "text":
            out.extend(b["lines"])
        else:
            out.append(b["title"])
            for i, l in enumerate(b["lines"]):
                out.append(("> " + l) if (i == 0 and b.get("first_quoted")) else l)
    return "\n".join(out)


ACTION_LABELS = {"hold": "보류", "query_more": "추가 조회", "propose": "제안", "no_alarm": "알람 없음", "undecidable": "판단 불가"}
_CLAUSE_KEY = re.compile(r"\[(P-[^\]]+)\]")


def _meets(grade: Optional[str], score: Optional[float], tau: Optional[float]) -> Optional[bool]:
    """ST4-03 citable condition: scoped A/B clause with score >= tau. None when tau is unknown."""
    if tau is None or score is None:
        return None
    return grade in ("A", "B") and float(score) >= float(tau)


def turn_sections(view: dict) -> dict:
    """Display-only layout of one turn (reviewer 2026-10-22): judgment → state line → clause list (grade, score, meets τ)
    → clause bodies → details. Built from the result's structured fields (decision, decision_after, draft, approval,
    work_order) and from the answer text split by answer_blocks(); nothing is reworded — the state line is the
    answer's first line verbatim, the judgment is action/rule/reason as recorded, clause titles/bodies are the
    narrator's own lines. Clause list source: decision(_after).evidence_snapshot + tau when the turn has a decision
    (the stored snapshot the decision used), otherwise the clause blocks of the answer (no criterion shown)."""
    blocks = answer_blocks(view.get("answer"))
    text_lines = [l for b in blocks if b["type"] == "text" for l in b["lines"]]
    state_line = next((l for l in text_lines if l.strip()), None)
    clause_blocks = [b for b in blocks if b["type"] == "clause"]
    blocks_by_key: dict[str, list[dict]] = {}
    for b in clause_blocks:                                              # the narrator prints one block per chunk, in order
        t = parse_clause_title(b.get("title"))
        if t["key"]:
            blocks_by_key.setdefault(t["key"], []).append(b)
    dec = view.get("decision_after") or view.get("decision")
    judgment = None
    if dec:
        judgment = {"action": dec.get("action"), "label": ACTION_LABELS.get(dec.get("action"), dec.get("action")), "rule": dec.get("rule"),
                    "reason": dec.get("reason"), "phase": ("after_query" if view.get("decision_after") else "initial"), "tau": dec.get("tau")}
    clauses: list[dict] = []
    snapshot = dec.get("evidence_snapshot") if dec else None
    if snapshot:
        groups: dict[str, dict] = {}
        for e in snapshot:                                               # i-th snapshot item of a key ↔ i-th narrator block of that key
            k = e.get("clause_key")
            g = groups.setdefault(k, {"clause_key": k, "grade": e.get("grade"), "title": None, "chunks": [], "n_items": 0})
            idx = g["n_items"]
            b = blocks_by_key.get(k, [])
            blk = b[idx] if idx < len(b) else None
            tt = parse_clause_title((blk or {}).get("title")) if blk else {"narrator_score": None, "title": None}
            if g["title"] is None and blk:
                g["title"] = blk["title"]
            g["chunks"].append({"index": idx + 1, "grade": e.get("grade"), "score": e.get("score"), "narrator_score": tt["narrator_score"],
                                "meets": _meets(e.get("grade"), e.get("score"), dec.get("tau")), "lines": (blk or {}).get("lines") or []})
            g["n_items"] += 1
        for g in groups.values():
            scores = [c["score"] for c in g["chunks"] if isinstance(c["score"], (int, float))]
            nscores = [c["narrator_score"] for c in g["chunks"] if isinstance(c["narrator_score"], (int, float))]
            g["score"] = max(scores) if scores else None                  # representative = max chunk score (labelled as such)
            g["narrator_score"] = max(nscores) if nscores else None
            g["meets"] = any(c["meets"] for c in g["chunks"]) if any(c["meets"] is not None for c in g["chunks"]) else None
            g["lines"] = g["chunks"][0]["lines"]                          # kept for callers that show one body; app.py renders every chunk
            clauses.append(g)
        n_meet = sum(1 for c in clauses if c["meets"])
        dup = len(snapshot) - len(clauses)
        criterion = (f"제안 근거 기준 {'충족' if n_meet else '미충족'} (A/B ∧ 최대 청크 score ≥ τ={dec.get('tau')}, 충족 {n_meet}/{len(clauses)} 조항"
                     + (f"; 스냅샷 {len(snapshot)}항목, 청크 중복 {dup}" if dup else "") + ")") if dec.get("tau") is not None else None
    else:
        for b in clause_blocks:
            t = parse_clause_title(b.get("title"))
            clauses.append({"clause_key": t["key"], "grade": t["grade"], "score": None, "narrator_score": t["narrator_score"], "n_items": 1, "meets": None,
                            "title": b.get("title"), "lines": b["lines"],
                            "chunks": [{"index": 1, "grade": t["grade"], "score": None, "narrator_score": t["narrator_score"], "meets": None, "lines": b["lines"]}]})
        criterion = None
    rest = [l for l in text_lines if l is not state_line]
    main_lines = [l for l in rest if is_template_line(l)]                   # recognised template lines only
    hidden = [l for l in rest if l.strip() and not is_template_line(l)]
    return {"headline": headline(view), "judgment": judgment, "state_line": state_line, "criterion": criterion, "clauses": clauses,
            "main_lines": main_lines, "other_lines": rest, "hidden_lines": hidden,
            "draft": view.get("draft"), "approval": view.get("approval"), "work_order": view.get("work_order"), "refusal": view.get("refusal")}


def contains_secret_like(obj: Any, needles: tuple[str, ...]) -> bool:
    """Debug guard used by tests: True if any needle string occurs anywhere in the rendered model."""
    if isinstance(obj, dict):
        return any(contains_secret_like(k, needles) or contains_secret_like(v, needles) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return any(contains_secret_like(v, needles) for v in obj)
    return isinstance(obj, str) and any(n and n in obj for n in needles)


def inflight_message(client_request_id: str, message: str) -> dict:
    return {"kind": "message", "client_request_id": client_request_id, "message": message}


def inflight_resume(client_request_id: str, parent_turn_id: str, decision: str, modifications: Optional[dict]) -> dict:
    if decision not in APPROVAL_DECISIONS:
        raise ValueError(f"decision must be one of {APPROVAL_DECISIONS}")
    return {"kind": "resume", "client_request_id": client_request_id, "parent_turn_id": parent_turn_id, "decision": decision, "modifications": modifications}


def inflight_from_pending(pending: dict) -> Optional[dict]:
    """Rebuild the inflight record from AgentSession.pending_turn() (ST4-25 step 2) — same ids, same payload."""
    if not pending:
        return None
    msg = pending.get("message")
    if isinstance(msg, str):
        try:
            import json
            msg = json.loads(msg)
        except Exception:
            msg = {"message": msg}
    msg = msg or {}
    if pending.get("kind") == "message":
        return inflight_message(pending["turn_id"], msg.get("message", ""))
    if pending.get("kind") == "resume":
        return inflight_resume(pending["turn_id"], msg.get("parent_turn_id"), msg.get("decision"), msg.get("modifications"))
    return None
