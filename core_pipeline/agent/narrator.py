"""core_pipeline/agent/narrator.py — answer text (ST4-15). The narrator is a WRITER only: numbers come from the
envelopes, decisions from decide_action, clause text is quoted verbatim from retrieve payload (contract §2.3: the LLM
never generates or augments clause text). S4a-04 ships the deterministic TemplateNarrator ($0); S4a-05 adds the LLM
narrator behind the same Protocol, with the clause blocks passed as fixed text.
"""
from __future__ import annotations

from typing import Optional, Protocol

REASON_KO = {
    "DEMO_LIMIT": "데모 범위 밖 요청입니다(선택 창 이후의 결과·다른 에피소드는 대화에 노출하지 않습니다).",
    "NO_ALARM": "저장된 알람이 없어 조치안을 만들 수 없습니다.",
    "NO_EVIDENCE": "결정 시점 스냅샷에 τ 이상인 A/B 조항이 없어 초안을 만들 수 없습니다.",
    "NO_PROPOSE_DECISION": "정책 결정이 propose가 아니어서 초안을 만들 수 없습니다.",
    "STALE_DECISION": "이 알람의 최신 결정이 아니어서 초안을 만들 수 없습니다.",
    "APPROVER_MISMATCH": "승인자가 이 세션의 소유자와 일치하지 않습니다.",
    "NO_APPROVAL": "승인 토큰 없이는 작업지시를 발행할 수 없습니다.",
}
ACTION_KO = {"undecidable": "판단 불가(게이트 BLOCK, 탐지 미호출)", "no_alarm": "알람 없음", "hold": "보류",
             "propose": "조치안 제안", "query_more": "추가 조회(확장·재검색)"}


class Narrator(Protocol):
    def narrate(self, state: dict) -> str: ...


def _fmt_p(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.4g}"


def _window_summary(state: dict) -> str:
    st = ((state.get("detect_after") or state.get("detect") or {}).get("state")) or ((state.get("gate") or {}).get("state")) or {}
    gp = ((state.get("gate") or {}).get("payload")) or {}
    if not st and not gp:
        return ""
    t = state.get("display_window")
    pred = st.get("expanded_set") if st.get("expanded_set") is not None else st.get("pred_set")
    lines = [f"창 {t}: gate={st.get('gate', gp.get('gate'))}, drift={st.get('drift', gp.get('drift'))}, "
             f"p={_fmt_p(st.get('p_value'))}, p_adj={_fmt_p(st.get('p_adj'))}, alarm={st.get('alarm')}, m_t={st.get('m_t')}"]
    if st.get("alarm"):
        lines.append(f"진단 집합(후보): {pred} (크기 {len(pred or [])}, top1={st.get('top1')})"
                     + (" — 확장 창 결과" if st.get("expanded_set") is not None else ""))
    return "\n".join(lines)


def _claims_contrast(state: dict) -> str:
    claims = state.get("user_claims") or []
    if not claims:
        return ""
    st = ((state.get("detect_after") or state.get("detect") or {}).get("state")) or {}
    pred = st.get("expanded_set") if st.get("expanded_set") is not None else st.get("pred_set")
    if not st:
        return f"사용자 주장 결함 {claims}: 탐지 결과가 없어 대조할 수 없습니다. 주장은 입력으로 쓰지 않습니다."
    inside = [c for c in claims if c in (pred or [])]
    outside = [c for c in claims if c not in (pred or [])]
    parts = [f"사용자 주장 결함 {claims} vs 탐지 집합 {pred}:"]
    if inside:
        parts.append(f"{inside}은(는) 집합에 포함됩니다.")
    if outside:
        parts.append(f"{outside}은(는) 집합에 없으므로 그 주장을 근거로 제안하지 않습니다.")
    return " ".join(parts)


def _clauses_block(env: Optional[dict], limit: int = 5) -> str:
    if not env or env.get("status") != "OK":
        return ""
    cl = (env.get("payload") or {}).get("clauses") or []
    out = []
    for c in cl[:limit]:
        out.append(f"[{c.get('clause_key')}] ({c.get('grade')}, score {c.get('score'):.3f}) {c.get('title') or ''}\n> {c.get('text') or ''}")
    g = env.get("grounding") or {}
    head = f"검색 mode={g.get('mode')}, τ={g.get('tau')}, g_AB={_fmt_p(g.get('g_AB'))}, g_C={_fmt_p(g.get('g_C'))}, ev_AB={g.get('ev_AB')}"
    return head + ("\n" + "\n".join(out) if out else "\n(조항 없음)")


class TemplateNarrator:
    def narrate(self, state: dict) -> str:
        if state.get("refusal"):
            code = state["refusal"].get("code")
            return REASON_KO.get(code, f"요청을 처리할 수 없습니다({code}).") + (f" {state['refusal'].get('message')}" if state["refusal"].get("message") else "")
        cat = state.get("category")
        parts: list[str] = []
        if state.get("turn_refs_discarded"):
            parts.append(f"요청한 참조 {state.get('turn_refs_note')}는 현재 상태와 달라 무시하고 선택 창 기준으로 답합니다.")
        if cat == "out_of_scope":
            return "그 요청은 이 에이전트의 권한 밖입니다(운전 조건·설정값·인터록 변경은 운전원/엔지니어와 인터록 우회 절차가 필요합니다). 도구를 호출하지 않았습니다."
        if cat == "methodology":
            return METHODOLOGY_KO
        if cat == "history":
            h = state.get("history") or {}
            return (f"이 에피소드(선택 창 {state.get('display_window')}까지): 관측 창 {h.get('n_windows')}개, 알람 창 {h.get('alarm_windows')}, "
                    f"BLOCK 창 {h.get('n_block')}개, 현재 alarm_id={state.get('alarm_id')}. 정책 결정 이력: {h.get('decisions')}")
        ws = _window_summary(state)
        if ws:
            parts.append(ws)
        cc = _claims_contrast(state)
        if cc:
            parts.append(cc)
        if cat == "evidence":
            parts.append(_clauses_block(state.get("retrieve")))
            parts.append("open 모드 결과는 참고용이며 제안의 근거(ev_AB)가 될 수 없습니다.")
            return "\n".join(p for p in parts if p)
        if cat in ("alarm", "diagnosis"):
            if not (((state.get("detect") or {}).get("state") or {}).get("alarm")):
                parts.append("이 창에서는 prefix-BH 알람이 없습니다.")
            return "\n".join(p for p in parts if p)
        # action
        dec = state.get("decision_after") or state.get("decision") or {}
        if dec:
            parts.append(f"정책 결정(E): {ACTION_KO.get(dec.get('action'), dec.get('action'))} — 규칙 {dec.get('rule')} ({dec.get('reason')})")
        q = state.get("query") or {}
        if q:
            parts.append(f"query 상태: {q.get('status')} (expand={q.get('expand_status')}, rerank={q.get('rerank_status')})")
        retr = state.get("retrieve_k10") or state.get("retrieve")
        if retr:
            parts.append(_clauses_block(retr))
        if state.get("draft") and state["draft"].get("status") == "OK":
            d = state["draft"]["payload"]
            parts.append(f"초안 {d.get('proposal_id')} v{d.get('draft_version')} 생성(sha {str(d.get('content_sha256'))[:12]}…).")
        if state.get("draft") and state["draft"].get("status") != "OK":
            r = state["draft"].get("reason") or {}
            parts.append("초안 거부: " + REASON_KO.get(r.get("code"), r.get("code", "")))
        ap = state.get("approval") or {}
        if ap:
            if ap.get("status") == "OK":
                parts.append(f"승인 결과: {ap.get('payload', {}).get('decision')}")
            else:
                parts.append("승인 거부: " + REASON_KO.get((ap.get("reason") or {}).get("code"), str((ap.get("reason") or {}).get("code"))))
        wo = state.get("work_order") or {}
        if wo:
            if wo.get("status") == "OK":
                parts.append(f"작업지시 발행: {wo['payload'].get('work_order_id')} (cmms_ref {wo['payload'].get('cmms_ref')})")
            else:
                parts.append("발행 거부: " + REASON_KO.get((wo.get("reason") or {}).get("code"), str((wo.get("reason") or {}).get("code"))))
        if dec.get("action") == "no_alarm":
            parts.append("선택 가능한 알람이 없어 조치안 경로로 진입하지 않습니다.")
        return "\n".join(p for p in parts if p)


METHODOLOGY_KO = (
    "탐지: 창(20샘플) 단위 Isolation Forest 점수 → split conformal p-value → prefix-BH(α=0.10, 인과적 휴리스틱, FDR 보증 없음; BLOCK 창은 검정 목록 제외). "
    "진단: LightGBM + split conformal 집합(목표 커버리지 0.90, 유효 집합 1≤s≤3). "
    "근거: 예측 집합의 결함→지침 매핑(A/B/C)으로 범위를 제한한 하이브리드 검색 + 리랭커 점수 [0,1], τ=0.5(개발셋 규칙 선택값, 일반화 미입증). "
    "정책 E(사전등록 §4.5): BLOCK→판단 불가, ¬알람→알람 없음, drift critical→보류, 유효 집합∧A/B 근거→제안, 창당 1회 추가 조회, 그 외 보류. "
    "작업지시는 사람 승인 없이는 발행되지 않습니다."
)
