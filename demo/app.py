"""demo/app.py — Streamlit demo of the conversational agent (S4a-06).

    streamlit run demo/app.py

The UI calls only start_session() and AgentSession.invoke / resume / pending_turn through demo.runtime (ST4-16):
  * the session secret lives in st.session_state only and is never rendered (a browser refresh ends the session);
  * every call is preceded by an in-flight record {kind: message|resume, client_request_id, payload} so that a Streamlit
    rerun during a turn resends the SAME request id through the SAME public wrapper (ST4-25) — never a new UUID;
  * the approval panel appears only when a result says waiting_approval; approve / modify / reject are the only
    decisions; modify edits the item wording only; a new draft version needs a new approval (ST4-27);
  * what is rendered goes through demo.views allow-lists (ST4-26/27); the LLM mode is fixed at startup and the cost
    panel shows four separate buckets (ST4-28); the runtime is created once per process (ST4-30) and a turn runs
    synchronously under a spinner (ST4-31).
Public mode (SGP_PUBLIC_DEMO=1, demo-public branch): owner is fixed to "public", the session start and every turn pass
demo.public_guard (session / day limits), the questions of outputs/public_demo_questions.json are offered as buttons
(document retrieval is replayed from stored results, so only those questions reach a stored result; the other categories
answer free text), a refused pre-computed retrieval is explained in a notice, the synthetic approval preset is labelled
in the sidebar and in every clause label, and the resource-restart button is hidden.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]                               # repo root on sys.path for `streamlit run demo/app.py`
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_pipeline.agent.manifest import entry_for_preset
from core_pipeline.agent.session import AuthError, start_session
from core_pipeline.agent.turns import TurnBusy, TurnConflict
from demo import public_guard as G
from demo import views as V
from demo.ledger_view import cost_buckets, rows_for_turn
from demo.runtime import DEFAULT_OWNER, DemoConfigError, build_runtime, fingerprint, load_config, mode_summary

load_dotenv(".env")
st.set_page_config(page_title="Process Reliability Agent Demo", layout="wide")


QUESTIONS_PATH = ROOT / "outputs" / "public_demo_questions.json"


@st.cache_resource(show_spinner="런타임 준비 중 (Skill 서버 기동)")
def get_runtime(fp: str):
    rt = build_runtime(load_config())
    import atexit
    atexit.register(rt.close)
    return rt


@st.cache_data
def public_questions() -> list[dict]:
    if not QUESTIONS_PATH.is_file():
        return []
    doc = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return [q for q in doc.get("questions", []) if q.get("text")]


def _session_synthetic() -> bool:
    return st.session_state.get("kind") == "synthetic"


def _reset_session_state() -> None:
    for k in ("thread_id", "secret", "preset", "display_window", "kind", "owner", "turns", "inflight", "pending_approval", "last_turn_id", "notice"):
        st.session_state.pop(k, None)


def _apply_result(result: dict, kind: str) -> None:
    """Common handling of invoke / resume results (ST4-25)."""
    status = result.get("status")
    view = V.result_view(result)
    if status == "done":
        st.session_state.turns.append({"role": "assistant", "view": view})
        st.session_state.pending_approval, st.session_state.inflight = None, None
        st.session_state.last_turn_id = result.get("turn_id")
        if cfg.public and G.public_limit_hit(view):
            st.session_state.notice = ("info", "공개 데모 한계: 이 질문의 문서 검색은 사전계산되어 있지 않아 근거 없이 답했습니다. 추천 질문 버튼의 문장을 그대로 쓰면 저장된 검색 결과가 재생됩니다.")
    elif status == "waiting_approval":
        st.session_state.pending_approval = view["interrupt"]
        st.session_state.inflight = None
        st.session_state.last_turn_id = result.get("turn_id")
        st.session_state.turns.append({"role": "assistant", "view": {"status": "waiting_approval", "turn_id": result.get("turn_id"),
                                                                     "answer": "작업지시 초안이 승인 대기 중입니다 (아래 승인 패널)."}})
    elif status in ("running", "lost_lease"):
        st.session_state.notice = ("warning", f"턴 {result.get('turn_id', '')[:8]}… 상태 {status}: 같은 요청 id로 재시도하세요 (재전송 버튼).")
    elif status == "failed":
        st.session_state.inflight = None
        st.session_state.turns.append({"role": "assistant", "view": view})
        st.session_state.notice = ("error", f"턴 실패: {result.get('error')}")
    else:
        st.session_state.inflight = None
        st.session_state.notice = ("error", f"알 수 없는 상태 {status!r}")


def _send(rt, inflight: dict) -> None:
    """Execute the in-flight record through the matching public wrapper; the record is already stored."""
    tid, secret = st.session_state.thread_id, st.session_state.secret
    try:
        with st.spinner("에이전트 실행 중…"):
            if inflight["kind"] == "message":
                out = rt.invoke(tid, secret, inflight["message"], inflight["client_request_id"])
            else:
                out = rt.resume(tid, secret, inflight["parent_turn_id"], inflight["decision"], inflight["client_request_id"],
                                modifications=inflight.get("modifications"))
    except TurnBusy as e:
        st.session_state.notice = ("warning", f"스레드에 미완료 턴이 있습니다: {e}")
        return
    except TurnConflict as e:
        st.session_state.inflight = None
        st.session_state.notice = ("error", f"요청 id 충돌: {e}")
        return
    except AuthError as e:
        st.session_state.notice = ("error", f"세션 인증 실패: {e}")
        return
    except Exception as e:                                              # the runner already recorded the failed turn
        st.session_state.inflight = None
        st.session_state.notice = ("error", f"{type(e).__name__}: {str(e)[:300]}")
        return
    _apply_result(out, inflight["kind"])


def _submit_message(rt, message: str) -> None:
    if cfg.public:
        why = G.check_turn(cfg.app_dsn, st.session_state.thread_id)
        if why:
            st.session_state.notice = ("warning", why)
            return
    rec = V.inflight_message(str(uuid.uuid4()), message)
    st.session_state.inflight = rec                                     # stored BEFORE the call (ST4-25)
    st.session_state.turns.append({"role": "user", "text": message, "turn_id": rec["client_request_id"]})
    _send(rt, rec)


def _submit_resume(rt, decision: str, modifications: dict | None) -> None:
    pa = st.session_state.pending_approval
    rec = V.inflight_resume(str(uuid.uuid4()), pa["turn_id"], decision, modifications)
    st.session_state.inflight = rec
    st.session_state.turns.append({"role": "user", "text": f"[승인 응답] {decision}" + (" (문안 수정)" if modifications else ""), "turn_id": rec["client_request_id"]})
    _send(rt, rec)


def _reconcile_pending(rt) -> None:
    """ST4-25 step 2: before accepting a new message, ask the DB which turn (if any) is open on this thread."""
    try:
        p = rt.pending(st.session_state.thread_id, st.session_state.secret)
    except AuthError as e:
        st.session_state.notice = ("error", f"세션 인증 실패: {e}")
        return
    if p is None:
        return
    if p["status"] == "waiting_approval" and st.session_state.pending_approval is None:
        rec = V.inflight_from_pending(p)                                # resend → stored interrupt, no re-execution (F1)
        if rec:
            st.session_state.inflight = rec
            _send(rt, rec)
    elif p["status"] in ("registered", "running") and not p.get("lease_valid") and st.session_state.inflight is None:
        st.session_state.inflight = V.inflight_from_pending(p)
        st.session_state.notice = ("warning", f"미완료 턴 {p['turn_id'][:8]}… ({p['status']}, lease 만료) — 재전송 버튼으로 복구하세요.")


# --------------------------------------------------------------------------------------------------------- boot
try:
    cfg = load_config()
except DemoConfigError as e:
    st.error(f"기동 거부: {e}")
    st.stop()
rt = get_runtime(fingerprint(cfg))
mode = mode_summary(cfg, rt.llm_cfg)
st.session_state.setdefault("turns", [])
st.session_state.setdefault("inflight", None)
st.session_state.setdefault("pending_approval", None)

# --------------------------------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("세션")
    if cfg.synthetic:
        st.error("합성 사례 모드 — 데모 아님 (test DB, 승인 계약 검증용)")
    if cfg.public:
        st.caption(f"공개 데모 · 앱 역할 {mode['app_user']} · manifest {Path(mode['manifest']).name}")
    else:
        st.caption(f"DB {mode['database']} · 앱 역할 {mode['app_user']} · manifest {Path(mode['manifest']).name}")
    if "thread_id" not in st.session_state:
        blocked = G.check_new_session(cfg.app_dsn) if cfg.public else None
        if blocked:
            st.warning(blocked)
        if cfg.public:
            owner = G.PUBLIC_OWNER
            st.caption(f"승인자(owner): {owner} (공개 데모에서는 고정)")
        else:
            owner = st.text_input("승인자(owner)", value=DEFAULT_OWNER)
        presets = {int(p["preset"]): p for p in V.visible_presets(rt.manifest)}
        choice = st.selectbox("사례", list(presets), format_func=lambda n: V.preset_label(presets[n]))
        if cfg.public and int(choice) in cfg.synthetic_presets:
            st.error("합성 사례: 실제 탐지 결과가 아닙니다. 근거 등급과 정책 결정을 인위적으로 '제안'으로 고정해 승인 게이트(approve / modify / reject)를 보여 주는 용도입니다.")
        if st.button("세션 시작", type="primary", disabled=(not owner.strip()) or bool(blocked)):
            e = entry_for_preset(rt.manifest, int(choice))
            tid, secret = start_session(cfg.app_dsn, owner.strip(), int(choice), e["episode_id"], int(e["display_window"]))
            _reset_session_state()
            st.session_state.update({"thread_id": tid, "secret": secret, "preset": int(choice), "display_window": int(e["display_window"]),
                                     "kind": e["kind"], "owner": owner.strip(), "turns": [], "inflight": None, "pending_approval": None})
            st.rerun()
    else:
        st.write(f"프리셋 {st.session_state.preset} · 창 {st.session_state.display_window} · {V.KIND_LABELS.get(st.session_state.kind, st.session_state.kind)}")
        st.caption(f"thread {st.session_state.thread_id[:14]}… · 승인자 {st.session_state.owner}")
        if _session_synthetic():
            st.error("합성 사례 세션: 실제 탐지 결과가 아닙니다 (승인 게이트 시연용).")
        if cfg.public:
            u = G.session_usage(cfg.app_dsn, st.session_state.thread_id)
            st.caption(f"이 세션 {u['turns']}/{G.SESSION_MAX_TURNS}턴 · {u['age_min']:.0f}/{G.SESSION_MAX_MINUTES}분")
        if st.button("세션 종료 (새 사례)"):
            _reset_session_state()
            st.rerun()
    st.divider()
    st.header("응답 모드 (기동 시 고정)")
    rmode = "규칙 분류" if mode["router"] == "rule" else "규칙 분류 + LLM 폴백"
    nmode = {"template": "템플릿 서술", "llm": "템플릿 + LLM 요약 (LLM 분류 턴만)", "llm_all": "템플릿 + LLM 요약 (전 턴)"}.get(mode["narrator"], mode["narrator"])
    st.write(f"{rmode} · {nmode}")
    if not mode["paid"]:
        st.write("LLM 호출: **비활성** — 답변은 템플릿과 조항 원문")
    elif mode["budget_set"]:
        st.write("LLM 호출: **활성** (장부 예약 경유)")
    else:
        st.write("LLM 호출: **거부** (`SGP_LLM_BUDGET_USD` 미설정)")
    if cfg.public:
        st.caption(f"활성화 시 사용할 모델: `{mode['model']}` (선정 규칙: 형식 오류 0, 턴당 비용 최소, 지연 최소)")
    else:
        st.caption(f"활성화 시 사용할 모델: `{mode['model']}` — 0-F 규칙 선정(형식 오류 0 → 턴당 비용 최소 → 지연 최소, ST0-23)")
    if mode["budget_set"]:
        st.caption(f"누적 상한 {mode['budget_usd']:.2f} USD (DSN 기준 누적 총액)")
    if not cfg.public:
        st.divider()
        st.header("비용 (장부, 읽기 전용)")
        try:
            b = cost_buckets(cfg.app_dsn)
            st.write(f"확정 정산 **{b['settled_usd']:.6f}** USD ({b['settled_rows']}행)")
            st.write(f"예약 중(PENDING) {b['pending_usd']:.6f} USD ({b['pending_rows']}행)")
            st.write(f"usage 미확인 {b['usage_missing_usd']:.6f} USD ({b['usage_missing_rows']}행) — 미확정")
            st.write(f"초과 미해소 {b['exceeded_unresolved_rows']}행 · 차단 행 {b['blocked_unresolved']}")
        except Exception as e:
            st.caption(f"장부 조회 실패: {type(e).__name__}")
    st.divider()
    if cfg.public:
        st.header("공개 데모 안내")
        st.caption("문서 근거 검색은 사전계산된 추천 질문에 대해 이루어집니다. 상태, 진단, 이력, 방법론, 승인은 사전계산과 무관하게 자유 입력으로 처리됩니다. "
                   "탐지 결과는 사전계산된 저장 행을 그대로 읽고, 작업지시 초안, 승인, 발행은 실시간으로 실행됩니다.")
    else:
        if st.button("리소스 재시작 (Skill 서버·풀)"):
            try:
                rt.close()
            finally:
                get_runtime.clear()
                _reset_session_state()
                st.rerun()

# --------------------------------------------------------------------------------------------------------- main
st.title("Process Reliability Agent Demo")
if "thread_id" not in st.session_state:
    st.info("왼쪽에서 사례를 고르고 세션을 시작하세요. 표시 창 이후의 정보는 대화에 들어오지 않습니다." + ("" if cfg.public else " (ST4-14)"))
    if cfg.public:
        st.markdown("공개 데모입니다. 사례를 고른 뒤 추천 질문 버튼으로 시작해 보세요. 문서 근거 검색은 사전계산된 질문만 재생하고, 나머지 범주(상태, 진단, 이력, 방법론, 작업지시·승인)는 자유 입력도 처리합니다. "
                    "마지막 사례는 승인 게이트를 보여 주기 위한 합성 사례이며 실제 탐지 결과가 아닙니다.")
    st.stop()

pending_notice = st.session_state.pop("notice", None)
if pending_notice and not cfg.public:                                    # public mode shows it next to the input, below the turns
    getattr(st, pending_notice[0])(pending_notice[1])

if st.session_state.inflight:                                            # a rerun interrupted a turn → resend the same id
    rec = st.session_state.inflight
    st.warning(f"진행 중 요청 {rec['client_request_id'][:8]}… ({rec['kind']})")
    if st.button("재전송 (같은 요청 id)"):
        _send(rt, rec)
        st.rerun()

for t in st.session_state.turns:
    if t["role"] == "user":
        with st.chat_message("user"):
            st.write(t["text"])
        continue
    v = t["view"]
    sec = V.turn_sections(v)
    with st.chat_message("assistant"):
        if v.get("refusal"):
            st.warning(f"거부 {v['refusal']['code']}: {v['refusal']['message']}")
        h, j = sec["headline"], sec["judgment"]
        if h:                                                            # 1. outcome first (issued > rejected > refused > decision)
            st.markdown(f"### {h['title']}")
        if j:
            if h and h["kind"] != "decision":
                st.caption(f"정책 결정: {j['label']} ({j['action']}) · 규칙 {j['rule']} · {j['reason']}" + (" · query 이후 재결정" if j["phase"] == "after_query" else ""))
            else:
                st.caption(f"{j['reason']}" + (" · query 이후 재결정" if j["phase"] == "after_query" else ""))
        if sec["state_line"]:                                            # 2. state line = the answer's first line, verbatim
            st.markdown(f"**상태** {sec['state_line']}")
        for line in sec["main_lines"]:                                   # recognised template lines only; the rest stays in the raw answer
            st.markdown(line)
        if cfg.public:                                                   # public demo: the whole template answer is visible (narrator text, no envelopes)
            for line in sec["hidden_lines"]:
                st.markdown(line)
        if sec["clauses"]:                                               # 3. clause list, 4. bodies on demand
            st.markdown("**검색된 조항**" + (f" — {sec['criterion']}" if sec["criterion"] else ""))
            for c in sec["clauses"]:
                with st.expander(V.clause_label(c, synthetic=(cfg.synthetic or _session_synthetic())), expanded=False):
                    st.caption(f"clause_key {c['clause_key']}")
                    for ch in c["chunks"]:                                # every retrieved chunk of the clause, in order
                        sc = f"스냅샷 {ch['score']:.3f}" if isinstance(ch["score"], (int, float)) else "스냅샷 —"
                        ns = f" · 검색 {ch['narrator_score']:.3f}" if isinstance(ch["narrator_score"], (int, float)) else ""
                        meets = "" if ch["meets"] is None else (" · 기준 충족" if ch["meets"] else " · 기준 미충족")
                        st.caption(f"청크 {ch['index']}/{c['n_items']} · {ch['grade'] or ''}등급 · {sc}{ns}{meets}")
                        if ch["lines"]:
                            with st.container(height=min(280, 40 + 24 * len(ch["lines"]))):
                                st.markdown("  \n".join(ch["lines"]))
                        else:
                            st.caption("이 턴의 답변에 이 청크의 본문이 없습니다 (결정 스냅샷의 키만 기록됨).")
        with st.expander("실행 상세 — 답변 원문·정책 입력·승인/발행 봉투·트레이스·장부", expanded=False):   # 5. details
            if sec["hidden_lines"] and not cfg.public:
                st.caption(f"기본 화면에 표시하지 않은 답변 줄 {len(sec['hidden_lines'])}개 — 아래 원문에 포함")
            st.text(v.get("answer") or "")
            cols = st.columns(2)
            cols[0].json({"category": v.get("category"), "decision": v.get("decision"), "decision_after": v.get("decision_after"), "refs_discarded": v.get("turn_refs_discarded")})
            cols[1].json({"query": v.get("query"), "draft": v.get("draft"), "approval": v.get("approval"), "work_order": v.get("work_order")})
            if v.get("trace"):
                st.dataframe(v["trace"], width="stretch", hide_index=True)
            if mode["paid"] and v.get("turn_id"):
                rows = rows_for_turn(cfg.app_dsn, st.session_state.thread_id, v["turn_id"])
                st.caption(f"이번 턴 장부 행 {len(rows)}건 (thread + stage_key로 연결)")
                if rows:
                    st.dataframe(rows, width="stretch", hide_index=True)

pa = st.session_state.pending_approval
if pa:
    st.subheader(f"승인 대기 — 작업지시 초안 v{pa['draft_version']}")
    f = pa["form"]
    st.caption(f"proposal {pa['proposal_id']} · sha256 {pa['content_sha256'][:16]}… · 창 {f.get('window_index')} · 정책 {f.get('policy')} · 승인자 {st.session_state.owner}")
    st.dataframe(f["action_items"], width="stretch", hide_index=True)
    st.write(pa.get("question"))
    c1, c2, c3 = st.columns(3)
    if c1.button("승인 (approve)", type="primary"):
        _submit_resume(rt, "approve", None)
        st.rerun()
    if c3.button("거부 (reject)"):
        _submit_resume(rt, "reject", None)
        st.rerun()
    with c2.expander("문안 수정 후 재제출 (modify)"):
        with st.form("modify_form"):
            texts = [st.text_area(f"{i + 1}. {it['action_type']} · {it['clause_key']}", value=it["item"] or "", key=f"mod_{pa['draft_version']}_{i}")
                     for i, it in enumerate(f["action_items"])]
            if st.form_submit_button("수정안 제출"):
                try:
                    mods = V.modifications_from_items(f["action_items"], texts)
                except ValueError as e:
                    st.error(str(e))
                else:
                    _submit_resume(rt, "modify", mods)
                    st.rerun()
    with st.expander("초안 상세 (근거 키·신호)", expanded=False):
        st.json({"p_adj": f.get("p_adj"), "pred_set": f.get("pred_set"), "evidence_keys": f.get("evidence_keys"), "decision_id": f.get("decision_id")})

if pending_notice and cfg.public:
    getattr(st, pending_notice[0])(pending_notice[1])

if cfg.public and not (pa or st.session_state.inflight):
    qs = [q for q in public_questions() if st.session_state.kind in (q.get("kinds") or [st.session_state.kind])]
    if qs:
        st.caption("추천 질문 (문서 근거 검색이 사전계산된 문장)")
        cols = st.columns(min(4, len(qs)))
        for i, q in enumerate(qs):
            if cols[i % len(cols)].button(q["text"], key=f"q_{q.get('id', i)}"):
                _reconcile_pending(rt)
                if not (st.session_state.inflight or st.session_state.pending_approval):
                    _submit_message(rt, q["text"].strip())
                st.rerun()

msg = st.chat_input("메시지 (예: 지금 어떤 상황이야? / 근거 조항 보여줘 / 작업지시 내줘)", disabled=bool(pa or st.session_state.inflight))
if msg:
    _reconcile_pending(rt)
    if st.session_state.inflight or st.session_state.pending_approval:
        st.rerun()
    _submit_message(rt, msg.strip())
    st.rerun()
