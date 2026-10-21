"""tests/integration/test_h0_servers.py — S4a-00/01: the two FastMCP servers through the client (the graph's path).
  H0-(d)  BLOCK window → no detection; the BLOCK row is committed by gate_check (m_t unchanged) and window t+1 proceeds
          (ST4-11); the refused call is logged as a violation.
  H0-(c)  no work order without approval: issue(None) → REFUSED(NO_APPROVAL); approve (thread owner only) → issue OK.
  S4a-00  detect → retrieve → decide_action chain across the process boundary (lightgbm ↔ torch never co-loaded).
  ST4-13  request ownership (same request_id: replay / different ctx → REQUEST_ID_REUSED), partial-commit completion
          (BLOCK window row, alarm rows), one logical query per window (spend_query → query_id for expand=2 and k=10),
          scoped k=5 computed once per alarm, approver must be the registered thread owner.

Runs ONLY against SGP_TEST_PG_DSN (ST4-09); destructive for two dev episodes' Stage 4a rows. Needs models + SGP_GATE_REF.
  SGP_TEST_PG_DSN=<test DSN> SGP_GATE_REF=<path> pytest tests/integration/test_h0_servers.py -x -s
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

import psycopg
import pytest

DSN = os.environ.get("SGP_TEST_PG_DSN", "")


def _ready():
    from core_pipeline.skills.servers.common import is_test_dsn
    if not DSN or not is_test_dsn(DSN) or DSN == os.environ.get("SGP_PG_DSN"):
        pytest.skip("SGP_TEST_PG_DSN (disposable test DB, distinct from SGP_PG_DSN) is required")
    if not Path(os.environ.get("SGP_GATE_REF", "data/processed/gate_reference_v1.parquet")).exists() or not Path("models/diagnoser_v1/lgb_model.txt").exists():
        pytest.skip("models / SGP_GATE_REF not available")


@pytest.fixture(scope="module")
def episodes():
    _ready()
    from core_pipeline.skills.replay_client import apply_stage4a_schema, clean_episode, dev_episodes
    apply_stage4a_schema(DSN)
    eps = dev_episodes(DSN, 2)
    assert len(eps) == 2, "need two dev episodes"
    for e in eps:
        clean_episode(DSN, e)
    return eps


def _q(sql, *args):
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchall()


def _count(sql, *args):
    return _q(sql, *args)[0][0]


def _exec(sql, *args):
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql, args)
        conn.commit()


def _register_thread(thread_id: str, owner: str):
    _exec("INSERT INTO mart.threads (thread_id, owner) VALUES (%s,%s) ON CONFLICT (thread_id) DO UPDATE SET owner=EXCLUDED.owner", thread_id, owner)


def test_h0_d_block_order_recovery_query_and_chain(episodes):
    from core_pipeline.skills.client import SkillClient
    from core_pipeline.skills.servers.common import new_ctx
    from core_pipeline.skills.replay_client import scoped_query
    from core_pipeline.policy.signals_adapter import signals_from_envelopes
    from core_pipeline.policy.decide_action import decide_action
    ep = episodes[0]
    thread = f"h0d-{uuid.uuid4().hex[:6]}"
    env = {"SGP_PG_DSN": DSN, "SGP_TEST_BLOCK_WINDOWS": f"{ep}:2"}

    async def go():
        async with SkillClient(env=env) as sk:
            # detect before gate is refused
            d = await sk.detect_diagnose(new_ctx(thread), str(ep), 0)
            assert d["status"] == "REFUSED" and d["reason"]["code"] == "GATE_NOT_RUN", d
            g0 = await sk.gate_check(new_ctx(thread), str(ep), 0)
            assert g0["status"] == "OK" and g0["payload"]["gate"] in ("PASS", "FLAG") and g0["state"] is None
            d0 = await sk.detect_diagnose(new_ctx(thread), str(ep), 0)
            assert d0["status"] == "OK" and d0["state"]["m_t"] == 1 and d0["payload"]["detection_run"] is True
            g5 = await sk.gate_check(new_ctx(thread), str(ep), 5)
            assert g5["status"] == "REFUSED" and g5["reason"]["code"] == "WINDOW_ORDER"
            await sk.gate_check(new_ctx(thread), str(ep), 1); d1 = await sk.detect_diagnose(new_ctx(thread), str(ep), 1)
            assert d1["status"] == "OK" and d1["state"]["m_t"] == 2
            # ---- window 2: injected input-contract violation → BLOCK row committed by gate_check
            g2 = await sk.gate_check(new_ctx(thread), str(ep), 2)
            assert g2["status"] == "OK" and g2["payload"]["gate"] == "BLOCK" and g2["payload"]["block_reasons"]
            assert g2["state"]["gate"] == "BLOCK" and g2["state"]["p_value"] is None and g2["state"]["alarm"] is None and g2["state"]["m_t"] == 2
            n_rows = _count("SELECT count(*) FROM mart.episode_windows WHERE episode_id=%s", ep)
            ctx_v = new_ctx(thread)
            d2 = await sk.detect_diagnose(ctx_v, str(ep), 2)
            assert d2["status"] == "REFUSED" and d2["reason"]["code"] == "GATE_BLOCK", d2
            assert _count("SELECT count(*) FROM mart.episode_windows WHERE episode_id=%s", ep) == n_rows, "refused call must write nothing"
            assert _count("SELECT count(*) FROM mart.agent_steps WHERE request_id=%s AND violation", ctx_v["request_id"]) == 1
            # ---- partial-commit recovery (ST4-13): gate row stored, window row lost → gate_check completes it
            _exec("DELETE FROM mart.episode_windows WHERE episode_id=%s AND window_index=2", ep)
            g2b = await sk.gate_check(new_ctx(thread), str(ep), 2)
            assert g2b["status"] == "OK" and g2b["payload"]["stored"] is True and g2b["state"]["gate"] == "BLOCK" and g2b["state"]["m_t"] == 2, g2b
            # ---- window 3 proceeds; m_3 = m_1 + 1
            g3 = await sk.gate_check(new_ctx(thread), str(ep), 3); d3 = await sk.detect_diagnose(new_ctx(thread), str(ep), 3)
            assert g3["status"] == "OK" and d3["status"] == "OK" and d3["state"]["m_t"] == 3, d3
            d3b = await sk.detect_diagnose(new_ctx(thread), str(ep), 3)
            assert d3b["state"]["reused"] is True and d3b["state"]["p_adj"] == d3["state"]["p_adj"]
            # ---- request ownership
            ctx_r = new_ctx(thread)
            a = await sk.gate_check(ctx_r, str(ep), 3); b = await sk.gate_check(ctx_r, str(ep), 3)
            assert b["trace"]["replayed"] is True and a["payload"] == b["payload"]
            c = await sk.gate_check(ctx_r, str(ep), 1)
            assert c["status"] == "REFUSED" and c["reason"]["code"] == "REQUEST_ID_REUSED"
            other_ctx = {**ctx_r, "thread_id": "someone-else"}                      # same id + args, different context
            c2 = await sk.gate_check(other_ctx, str(ep), 3)
            assert c2["status"] == "REFUSED" and c2["reason"]["code"] == "REQUEST_ID_REUSED"
            assert _count("SELECT count(*) FROM mart.agent_steps WHERE status='PENDING'") == 0
            # ---- one logical query per window: spend → expand (window 3 after a BLOCK predecessor is refused, query stays spent)
            e_no = await sk.detect_diagnose(new_ctx(thread), str(ep), 3, expand=2)
            assert e_no["status"] == "REFUSED" and e_no["reason"]["code"] == "QUERY_REQUIRED"
            ctx_q = new_ctx(thread)
            sq = await sk.spend_query(ctx_q, str(ep), 3)
            assert sq["status"] == "OK" and sq["payload"]["query_id"] == ctx_q["request_id"] and sq["state"]["q_used"] is True, sq
            sq_again = await sk.spend_query(new_ctx(thread), str(ep), 3)
            assert sq_again["status"] == "REFUSED" and sq_again["reason"]["code"] == "QUERY_USED"
            e3 = await sk.detect_diagnose(new_ctx(thread), str(ep), 3, expand=2, query_id=sq["payload"]["query_id"])
            assert e3["status"] == "REFUSED" and e3["reason"]["code"] == "EXPANSION_INVALID", e3
            (q_used, qrid, exp), = _q("SELECT q_used, query_request_id, expanded_set FROM mart.episode_windows WHERE episode_id=%s AND window_index=3", ep)
            assert q_used is True and str(qrid) == sq["payload"]["query_id"] and exp is None
            e_wrong = await sk.detect_diagnose(new_ctx(thread), str(ep), 3, expand=2, query_id=str(uuid.uuid4()))
            assert e_wrong["status"] == "REFUSED" and e_wrong["reason"]["code"] == "QUERY_USED"
            # window 1 (predecessor 0 non-BLOCK): expansion runs once; a second call with the same query_id returns the stored result
            sq1 = await sk.spend_query(new_ctx(thread), str(ep), 1)
            e1 = await sk.detect_diagnose(new_ctx(thread), str(ep), 1, expand=2, query_id=sq1["payload"]["query_id"])
            assert e1["status"] == "OK" and e1["payload"]["detection_run"] is False and e1["payload"]["diagnosis"]["calibration_window"] == 40, e1
            assert e1["state"]["expanded_set"] is not None and e1["state"]["q_used"] is True
            e1b = await sk.detect_diagnose(new_ctx(thread), str(ep), 1, expand=2, query_id=sq1["payload"]["query_id"])
            assert e1b["status"] == "OK" and e1b["state"]["reused"] is True and e1b["state"]["expanded_set"] == e1["state"]["expanded_set"]
            e0 = await sk.spend_query(new_ctx(thread), str(ep), 0)
            assert e0["status"] == "OK" and e0["payload"]["expand_allowed"] is False
            e0x = await sk.detect_diagnose(new_ctx(thread), str(ep), 0, expand=2, query_id=e0["payload"]["query_id"])
            assert e0x["status"] == "REFUSED" and e0x["reason"]["code"] == "NO_EXPANSION"
            # ---- S4a-00 chain: detect → retrieve → decide_action across the two processes
            sel = d3
            aid = sel["payload"]["alarm_id"]
            r = await sk.retrieve(new_ctx(thread), scoped_query(sel["state"]["pred_set"]), "scoped", sel["state"]["pred_set"], 5, alarm_id=aid)
            assert r["status"] == "OK" and r["grounding"]["mode"] == "scoped" and r["grounding"]["tau"] == 0.5 and r["payload"]["reused"] is False, r
            dec = decide_action(signals_from_envelopes(sel, r), "E")
            assert dec.action in ("no_alarm", "propose", "hold", "query_more"), dec
            if aid:
                assert _count("SELECT count(*) FROM mart.evidence WHERE alarm_id=%s AND mode='scoped' AND k=5", aid) == len(r["payload"]["clauses"])
                r_again = await sk.retrieve(new_ctx(thread), "다른 질의문", "scoped", sel["state"]["pred_set"], 5, alarm_id=aid)
                assert r_again["payload"]["reused"] is True and r_again["payload"]["clauses"] == r["payload"]["clauses"]
                assert _count("SELECT count(*) FROM mart.evidence WHERE alarm_id=%s AND mode='scoped' AND k=5", aid) == len(r["payload"]["clauses"]), "no duplicate evidence"
                wrong = await sk.retrieve(new_ctx(thread), "x y z", "scoped", [19], 5, alarm_id=aid)
                assert wrong["status"] == "REFUSED" and wrong["reason"]["code"] == "ARG_NOT_IN_STATE"
                k10 = await sk.retrieve(new_ctx(thread), scoped_query(sel["state"]["pred_set"]), "scoped", sel["state"]["pred_set"], 10, alarm_id=aid)
                assert k10["status"] == "REFUSED" and k10["reason"]["code"] == "QUERY_REQUIRED"
                k10b = await sk.retrieve(new_ctx(thread), scoped_query(sel["state"]["pred_set"]), "scoped", sel["state"]["pred_set"], 10, alarm_id=aid, query_id=sq["payload"]["query_id"])
                assert k10b["status"] == "OK" and k10b["payload"]["k"] == 10, k10b
                # partial-commit recovery: alarm rows lost after observe() → the next detect completes them
                _exec("DELETE FROM mart.evidence WHERE alarm_id=%s", aid); _exec("DELETE FROM mart.diagnoses WHERE alarm_id=%s", aid); _exec("DELETE FROM mart.alarms WHERE alarm_id=%s", aid)
                d3c = await sk.detect_diagnose(new_ctx(thread), str(ep), 3)
                assert d3c["status"] == "OK" and d3c["payload"]["alarm_id"] == aid
                assert _count("SELECT count(*) FROM mart.alarms a JOIN mart.diagnoses d USING (alarm_id) WHERE alarm_id=%s", aid) == 1
            o = await sk.retrieve(new_ctx(thread), "경보 우선순위 부여 방법", "open", [], 5)
            assert o["status"] == "OK" and o["grounding"]["ev_AB"] is False
            bad = await sk.retrieve(new_ctx(thread, caller_role="restricted"), "경보 발생 시 조치", "open", [], 5)
            assert bad["status"] == "OK"
    asyncio.run(go())


def test_h0_c_no_work_order_without_approval(episodes):
    from core_pipeline.skills.client import SkillClient
    from core_pipeline.skills.servers.common import alarm_id_of, new_ctx
    from core_pipeline.skills.state_store import DetectionStateStore
    from core_pipeline.skills.work_order import record_decision
    ep = episodes[1]
    meta = json.loads(Path("outputs/model_meta/detector_v1.json").read_text())
    store = DetectionStateStore(DSN, meta["version"], meta["input_contract_sha256"])
    rec = store.observe(ep, 0, "PASS", "ok", False, 0.001, [4], 4)
    assert rec.alarm is True
    aid = alarm_id_of(ep, 0, meta["version"])
    _exec("""INSERT INTO mart.alarms (alarm_id, episode_id, window_index, gate, drift, p_value, p_adj, m_t, alarm, detector)
             VALUES (%s,%s,0,'PASS','ok',0.001,0.001,1,true,%s) ON CONFLICT DO NOTHING""", aid, ep, meta["version"])
    snap = [{"clause_key": "P-53-2012:0:6.7", "grade": "A", "score": 0.98}, {"clause_key": "P-116-2012:0:5.1", "grade": "C", "score": 0.7}]
    sig = {"gate": "PASS", "drift": "ok", "alarm": True, "s": 1, "g_AB": 0.98, "g_C": 0.7, "q_used": False, "tau": 0.5, "retrieve_mode": "scoped"}
    did = record_decision(DSN, aid, "E", sig, "propose", 4, evidence=snap)          # synthetic propose: approval mechanics only
    thread, other = f"h0c-{uuid.uuid4().hex[:6]}", f"other-{uuid.uuid4().hex[:6]}"
    _register_thread(thread, "oracle")
    item = [{"action_type": "inspect", "item": "반응기 냉각수 제어 루프 점검", "clause_key": "P-53-2012:0:6.7"}]

    async def go():
        async with SkillClient(env={"SGP_PG_DSN": DSN}) as sk:
            dr = await sk.work_order_draft(new_ctx(thread), alarm_id=aid, episode_id=str(ep), window_index=0, policy="E", decision_id=str(did), action_items=item)
            assert dr["status"] == "OK" and dr["payload"]["draft_version"] == 1, dr
            draft = dr["payload"]
            # G5 / H0-(c): no token → no work order
            no = await sk.work_order_issue(new_ctx(thread), draft, None)
            assert no["status"] == "REFUSED" and no["reason"]["code"] == "NO_APPROVAL", no
            assert _count("SELECT count(*) FROM mart.work_orders WHERE proposal_id=%s", draft["proposal_id"]) == 0
            # citation outside the decision's A/B snapshot → refused; forbidden wording → refused
            bad = await sk.work_order_draft(new_ctx(thread), alarm_id=aid, episode_id=str(ep), window_index=0, policy="E", decision_id=str(did),
                                            action_items=[{"action_type": "inspect", "item": "x", "clause_key": "P-116-2012:0:5.1"}])
            assert bad["status"] == "REFUSED", bad
            forb = await sk.work_order_draft(new_ctx(thread), alarm_id=aid, episode_id=str(ep), window_index=0, policy="E", decision_id=str(did),
                                             action_items=[{"action_type": "inspect", "item": "인터록 해제 후 점검", "clause_key": "P-53-2012:0:6.7"}])
            assert forb["status"] == "REFUSED", forb
            # approval boundary: only the registered thread owner may approve; a tool call cannot invent an approver
            imp = await sk.work_order_approve(new_ctx(thread), draft, "agent")
            assert imp["status"] == "REFUSED" and imp["reason"]["code"] == "APPROVER_MISMATCH", imp
            unreg = await sk.work_order_approve(new_ctx(other), draft, "oracle")
            assert unreg["status"] == "REFUSED" and unreg["reason"]["code"] == "APPROVER_MISMATCH"
            tam = dict(draft); tam["form"] = {**draft["form"], "action_items": [{**item[0], "item": "다른 내용"}]}
            t1 = await sk.work_order_approve(new_ctx(thread), tam, "oracle")
            assert t1["status"] == "REFUSED" and t1["reason"]["code"] == "APPROVAL_MISMATCH", t1
            ap = await sk.work_order_approve(new_ctx(thread), draft, "oracle")
            assert ap["status"] == "OK" and ap["payload"]["approval_token"], ap
            tok = ap["payload"]["approval_token"]
            wrong_thread = await sk.work_order_issue(new_ctx(other), draft, tok)
            assert wrong_thread["status"] == "REFUSED" and wrong_thread["reason"]["code"] == "THREAD_MISMATCH"
            ok = await sk.work_order_issue(new_ctx(thread), draft, tok)
            assert ok["status"] == "OK" and ok["payload"]["idempotent_replay"] is False, ok
            again = await sk.work_order_issue(new_ctx(thread), draft, tok)
            assert again["status"] == "OK" and again["payload"]["idempotent_replay"] is True and again["payload"]["work_order_id"] == ok["payload"]["work_order_id"]
            assert _count("SELECT count(*) FROM mart.work_orders WHERE proposal_id=%s", draft["proposal_id"]) == 1
            st = await sk.work_order_status(new_ctx(thread), draft["proposal_id"])
            assert st["status"] == "OK" and len(st["payload"]["work_orders"]) == 1 and st["payload"]["approvals"][0]["token_used"] is True
            record_decision(DSN, aid, "E", sig, "hold", 6, evidence=snap)
            stale = await sk.work_order_draft(new_ctx(thread), alarm_id=aid, episode_id=str(ep), window_index=0, policy="E", decision_id=str(did), action_items=item)
            assert stale["status"] == "REFUSED" and stale["reason"]["code"] == "STALE_DECISION", stale
    asyncio.run(go())
