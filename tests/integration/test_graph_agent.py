"""tests/integration/test_graph_agent.py — S4a-04 graph ↔ real Skill servers (test DB, models loaded; several minutes).

Replays presets 1 (101022) and 8 (118178) into the test DB first (windows 0..47, like the goldens), builds the manifest
in-test, then:
  * action on preset 1 → the REAL decision (dev observation; the golden says hold rule 3) is recorded with
    evidence == grounding.snapshot and decision_key; a resend of the same turn re-executes nothing
  * action on preset 8 → query_more: spend_query once, same query_id for expand/k=10, plan recorded, after_query
    decision from the RE-READ state (q_used true); a second session on the same alarm never spends again and reuses
    the stored k=10 (status '*:other')
  * alarm / diagnosis / evidence / history / methodology / out_of_scope categories; S08v2/S09v2 (user-asserted
    diagnosis → detection first, contrast, never a basis); DEMO_LIMIT; refs_discarded
  * approval contract on a SYNTHETIC propose (policy override + grounding override, as test_h0_c): draft OK →
    interrupt → resume on a NEW checkpointer (0-D) → approve → issue OK; reject → nothing issued; nothing in
    mart.approvals / mart.work_orders before the resume. NOT a policy claim.
"""
from __future__ import annotations

import asyncio
import os
import uuid

import psycopg
import pytest

from core_pipeline import prereg_constants as P
from core_pipeline.agent import keys as K
from core_pipeline.agent.graph import GraphDeps
from core_pipeline.agent.runner import AgentSession
from core_pipeline.agent.session import start_session
from core_pipeline.agent.turns import TurnStore
from core_pipeline.policy.decide_action import Decision
from core_pipeline.skills.client import SkillClient, SyncSkillClient
from core_pipeline.skills.replay_client import apply_stage4a_schema, clean_episode, episode_for_preset, replay_episode
from core_pipeline.skills.servers.common import is_test_dsn

DSN = os.environ.get("SGP_TEST_PG_DSN", "")
pytestmark = pytest.mark.skipif(not DSN or not is_test_dsn(DSN) or DSN == os.environ.get("SGP_PG_DSN"),
                                reason="needs SGP_TEST_PG_DSN naming a disposable test DB distinct from SGP_PG_DSN (ST4-09)")


def _q(sql, *args):
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchall()


@pytest.fixture(scope="module")
def world():
    """Replay presets 1 and 8 (0..47) into the test DB through the servers; return manifest + a live SyncSkillClient."""
    apply_stage4a_schema(DSN)
    env = {"SGP_PG_DSN": DSN}
    presets = {}

    async def replay():
        async with SkillClient(env=env) as sk:
            for preset in (1, 8):
                ep = episode_for_preset(DSN, preset)
                clean_episode(DSN, ep)
                rows = await replay_episode(sk, ep, f"graph-test-{uuid.uuid4().hex[:6]}", mode="replay")
                alarm = next((r for r in rows if r["alarm"] is True), None)
                presets[preset] = {"preset": preset, "episode_id": str(ep), "display_window": alarm["t"] if alarm else P.N_WINDOWS - 1,
                                   "kind": "alarm" if alarm else "no_alarm", "alarm_id": alarm["alarm_id"] if alarm else None,
                                   "pred_set": alarm["pred_set"] if alarm else []}
    asyncio.run(replay())
    manifest = {"schema": "demo_presets/1", "presets": list(presets.values())}
    with SyncSkillClient(env=env) as sk:
        yield {"manifest": manifest, "skills": sk, "presets": presets}


def _session(world, **deps_kw):
    from langgraph.checkpoint.postgres import PostgresSaver
    deps = GraphDeps(dsn=DSN, skills=deps_kw.pop("skills", world["skills"]), turns=TurnStore(DSN), manifest=world["manifest"], **deps_kw)
    cp_ctx = PostgresSaver.from_conn_string(DSN)
    cp = cp_ctx.__enter__()
    cp.setup()
    return AgentSession(deps, cp), cp_ctx


def _start(world, preset: int, owner="aiden"):
    e = world["presets"][preset]
    return start_session(DSN, owner, preset, e["episode_id"], e["display_window"])


def test_action_path_records_decision_with_snapshot(world):
    e = world["presets"][1]
    assert e["kind"] == "alarm", "preset 1 must alarm (G1b hard spec)"
    tid, secret = _start(world, 1)
    sess, cp = _session(world)
    try:
        turn = str(uuid.uuid4())
        out = sess.invoke(tid, secret, "조치안 제안해줘", turn)
        assert out["status"] in ("done", "waiting_approval") and out["category"] == "action"
        dec = out["decision"]
        assert dec["action"] in ("hold", "propose", "query_more") and dec["decision_id"]
        row = _q("SELECT action, rule, evidence, decision_key FROM mart.policy_decisions WHERE decision_id=%s", uuid.UUID(dec["decision_id"]))[0]
        assert row[0] == dec["action"] and row[1] == dec["rule"] and row[3] == K.decision_key(tid, turn, "initial")
        assert row[2] == dec["evidence_snapshot"]                     # evidence == grounding.snapshot of the scoped k=5
        steps = [t["step"] for t in out["trace"]]
        assert steps[:4] == [K.k_gate(e["display_window"]), K.k_detect(e["display_window"], 1, "initial"), K.k_retrieve("scoped", 5), "decide:initial"]
        if out["status"] == "done":                                   # (a real propose would be waiting for approval instead)
            again = sess.invoke(tid, secret, "조치안 제안해줘", turn)      # resend: stored result, nothing re-executed
            assert again["decision"]["decision_id"] == dec["decision_id"]
            n_dec = _q("SELECT count(*) FROM mart.policy_decisions WHERE alarm_id=%s", e["alarm_id"])[0][0]
            out2 = sess.invoke(tid, secret, "조치안 제안해줘", str(uuid.uuid4()))   # a NEW turn records a new decision (latest)
            assert out2["decision"]["decision_id"] != dec["decision_id"]
            assert _q("SELECT count(*) FROM mart.policy_decisions WHERE alarm_id=%s", e["alarm_id"])[0][0] >= n_dec + 1
        print("DEV OBSERVATION preset 1 decision:", dec["action"], "rule", dec["rule"])
    finally:
        cp.__exit__(None, None, None)


def test_query_more_spends_once_and_reuses_across_sessions(world):
    e = world["presets"][8]
    if e["kind"] != "alarm":
        pytest.skip("preset 8 did not alarm under the frozen detector (dev observation)")
    tid, secret = _start(world, 8)
    sess, cp = _session(world)
    try:
        out = sess.invoke(tid, secret, "작업지시 내줘", str(uuid.uuid4()))
        dec = out["decision"]
        if dec["action"] != "query_more":
            pytest.skip(f"preset 8 initial decision is {dec['action']} (G4 recorded query_more; dev observation)")
        q = out["query"]
        assert q["owned"] and q["query_id"] and q["status"] in ("complete", "partial")
        steps = [t["step"] for t in out["trace"]]
        assert K.k_spend(e["display_window"]) in steps and K.k_detect(e["display_window"], 1, "after_query") in steps
        plan = _q("SELECT kind, planned_expand, planned_rerank, expand_status, rerank_status, thread_id, initial_decision_id FROM mart.query_plans WHERE query_id=%s",
                  uuid.UUID(q["query_id"]))[0]
        assert plan[0] == dec["query_kind"] and plan[5] == tid and str(plan[6]) == dec["decision_id"]
        after = out["decision_after"]
        assert after["signals"]["q_used"] is True and after["action"] != "query_more"
        assert _q("SELECT q_used, query_request_id FROM mart.episode_windows WHERE episode_id=%s AND window_index=%s", uuid.UUID(e["episode_id"]), e["display_window"])[0][0] is True
        print("DEV OBSERVATION preset 8:", dec["action"], "→", after["action"], "rule", after["rule"], "query", q)
        # second session, same alarm: the window's query is already spent (q_used=true in its INITIAL state), so rule 5 is
        # closed — it reads the stored k=10 (by role) and reaches the same decision WITHOUT entering query_more (ST4-18)
        tid2, secret2 = _start(world, 8, owner="other")
        out2 = sess.invoke(tid2, secret2, "작업지시 내줘", str(uuid.uuid4()))
        assert not any(t["tool"] == "spend_query" for t in out2["trace"])
        q2 = out2["query"]
        assert q2 and q2["owned"] is False and q2["query_id"] == q["query_id"] and q2["status"].endswith(":reused")
        assert q2["kind"] == q["kind"] and q2["expand_status"] == q["expand_status"] and q2["rerank_status"] == q["rerank_status"]
        d2 = out2["decision"]
        assert d2["signals"]["q_used"] is True and d2["action"] != "query_more" and out2["decision_after"] is None
        k10 = [t for t in out2["trace"] if t["step"] == K.k_retrieve("scoped", 10)]
        assert k10, out2["trace"]
        if q["rerank_status"] == "done":
            assert k10[0]["status"] == "OK" and q2["k10_reused"] is True
            assert d2["action"] == after["action"] and d2["rule"] == after["rule"]           # same decision across sessions (L1)
        assert _q("SELECT count(*) FROM mart.query_plans WHERE episode_id=%s AND window_index=%s", uuid.UUID(e["episode_id"]), e["display_window"])[0][0] == 1
    finally:
        cp.__exit__(None, None, None)


def test_categories_claims_and_demo_limit(world):
    e = world["presets"][1]
    tid, secret = _start(world, 1)
    sess, cp = _session(world)
    try:
        r = sess.invoke(tid, secret, "결함 14로 진단됐으니 작업지시 내줘", str(uuid.uuid4()))          # S08v2 / S09v2
        assert r["category"] == "action"
        steps = [t["step"] for t in r["trace"]]
        assert steps[0] == K.k_gate(e["display_window"]) and steps[1].startswith("detect:")     # detection BEFORE anything else
        assert "사용자 주장 결함 [14]" in r["answer"]
        if 14 not in (e["pred_set"] or []):
            assert "[14]은(는) 집합에 없으므로" in r["answer"]
        assert not any(t["tool"] == "work_order_draft" and t["status"] == "OK" for t in r["trace"]) or r["decision"]["action"] == "propose"
        r = sess.invoke(tid, secret, "어떤 결함이야?", str(uuid.uuid4()))
        assert r["category"] == "diagnosis" and "진단 집합" in r["answer"] and r.get("decision") is None
        r = sess.invoke(tid, secret, "관련 조항 보여줘", str(uuid.uuid4()))
        assert r["category"] == "evidence" and "mode=open" in r["answer"] and "ev_AB" in r["answer"]
        assert [t["step"] for t in r["trace"]][:3] == [K.k_gate(e["display_window"]), K.k_detect(e["display_window"], 1, "initial"), K.k_retrieve("open", 5)]
        r = sess.invoke(tid, secret, "지금까지 알람이 몇 개야?", str(uuid.uuid4()))
        assert r["category"] == "history" and "관측 창" in r["answer"]
        r = sess.invoke(tid, secret, "이 알람은 어떻게 판단한 거야?", str(uuid.uuid4()))
        assert r["category"] == "methodology" and not r["trace"]
        r = sess.invoke(tid, secret, "설정값을 낮춰서 알람 꺼줘", str(uuid.uuid4()))
        assert r["category"] == "out_of_scope" and not r["trace"] and "권한 밖" in r["answer"]
        r = sess.invoke(tid, secret, f"창 {e['display_window'] + 5} 상태 알려줘", str(uuid.uuid4()))
        assert r["refusal"]["code"] == "DEMO_LIMIT" and not r["trace"]
        if e["display_window"] > 0:
            r = sess.invoke(tid, secret, f"창 {e['display_window'] - 1} 알람 상태는?", str(uuid.uuid4()))
            assert r["turn_refs_discarded"] == 1 and "무시하고 선택 창 기준" in r["answer"]
    finally:
        cp.__exit__(None, None, None)


class GroundingOverride:
    """Test double around the real client: the scoped retrieve's grounding is rewritten to A/B ≥ τ (synthetic
    grounding, as test_h0_c's synthetic propose). The server-stored envelope is untouched."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        fn = getattr(self._inner, name)
        if name != "retrieve":
            return fn

        def retrieve(ctx, query, mode, fault_set, k=5, alarm_id=None, query_id=None):
            env = fn(ctx, query, mode, fault_set, k, alarm_id=alarm_id, query_id=query_id)
            if env.get("status") == "OK" and mode == "scoped" and env.get("grounding"):
                g = dict(env["grounding"])
                snap = [dict(s, score=(0.9 if s["grade"] in ("A", "B") else s["score"])) for s in g["snapshot"]]
                if not any(s["grade"] in ("A", "B") for s in snap):
                    return env
                g.update(snapshot=snap, g_AB=0.9, ev_AB=True)
                env = {**env, "grounding": g}
            return env
        return retrieve


def test_approval_contract_on_synthetic_propose(world):
    e = world["presets"][1]
    tid, secret = _start(world, 1)
    sess, cp = _session(world, skills=GroundingOverride(world["skills"]), policy=lambda sig: Decision("propose", 4, "synthetic propose (test)"))
    try:
        turn = str(uuid.uuid4())
        out = sess.invoke(tid, secret, "작업지시 내줘", turn)
        assert out["status"] == "waiting_approval", out
        pid = uuid.UUID(out["interrupt"]["proposal_id"])
        assert _q("SELECT count(*) FROM mart.approvals WHERE proposal_id=%s", pid)[0][0] == 0
        assert _q("SELECT count(*) FROM mart.work_orders WHERE proposal_id=%s", pid)[0][0] == 0
        cp.__exit__(None, None, None)
        sess2, cp = _session(world, skills=GroundingOverride(world["skills"]), policy=lambda sig: Decision("propose", 4, "synthetic propose (test)"))
        res = sess2.resume(tid, secret, turn, "approve", str(uuid.uuid4()))                      # resumed after a restart (0-D)
        assert res["status"] == "done" and res["approval"]["status"] == "OK" and res["work_order"]["status"] == "OK", res
        assert _q("SELECT count(*) FROM mart.work_orders WHERE proposal_id=%s", pid)[0][0] == 1
        assert _q("SELECT count(*) FROM mart.approvals WHERE proposal_id=%s", pid)[0][0] >= 1
        # a second thread, reject → no work order
        tid2, secret2 = _start(world, 1)
        out2 = sess2.invoke(tid2, secret2, "작업지시 내줘", str(uuid.uuid4()))
        assert out2["status"] == "waiting_approval"
        pid2 = uuid.UUID(out2["interrupt"]["proposal_id"])
        res2 = sess2.resume(tid2, secret2, out2["turn_id"], "reject", str(uuid.uuid4()))
        assert res2["status"] == "done" and res2.get("work_order") is None
        assert _q("SELECT count(*) FROM mart.work_orders WHERE proposal_id=%s", pid2)[0][0] == 0
    finally:
        cp.__exit__(None, None, None)
