"""tests/integration/test_graph_turns_recovery.py — S4a-04 turn registry, lease and recovery (ST4-16, reviewer 2026-10-21).

Test DB only (SGP_TEST_PG_DSN, name contains 'test'). No servers, no models: a FakeSkills double with the envelope
shapes of contract v1.0 so the only thing under test is the graph runtime + turn state machine:
  * registered but never run → resend executes
  * completed checkpoint, DB record missing → resend recovers the result WITHOUT re-running any Skill call
  * lease expired while the old executor is alive → old executor's next write is refused (LostLease), new one completes;
    its CHECKPOINT write is refused too (LeasedCheckpointer); a NEW turn is refused until the expired turn is recovered
  * resume close is one transaction; a resent finished modify returns its own result and never rewinds the parent's
    newer interrupt (modify v1 → modify v2 → resend v1's request)
  * waiting_approval → new turn refused (TurnBusy); resend returns the interrupt; resume closes; double resume = stored
  * done turn resent after a later turn → that turn's own result, not the thread's latest checkpoint
  * wrong secret → AuthError before any registration
  * record_decision(decision_key): same input → same id; different input → DECISION_MISMATCH (synthetic alarm row)
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

import psycopg
import pytest

from core_pipeline.agent import keys as K
from core_pipeline.agent.graph import GraphDeps
from core_pipeline.agent.runner import AgentSession
from core_pipeline.agent.session import AuthError, SessionAuth, session_scope, start_session, verify_secret
from core_pipeline.agent.turns import LostLease, TurnBusy, TurnConflict, TurnStore
from core_pipeline.policy.decide_action import Decision
from core_pipeline.skills.replay_client import apply_stage4a_schema, dev_episodes
from core_pipeline.skills.servers.common import is_test_dsn

DSN = os.environ.get("SGP_TEST_PG_DSN", "")
pytestmark = pytest.mark.skipif(not DSN or not is_test_dsn(DSN) or DSN == os.environ.get("SGP_PG_DSN"),
                                reason="needs SGP_TEST_PG_DSN naming a disposable test DB distinct from SGP_PG_DSN (ST4-09)")

DET = "detector_v1"
CONTRACT = "808601386456fa9bdd9cc31b186058d4af426f0b887fb27d0caa8fc99b226515"
SNAP = [{"clause_key": "P-132-2013:0:4.4", "grade": "A", "score": 0.91}, {"clause_key": "P-129-2013:0:5.2.2", "grade": "C", "score": 0.3}]


class FakeSkills:
    """Deterministic envelopes keyed by step; counts executions per request_id so 'no re-execution' is observable."""

    def __init__(self, episode_id: str, alarm: bool = True):
        self.ep, self.alarm, self.calls = episode_id, alarm, []
        self.seen: dict[str, dict] = {}
        self.q_used, self.query_request_id, self.expanded_set = False, None, None       # window state (shared, like the DB)
        self.versions: dict[str, int] = {}

    def _env(self, ctx, tool, payload=None, state=None, grounding=None, status="OK", reason=None):
        rid = ctx["request_id"]
        if rid in self.seen:
            out = dict(self.seen[rid]); out["trace"] = {**out["trace"], "replayed": True}
            return out
        self.calls.append((tool, rid))
        env = {"status": status, "payload": payload, "grounding": grounding, "state": state,
               "reason": ({"code": reason, "message": reason} if reason else None),
               "trace": {"skill": "fake", "tool": tool, "contract_version": "1.0", "versions": {"detector": DET, "contract_sha256": CONTRACT}, "replayed": False}}
        self.seen[rid] = env
        return env

    def gate_check(self, ctx, episode_id, window_index):
        return self._env(ctx, "gate_check", {"gate": "PASS", "drift": "ok", "stored": True})

    def detect_diagnose(self, ctx, episode_id, window_index, expand=1, query_id=None):
        aid = f"{episode_id}:{window_index}:{DET}" if self.alarm else None
        if expand == 2:
            if query_id != self.query_request_id:
                return self._env(ctx, "detect_diagnose", status="REFUSED", reason="QUERY_USED" if query_id else "QUERY_REQUIRED")
            self.expanded_set = [1, 4]
        st = {"episode_id": episode_id, "window_index": window_index, "detector_version": DET, "contract_sha256": CONTRACT, "gate": "PASS", "drift": "ok",
              "drift_assessed": True, "p_value": 0.001, "p_adj": 0.002, "m_t": 9, "alarm": self.alarm, "pred_set": [1] if self.alarm else [], "set_size": 1 if self.alarm else 0,
              "top1": 1, "q_used": self.q_used, "query_request_id": self.query_request_id, "expanded_set": self.expanded_set, "alarm_id": aid, "reused": True}
        return self._env(ctx, "detect_diagnose", {"alarm": self.alarm, "alarm_id": aid, "diagnosis": {"set": st["expanded_set"] or st["pred_set"]}}, state=st)

    def retrieve(self, ctx, query, mode, fault_set, k=5, alarm_id=None, query_id=None):
        g = {"mode": mode, "tau": 0.5, "g_AB": 0.91 if mode == "scoped" else None, "g_C": 0.3, "ev_AB": mode == "scoped", "ev_C": False, "snapshot": SNAP}
        clauses = [{"chunk_key": "c1", "clause_key": "P-132-2013:0:4.4", "guide_no": "P-132", "title": "제목", "text": "본문", "page": 1, "grade": "A", "rank": 1, "score": 0.91, "rrf": 0.1}]
        return self._env(ctx, "retrieve", {"clauses": clauses, "k": k, "mode": mode, "scope_size": 1, "n_candidates": 1, "alarm_id": alarm_id, "reused": False}, grounding=g)

    def spend_query(self, ctx, episode_id, window_index):
        if self.q_used and self.query_request_id != ctx["request_id"]:
            return self._env(ctx, "spend_query", status="REFUSED", reason="QUERY_USED")
        self.q_used, self.query_request_id = True, ctx["request_id"]
        return self._env(ctx, "spend_query", {"query_id": ctx["request_id"], "q_used": True, "expand_allowed": True})

    def work_order_draft(self, ctx, **kw):
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, kw["decision_id"]))
        if ctx["request_id"] not in self.seen:
            self.versions[pid] = self.versions.get(pid, 0) + 1
        return self._env(ctx, "work_order_draft", {"proposal_id": pid, "draft_version": self.versions[pid],
                                                   "content_sha256": f"{self.versions[pid]:064x}", "form": {"items": kw["action_items"]}})

    def work_order_approve(self, ctx, draft, approver, decision="approve", modifications=None):
        tok = str(uuid.uuid4()) if decision == "approve" else None
        return self._env(ctx, "work_order_approve", {"approval_token": tok, "decision": decision, "approver": approver})

    def work_order_issue(self, ctx, draft, approval_token):
        if not approval_token:
            return self._env(ctx, "work_order_issue", status="REFUSED", reason="NO_APPROVAL")
        return self._env(ctx, "work_order_issue", {"work_order_id": "WO-1", "cmms_ref": "CMMS-1", "idempotent_replay": False})


def _q(sql, *args):
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchall() if cur.description else None


def _exec(sql, *args):
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql, args)
        conn.commit()


@pytest.fixture(scope="module")
def episode():
    apply_stage4a_schema(DSN)
    return str(dev_episodes(DSN, 1)[0])


def _manifest(ep: str, alarm: bool) -> dict:
    return {"schema": "demo_presets/1", "presets": [{"preset": 1, "episode_id": ep, "display_window": 8, "kind": "alarm" if alarm else "no_alarm",
                                                     "alarm_id": (f"{ep}:8:{DET}" if alarm else None)}]}


def _deps(ep: str, alarm: bool, propose: bool):
    fake = FakeSkills(ep, alarm)
    policy = (lambda sig: Decision("propose", 4, "synthetic propose")) if propose else None
    kw = {"policy": policy} if policy else {}
    return GraphDeps(dsn=DSN, skills=fake, turns=TurnStore(DSN, lease_seconds=5), manifest=_manifest(ep, alarm),
                     record_decision=lambda *a, **k: uuid.uuid4(), **kw), fake


def _session(deps):
    from langgraph.checkpoint.postgres import PostgresSaver
    cp_ctx = PostgresSaver.from_conn_string(DSN)
    cp = cp_ctx.__enter__()
    cp.setup()
    return AgentSession(deps, cp), cp_ctx


def test_wrong_secret_registers_nothing(episode):
    deps, fake = _deps(episode, False, False)
    tid, secret = start_session(DSN, "aiden", 1, episode, 8)
    sess, cp = _session(deps)
    try:
        with pytest.raises(AuthError):
            sess.invoke(tid, secret + "x", "안녕", str(uuid.uuid4()))
        assert deps.turns.list_thread(tid) == [] and fake.calls == []
    finally:
        cp.__exit__(None, None, None)


def test_registered_not_run_then_resend_runs(episode):
    deps, fake = _deps(episode, False, False)
    tid, secret = start_session(DSN, "aiden", 1, episode, 8)
    turn = str(uuid.uuid4())
    deps.turns.register(tid, turn, "message", {"message": "지금 알람 상태는?"}, str(uuid.uuid4()))      # crashed before running
    sess, cp = _session(deps)
    try:
        out = sess.invoke(tid, secret, "지금 알람 상태는?", turn)
        assert out["status"] == "done" and out["category"] == "alarm" and "알람이 없습니다" in out["answer"]
        assert deps.turns.get(tid, turn).status == "done"
        with pytest.raises(TurnConflict):
            sess.invoke(tid, secret, "다른 메시지", turn)
    finally:
        cp.__exit__(None, None, None)


def test_completed_checkpoint_without_db_record_is_recovered_not_rerun(episode):
    deps, fake = _deps(episode, True, False)
    tid, secret = start_session(DSN, "aiden", 1, episode, 8)
    turn = str(uuid.uuid4())
    sess, cp = _session(deps)
    try:
        t = deps.turns.register(tid, turn, "message", {"message": "지금 알람 상태는?"}, str(uuid.uuid4()))
        executor = str(uuid.uuid4())
        assert deps.turns.claim(tid, turn, executor) is not None
        row = verify_secret(DSN, tid, secret)
        with session_scope(SessionAuth(tid, row.owner, secret, executor, turn)):
            sess.graph.invoke({"thread_id": tid, "turn_id": turn, "run_uuid": t.run_uuid, "message": "지금 알람 상태는?", "caller_role": "engineer"},
                              {"configurable": {"thread_id": tid}})                 # checkpoint complete; "crash" before finish()
        n = len(fake.calls)
        _exec("UPDATE mart.graph_turns SET lease_until=now() - interval '1 second' WHERE thread_id=%s AND turn_id=%s", tid, uuid.UUID(turn))
        out = sess.invoke(tid, secret, "지금 알람 상태는?", turn)
        assert out["status"] == "done" and out["answer"] and len(fake.calls) == n         # recovered, not re-executed
        assert deps.turns.get(tid, turn).status == "done"
    finally:
        cp.__exit__(None, None, None)


def test_lease_expired_with_old_executor_alive(episode):
    deps, fake = _deps(episode, False, False)
    tid, secret = start_session(DSN, "aiden", 1, episode, 8)
    turn = str(uuid.uuid4())
    sess, cp = _session(deps)
    try:
        deps.turns.register(tid, turn, "message", {"message": "안녕"}, str(uuid.uuid4()))
        old = str(uuid.uuid4())
        assert deps.turns.claim(tid, turn, old) is not None
        _exec("UPDATE mart.graph_turns SET lease_until=now() - interval '1 second' WHERE thread_id=%s AND turn_id=%s", tid, uuid.UUID(turn))
        out = sess.invoke(tid, secret, "안녕", turn)                                  # new executor reclaims and completes
        assert out["status"] == "done"
        with pytest.raises(LostLease):
            deps.turns.touch(tid, turn, old)                                          # the slow old executor cannot continue
        with pytest.raises(LostLease):
            deps.turns.finish(tid, turn, old, "done", None, {"answer": "stale"})       # nor overwrite the result
        assert deps.turns.get(tid, turn).result["answer"] == out["answer"]
    finally:
        cp.__exit__(None, None, None)


def test_expired_turn_blocks_new_turns_until_recovered(episode):
    deps, fake = _deps(episode, False, False)
    tid, secret = start_session(DSN, "aiden", 1, episode, 8)
    a = str(uuid.uuid4())
    sess, cp = _session(deps)
    try:
        deps.turns.register(tid, a, "message", {"message": "안녕"}, str(uuid.uuid4()))
        old = str(uuid.uuid4())
        assert deps.turns.claim(tid, a, old) is not None
        _exec("UPDATE mart.graph_turns SET lease_until=now() - interval '1 second' WHERE thread_id=%s AND turn_id=%s", tid, uuid.UUID(a))
        with pytest.raises(TurnBusy, match="resend"):                                     # expired running turn still blocks
            sess.invoke(tid, secret, "새 질문", str(uuid.uuid4()))
        p = sess.pending_turn(tid, secret)
        assert p["turn_id"] == a and p["status"] == "running" and p["lease_valid"] is False
        assert sess.invoke(tid, secret, "안녕", a)["status"] == "done"                     # recover by resending
        assert sess.pending_turn(tid, secret) is None
        assert sess.invoke(tid, secret, "새 질문", str(uuid.uuid4()))["status"] == "done"
        # the old executor can neither continue nor write a checkpoint
        config = {"configurable": {"thread_id": tid}}
        n_ck = sum(1 for _ in sess.checkpointer.list(config))
        with session_scope(SessionAuth(tid, "aiden", secret, old, a)):
            with pytest.raises(LostLease):
                sess.checkpointer.put_writes(config, [], "task-x")
            with pytest.raises(LostLease):
                sess.graph.invoke({"thread_id": tid, "turn_id": a, "run_uuid": str(uuid.uuid4()), "message": "안녕", "caller_role": "engineer"}, config)
        assert sum(1 for _ in sess.checkpointer.list(config)) == n_ck
    finally:
        cp.__exit__(None, None, None)


def test_waiting_approval_serialises_and_resume_closes(episode):
    deps, fake = _deps(episode, True, True)
    tid, secret = start_session(DSN, "aiden", 1, episode, 8)
    turn = str(uuid.uuid4())
    sess, cp = _session(deps)
    try:
        out = sess.invoke(tid, secret, "작업지시 내줘", turn)
        assert out["status"] == "waiting_approval" and out["interrupt"]["proposal_id"]
        assert _q("SELECT count(*) FROM mart.graph_turns WHERE thread_id=%s AND status='waiting_approval'", tid)[0][0] == 1
        with pytest.raises(TurnBusy):
            sess.invoke(tid, secret, "다른 질문", str(uuid.uuid4()))                     # new turn refused while waiting
        again = sess.invoke(tid, secret, "작업지시 내줘", turn)                           # resend → the interrupt, no re-execution
        assert again["status"] == "waiting_approval" and again["interrupt"]["proposal_id"] == out["interrupt"]["proposal_id"]
        assert not any(t == "work_order_approve" for t, _ in fake.calls)
        cp.__exit__(None, None, None)
        sess2, cp = _session(deps)                                                        # simulated restart: new checkpointer + graph
        rid = str(uuid.uuid4())
        res = sess2.resume(tid, secret, turn, "approve", rid)
        assert res["status"] == "done" and res["work_order"]["status"] == "OK"
        approvers = [fake.seen[r]["payload"]["approver"] for t, r in fake.calls if t == "work_order_approve"]
        assert approvers == ["aiden"]                                                       # approver = DB owner
        n = len(fake.calls)
        assert sess2.resume(tid, secret, turn, "approve", rid)["work_order"]["status"] == "OK" and len(fake.calls) == n   # idempotent
        with pytest.raises(TurnBusy):
            sess2.resume(tid, secret, turn, "reject", str(uuid.uuid4()))                  # parent is done
        assert deps.turns.get(tid, turn).status == "done"
        out2 = sess2.invoke(tid, secret, "지금 알람 상태는?", str(uuid.uuid4()))             # thread is free again
        assert out2["status"] == "done"
    finally:
        cp.__exit__(None, None, None)


def test_resent_old_modify_does_not_rewind_parent(episode):
    deps, fake = _deps(episode, True, True)
    tid, secret = start_session(DSN, "aiden", 1, episode, 8)
    turn = str(uuid.uuid4())
    sess, cp = _session(deps)
    try:
        out = sess.invoke(tid, secret, "작업지시 내줘", turn)
        assert out["status"] == "waiting_approval" and out["interrupt"]["draft_version"] == 1
        mods = {"action_items": [{"action_type": "inspect", "item": "수정 항목", "clause_key": SNAP[0]["clause_key"]}]}
        r1 = str(uuid.uuid4())
        m1 = sess.resume(tid, secret, turn, "modify", r1, modifications=mods)
        assert m1["status"] == "waiting_approval" and m1["interrupt"]["draft_version"] == 2
        m2 = sess.resume(tid, secret, turn, "modify", str(uuid.uuid4()), modifications=mods)
        assert m2["status"] == "waiting_approval" and m2["interrupt"]["draft_version"] == 3
        parent_before = deps.turns.get(tid, turn)
        assert parent_before.status == "waiting_approval" and parent_before.result["draft_version"] == 3
        again = sess.resume(tid, secret, turn, "modify", r1, modifications=mods)          # resend of the FIRST modify
        assert again["interrupt"]["draft_version"] == 2                                   # its own stored result ...
        parent_after = deps.turns.get(tid, turn)
        assert parent_after.result == parent_before.result and parent_after.status == "waiting_approval"   # ... parent untouched (still v3)
        assert deps.turns.get(tid, r1).status == "done"
        fin = sess.resume(tid, secret, turn, "approve", str(uuid.uuid4()))
        assert fin["status"] == "done" and fin["work_order"]["status"] == "OK" and fin["draft"]["payload"]["draft_version"] == 3
    finally:
        cp.__exit__(None, None, None)


def test_reject_issues_nothing(episode):
    deps, fake = _deps(episode, True, True)
    tid, secret = start_session(DSN, "aiden", 1, episode, 8)
    turn = str(uuid.uuid4())
    sess, cp = _session(deps)
    try:
        assert sess.invoke(tid, secret, "작업지시 내줘", turn)["status"] == "waiting_approval"
        res = sess.resume(tid, secret, turn, "reject", str(uuid.uuid4()))
        assert res["status"] == "done" and res.get("work_order") is None
        assert not any(t == "work_order_issue" for t, _ in fake.calls)
    finally:
        cp.__exit__(None, None, None)


def test_done_turn_resend_returns_its_own_result(episode):
    deps, fake = _deps(episode, False, False)
    tid, secret = start_session(DSN, "aiden", 1, episode, 8)
    sess, cp = _session(deps)
    try:
        a, b = str(uuid.uuid4()), str(uuid.uuid4())
        ra = sess.invoke(tid, secret, "지금 알람 상태는?", a)
        rb = sess.invoke(tid, secret, "이 알람은 어떻게 판단한 거야?", b)
        assert ra["category"] == "alarm" and rb["category"] == "methodology"
        again = sess.invoke(tid, secret, "지금 알람 상태는?", a)
        assert again["category"] == "alarm" and again["answer"] == ra["answer"]
    finally:
        cp.__exit__(None, None, None)


def test_record_decision_key_idempotent_and_mismatch(episode):
    from core_pipeline.skills.work_order import record_decision
    aid = f"{episode}:8:{DET}-synthetic-{uuid.uuid4().hex[:6]}"
    _exec("""INSERT INTO mart.alarms (alarm_id, episode_id, window_index, gate, drift, p_value, p_adj, m_t, alarm, detector)
             VALUES (%s,%s,8,'PASS','ok',0.001,0.002,9,true,%s)""", aid, uuid.UUID(episode), DET)
    try:
        sig = {"gate": "PASS", "drift": "ok", "alarm": True, "s": 1, "g_AB": 0.9, "g_C": None, "q_used": False, "tau": 0.5, "retrieve_mode": "scoped"}
        key = K.decision_key("s-x", "t-x", "initial")
        d1 = record_decision(DSN, aid, "E", sig, "propose", 4, evidence=SNAP, decision_key=key)
        d2 = record_decision(DSN, aid, "E", sig, "propose", 4, evidence=SNAP, decision_key=key)
        assert d1 == d2
        with pytest.raises(ValueError, match="DECISION_MISMATCH"):
            record_decision(DSN, aid, "E", sig, "hold", 6, evidence=SNAP, decision_key=key)
        with pytest.raises(ValueError, match="DECISION_MISMATCH"):
            record_decision(DSN, aid, "E", sig, "propose", 4, evidence=SNAP[:1], decision_key=key)     # same signals, other snapshot
        d3 = record_decision(DSN, aid, "E", sig, "propose", 4, evidence=SNAP)                         # no key: a fresh row
        assert d3 != d1
        assert _q("SELECT count(*) FROM mart.policy_decisions WHERE alarm_id=%s", aid)[0][0] == 2
    finally:
        _exec("DELETE FROM mart.policy_decisions WHERE alarm_id=%s", aid)
        _exec("DELETE FROM mart.alarms WHERE alarm_id=%s", aid)
