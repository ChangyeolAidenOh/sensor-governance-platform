"""tests/integration/test_graph_driver_equivalence.py — ST4-50 S4b-03: chat graph query_more node ↔ replay driver E path on the
SAME initial state, real Skill servers, test DB, preset 8 (118178), window = first alarm window. $0, no LLM.

Procedure: (1) clean → replay 0..47 → graph session "작업지시 내줘" on the display window (the graph spends the query);
collect its initial / after_query rows from mart.policy_decisions (decision_key = thread:turn:phase), its query plan and
its trace. (2) clean → the driver runs windows 0..t itself (its own spend); collect the same things from
mart.policy_decisions (decision_key = eval:<run>:E:<ep>:<t>:<phase>), mart.query_plans and mart.agent_steps.
Compared: initial and after_query Signals / action / rule / evidence snapshot, plan kind + expand/rerank status, the
ordered Skill call sequence (tool, k). Run-specific UUIDs are compared by ownership (each side's query_request_id is
its own spend request), never by string equality.

If this test disagrees, do NOT update expectations first: tests/unit/test_replay_driver_fake.py pins the expected
decision per branch from the pre-registered rule table — use it and docs/skill_contract.md to decide which side drifted."""
from __future__ import annotations

import asyncio
import json
import math
import os
import uuid

import psycopg
import pytest

from core_pipeline import prereg_constants as P
from core_pipeline.agent import keys as GK
from core_pipeline.agent.plans import spend_origin
from core_pipeline.agent.graph import GraphDeps
from core_pipeline.agent.runner import AgentSession
from core_pipeline.agent.session import start_session
from core_pipeline.agent.turns import TurnStore
from core_pipeline.eval import keys as EK
from core_pipeline.eval.archive import archive_run
from core_pipeline.eval.eval_runs import set_status, start_run
from core_pipeline.eval.replay_driver import ReplayDriver, RunInfo, make_db_deps
from core_pipeline.skills.client import SkillClient, SyncSkillClient
from core_pipeline.skills.replay_client import apply_stage4a_schema, clean_episode, episode_for_preset, replay_episode
from core_pipeline.skills.servers.common import is_test_dsn

DSN = os.environ.get("SGP_TEST_PG_DSN", "")
pytestmark = pytest.mark.skipif(not DSN or not is_test_dsn(DSN) or DSN == os.environ.get("SGP_PG_DSN"),
                                reason="needs SGP_TEST_PG_DSN naming a disposable test DB distinct from SGP_PG_DSN (ST4-09)")
PRESET = 8
COMMIT_TAG = "test-s4b03"
TOL = 1e-6
SKILL_TOOLS = {"gate_check", "detect_diagnose", "spend_query", "retrieve", "work_order_draft", "work_order_approve", "work_order_issue", "work_order_status"}


def _q(sql, *args):
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchall()


def _meta():
    m = json.loads(open("outputs/model_meta/detector_v1.json").read())
    return m["version"], m["input_contract_sha256"]


def _close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and math.isclose(float(a), float(b), abs_tol=TOL)
    return a == b


def _same_signals(x: dict, y: dict) -> list[str]:
    return [k for k in sorted(set(x) | set(y)) if not _close(x.get(k), y.get(k))]


def _same_evidence(x: list, y: list) -> bool:
    return len(x) == len(y) and all(a["clause_key"] == b["clause_key"] and a["grade"] == b["grade"] and _close(a["score"], b["score"]) for a, b in zip(x, y))


def _decision(key: str):
    rows = _q("SELECT action, rule, signals, evidence, decision_id FROM mart.policy_decisions WHERE decision_key=%s", key)
    if not rows:
        return None
    a, r, s, e, d = rows[0]
    return {"action": a, "rule": r, "signals": s if isinstance(s, dict) else json.loads(s), "evidence": e if isinstance(e, list) else json.loads(e or "[]"),
            "decision_id": str(d)}


def _plan(query_id):
    rows = _q("SELECT kind, planned_expand, planned_rerank, expand_status, rerank_status, initial_decision_id FROM mart.query_plans WHERE query_id=%s",
              uuid.UUID(str(query_id)))
    return None if not rows else {"kind": rows[0][0], "planned_expand": rows[0][1], "planned_rerank": rows[0][2], "expand_status": rows[0][3],
                                  "rerank_status": rows[0][4], "initial_decision_id": str(rows[0][5]) if rows[0][5] else None}


@pytest.fixture(scope="module")
def world():
    apply_stage4a_schema(DSN)
    active = _q("SELECT eval_run_id, status FROM mart.eval_runs WHERE status IN ('running','incomplete')")
    if active:
        pytest.skip(f"active eval run {active[0][0]} ({active[0][1]}) on the test DB; archive it first")
    env = {"SGP_PG_DSN": DSN}
    ep = episode_for_preset(DSN, PRESET)
    with SyncSkillClient(env=env) as sk:
        yield {"skills": sk, "env": env, "ep": ep, "detector_contract": _meta()}
    clean_episode(DSN, ep)
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM mart.policy_events WHERE eval_run_id IN (SELECT eval_run_id FROM mart.eval_runs WHERE code_commit=%s)", (COMMIT_TAG,))
        cur.execute("DELETE FROM mart.eval_runs WHERE code_commit=%s", (COMMIT_TAG,))
        conn.commit()


def _graph_side(world) -> dict:
    """Fresh state → graph session on the first alarm window → what the graph recorded."""
    ep = world["ep"]
    clean_episode(DSN, ep)

    async def replay():
        async with SkillClient(env=world["env"]) as sk:
            return await replay_episode(sk, ep, f"equiv-{uuid.uuid4().hex[:6]}", mode="replay")
    rows = asyncio.run(replay())
    alarm = next((r for r in rows if r["alarm"] is True), None)
    if alarm is None:
        pytest.skip("preset 8 did not alarm under the frozen detector")
    t = alarm["t"]
    manifest = {"schema": "demo_presets/1", "presets": [{"preset": PRESET, "episode_id": str(ep), "display_window": t, "kind": "alarm",
                                                          "alarm_id": alarm["alarm_id"], "pred_set": alarm["pred_set"]}]}
    from langgraph.checkpoint.postgres import PostgresSaver
    deps = GraphDeps(dsn=DSN, skills=world["skills"], turns=TurnStore(DSN), manifest=manifest)
    cp_ctx = PostgresSaver.from_conn_string(DSN)
    cp = cp_ctx.__enter__()
    try:
        cp.setup()
        sess = AgentSession(deps, cp)
        tid, secret = start_session(DSN, "aiden", PRESET, str(ep), t)
        turn = str(uuid.uuid4())
        out = sess.invoke(tid, secret, "작업지시 내줘", turn)
    finally:
        cp_ctx.__exit__(None, None, None)
    initial = _decision(GK.decision_key(tid, turn, "initial"))
    assert initial is not None and initial["action"] == out["decision"]["action"]
    after = _decision(GK.decision_key(tid, turn, "after_query")) if initial["action"] == "query_more" else None
    q = out.get("query") or {}
    calls = [(x["tool"], int(x["step"].rsplit(":k", 1)[1]) if x.get("step", "").startswith("retrieve:") else None)
             for x in out["trace"] if x.get("tool") in SKILL_TOOLS]                      # the graph also traces decide_action
    spent = _q("SELECT q_used, query_request_id FROM mart.episode_windows WHERE episode_id=%s AND window_index=%s", ep, t)[0]
    origin = spend_origin(DSN, str(spent[1])) if spent[1] else None
    return {"t": t, "initial": initial, "after": after, "plan": _plan(q["query_id"]) if q.get("query_id") else None, "calls": calls,
            "query_owned": bool(q.get("owned")), "q_used": spent[0], "query_request_id": str(spent[1]) if spent[1] else None,
            "owner_thread": origin["thread_id"] if origin else None, "my_thread": tid}


def _driver_side(world, t: int) -> dict:
    """Fresh state → the driver runs windows 0..t → what the driver recorded."""
    ep = world["ep"]
    clean_episode(DSN, ep)
    detector, contract = world["detector_contract"]
    run = start_run(DSN, [ep], COMMIT_TAG, detector, contract, {"contract_sha256": contract, "detector": detector, "k_primary": 5}, prereg_tag="prereg-v1")
    info = RunInfo(run.eval_run_id, detector, contract, "engineer")
    deps = make_db_deps(DSN, world["skills"], info, os.environ.get("SGP_PROCESSED_DIR", "data/processed/TEP"),
                        os.environ.get("SGP_DETECTOR_DIR", "models/detector_v1"), expected_k5_versions=run.expected_k5_versions)
    driver = ReplayDriver(deps, info)
    try:
        for w in range(t + 1):
            driver.run_window(ep, w)
    finally:
        set_status(DSN, run.eval_run_id, "incomplete")
        archive_run(DSN, run.eval_run_id)
    initial = _decision(EK.decision_key(run.eval_run_id, "E", ep, t, "initial"))
    after = _decision(EK.decision_key(run.eval_run_id, "E", ep, t, "after_query"))
    spent = _q("SELECT q_used, query_request_id FROM mart.episode_windows WHERE episode_id=%s AND window_index=%s", ep, t)[0]
    steps = _q("""SELECT input->>'tool', input->'args'->>'k', request_id FROM mart.agent_steps WHERE run_uuid=%s
                  AND ((input->'args'->>'window_index')::int=%s OR input->'args'->>'alarm_id'=%s) ORDER BY step_id""",
               uuid.UUID(run.eval_run_id), t, f"{ep}:{t}:{detector}")
    calls = [(tool, int(k) if k else None) for tool, k, _ in steps]
    origin = spend_origin(DSN, str(spent[1])) if spent[1] else None
    return {"t": t, "initial": initial, "after": after, "plan": _plan(spent[1]) if spent[1] else None, "calls": calls, "q_used": spent[0],
            "query_request_id": str(spent[1]) if spent[1] else None, "owner_thread": origin["thread_id"] if origin else None,
            "my_thread": EK.thread_id(run.eval_run_id), "spend_request_id": str(EK.request_id(run.eval_run_id, ep, t, EK.step_keys(t)["spend"]))}


def test_graph_and_driver_agree_on_the_same_initial_state(world):
    g = _graph_side(world)
    d = _driver_side(world, g["t"])
    # initial decision: same Signals (snapshot-derived vs envelope-derived), action, rule, evidence (k=5 grounding)
    assert _same_signals(g["initial"]["signals"], d["initial"]["signals"]) == []
    assert (g["initial"]["action"], g["initial"]["rule"]) == (d["initial"]["action"], d["initial"]["rule"])
    assert _same_evidence(g["initial"]["evidence"], d["initial"]["evidence"])
    print("EQUIV initial:", g["initial"]["action"], "rule", g["initial"]["rule"], "g_AB", g["initial"]["signals"].get("g_AB"))
    if g["initial"]["action"] != "query_more":
        assert d["after"] is None and g["after"] is None
        return
    # query path: ownership relations (each side owns its own spend), one plan, same plan outcome
    assert g["query_owned"] and g["q_used"] is True and d["q_used"] is True
    assert g["owner_thread"] == g["my_thread"] and d["owner_thread"] == d["my_thread"]          # each side owns its own spend
    assert d["query_request_id"] == d["spend_request_id"]                                        # driver: the deterministic spend id
    assert g["plan"] is not None and d["plan"] is not None
    for k in ("kind", "planned_expand", "planned_rerank", "expand_status", "rerank_status"):
        assert g["plan"][k] == d["plan"][k], (k, g["plan"], d["plan"])
    assert g["plan"]["initial_decision_id"] == g["initial"]["decision_id"] and d["plan"]["initial_decision_id"] == d["initial"]["decision_id"]
    # after_query decision: same Signals (q_used true, s from the re-read state), action, rule, evidence actually used
    assert g["after"] is not None and d["after"] is not None
    assert _same_signals(g["after"]["signals"], d["after"]["signals"]) == []
    assert (g["after"]["action"], g["after"]["rule"]) == (d["after"]["action"], d["after"]["rule"])
    assert _same_evidence(g["after"]["evidence"], d["after"]["evidence"])
    assert g["after"]["signals"]["q_used"] is True and g["after"]["action"] != "query_more"
    # same ordered Skill call sequence (tool, k) for the window
    assert g["calls"] == d["calls"], (g["calls"], d["calls"])
    print("EQUIV after_query:", g["after"]["action"], "rule", g["after"]["rule"], "plan", g["plan"], "calls", g["calls"])
