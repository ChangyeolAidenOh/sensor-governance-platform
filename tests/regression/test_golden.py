"""tests/regression/test_golden.py — golden REGRESSION (phase 2, ST4-12): replays each frozen case through the Skill
servers and compares against tests/regression/golden/<case>.json. Fixed window, fixed versions, fixed expectations.
No re-selection, no xfail. A version mismatch fails with 'golden stale' — regenerate deliberately with
scripts/make_goldens.py and commit the new file with a decision-log entry.

Runs only with SGP_TEST_PG_DSN (destructive for the demo episodes' Stage 4a rows) and the model files present.
  SGP_TEST_PG_DSN=... SGP_GATE_REF=... pytest tests/regression -m golden
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import uuid
from pathlib import Path
from uuid import UUID

import pytest

GOLDEN_DIR = Path("tests/regression/golden")
DSN = os.environ.get("SGP_TEST_PG_DSN", "")
pytestmark = pytest.mark.golden

REQUIRED = ["G1", "G1b", "G2", "G3", "G4"]          # every case must be frozen; a missing or failed case fails the suite
CASES = REQUIRED


def _skip_unless_ready():
    from core_pipeline.skills.servers.common import is_test_dsn
    if not DSN or not is_test_dsn(DSN) or DSN == os.environ.get("SGP_PG_DSN"):
        pytest.skip("SGP_TEST_PG_DSN (disposable test DB, distinct from SGP_PG_DSN) is required")
    if not Path(os.environ.get("SGP_GATE_REF", "data/processed/gate_reference_v1.parquet")).exists() or not Path("models/retrieval_v1/dense.npy").exists():
        pytest.skip("models / SGP_GATE_REF not available")


def _close(a, b, tol=1e-9):
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=tol)


@pytest.mark.parametrize("case", CASES)
def test_golden(case):
    _skip_unless_ready()
    failed = GOLDEN_DIR / f"{case}.failed.json"
    if failed.exists():
        pytest.fail(f"{case}: generation recorded a spec failure — {json.loads(failed.read_text())['spec_failure']}")
    if not (GOLDEN_DIR / f"{case}.json").exists():
        pytest.fail(f"{case}: golden not generated (scripts/make_goldens.py)")
    from core_pipeline.skills.client import SkillClient
    from core_pipeline.skills.replay_client import apply_stage4a_schema, clean_episode, replay_episode
    from core_pipeline.skills.servers.common import new_ctx
    from core_pipeline.policy.signals_adapter import signals_from_envelopes
    from core_pipeline.policy.decide_action import decide_action

    gold = json.loads((GOLDEN_DIR / f"{case}.json").read_text(encoding="utf-8"))
    ep = UUID(gold["episode_id"])
    apply_stage4a_schema(DSN)
    clean_episode(DSN, ep)
    env = {"SGP_PG_DSN": DSN}
    if gold["kind"] == "block_sequence":
        env["SGP_TEST_BLOCK_WINDOWS"] = f"{ep}:2"
    thread = f"regr-{case}-{uuid.uuid4().hex[:6]}"

    async def go():
        async with SkillClient(env=env) as sk:
            t_sel = gold["selected_window"]
            t_max = t_sel if gold["kind"] == "block_sequence" else len(gold["windows"]) - 1      # no_alarm / alarm: the whole episode
            rows = await replay_episode(sk, ep, thread, t_max=t_max)
            # versions must match the frozen ones (detector / diagnoser / gate calib / contract) — else the golden is stale
            det_any = await sk.detect_diagnose(new_ctx(thread), str(ep), next(r["t"] for r in rows if r["gate"] != "BLOCK"))
            live = det_any["trace"]["versions"]
            for k in ("detector", "diagnoser", "calib_sha256", "contract_sha256", "alpha", "tau"):
                assert live.get(k) == gold["versions"].get(k), f"golden stale: {k} {live.get(k)} != {gold['versions'].get(k)}"
            assert len(rows) == len(gold["windows"][: t_max + 1])
            for r_live, r_gold in zip(rows, gold["windows"][: t_max + 1]):
                for k in ("t", "gate", "drift", "alarm", "m_t", "pred_set", "top1"):
                    assert r_live[k] == r_gold[k], f"{case} window {r_gold['t']} {k}: {r_live[k]} != {r_gold[k]}"
                assert _close(r_live["p_value"], r_gold["p_value"]) and _close(r_live["p_adj"], r_gold["p_adj"]), (case, r_gold["t"])
            if gold["kind"] == "block_sequence":
                det = await sk.detect_diagnose(new_ctx(thread), str(ep), 2)
                assert det["status"] == "REFUSED" and det["reason"]["code"] == "GATE_BLOCK"
                return
            sel = rows[t_sel]
            det = await sk.detect_diagnose(new_ctx(thread), str(ep), t_sel)
            retr = await sk.retrieve(new_ctx(thread), gold["retrieve"]["query"], "scoped", sel["pred_set"], 5, alarm_id=sel["alarm_id"])
            assert retr["status"] == "OK"
            g_live, g_gold = retr["grounding"], gold["retrieve"]["grounding"]
            for k in ("retrieve.embedder", "retrieve.reranker"):
                assert retr["trace"]["versions"][k.split(".", 1)[1]] == gold["versions"][k], f"golden stale: {k}"
            assert g_live["ev_AB"] == g_gold["ev_AB"] and g_live["ev_C"] == g_gold["ev_C"], (g_live, g_gold)
            assert _close(g_live["g_AB"], g_gold["g_AB"], 1e-6) and _close(g_live["g_C"], g_gold["g_C"], 1e-6)
            assert [c["clause_key"] for c in retr["payload"]["clauses"]] == gold["retrieve"]["top_keys"]
            dec = decide_action(signals_from_envelopes(det, retr), "E")
            assert dec.action == gold["decision"]["action"] and dec.rule == gold["decision"]["rule"]
            if gold["kind"] == "no_alarm":
                from core_pipeline.skills.servers.common import alarm_id_of
                d = await sk.work_order_draft(new_ctx(thread), alarm_id=alarm_id_of(ep, t_sel, live["detector"]), episode_id=str(ep), window_index=t_sel,
                                              policy="E", decision_id=str(uuid.uuid4()), action_items=[{"action_type": "inspect", "item": "x", "clause_key": "P-116-2012:0:5.1"}])
                assert d["status"] == "REFUSED" and d["reason"]["code"] == "NO_ALARM"
            else:
                from core_pipeline.skills.work_order import record_decision
                from core_pipeline.policy.signals_adapter import signals_dict
                import psycopg
                did = record_decision(DSN, sel["alarm_id"], "E", signals_dict(signals_from_envelopes(det, retr)), dec.action, dec.rule, evidence=g_live["snapshot"])
                ab = [e for e in g_live["snapshot"] if e["grade"] in ("A", "B") and e["score"] >= g_live["tau"]]
                key = ab[0]["clause_key"] if ab else g_live["snapshot"][0]["clause_key"]
                dr = await sk.work_order_draft(new_ctx(thread), alarm_id=sel["alarm_id"], episode_id=str(ep), window_index=t_sel, policy="E",
                                               decision_id=str(did), action_items=[{"action_type": "inspect", "item": "점검", "clause_key": key}])
                if gold["cmms"]["decision"] == "propose":
                    assert dr["status"] == "OK", dr
                    if gold["cmms"].get("g5"):
                        with psycopg.connect(DSN) as conn, conn.cursor() as cur:
                            cur.execute("INSERT INTO mart.threads (thread_id, owner) VALUES (%s,'oracle') ON CONFLICT (thread_id) DO NOTHING", (thread,)); conn.commit()
                        no = await sk.work_order_issue(new_ctx(thread), dr["payload"], None)
                        assert no["status"] == "REFUSED" and no["reason"]["code"] == "NO_APPROVAL"
                        ap = await sk.work_order_approve(new_ctx(thread), dr["payload"], "oracle")
                        ok = await sk.work_order_issue(new_ctx(thread), dr["payload"], ap["payload"]["approval_token"])
                        assert ok["status"] == "OK"
                else:
                    assert dr["status"] == "REFUSED" and dr["reason"]["code"] == "NO_PROPOSE_DECISION", dr
    asyncio.run(go())
