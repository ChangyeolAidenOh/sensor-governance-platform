"""scripts/make_goldens.py — golden GENERATION (phase 1 of two, ST4-12). Fixed models + contract v1.0, sequential replay
from window 0 through the Skill servers, selection rule applied ONCE here, spec assertions checked, then frozen to
tests/regression/golden/<case>.json. Phase 2 (tests/regression/test_golden.py) only compares against the frozen file;
it never re-selects a window and never turns a failure into an xfail.

Cases (contract v1.0 §4):
  G1   preset 2 (fault 4)  : first prefix-BH alarm window in 8..47 → alarm case; none → preserved as a no_alarm case
  G1b  preset 1 (fault 1)  : alarm window must exist (hard spec), 1 ∈ set, scoped ev_AB true
  G2   preset 9 (normal)   : first non-alarm window in 8..47; full alarm vector recorded as a dev observation
  G3   preset 10 (normal)  : BLOCK injected at window 2 (test-only hook) → BLOCK row, detect refused, window 3 proceeds
  G4   preset 8 (fault 18) : real result frozen; the {18} → C-only → ev_AB=false CONTRACT check is tests/unit/test_scope_contract.py
  G5   on G1b's alarm, ONLY if the real policy decision is 'propose': issue without token → NO_APPROVAL; approve → OK.
       The policy decision is recorded AS COMPUTED (dec.action / dec.rule, ST4-13) — never forced to 'propose'; the
       approval mechanics on a synthetic propose are tests/integration/test_h0_servers.py::test_h0_c_… instead.

Usage (test DB only — destructive for the demo episodes' Stage 4a rows):
  SGP_TEST_PG_DSN=postgresql://postgres:postgres@localhost:5436/sgp_test SGP_GATE_REF=<path> \
  python -m scripts.make_goldens --out tests/regression/golden [--cases G1,G1b,G2,G3,G4,G5]
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from core_pipeline import prereg_constants as P
from core_pipeline.policy.decide_action import decide_action
from core_pipeline.policy.signals_adapter import signals_dict, signals_from_envelopes
from core_pipeline.skills.client import SkillClient
from core_pipeline.skills.replay_client import (apply_stage4a_schema, clean_episode, episode_for_preset, replay_episode, scoped_query)
from core_pipeline.skills.servers.common import alarm_id_of, is_test_dsn, new_ctx
from core_pipeline.skills.work_order import record_decision
import psycopg


def _register_thread(dsn: str, thread_id: str, owner: str) -> None:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO mart.threads (thread_id, owner) VALUES (%s,%s) ON CONFLICT (thread_id) DO UPDATE SET owner=EXCLUDED.owner", (thread_id, owner))
        conn.commit()

CASES = {"G1": 2, "G1b": 1, "G2": 9, "G3": 10, "G4": 8}
AB_GUIDES = {1: {"P-132", "P-167", "P-179"}, 4: {"P-53", "P-65", "P-38"}}


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _versions(rows_env: dict, retr_env: dict | None) -> dict:
    v = dict(rows_env["trace"]["versions"])
    if retr_env:
        v.update({f"retrieve.{k}": val for k, val in retr_env["trace"]["versions"].items()})
    return v


async def run_case(case: str, dsn: str, out_dir: Path) -> dict:
    doc: dict = {}
    try:
        return await _run_case(case, dsn, out_dir, doc)
    except AssertionError as e:
        e.doc = doc
        raise


async def _run_case(case: str, dsn: str, out_dir: Path, doc: dict) -> dict:
    preset = CASES[case]
    ep = episode_for_preset(dsn, preset)
    clean_episode(dsn, ep)
    thread = f"golden-{case}-{uuid.uuid4().hex[:8]}"
    env = {"SGP_PG_DSN": dsn}
    if case == "G3":
        env["SGP_TEST_BLOCK_WINDOWS"] = f"{ep}:2"
    doc.update({"case": case, "preset": preset, "episode_id": str(ep), "contract_version": "1.0", "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "code_commit": _git_sha(), "spec": [], "observations": {}})
    async with SkillClient(env=env) as sk:
        last_env: dict = {}
        def keep(t, g, d):
            last_env.update(d or g)
        if case == "G3":
            rows = await replay_episode(sk, ep, thread, t_max=3, on_window=keep)
            r2, r3 = rows[2], rows[3]
            assert r2["gate"] == "BLOCK" and r2["p_value"] is None and r2["alarm"] is None, r2
            assert r2["m_t"] == rows[1]["m_t"], "BLOCK must leave m_t unchanged"
            det = await sk.detect_diagnose(new_ctx(thread), str(ep), 2)
            assert det["status"] == "REFUSED" and det["reason"]["code"] == "GATE_BLOCK", det
            assert r3["gate"] != "BLOCK" and r3["m_t"] == rows[1]["m_t"] + 1, r3
            doc["spec"] = ["window 2 BLOCK row committed by gate_check (p/p_adj/alarm null, m_t unchanged)",
                           "detect_diagnose(2) → REFUSED(GATE_BLOCK)", "window 3 accepted, m_3 = m_1 + 1"]
            doc.update(windows=rows, selected_window=3, kind="block_sequence", versions=_versions(last_env, None))
            return doc
        rows = await replay_episode(sk, ep, thread, on_window=keep)
        active = [r for r in rows if r["t"] >= P.ONSET_WINDOW]
        alarm_rows = [r for r in active if r["alarm"] is True]
        doc["observations"] = {"n_alarm_windows_total": sum(1 for r in rows if r["alarm"]), "alarm_windows": [r["t"] for r in rows if r["alarm"]],
                               "n_block": sum(1 for r in rows if r["gate"] == "BLOCK")}
        if case == "G2":
            sel = next(r for r in active if r["alarm"] is False)
            kind = "no_alarm"
            doc["spec"] = ["selected window: first non-alarm window t ≥ 8", "decide_action(E) → no_alarm", "work_order_draft → REFUSED(NO_ALARM)"]
        elif alarm_rows:
            sel, kind = alarm_rows[0], "alarm"
            doc["spec"] = ["selected window: first prefix-BH alarm window in 8..47"]
        else:
            if case == "G1b":
                raise AssertionError("G1b spec (hard): an alarm window must exist in 8..47 for the fault-1 demo episode — nothing frozen")
            sel, kind = active[0], "no_alarm"
            doc["spec"] = ["no prefix-BH alarm in 8..47 under the frozen detector — preserved as a no_alarm case (ST4-12)",
                           "work_order_draft → REFUSED(NO_ALARM): no draft without a stored alarm"]
        t = sel["t"]
        det = await sk.detect_diagnose(new_ctx(thread), str(ep), t)          # stored row (reused) — same values
        assert det["status"] == "OK" and det["state"]["reused"] is True
        retr = await sk.retrieve(new_ctx(thread), scoped_query(sel["pred_set"]), "scoped", sel["pred_set"], 5, alarm_id=sel["alarm_id"])
        assert retr["status"] == "OK", retr
        sig = signals_from_envelopes(det, retr)
        dec = decide_action(sig, "E")
        doc.update(windows=rows, selected_window=t, kind=kind, versions=_versions(det, retr),
                   retrieve={"query": scoped_query(sel["pred_set"]), "grounding": retr["grounding"],
                             "top_keys": [c["clause_key"] for c in retr["payload"]["clauses"]], "scope_size": retr["payload"]["scope_size"]},
                   decision={"signals": signals_dict(sig), "action": dec.action, "rule": dec.rule})
        fault = {1: 1, 2: 4, 8: 18}.get(preset)
        if kind == "alarm":
            assert sel["alarm"] is True
            if case == "G1b":
                assert 1 in sel["pred_set"], f"G1b spec: 1 ∈ set, got {sel['pred_set']}"
                doc["spec"] += ["1 ∈ diagnosis set (hard)"]
                # grounding is a real-data OBSERVATION, not a hard spec (reviewer 2026-10-21): freeze whichever way it goes
                met = retr["grounding"]["ev_AB"] is True and any(k.split("-")[0] + "-" + k.split("-")[1] in AB_GUIDES[1] for k in doc["retrieve"]["top_keys"] if k.startswith("P-"))
                doc["observations"]["g1b_grounding_expectation_met"] = met
                doc["spec"] += ["scoped ev_AB and P-132/P-167/P-179 in top-5: expected, recorded as observation ("
                                + ("met" if met else "NOT met — valid non-grounding under tau=0.5 with the §2.3 query template; approval path covered by test_h0_servers") + ")"]
            if case == "G1":
                doc["spec"] += [f"fault 4 case: set={sel['pred_set']}, ev_AB={retr['grounding']['ev_AB']} (recorded, contract §4 G1 expectation 4 ∈ set and ev_AB true)"]
                doc["observations"]["g1_contract_expectation_met"] = (4 in sel["pred_set"]) and retr["grounding"]["ev_AB"] is True
            if case == "G4":
                doc["spec"] += [f"fault 18 case: set={sel['pred_set']} — C-only iff set ⊆ {{16..20}}; ev_AB={retr['grounding']['ev_AB']} recorded"]
                doc["observations"]["set_within_16_20"] = all(16 <= f <= 20 for f in sel["pred_set"])
            # CMMS contract on this alarm with the REAL policy decision (never forced to 'propose', ST4-13)
            did = record_decision(dsn, sel["alarm_id"], "E", signals_dict(sig), dec.action, dec.rule, evidence=retr["grounding"]["snapshot"])
            ab = [e for e in retr["grounding"]["snapshot"] if e["grade"] in ("A", "B") and e["score"] >= retr["grounding"]["tau"]]
            item = [{"action_type": "inspect", "item": "관련 계측·제어 루프 점검", "clause_key": (ab[0]["clause_key"] if ab else retr["grounding"]["snapshot"][0]["clause_key"])}]
            draft = await sk.work_order_draft(new_ctx(thread), alarm_id=sel["alarm_id"], episode_id=str(ep), window_index=t, policy="E",
                                              decision_id=str(did), action_items=item)
            if dec.action == "propose":
                assert retr["grounding"]["ev_AB"] is True and draft["status"] == "OK", draft
                doc["cmms"] = {"decision": "propose", "draft": "OK", "draft_version": draft["payload"]["draft_version"]}
                doc["spec"] += ["decision propose → work_order_draft OK"]
                if case == "G1b":                                                     # G5 on the real propose
                    _register_thread(dsn, thread, "oracle")
                    no_tok = await sk.work_order_issue(new_ctx(thread), draft["payload"], None)
                    assert no_tok["status"] == "REFUSED" and no_tok["reason"]["code"] == "NO_APPROVAL", no_tok
                    ap = await sk.work_order_approve(new_ctx(thread), draft["payload"], "oracle")
                    assert ap["status"] == "OK" and ap["payload"]["approval_token"]
                    iss = await sk.work_order_issue(new_ctx(thread), draft["payload"], ap["payload"]["approval_token"])
                    assert iss["status"] == "OK", iss
                    doc["cmms"]["g5"] = {"issue_without_token": "REFUSED(NO_APPROVAL)", "issue_with_token": "OK"}
                    doc["spec"] += ["G5: work_order_issue without token → REFUSED(NO_APPROVAL); with token → OK"]
            else:
                assert draft["status"] == "REFUSED" and draft["reason"]["code"] == "NO_PROPOSE_DECISION", draft
                doc["cmms"] = {"decision": dec.action, "draft": "REFUSED(NO_PROPOSE_DECISION)"}
                doc["spec"] += [f"decision {dec.action} (rule {dec.rule}) → work_order_draft REFUSED(NO_PROPOSE_DECISION)"]
                if case == "G1b":
                    doc["observations"]["g5_not_in_golden"] = "real decision is not propose; approval mechanics covered by test_h0_servers (synthetic propose)"
        else:
            # no alarm row exists, so no policy_decisions row can be written (FK); the draft is refused upstream (NO_ALARM)
            draft = await sk.work_order_draft(new_ctx(thread), alarm_id=alarm_id_of(ep, t, det["trace"]["versions"]["detector"]), episode_id=str(ep),
                                              window_index=t, policy="E", decision_id=str(uuid.uuid4()), action_items=[{"action_type": "inspect", "item": "x", "clause_key": "P-116-2012:0:5.1"}])
            assert draft["status"] == "REFUSED" and draft["reason"]["code"] == "NO_ALARM", draft
            assert dec.action == "no_alarm", dec
            doc["cmms"] = {"draft": "REFUSED(NO_ALARM)"}
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/regression/golden")
    ap.add_argument("--cases", default="G1,G1b,G2,G3,G4")
    a = ap.parse_args()
    dsn = os.environ.get("SGP_TEST_PG_DSN", "")
    if not dsn or not is_test_dsn(dsn) or dsn == os.environ.get("SGP_PG_DSN"):
        sys.exit("SGP_TEST_PG_DSN must name a disposable *test* database distinct from SGP_PG_DSN (ST4-09)")
    apply_stage4a_schema(dsn)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    failed = []
    for case in a.cases.split(","):
        case = case.strip()
        try:
            doc = asyncio.run(run_case(case, dsn, out))
        except AssertionError as e:                       # spec not met: keep the observation, freeze nothing
            doc = getattr(e, "doc", None) or {"case": case, "error": str(e)}
            doc["spec_failure"] = str(e)
            (out / f"{case}.failed.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            print(f"{case}: SPEC FAILED — {e}  (observation in {case}.failed.json, not a golden)")
            failed.append(case)
            continue
        (out / f"{case}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"{case}: kind={doc.get('kind')} window={doc.get('selected_window')} decision={(doc.get('decision') or {}).get('action')} spec={len(doc['spec'])} obs={doc.get('observations')}")
    if failed:
        sys.exit(f"spec failures: {failed}")


if __name__ == "__main__":
    main()
