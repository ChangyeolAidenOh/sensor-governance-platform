"""tests/unit/test_baselines_static.py — ST4-50 completion criterion 2 (call boundary) for the baseline path, statically:
baselines.py must not reach any query_more output or any DB / Skill / state code, and PolicyInput must not even have
such fields. Plus the behavioural contract of signals_from_snapshot and the equivalence with signals_from_envelopes on
a pre-query envelope pair (chat graph E-initial == replay E-initial)."""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

from core_pipeline.policy.baselines import decide_baseline, decide_e_initial, evidence_for, signals_from_snapshot
from core_pipeline.policy.decide_action import Signals, decide_action
from core_pipeline.policy.policy_input import K5Grounding, PolicyInput
from core_pipeline.policy.signals_adapter import signals_from_block_gate, signals_from_envelopes

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "core_pipeline" / "policy" / "baselines.py"

FORBIDDEN_FIELDS = {"expanded_set", "q_used", "query_request_id", "detect_x", "detect_after", "retrieve_k10", "query"}
FORBIDDEN_ATTRS = FORBIDDEN_FIELDS | {"spend_query", "detect_diagnose", "retrieve", "record_decision", "connect", "cursor", "execute"}
FORBIDDEN_CALLS = {"spend_query", "detect_diagnose", "retrieve", "record_decision", "connect", "open", "exec", "eval"}
ALLOWED_IMPORTS = {"__future__", "core_pipeline.policy.decide_action", "core_pipeline.policy.policy_input"}

CONTRACT = "c" * 64
DETECTOR = "detector_v1"
EP = "11111111-1111-1111-1111-111111111111"
AID = f"{EP}:8:{DETECTOR}"


def _k5(g_ab=0.7, g_c=0.2, tau=0.5):
    return K5Grounding(request_id="22222222-2222-2222-2222-222222222222", fault_set=[1], mode="scoped", tau=tau, g_ab=g_ab, g_c=g_c,
                       ev_ab=(None if tau is None else (g_ab is not None and g_ab >= tau)), ev_c=(None if tau is None else (g_c is not None and g_c >= tau)),
                       snapshot=[{"clause_key": "P-132-2013:0:4.8", "grade": "A", "score": g_ab or 0.0}],
                       versions={"contract_sha256": CONTRACT, "detector": DETECTOR, "tau": tau})


def _snap(alarm=True, gate="PASS", drift="ok", pred_set=(1,), k5="auto"):
    k = _k5() if (k5 == "auto" and alarm) else (None if k5 == "auto" else k5)
    return PolicyInput(EP, 8, DETECTOR, CONTRACT, "engineer", gate, drift, True, 0.01 if gate != "BLOCK" else None,
                       0.02 if gate != "BLOCK" else None, 9, alarm if gate != "BLOCK" else None, AID if alarm else None,
                       list(pred_set), len(pred_set), pred_set[0] if pred_set else None, k, None, "")


# ----------------------------------------------------------------------------------------------- static boundary
def test_baselines_source_has_no_query_state_access_and_no_io():
    tree = ast.parse(BASELINES.read_text(encoding="utf-8"))
    bad_attrs = sorted({n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in FORBIDDEN_ATTRS})
    bad_names = sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in FORBIDDEN_ATTRS})
    bad_calls = sorted({(n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", None))
                        for n in ast.walk(tree) if isinstance(n, ast.Call)} & FORBIDDEN_CALLS)
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    assert not bad_attrs and not bad_names and not bad_calls, (bad_attrs, bad_names, bad_calls)
    assert mods <= ALLOWED_IMPORTS, mods - ALLOWED_IMPORTS


def test_policy_input_has_no_query_state_fields():
    fields = {f.name for f in dataclasses.fields(PolicyInput)} | {f.name for f in dataclasses.fields(K5Grounding)}
    assert not (fields & FORBIDDEN_FIELDS), fields & FORBIDDEN_FIELDS


# ----------------------------------------------------------------------------------------------- behaviour
def test_signals_from_snapshot_uses_pred_set_and_never_q_used():
    sig = signals_from_snapshot(_snap(pred_set=(1, 2)))
    assert sig.s == 2 and sig.q_used is False and sig.retrieve_mode == "scoped" and sig.tau == 0.5 and sig.g_AB == 0.7
    sig0 = signals_from_snapshot(_snap(alarm=False))
    assert sig0.alarm is False and sig0.g_AB is None and sig0.tau is None and sig0.retrieve_mode is None and sig0.ev_AB is None


def test_block_snapshot_matches_block_gate_signals():
    snap = _snap(alarm=False, gate="BLOCK", pred_set=())
    gate_env = {"status": "OK", "payload": {"gate": "BLOCK", "drift": "ok"}, "state": {"gate": "BLOCK", "drift": "ok", "q_used": False}}
    assert signals_from_snapshot(snap) == signals_from_block_gate(gate_env)
    for p in ("B1", "B3"):
        assert decide_baseline(snap, p).action == "undecidable"
    assert decide_e_initial(snap).rule == 1


def test_baseline_decisions_follow_prereg_rows():
    assert decide_baseline(_snap(alarm=False), "B1").action == "no_alarm"
    assert decide_baseline(_snap(), "B1").action == "propose"
    assert decide_baseline(_snap(drift="critical"), "B1").action == "propose"           # B1 ignores drift
    assert decide_baseline(_snap(), "B3").action == "propose"                           # ev_AB true
    assert decide_baseline(_snap(k5=_k5(g_ab=0.1)), "B3").action == "hold"              # ev_AB false
    assert decide_baseline(_snap(k5=_k5(g_ab=0.9, tau=None)), "B3").action == "hold"    # tau unfixed → ev_AB None
    assert decide_e_initial(_snap(drift="critical")).rule == 3
    assert decide_e_initial(_snap(k5=_k5(g_ab=0.1))).action == "query_more"            # rule 5 open: q_used False by construction
    assert evidence_for(_snap())[0]["grade"] == "A" and evidence_for(_snap(alarm=False)) == []
    try:
        decide_baseline(_snap(), "E")
        assert False, "E is not a baseline"
    except ValueError:
        pass


def test_e_initial_equals_chat_graph_signals_on_pre_query_envelopes():
    """signals_from_snapshot(snap) == signals_from_envelopes(detect, retrieve) for the envelopes BEFORE any query."""
    snap = _snap(pred_set=(1,))
    detect = {"status": "OK", "payload": {"alarm_id": AID},
              "state": {"gate": "PASS", "drift": "ok", "alarm": True, "pred_set": [1], "expanded_set": None, "q_used": False,
                        "alarm_id": AID, "contract_sha256": CONTRACT}}
    retrieve = {"status": "OK", "payload": {"alarm_id": AID, "k": 5, "mode": "scoped"},
                "grounding": {"mode": "scoped", "tau": 0.5, "g_AB": 0.7, "g_C": 0.2, "ev_AB": True, "ev_C": False, "snapshot": snap.k5.snapshot},
                "trace": {"versions": {"contract_sha256": CONTRACT}}}
    assert signals_from_snapshot(snap) == signals_from_envelopes(detect, retrieve)
    assert decide_e_initial(snap) == decide_action(signals_from_envelopes(detect, retrieve), "E")
