"""core_pipeline/policy/baselines.py — B1 / B3 / E-initial decisions from the first-observation snapshot only (ST4-50).

Pure functions. This module must not import DB, Skill-client, state-store or replay code and must not touch any
query_more output (expanded_set, detect_x, detect_after, retrieve_k10, spend_query): tests/unit/test_baselines_static.py
checks that with an AST walk; the PolicyInput type has no such fields in the first place.
"""
from __future__ import annotations

from core_pipeline.policy.decide_action import Decision, Signals, decide_action
from core_pipeline.policy.policy_input import PolicyInput

BASELINE_POLICIES = ("B1", "B3")


def signals_from_snapshot(snap: PolicyInput) -> Signals:
    """Signals of the window as first observed: s = |pred_set| (never an expanded set), q_used = False (no query yet),
    grounding from the first scoped k=5 envelope or None when the window has no alarm."""
    k5 = snap.k5
    return Signals(gate=snap.gate, drift=snap.drift, alarm=bool(snap.alarm) if snap.alarm is not None else False,
                   s=len(snap.pred_set), g_AB=k5.g_ab if k5 else None, g_C=k5.g_c if k5 else None, q_used=False,
                   tau=k5.tau if k5 else None, retrieve_mode=k5.mode if k5 else None)


def decide_baseline(snap: PolicyInput, policy: str) -> Decision:
    if policy not in BASELINE_POLICIES:
        raise ValueError(f"decide_baseline handles {BASELINE_POLICIES}, got {policy!r}")
    return decide_action(signals_from_snapshot(snap), policy)


def decide_e_initial(snap: PolicyInput) -> Decision:
    """E's initial decision from the same input B1/B3 see; the after_query decision is the driver's E path."""
    return decide_action(signals_from_snapshot(snap), "E")


def evidence_for(snap: PolicyInput) -> list[dict]:
    """The evidence snapshot a B3 / E-initial decision records: the first k=5 grounding snapshot, [] without an alarm."""
    return list(snap.k5.snapshot) if snap.k5 else []
