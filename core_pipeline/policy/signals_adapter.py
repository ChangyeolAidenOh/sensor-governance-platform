"""core_pipeline/policy/signals_adapter.py — the ONLY way Signals are built in Stage 4a (ST4-07): from validated Skill
envelopes, never from LLM text. retrieve_mode stays None without a scoped retrieve, which makes ev_AB None (not grounding).
S4a-04: signals_from_block_gate builds the rule-1 Signals from a gate_check envelope whose committed state is BLOCK
(detect_diagnose is never called on a BLOCK window, contract §1.1), so the graph reaches `undecidable` through
decide_action like every other action and never hard-codes it.
"""
from __future__ import annotations

from typing import Optional

from core_pipeline.policy.decide_action import Signals


def signals_from_envelopes(detect: dict, retrieve: Optional[dict] = None) -> Signals:
    """detect = detect_diagnose envelope (status OK, with state); retrieve = retrieve envelope or None."""
    if detect.get("status") != "OK" or not detect.get("state"):
        raise ValueError("signals need an OK detect_diagnose envelope with a committed state row")
    st = detect["state"]
    pred = st["expanded_set"] if st.get("expanded_set") is not None else st["pred_set"]
    g_ab = g_c = tau = None
    mode = None
    if retrieve is not None and retrieve.get("status") == "OK" and retrieve.get("grounding"):
        # the two envelopes must describe the same event under the same contract (ST4-13, reviewer point 7)
        r_alarm = (retrieve.get("payload") or {}).get("alarm_id")
        d_alarm = st.get("alarm_id") or (detect.get("payload") or {}).get("alarm_id")
        if r_alarm != d_alarm:
            raise ValueError(f"retrieve alarm_id {r_alarm!r} != detect alarm_id {d_alarm!r}")
        r_contract = ((retrieve.get("trace") or {}).get("versions") or {}).get("contract_sha256")
        if r_contract and st.get("contract_sha256") and r_contract != st["contract_sha256"]:
            raise ValueError("retrieve and detect envelopes come from different input contracts")
        gr = retrieve["grounding"]
        mode, g_ab, g_c, tau = gr["mode"], gr["g_AB"], gr["g_C"], gr["tau"]
    return Signals(gate=st["gate"], drift=st["drift"], alarm=bool(st["alarm"]) if st["alarm"] is not None else False,
                   s=len(pred), g_AB=g_ab, g_C=g_c, q_used=bool(st["q_used"]), tau=tau, retrieve_mode=mode)


def signals_from_block_gate(gate: dict) -> Signals:
    """gate = gate_check envelope (status OK) whose payload.gate == BLOCK and whose state is the committed BLOCK row."""
    if gate.get("status") != "OK" or (gate.get("payload") or {}).get("gate") != "BLOCK":
        raise ValueError("signals_from_block_gate needs an OK gate_check envelope with gate == BLOCK")
    st = gate.get("state") or {}
    if st and st.get("gate") != "BLOCK":
        raise ValueError("gate envelope state is not a BLOCK row")
    drift = (st.get("drift") if st else None) or (gate.get("payload") or {}).get("drift") or "ok"
    return Signals(gate="BLOCK", drift=drift, alarm=False, s=0, g_AB=None, g_C=None, q_used=bool(st.get("q_used")) if st else False,
                   tau=None, retrieve_mode=None)


def signals_dict(sig: Signals) -> dict:
    return {"gate": sig.gate, "drift": sig.drift, "alarm": sig.alarm, "s": sig.s, "g_AB": sig.g_AB, "g_C": sig.g_C,
            "q_used": sig.q_used, "tau": sig.tau, "retrieve_mode": sig.retrieve_mode}
