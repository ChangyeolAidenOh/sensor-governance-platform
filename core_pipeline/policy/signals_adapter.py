"""core_pipeline/policy/signals_adapter.py — the ONLY way Signals are built in Stage 4a (ST4-07): from validated Skill
envelopes, never from LLM text. retrieve_mode stays None without a scoped retrieve, which makes ev_AB None (not grounding).
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


def signals_dict(sig: Signals) -> dict:
    return {"gate": sig.gate, "drift": sig.drift, "alarm": sig.alarm, "s": sig.s, "g_AB": sig.g_AB, "g_C": sig.g_C,
            "q_used": sig.q_used, "tau": sig.tau, "retrieve_mode": sig.retrieve_mode}
