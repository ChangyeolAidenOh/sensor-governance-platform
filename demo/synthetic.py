"""demo/synthetic.py — the synthetic-propose doubles for the approval-UI check on the TEST DB (ST4-29).

Verbatim reuse of tests/integration/test_graph_agent.py (GroundingOverride + the forced-propose policy): the server-stored
retrieve envelope is untouched; only the grounding the GRAPH sees is rewritten to A/B ≥ τ, and the policy is replaced by
a constant propose. This exercises the approval contract (draft → interrupt → approve/modify/reject → issue) and nothing
else — it is NOT a demo case and NOT a policy result. runtime.py refuses to build with these unless both DSNs point at a
database whose name contains "test".
"""
from __future__ import annotations

from core_pipeline.policy.decide_action import Decision

SYNTHETIC_REASON = "synthetic propose (demo on the test DB; approval-contract check, not a policy result)"
SYNTHETIC_RULE = 4


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


def synthetic_policy(sig) -> Decision:
    return Decision("propose", SYNTHETIC_RULE, SYNTHETIC_REASON)
