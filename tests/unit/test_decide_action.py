"""S4a-02 / H0-(a): exhaustive enumeration of a SELECTED boundary grid of the signal space (gate × drift × alarm ×
s{0,1,3,4} × g_AB{None,<τ,=τ,>τ} × g_C × q_used × τ{None,0.5}) for all four policies. Expected outcomes are derived
from the pre-registered table re-stated in the test; valid_s/ev_AB are also checked directly at the boundaries. This is
a grid verification, not an independent specification proof."""
import itertools
import pytest
from core_pipeline.policy.decide_action import Signals, decide_action, Decision
from core_pipeline import prereg_constants as P

TAU = 0.5
GATES = ["PASS", "FLAG", "BLOCK"]
DRIFTS = ["ok", "critical"]
S_VALUES = [0, 1, 3, 4]                      # empty, min valid, max valid, over
G_VALUES = [None, TAU - 0.01, TAU, TAU + 0.3]  # no A/B, below, at, above
TAUS = [None, TAU]


def all_signals(gate_values=GATES):
    for gate, drift, alarm, s, g, gc, q, tau in itertools.product(gate_values, DRIFTS, [False, True], S_VALUES, G_VALUES, [None, 0.9], [False, True], TAUS):
        yield Signals(gate, drift, alarm, s, g, gc, q, tau, retrieve_mode="scoped")


def test_rule_priority_E():
    for sig in all_signals():
        d = decide_action(sig, "E")
        if sig.gate == "BLOCK":
            assert d.action == "undecidable" and d.rule == 1, sig          # BLOCK beats everything, even alarm+grounding
        elif not sig.alarm:
            assert d.action == "no_alarm" and d.rule == 2, sig
        elif sig.drift == "critical":
            assert d.action == "hold" and d.rule == 3, sig                  # critical beats a satisfied proposal condition
        elif sig.valid_s and sig.ev_AB is True:
            assert d.action == "propose" and d.rule == 4, sig
        elif not sig.q_used:
            assert d.action == "query_more" and d.rule == 5, sig
            assert d.query_kind == ("both" if (not sig.valid_s and sig.ev_AB is not True) else ("expand" if not sig.valid_s else "rerank"))
        else:
            assert d.action == "hold" and d.rule == 6, sig


def test_ev_AB_none_is_never_grounding():
    for sig in all_signals():
        if sig.tau is None:
            assert sig.ev_AB is None
            d = decide_action(sig, "E")
            assert d.action != "propose", sig                                # unfixed tau can never produce a proposal
            if sig.alarm and sig.gate != "BLOCK" and sig.drift == "ok" and sig.q_used:
                assert d.action == "hold"


def test_reentry_after_query_more_can_propose_but_not_query_again():
    base = Signals("PASS", "ok", True, 0, None, None, False, TAU, "scoped")
    assert decide_action(base, "E").action == "query_more"
    improved = Signals("PASS", "ok", True, 1, 0.7, None, True, TAU, "scoped")             # after expand+rerank: valid set + grounding
    assert decide_action(improved, "E") == Decision("propose", 4, "valid set ∧ ev_AB")
    still_bad = Signals("PASS", "ok", True, 0, 0.1, None, True, TAU, "scoped")
    d = decide_action(still_bad, "E")
    assert d.action == "hold" and d.rule == 6                                   # q_used only closes rule 5


def test_set_size_boundaries():
    for s, valid in ((0, False), (1, True), (P.SET_SIZE_MAX, True), (P.SET_SIZE_MAX + 1, False)):
        d = decide_action(Signals("PASS", "ok", True, s, 0.9, None, True, TAU, "scoped"), "E")
        assert (d.action == "propose") == valid, s


def test_g_at_tau_counts_as_grounding_and_below_does_not():
    assert decide_action(Signals("PASS", "ok", True, 1, TAU, None, True, TAU, "scoped"), "E").action == "propose"
    assert decide_action(Signals("PASS", "ok", True, 1, TAU - 1e-9, None, True, TAU, "scoped"), "E").action == "hold"


def test_c_only_cannot_ground():
    d = decide_action(Signals("PASS", "ok", True, 1, None, 0.99, True, TAU, "scoped"), "E")
    assert d.action == "hold" and "C-only" in d.reason


def test_baselines_have_their_own_rows():
    for sig in all_signals():
        b1, b3 = decide_action(sig, "B1"), decide_action(sig, "B3")
        if sig.gate == "BLOCK":
            assert b1.action == b3.action == "undecidable"
        elif not sig.alarm:
            assert b1.action == b3.action == "no_alarm"
        else:
            assert b1.action == "propose"                                        # B1 ignores drift, set, grounding
            assert b3.action == ("propose" if sig.ev_AB is True else "hold")     # B3: grounding gate only, no query, ignores drift/set
            assert b3.action != "query_more"
    for sig in all_signals(gate_values=[None, "BLOCK"]):                         # B2 never sees the gate
        b2 = decide_action(sig, "B2")
        assert b2.action == ("propose" if sig.alarm else "no_alarm")


def test_open_mode_never_grounds():
    hi = Signals("PASS", "ok", True, 1, 0.99, None, True, TAU, retrieve_mode="open")
    assert hi.ev_AB is False and decide_action(hi, "E").action == "hold" and decide_action(hi, "B3").action == "hold"
    assert Signals("PASS", "ok", True, 1, 0.99, None, True, TAU, retrieve_mode=None).ev_AB is False
    missing = Signals("PASS", "ok", True, 1, 0.99, None, True, TAU)                  # mode omitted → default None → no proposal
    assert missing.retrieve_mode is None and decide_action(missing, "E").action != "propose" and decide_action(missing, "B3").action == "hold"


def test_signal_space_size_is_exhaustive():
    assert sum(1 for _ in all_signals()) == 3 * 2 * 2 * 4 * 4 * 2 * 2 * 2
