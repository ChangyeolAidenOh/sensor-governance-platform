"""tests/unit/test_scope_contract.py — G4 CONTRACT half (ST4-12, reviewer (g)): the mapping as a hard filter.
  * predicted set {18} (any of 16..20) → scope = common clauses only → no A/B → ev_AB can only be false
  * empty set → C only;  a set that mixes 1..15 with 16..20 → A/B of the 1..15 members are in scope
  * valid_s and ev_AB are independent signals (size-4 set may still carry A/B)
Needs only outputs/fault_to_guidance_map.json (no models, no DB)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core_pipeline.policy.decide_action import Signals
from core_pipeline.policy.signals_adapter import signals_from_envelopes
from core_pipeline.retrieval.scope import FaultScope

FMAP = Path("outputs/fault_to_guidance_map.json")


def _val(x):
    return x() if callable(x) else x          # Signals.valid_s / ev_AB may be properties or methods
pytestmark = pytest.mark.skipif(not FMAP.exists(), reason="fault map not present")


@pytest.fixture(scope="module")
def scope():
    return FaultScope(FMAP)


@pytest.fixture(scope="module")
def common():
    m = json.loads(FMAP.read_text(encoding="utf-8"))
    cc = m["common_clauses"]
    return set(cc.keys()) if isinstance(cc, dict) else set(cc)


@pytest.mark.parametrize("f", [16, 17, 18, 19, 20])
def test_unknown_faults_are_c_only(scope, common, f):
    keys = scope.scope_keys([f])
    assert keys and keys <= common, f"fault {f}: scope {keys - common} outside the common clauses"
    assert all(scope.grade(k, [f]) == "C" for k in keys)


def test_empty_set_is_c_only(scope, common):
    assert scope.scope_keys([]) == common


def test_mixed_set_keeps_ab_of_known_members(scope, common):
    keys = scope.scope_keys([4, 18])
    assert keys > common
    assert any(scope.grade(k, [4, 18]) in ("A", "B") for k in keys)


def test_size_four_set_can_still_have_ab(scope):
    keys = scope.scope_keys([1, 4, 6, 7])
    assert any(scope.grade(k, [1, 4, 6, 7]) in ("A", "B") for k in keys)
    assert not _val(Signals("PASS", "ok", True, 4, 0.9, 0.9, False, 0.5, "scoped").valid_s)


def test_signals_adapter_without_retrieve_never_grounds():
    det = {"status": "OK", "state": {"gate": "PASS", "drift": "ok", "alarm": True, "pred_set": [18], "expanded_set": None, "q_used": False}}
    sig = signals_from_envelopes(det, None)
    assert sig.retrieve_mode is None and _val(sig.ev_AB) is None and sig.s == 1


def test_signals_adapter_open_mode_never_grounds():
    det = {"status": "OK", "state": {"gate": "PASS", "drift": "ok", "alarm": True, "pred_set": [4], "expanded_set": None, "q_used": False, "alarm_id": "a"}}
    ret = {"status": "OK", "payload": {"alarm_id": "a"}, "grounding": {"mode": "open", "tau": 0.5, "g_AB": 0.99, "g_C": 0.9, "ev_AB": False, "ev_C": True}}
    assert _val(signals_from_envelopes(det, ret).ev_AB) is not True


def test_signals_adapter_uses_expanded_set_when_present():
    det = {"status": "OK", "state": {"gate": "FLAG", "drift": "ok", "alarm": True, "pred_set": [], "expanded_set": [4, 11], "q_used": True, "alarm_id": "a"}}
    ret = {"status": "OK", "payload": {"alarm_id": "a"}, "grounding": {"mode": "scoped", "tau": 0.5, "g_AB": 0.6, "g_C": 0.4, "ev_AB": True, "ev_C": False}}
    sig = signals_from_envelopes(det, ret)
    assert sig.s == 2 and sig.q_used is True and _val(sig.ev_AB) is True


def test_signals_adapter_refuses_mismatched_events():
    det = {"status": "OK", "state": {"gate": "PASS", "drift": "ok", "alarm": True, "pred_set": [4], "expanded_set": None, "q_used": False, "alarm_id": "a", "contract_sha256": "c1"}}
    ret = {"status": "OK", "payload": {"alarm_id": "b"}, "grounding": {"mode": "scoped", "tau": 0.5, "g_AB": 0.9, "g_C": 0.1, "ev_AB": True, "ev_C": False}}
    with pytest.raises(ValueError):
        signals_from_envelopes(det, ret)
    ret2 = {"status": "OK", "payload": {"alarm_id": "a"}, "trace": {"versions": {"contract_sha256": "c2"}}, "grounding": ret["grounding"]}
    with pytest.raises(ValueError):
        signals_from_envelopes(det, ret2)
