"""tests/unit/test_action_cost.py — H1 cost definitions pinned by the eval spec v3 §1 examples (ST4-51). No DB.

Window spec mini-language for one policy: 'P' propose, 'H' hold, 'Q>H' / 'Q>P' query_more then hold / propose (E only),
'N' no_alarm (default), 'U' undecidable (gate BLOCK). Alarm and gate are shared by E / B1 / B3 (one snapshot); B2 has
its own alarm set.
"""
import json
from fractions import Fraction

import pytest

from core_pipeline.eval import action_cost as AC

RUN = "00000000-0000-4000-8000-000000000001"
EP = "00000000-0000-4000-8000-0000000000aa"


def window_events(ep, t, spec, policy, alarm, gate):
    if spec.startswith("Q>"):
        aft = {"H": "hold", "P": "propose"}[spec[2:]]
        return [dict(policy=policy, episode_id=ep, window_index=t, phase="initial", action="query_more", alarm=True, gate=gate),
                dict(policy=policy, episode_id=ep, window_index=t, phase="after_query", action=aft, alarm=True, gate=gate)]
    act = {"P": "propose", "H": "hold", "N": "no_alarm", "U": "undecidable"}[spec]
    return [dict(policy=policy, episode_id=ep, window_index=t, phase="initial", action=act, alarm=alarm, gate=gate)]


def episode_events(ep, specs, alarms=(), blocks=(), b2_alarms=()):
    """specs: {policy: {t: spec}}; windows not listed are no_alarm (or undecidable inside blocks)."""
    out = []
    for pol in ("E", "B1", "B3"):
        for t in range(48):
            gate = "BLOCK" if t in blocks else "PASS"
            default = "U" if t in blocks else "N"
            spec = specs.get(pol, {}).get(t, default)
            out += window_events(ep, t, spec, pol, t in alarms, gate)
    for t in range(48):
        spec = specs.get("B2", {}).get(t, "P" if t in b2_alarms else "N")
        out += window_events(ep, t, spec, "B2", t in b2_alarms, None)
    for e in out:
        e["eval_run_id"] = RUN
        e["rule"] = 0
    return out


def costs_of(events, is_fault, ep=EP):
    outs = AC.outcomes_by_episode(events, [ep])
    return {p: AC.episode_cost(outs[(p, ep)], is_fault) for p in AC.POLICIES}


def spec_example_events():
    """eval spec v3 §1 fixed example: fault episode, unlisted windows have no cost event."""
    alarms = {3, 8, 9, 10, 11}
    return episode_events(EP, {"E": {3: "P", 8: "H", 9: "H", 10: "Q>H", 11: "Q>P"},
                               "B1": {3: "P", 8: "P", 9: "P", 10: "P", 11: "P"},
                               "B3": {3: "P", 8: "P", 9: "P", 10: "P", 11: "P"}}, alarms=alarms)


def test_coefficients_come_from_prereg_constants_as_exact_decimals():
    assert (AC.C_FP, AC.C_FN, AC.C_HOLD, AC.C_QUERY) == (1, Fraction(3, 10), Fraction(1, 10), Fraction(1, 20))
    assert AC.C_FP_GRID == (Fraction(1, 2), 1, 2)
    assert AC.C_FN * AC.N_ACTIVE == 12


def test_spec_example_E_is_2_3():
    c = costs_of(spec_example_events(), True)["E"]
    assert c.total() == Fraction(23, 10)
    assert c.components() == {"fp": 1, "fn": Fraction(9, 10), "hold": Fraction(3, 10), "query": Fraction(1, 10)}
    assert c.T == 11


def test_spec_example_B1_is_1_0_and_counts_proposals_after_T():
    c = costs_of(spec_example_events(), True)["B1"]
    assert c.total() == 1 and c.T == 8 and c.n_after_T == 3


def test_pre_injection_propose_does_not_end_the_delay():
    c = costs_of(spec_example_events(), True)["E"]
    assert c.n_fp == 1 and c.n_fn_windows == 3


def test_query_more_then_hold_costs_both():
    ev = episode_events(EP, {"E": {12: "Q>H"}, "B1": {12: "P"}, "B3": {12: "P"}}, alarms={12})
    c = costs_of(ev, False)["E"]
    assert c.total() == Fraction(15, 100) and (c.n_query, c.n_hold) == (1, 1)


def test_registered_consequence_b_after_query_propose_sets_T():
    ev = episode_events(EP, {"E": {15: "Q>P"}, "B1": {15: "P"}, "B3": {15: "P"}}, alarms={15})
    c = costs_of(ev, True)["E"]
    assert c.T == 15 and c.total() == Fraction(3, 10) * 7 + Fraction(1, 20)


def test_normal_episode_has_no_delay_and_charges_each_propose_and_hold():
    ev = episode_events(EP, {"E": {4: "H", 30: "P"}, "B1": {4: "P", 30: "P"}, "B3": {4: "H", 30: "P"}}, alarms={4, 30})
    c = costs_of(ev, False)
    assert c["E"].total() == Fraction(11, 10) and c["E"].n_fn_windows == 0 and c["E"].T is None
    assert c["B1"].total() == 2 and c["B3"].total() == Fraction(11, 10)


def test_no_T_and_window_7_propose_is_13_0():
    ev = episode_events(EP, {"E": {7: "P"}, "B1": {7: "P"}, "B3": {7: "P"}}, alarms={7})
    c = costs_of(ev, True)["E"]
    assert c.T is None and c.total() == 13


def test_block_windows_8_to_10_and_T_12_give_delay_1_2():
    ev = episode_events(EP, {"E": {12: "P"}, "B1": {12: "P"}, "B3": {12: "P"}}, alarms={12}, blocks={8, 9, 10})
    c = costs_of(ev, True)["E"]
    assert c.T == 12 and c.components()["fn"] == Fraction(6, 5) and c.n_block == 3 and c.total() == Fraction(6, 5)


def test_registered_consequence_a_active_no_alarm_windows_accumulate_delay():
    ev = episode_events(EP, {"E": {20: "P"}, "B1": {20: "P"}, "B3": {20: "P"}}, alarms={20})
    assert costs_of(ev, True)["E"].total() == Fraction(36, 10)


def test_registered_consequence_c_b2_on_a_normal_episode_is_c_fp_only():
    c = costs_of(episode_events(EP, {}, b2_alarms={2, 30}), False)["B2"]
    assert c.total() == 2 and (c.n_fn_windows, c.n_hold, c.n_query) == (0, 0, 0)


def test_b2_runs_through_block_windows():
    ev = episode_events(EP, {}, blocks={8, 9}, b2_alarms={9})
    c = costs_of(ev, True)["B2"]
    assert c.T == 9 and c.total() == Fraction(3, 10)


def test_sensitivity_grid_changes_only_the_fp_term():
    c = costs_of(spec_example_events(), True)["E"]
    rest = c.total() - c.components()["fp"]
    assert [c.total(x) - rest for x in AC.C_FP_GRID] == [Fraction(1, 2), 1, 2]


def _drop(events, pred):
    return [e for e in events if not pred(e)]


def test_missing_after_query_is_incomplete():
    ev = _drop(spec_example_events(), lambda e: e["phase"] == "after_query" and e["window_index"] == 10)
    with pytest.raises(AC.CostInputError) as ex:
        AC.outcomes_by_episode(ev, [EP])
    assert ex.value.code == "INCOMPLETE"


def test_missing_initial_is_incomplete():
    ev = _drop(spec_example_events(), lambda e: e["policy"] == "B2" and e["window_index"] == 47)
    with pytest.raises(AC.CostInputError) as ex:
        AC.outcomes_by_episode(ev, [EP])
    assert ex.value.code == "INCOMPLETE"


def test_after_query_without_query_more_is_incomplete():
    ev = spec_example_events() + [dict(eval_run_id=RUN, policy="E", episode_id=EP, window_index=8, phase="after_query",
                                       action="hold", alarm=True, gate="PASS", rule=6)]
    with pytest.raises(AC.CostInputError) as ex:
        AC.outcomes_by_episode(ev, [EP])
    assert ex.value.code == "INCOMPLETE"


@pytest.mark.parametrize("patch,code", [
    (dict(policy="E", window_index=20, action="hold", alarm=False), "INVARIANT_ALARM"),
    (dict(policy="B1", window_index=20, action="query_more", alarm=True), "INVARIANT_ACTION"),
    (dict(policy="B2", window_index=20, action="hold", alarm=True), "INVARIANT_ACTION"),
    (dict(policy="E", window_index=20, action="undecidable", alarm=False), "INVARIANT_BLOCK"),
])
def test_invariants_refuse(patch, code):
    ev = spec_example_events()
    for e in ev:
        if e["policy"] == patch["policy"] and e["window_index"] == patch["window_index"] and e["phase"] == "initial":
            e.update(patch)
    with pytest.raises(AC.CostInputError) as ex:
        AC.outcomes_by_episode(ev, [EP])
    assert ex.value.code == code


def test_after_query_for_a_baseline_is_refused():
    ev = spec_example_events() + [dict(eval_run_id=RUN, policy="B3", episode_id=EP, window_index=8, phase="after_query",
                                       action="hold", alarm=True, gate="PASS", rule=6)]
    with pytest.raises(AC.CostInputError) as ex:
        AC.outcomes_by_episode(ev, [EP])
    assert ex.value.code == "INVARIANT_AFTER_QUERY_POLICY"


def test_snapshot_policies_must_share_alarm_and_gate():
    ev = spec_example_events()
    for e in ev:
        if e["policy"] == "B1" and e["window_index"] == 3:
            e.update(action="no_alarm", alarm=False)
    with pytest.raises(AC.CostInputError) as ex:
        AC.outcomes_by_episode(ev, [EP])
    assert ex.value.code == "INVARIANT_SHARED_SNAPSHOT"


def test_unplanned_episode_is_refused():
    with pytest.raises(AC.CostInputError) as ex:
        AC.outcomes_by_episode(spec_example_events(), ["00000000-0000-4000-8000-0000000000bb"])
    assert ex.value.code == "EVENT_NOT_PLANNED"


# ----------------------------------------------------------------------------------------------------- whole run
def ep_id(run_id, fault):
    return f"00000000-0000-4000-8000-{run_id:06d}{fault:06d}"


def manifest(run_ids):
    return {"schema": AC.MANIFEST_SCHEMA, "episodes": [
        {"episode_id": ep_id(r, f), "run_id": r, "fault_scenario": f, "engine_id": 100000 + 1000 * f + r} for r in run_ids for f in range(21)]}


def synthetic_run(run_ids, b1_equal_to_e=False):
    """Normal: false alarms at 4,5 (E holds, B1 proposes, B3 holds; with b1_equal_to_e all three propose). Fault: alarms 8..47, E query_more->propose at 8,
    B3 holds 8,9 then proposes, B1 proposes from 8. B2 proposes only at window 1 (pre-injection): no T, delay 12.0."""
    events = []
    for r in run_ids:
        for f in range(21):
            ep = ep_id(r, f)
            if f == 0:
                alarms = {4, 5}
                if b1_equal_to_e:
                    specs = {"E": {4: "P", 5: "P"}, "B1": {4: "P", 5: "P"}, "B3": {4: "P", 5: "P"}}
                else:
                    specs = {"E": {4: "H", 5: "H"}, "B1": {4: "P", 5: "P"}, "B3": {4: "H", 5: "H"}}
            else:
                alarms = set(range(8, 48))
                e = {t: "P" for t in range(9, 48)}
                e[8] = "Q>P"
                b1 = {t: "P" for t in range(8, 48)}
                b3 = {t: "P" for t in range(10, 48)}
                b3.update({8: "H", 9: "H"})
                specs = {"E": e, "B1": b1, "B3": b3}
            events += episode_events(ep, specs, alarms=alarms, b2_alarms={1})
    return events


def run_doc(run_ids, status="complete"):
    return {"eval_run_id": RUN, "status": status, "planned_episodes": [ep_id(r, f) for r in run_ids for f in range(21)]}


POOL = list(range(101, 161))


def test_full_conformant_run_passes_when_e_is_cheaper_than_every_baseline():
    rids = POOL[:40]
    meta = AC.parse_manifest(manifest(rids))
    out = AC.compute(run_doc(rids), synthetic_run(rids), meta, POOL)
    assert out["prereg_conformant"] is True and out["n_episodes"] == 840
    p = out["primary"]
    assert p["B1"]["mean"] == pytest.approx((1.8 - 20 * 0.05) / 21)
    assert p["B2"]["mean"] == pytest.approx((0.8 + 20 * (1.0 + 12.0 - 0.05)) / 21)
    assert p["B3"]["mean"] == pytest.approx(20 * 0.75 / 21)
    assert all(v["lower"] == pytest.approx(v["mean"]) for v in p.values())
    assert out["h1_pass"] is True and out["h1_failed_comparisons"] == []
    assert out["per_policy"]["B1"]["oracle_labels"] == {"approve_active_window_proposals": 40 * 20 * 40,
                                                        "reject_nonfault_window_proposals": 40 * 2}
    assert out["by_fault"]["normal"]["B1"]["mean"] == pytest.approx(1.8)
    json.dumps(out)


def test_a_zero_lower_bound_fails_h1_and_names_the_comparison():
    rids = POOL[:40]
    out = AC.compute(run_doc(rids), synthetic_run(rids, b1_equal_to_e=True), AC.parse_manifest(manifest(rids)), POOL)
    assert out["primary"]["B1"]["mean"] == pytest.approx(-20 * 0.05 / 21) and out["primary"]["B1"]["lower"] <= 0
    assert out["primary"]["B3"]["lower"] > 0 and out["primary"]["B2"]["lower"] > 0
    assert out["h1_pass"] is False and out["h1_failed_comparisons"] == ["B1"]


def test_incomplete_run_status_is_refused():
    rids = POOL[:40]
    with pytest.raises(AC.CostInputError) as ex:
        AC.compute(run_doc(rids, "incomplete"), [], AC.parse_manifest(manifest(rids)), POOL)
    assert ex.value.code == "RUN_NOT_COMPLETE"


def test_conformance_rules():
    with pytest.raises(AC.CostInputError) as ex:
        AC.blocks_of(AC.parse_manifest(manifest(POOL[:30])), POOL, False)
    assert ex.value.code == "NOT_PREREG_CONFORMANT"
    with pytest.raises(AC.CostInputError):
        AC.blocks_of(AC.parse_manifest(manifest(POOL[1:41])), POOL, False)
    doc = manifest(POOL[:40])
    doc["episodes"] = [e for e in doc["episodes"] if not (e["run_id"] == POOL[3] and e["fault_scenario"] == 7)]
    with pytest.raises(AC.CostInputError):
        AC.blocks_of(AC.parse_manifest(doc), POOL, False)
    run_ids, problems, ok = AC.blocks_of(AC.parse_manifest(manifest(POOL[:3])), POOL, True)
    assert ok is False and problems and run_ids == POOL[:3]


def test_block_order_follows_the_pool_not_numeric_order():
    pool = list(reversed(POOL))
    run_ids, _, ok = AC.blocks_of(AC.parse_manifest(manifest(pool[:40])), pool, False)
    assert ok and run_ids == pool[:40]


def test_nonconformant_input_gets_no_verdict():
    rids = POOL[:2]
    out = AC.compute(run_doc(rids), synthetic_run(rids), AC.parse_manifest(manifest(rids)), POOL, allow_nonconformant=True)
    assert out["prereg_conformant"] is False and out["h1_pass"] is None and out["h1_failed_comparisons"] is None


def test_manifest_engine_rule_is_checked():
    doc = manifest(POOL[:1])
    doc["episodes"][5]["engine_id"] += 1000
    with pytest.raises(AC.CostInputError) as ex:
        AC.parse_manifest(doc)
    assert ex.value.code == "MANIFEST_ENGINE_RULE"


def test_h1_result_file_roundtrip(tmp_path):
    rids = POOL[:40]
    exp = tmp_path / "policy_events.json"
    exp.write_text(json.dumps({"eval_run": run_doc(rids), "policy_events": synthetic_run(rids)}), encoding="utf-8")
    man = tmp_path / "episode_manifest.json"
    man.write_text(json.dumps(manifest(rids)), encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"groups": {"evaluation_pool": {"run_ids": POOL}}}), encoding="utf-8")
    with pytest.raises(AC.CostInputError):
        AC.h1_result(exp, man, split, expect_export_sha256="0" * 64)
    out = tmp_path / "h1.json"
    assert AC.main(["h1", "--export", str(exp), "--episodes", str(man), "--split", str(split), "--out", str(out),
                    "--expect-export-sha256", AC.file_sha256(exp)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema"] == AC.SCHEMA and doc["h1_pass"] is True and doc["provenance"]["export_sha256"] == AC.file_sha256(exp)
    with pytest.raises(SystemExit):
        AC.main(["h1", "--export", str(exp), "--episodes", str(man), "--split", str(split), "--out", str(out)])
