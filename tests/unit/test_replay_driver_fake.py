"""tests/unit/test_replay_driver_fake.py — ST4-50 S4b-02: the replay driver's window logic in memory (FakeSkills, no DB).

Pins, per branch, the EXPECTED decisions from the pre-registered rule table (§4.5) — not merely that two
implementations agree — and the driver's recovery contract: identical events after an abort at any of the three
query_more points, one logical spend per window, B1/B3 untouched by E's query, no query_more call before the
baseline events, B2 independent of gate / detection state. If a graph↔driver equivalence test ever disagrees, these
expected-decision tests decide which side drifted (never update expectations first)."""
from __future__ import annotations

import ast
import json
import os
import uuid
from pathlib import Path

import pytest

from core_pipeline.eval import keys as K
from core_pipeline.eval.policy_events import EventMismatch, MemoryEventStore
from core_pipeline.eval.replay_driver import DriverDeps, DriverError, ReplayDriver, RunInfo, assert_no_llm_env, k5_envelope
from core_pipeline.policy.baselines import decide_baseline
from core_pipeline.policy.decide_action import decide_action
from core_pipeline.policy.initial_snapshot import SnapshotError, alarm_id_of, validate_k5
from core_pipeline.policy.policy_input import PolicyInput
from core_pipeline.policy.signals_adapter import signals_from_envelopes

ROOT = Path(__file__).resolve().parents[2]
CONTRACT, DETECTOR, TAU = "c" * 64, "detector_v1", 0.5
EP = "11111111-1111-1111-1111-111111111111"
RUN = "33333333-3333-3333-3333-333333333333"
VERSIONS = {"embedder": "bge-m3", "reranker": "bge-reranker-v2-m3", "tau": TAU, "fault_map_sha256": "f" * 64, "k_primary": 5, "k_query_more": 10,
            "contract_sha256": CONTRACT, "detector": DETECTOR}


class Crash(RuntimeError):
    pass


def spec(gate="PASS", drift="ok", alarm=True, pred_set=(1,), expanded=(1,), expand_status="OK", g_ab5=0.1, g_c5=0.2, g_ab10=0.8, g_c10=0.2,
         k10_status="OK", b2=True):
    return dict(gate=gate, drift=drift, alarm=alarm, pred_set=list(pred_set), expanded=list(expanded), expand_status=expand_status,
                g_ab5=g_ab5, g_c5=g_c5, g_ab10=g_ab10, g_c10=g_c10, k10_status=k10_status, b2=b2)


class FakeSkills:
    """Stateful stand-in for the two Skill servers with ST4-13 replay-by-request_id and crash injection."""

    def __init__(self, windows: dict, fail_at: set | None = None):
        self.w = windows
        self.state = {t: {"q_used": False, "qrid": None, "expanded_set": None} for t in windows}
        self.calls: list[tuple] = []
        self.replays: dict[str, dict] = {}
        self.fail_at = set(fail_at or ())
        self.k5_envelopes: dict[int, tuple[str, dict, dict]] = {}

    def _env(self, status, payload=None, grounding=None, state=None, reason=None, ctx=None):
        return {"status": status, "payload": payload, "grounding": grounding, "state": state,
                "reason": None if status == "OK" else {"code": reason, "message": reason},
                "trace": {"request_id": ctx["request_id"], "versions": dict(VERSIONS), "replayed": False}}

    def _replay(self, ctx):
        env = self.replays.get(ctx["request_id"])
        if env is not None:
            env = json.loads(json.dumps(env)); env["trace"]["replayed"] = True
        return env

    def _store(self, ctx, env):
        self.replays[ctx["request_id"]] = env
        return env

    def _state(self, t):
        s, sp = self.state[t], self.w[t]
        return {"gate": sp["gate"], "drift": sp["drift"], "alarm": None if sp["gate"] == "BLOCK" else sp["alarm"], "pred_set": list(sp["pred_set"]),
                "expanded_set": None if s["expanded_set"] is None else list(s["expanded_set"]), "q_used": s["q_used"],
                "query_request_id": s["qrid"], "alarm_id": alarm_id_of(EP, t, DETECTOR) if (sp["gate"] != "BLOCK" and sp["alarm"]) else None,
                "contract_sha256": CONTRACT, "detector_version": DETECTOR}

    def _crash(self, point):
        if point in self.fail_at:
            self.fail_at.discard(point)
            raise Crash(point)

    def gate_check(self, ctx, ep, t):
        self.calls.append(("gate_check", t, ctx["request_id"]))
        r = self._replay(ctx)
        if r: return r
        sp = self.w[t]
        return self._store(ctx, self._env("OK", {"gate": sp["gate"], "drift": sp["drift"]}, state=self._state(t) if sp["gate"] == "BLOCK" else None, ctx=ctx))

    def detect_diagnose(self, ctx, ep, t, expand=1, query_id=None):
        self.calls.append(("detect_diagnose", t, ctx["request_id"], expand))
        r = self._replay(ctx)
        if r: return r
        sp, s = self.w[t], self.state[t]
        if sp["gate"] == "BLOCK":
            return self._store(ctx, self._env("REFUSED", reason="GATE_BLOCK", ctx=ctx))
        if expand == 1:
            return self._store(ctx, self._env("OK", {"alarm_id": self._state(t)["alarm_id"]}, state=self._state(t), ctx=ctx))
        if t == 0:
            return self._store(ctx, self._env("REFUSED", reason="NO_EXPANSION", ctx=ctx))
        if not s["q_used"] or s["qrid"] != query_id:
            return self._store(ctx, self._env("REFUSED", reason="QUERY_USED", ctx=ctx))
        if sp["expand_status"] != "OK":
            return self._store(ctx, self._env("REFUSED", reason=sp["expand_status"], ctx=ctx))
        if s["expanded_set"] is None:
            s["expanded_set"] = list(sp["expanded"])
        env = self._store(ctx, self._env("OK", {"alarm_id": self._state(t)["alarm_id"]}, state=self._state(t), ctx=ctx))
        self._crash("after_expand")
        return env

    def spend_query(self, ctx, ep, t):
        self.calls.append(("spend_query", t, ctx["request_id"]))
        r = self._replay(ctx)
        if r: return r
        s = self.state[t]
        if s["q_used"] and s["qrid"] != ctx["request_id"]:
            return self._store(ctx, self._env("REFUSED", reason="QUERY_USED", ctx=ctx))
        s["q_used"], s["qrid"] = True, ctx["request_id"]
        env = self._store(ctx, self._env("OK", {"query_id": ctx["request_id"], "q_used": True, "expand_allowed": t > 0}, state=self._state(t), ctx=ctx))
        self._crash("after_spend")
        return env

    def retrieve(self, ctx, query, mode, fault_set, k, alarm_id=None, query_id=None):
        self.calls.append(("retrieve", k, ctx["request_id"], list(fault_set)))
        r = self._replay(ctx)
        if r: return r
        t = int(alarm_id.split(":")[1])
        sp, s = self.w[t], self.state[t]
        current = s["expanded_set"] if s["expanded_set"] is not None else sp["pred_set"]
        if sorted(fault_set) != sorted(current):
            return self._store(ctx, self._env("REFUSED", reason="ARG_NOT_IN_STATE", ctx=ctx))
        if k == 10:
            if not s["q_used"] or s["qrid"] != query_id:
                return self._store(ctx, self._env("REFUSED", reason="QUERY_REQUIRED", ctx=ctx))
            if sp["k10_status"] != "OK":
                return self._store(ctx, self._env("ERROR", reason=sp["k10_status"], ctx=ctx))
            g_ab, g_c = sp["g_ab10"], sp["g_c10"]
        else:
            g_ab, g_c = sp["g_ab5"], sp["g_c5"]
        gr = {"mode": "scoped", "tau": TAU, "g_AB": g_ab, "g_C": g_c, "ev_AB": g_ab >= TAU, "ev_C": g_c >= TAU,
              "snapshot": [{"clause_key": f"P-1:0:{k}", "grade": "A", "score": g_ab}]}
        env = self._store(ctx, self._env("OK", {"clauses": [], "k": k, "mode": "scoped", "alarm_id": alarm_id, "reused": False}, gr, ctx=ctx))
        if k == 5:
            bound = {"ctx": ctx, "skill": "skills_retrieve", "tool": "retrieve", "contract_version": "1.0",
                     "args": {"query": query, "mode": mode, "fault_set": list(fault_set), "k": k, "alarm_id": alarm_id, "query_id": None}}
            self.k5_envelopes[t] = (ctx["request_id"], bound, env)
        else:
            self._crash("after_k10")
        return env


class MemoryWorld:
    """In-memory freeze / load / decisions / plans mirroring the DB semantics the driver relies on."""

    def __init__(self, fake: FakeSkills):
        self.fake, self.snapshots, self.decisions, self.plans, self.marks = fake, {}, {}, {}, []
        self.events = MemoryEventStore()

    def freeze(self, ep, t):
        if t in self.snapshots:
            return self.snapshots[t]
        sp, s = self.fake.w[t], self.fake.state[t]
        if s["q_used"] or s["expanded_set"] is not None:
            raise SnapshotError("SNAPSHOT_TOO_LATE", "spent")
        block = sp["gate"] == "BLOCK"
        alarm = None if block else sp["alarm"]
        aid = alarm_id_of(EP, t, DETECTOR) if alarm else None
        k5 = None
        if alarm:
            if t not in self.fake.k5_envelopes:
                raise SnapshotError("SNAPSHOT_K5_MISSING", "no k5")
            rid, bound, out = self.fake.k5_envelopes[t]
            k5 = validate_k5(rid, bound, out, sp["pred_set"], aid, CONTRACT)
        pi = PolicyInput(EP, t, DETECTOR, CONTRACT, "engineer", sp["gate"], sp["drift"], True, None if block else 0.01, None if block else 0.02,
                         t + 1, alarm, aid, [] if block else list(sp["pred_set"]), 0 if block else len(sp["pred_set"]),
                         None if (block or not sp["pred_set"]) else sp["pred_set"][0], k5, str(uuid.uuid4()), "")
        self.snapshots[t] = PolicyInput(**{**pi.__dict__, "content_sha256": pi.content_hash()})
        return self.snapshots[t]

    def try_load(self, ep, t):
        return self.snapshots.get(t)

    def record_decision(self, alarm_id, policy, signals, action, rule, evidence, key):
        if key not in self.decisions:
            self.decisions[key] = str(uuid.uuid5(K.NS, key))
        return self.decisions[key]

    def plan_write(self, query_id, ep, t, thread_id, turn_id, initial_decision_id, kind):
        self.plans.setdefault(query_id, {"t": t, "thread_id": thread_id, "turn_id": turn_id, "initial_decision_id": initial_decision_id, "kind": kind})

    def plan_mark(self, query_id, step, status, reason):
        self.marks.append((query_id, step, status, reason))

    def deps(self):
        return DriverDeps(skills=self.fake, events=self.events, freeze=self.freeze, try_load=self.try_load, record_decision=self.record_decision,
                          decision_sha=lambda d: None, plan_write=self.plan_write, plan_mark=self.plan_mark,
                          b2_alarm=lambda ep, t: self.fake.w[t]["b2"], fault_map_path=str(ROOT / "outputs" / "fault_to_guidance_map.json"))


RUN_INFO = RunInfo(RUN, DETECTOR, CONTRACT, "engineer")


def _world(windows, fail_at=None):
    fake = FakeSkills(windows, fail_at)
    world = MemoryWorld(fake)
    return world, ReplayDriver(world.deps(), RUN_INFO)


def _calls(fake, tool, t=None, k=None):
    return [c for c in fake.calls if c[0] == tool and (t is None or c[1] == t) and (k is None or c[1] == k)]


def _ev(world, policy, phase, t):
    return world.events.rows[(RUN, policy, EP, t, phase)]


# ---------------------------------------------------------------------------------------------------- static guards
def test_driver_modules_import_no_narrator_graph_or_llm():
    for f in ("core_pipeline/eval/replay_driver.py", "core_pipeline/eval/b2.py", "core_pipeline/eval/keys.py", "core_pipeline/eval/policy_events.py"):
        tree = ast.parse((ROOT / f).read_text(encoding="utf-8"))
        mods = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        bad = {m for m in mods if any(x in m for x in ("agent.graph", "agent.narrator", "agent.router", "llm_factory", "llm_narrator", "llm_router"))}
        assert not bad, (f, bad)


def test_llm_env_refused(monkeypatch):
    monkeypatch.setenv("SGP_LLM_NARRATOR", "llm")
    with pytest.raises(DriverError) as e:
        assert_no_llm_env()
    assert e.value.code == "LLM_ENV_SET"
    monkeypatch.delenv("SGP_LLM_NARRATOR")
    monkeypatch.setenv("SGP_LLM_ROUTER", "rule")
    assert_no_llm_env()


def test_request_ids_are_deterministic_and_distinguish_initial_from_after_query():
    ks = K.step_keys(8)
    a = K.request_id(RUN, EP, 8, ks["detect_initial"])
    assert a == K.request_id(RUN, EP, 8, ks["detect_initial"]) and a != K.request_id(RUN, EP, 8, ks["detect_after"])
    assert K.request_id(RUN, EP, 8, ks["spend"]) != K.request_id(RUN, EP, 9, ks["spend"])
    assert K.ctx(RUN, EP, 8, ks["spend"], "engineer")["mode"] == "replay"


# ---------------------------------------------------------------------------------------------------- branches
def test_rerank_only_branch():
    world, drv = _world({8: spec(pred_set=(1,), g_ab5=0.1, g_ab10=0.8)})
    out = drv.run_window(EP, 8)
    assert (out["B1"], out["B3"], out["E_initial"], out["E_after"], out["B2"]) == ("propose", "hold", "query_more", "propose", "propose")
    assert _ev(world, "E", "initial", 8).query_kind == "rerank" and _ev(world, "E", "after_query", 8).rule == 4
    assert _ev(world, "E", "after_query", 8).evidence_stage == "k10" and _ev(world, "E", "after_query", 8).evidence[0]["clause_key"] == "P-1:0:10"
    assert not _calls(world.fake, "detect_diagnose", 8) or all(c[3] == 1 for c in _calls(world.fake, "detect_diagnose", 8))
    assert len(_calls(world.fake, "spend_query", 8)) == 1 and len(_calls(world.fake, "retrieve", k=10)) == 1
    assert [m[1:] for m in world.marks] == [("rerank", "done", None)]


def test_expand_only_branch_uses_frozen_k5_as_evidence():
    world, drv = _world({8: spec(pred_set=(), expanded=(1,), g_ab5=0.8)})
    out = drv.run_window(EP, 8)
    assert (out["B1"], out["B3"], out["E_initial"], out["E_after"]) == ("propose", "propose", "query_more", "propose")
    e0, e1 = _ev(world, "E", "initial", 8), _ev(world, "E", "after_query", 8)
    assert e0.query_kind == "expand" and e0.signals["s"] == 0 and e1.signals["s"] == 1 and e1.rule == 4
    assert e1.evidence_stage == "k5" and e1.evidence_request_id == world.snapshots[8].k5.request_id
    assert not _calls(world.fake, "retrieve", k=10) and [m[1:] for m in world.marks] == [("expand", "done", None)]


def test_both_branch():
    world, drv = _world({8: spec(pred_set=(), expanded=(1,), g_ab5=0.1, g_ab10=0.8)})
    out = drv.run_window(EP, 8)
    assert out["E_initial"] == "query_more" and _ev(world, "E", "initial", 8).query_kind == "both" and out["E_after"] == "propose"
    assert _ev(world, "E", "after_query", 8).evidence_stage == "k10"
    assert [m[1:] for m in world.marks] == [("expand", "done", None), ("rerank", "done", None)]


def test_t0_cannot_expand_and_holds_on_invalid_set():
    world, drv = _world({0: spec(pred_set=(), g_ab5=0.1, g_ab10=0.8)})
    out = drv.run_window(EP, 0)
    assert out["E_initial"] == "query_more" and out["E_after"] == "hold"
    assert _ev(world, "E", "after_query", 0).rule == 6 and "indeterminate" in _ev(world, "E", "after_query", 0).reason
    assert [m[1:] for m in world.marks] == [("expand", "skipped", "NO_EXPANSION"), ("rerank", "done", None)]
    assert not [c for c in _calls(world.fake, "detect_diagnose", 0) if c[3] == 2]


def test_expansion_failure_is_marked_and_policy_decides_on_what_is_available():
    world, drv = _world({8: spec(pred_set=(), expand_status="EXPANSION_INVALID", g_ab5=0.1, g_ab10=0.8)})
    out = drv.run_window(EP, 8)
    assert out["E_after"] == "hold" and _ev(world, "E", "after_query", 8).rule == 6
    assert ("expand", "failed", "EXPANSION_INVALID") in [m[1:] for m in world.marks] and _ev(world, "E", "after_query", 8).evidence_stage == "k10"


def test_k10_failure_falls_back_to_frozen_k5():
    world, drv = _world({8: spec(pred_set=(1,), g_ab5=0.1, k10_status="INDEX_MISMATCH")})
    out = drv.run_window(EP, 8)
    e1 = _ev(world, "E", "after_query", 8)
    assert out["E_after"] == "hold" and e1.rule == 6 and e1.evidence_stage == "k5" and e1.signals["g_AB"] == 0.1
    assert ("rerank", "failed", "INDEX_MISMATCH") in [m[1:] for m in world.marks]


def test_block_and_no_alarm_windows():
    world, drv = _world({3: spec(gate="BLOCK", b2=True), 4: spec(alarm=False, b2=False)})
    o3, o4 = drv.run_window(EP, 3), drv.run_window(EP, 4)
    assert (o3["B1"], o3["B3"], o3["E_initial"], o3["B2"]) == ("undecidable", "undecidable", "undecidable", "propose")
    assert (o4["B1"], o4["B3"], o4["E_initial"], o4["B2"]) == ("no_alarm", "no_alarm", "no_alarm", "no_alarm")
    assert not _calls(world.fake, "detect_diagnose", 3) and not _calls(world.fake, "retrieve")
    assert "E_after" not in o3 and "E_after" not in o4 and not world.decisions


def test_after_query_signals_match_independent_graph_style_computation():
    world, drv = _world({8: spec(pred_set=(), expanded=(1,), g_ab5=0.1, g_ab10=0.8)})
    drv.run_window(EP, 8)
    ks = K.step_keys(8)
    da = world.fake.replays[str(K.request_id(RUN, EP, 8, ks["detect_after"]))]
    k10 = world.fake.replays[str(K.request_id(RUN, EP, 8, ks["retrieve_k10"]))]
    retr = k10 if k10["status"] == "OK" else k5_envelope(world.snapshots[8])
    dec = decide_action(signals_from_envelopes(da, retr), "E")
    e1 = _ev(world, "E", "after_query", 8)
    assert (e1.action, e1.rule) == (dec.action, dec.rule)
    k10fail, drv2 = _world({8: spec(pred_set=(), expanded=(1,), g_ab5=0.1, g_ab10=0.8, k10_status="INDEX_MISMATCH")})
    drv2.run_window(EP, 8)
    assert _ev(k10fail, "E", "after_query", 8).signals["g_AB"] == 0.1          # k5 fallback, never the failed envelope


# ---------------------------------------------------------------------------------------------------- isolation & recovery
def test_baselines_recorded_before_any_query_call_and_unchanged_after():
    world, drv = _world({8: spec(pred_set=(), expanded=(1, 2), g_ab5=0.1, g_ab10=0.9)})
    seen = {}
    orig = world.events.record

    def record(ev):
        seen[(ev.policy, ev.phase)] = len(world.fake.calls)
        return orig(ev)
    world.events.record = record
    drv.run_window(EP, 8)
    first_spend = next(i for i, c in enumerate(world.fake.calls) if c[0] == "spend_query")
    assert all(seen[(p, "initial")] <= first_spend for p in ("B1", "B3", "E"))
    assert world.fake.state[8]["expanded_set"] == [1, 2]
    for p in ("B1", "B3"):
        again = decide_baseline(world.snapshots[8], p)
        assert (again.action, again.rule) == (_ev(world, p, "initial", 8).action, _ev(world, p, "initial", 8).rule)
    assert _ev(world, "B3", "initial", 8).signals["s"] == 0 and _ev(world, "B3", "initial", 8).evidence_stage == "k5"


@pytest.mark.parametrize("point", ["after_spend", "after_expand", "after_k10"])
def test_abort_then_resume_yields_identical_events_and_one_logical_spend(point):
    ref, drv_ref = _world({8: spec(pred_set=(), expanded=(1,), g_ab5=0.1, g_ab10=0.8)})
    drv_ref.run_window(EP, 8)
    world, drv = _world({8: spec(pred_set=(), expanded=(1,), g_ab5=0.1, g_ab10=0.8)}, fail_at={point})
    with pytest.raises(Crash):
        drv.run_window(EP, 8)
    assert (RUN, "E", EP, 8, "after_query") not in world.events.rows
    n_before = len(world.events.rows)
    drv_resumed = ReplayDriver(world.deps(), RUN_INFO)
    out = drv_resumed.run_window(EP, 8)
    assert out["E_after"] == "propose" and len(world.events.rows) == 5 and n_before < 5
    assert {k: v.event_sha256 for k, v in world.events.rows.items()} == {k: v.event_sha256 for k, v in ref.events.rows.items()}
    spends = _calls(world.fake, "spend_query", 8)
    assert len({c[2] for c in spends}) == 1 and world.fake.state[8]["qrid"] == spends[0][2]
    assert len(world.plans) == 1
    n_calls = len(world.fake.calls)
    drv_resumed.run_window(EP, 8)
    assert len(world.events.rows) == 5 and len(world.fake.calls) == n_calls


def test_resume_does_not_re_observe_when_snapshot_exists():
    world, drv = _world({8: spec(pred_set=(1,), g_ab5=0.1, g_ab10=0.8)})
    drv.run_window(EP, 8)
    n = len(world.fake.calls)
    ReplayDriver(world.deps(), RUN_INFO).run_window(EP, 8)
    assert len(world.fake.calls) == n                               # ledger + snapshot answered everything


def test_foreign_query_ownership_and_lost_first_observation_are_errors():
    world, drv = _world({8: spec(pred_set=(1,), g_ab5=0.1)})
    world.fake.state[8].update({"q_used": True, "qrid": "other-request"})
    with pytest.raises(DriverError) as e:
        drv.run_window(EP, 8)
    assert e.value.code == "STATE_ALREADY_EXPANDED"
    world3, drv3 = _world({8: spec(pred_set=(1,), g_ab5=0.1)})
    world3.fake.retrieve(K.ctx(RUN, EP, 8, K.step_keys(8)["retrieve_k5"], "engineer"), "q", "scoped", [1], 5, alarm_id=alarm_id_of(EP, 8, DETECTOR))
    world3.freeze(EP, 8)
    world3.fake.state[8].update({"q_used": True, "qrid": "other-request"})
    with pytest.raises(DriverError) as e:
        drv3.run_window(EP, 8)
    assert e.value.code == "QUERY_OWNED_ELSEWHERE"


def test_event_mismatch_is_refused():
    world, drv = _world({8: spec(pred_set=(1,), g_ab5=0.1, g_ab10=0.8)})
    drv.run_window(EP, 8)
    ev = _ev(world, "B1", "initial", 8)
    with pytest.raises(EventMismatch):
        world.events.record(ev.__class__(**{**ev.__dict__, "action": "hold"}))


def test_b2_is_independent_of_gate_and_e_state():
    world, drv = _world({8: spec(gate="BLOCK", b2=True), 9: spec(pred_set=(), expanded=(1,), g_ab5=0.1, g_ab10=0.8, b2=False)})
    drv.run_window(EP, 8)
    drv.run_window(EP, 9)
    assert _ev(world, "B2", "initial", 8).action == "propose" and _ev(world, "B2", "initial", 8).gate is None
    assert _ev(world, "B2", "initial", 9).action == "no_alarm" and _ev(world, "B2", "initial", 9).snapshot_sha256 is None
    assert _ev(world, "B2", "initial", 9).signals["q_used"] is False and _ev(world, "B2", "initial", 9).signals["gate"] is None
