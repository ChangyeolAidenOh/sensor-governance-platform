"""core_pipeline/eval/replay_driver.py — Stage 4b event-driven replay: trigger → gate → detect → k=5 → freeze →
B1 / B3 / E-initial events → E query_more path → B2 (ST4-50 S4b-02, option C with a driver-owned E path).

The chat graph (core_pipeline/agent/graph.py) is left untouched; this driver re-implements E's query_more ORDER
(spend once → plan → expand=2 → detect after_query → k=10 → decide again) against the same Skill servers, and
tests/unit/test_replay_driver_fake.py pins the per-branch decisions. Rules enforced here, not by convention:
  * every Skill call uses a deterministic request_id (eval/keys.py): a resumed run re-sends the same ids and gets the
    stored envelopes back — one logical spend per window, never a second spend under a new id;
  * resume looks at the ledger and the snapshot FIRST; gate → detect → k=5 are only called when no snapshot exists,
    and k=5 is always requested on the FIRST pred_set (a window found already expanded without a snapshot is an error,
    never reconstructed);
  * B1 / B3 / E-initial are computed from the snapshot only (policy/baselines.py); B2 from raw samples only (eval/b2.py);
  * the after_query decision uses the k=10 envelope only when its status is OK, otherwise the frozen k=5 grounding —
    the same selection the graph makes (graph.py _decide);
  * no narrator is constructed and an LLM router / narrator environment refuses to start (prereg §4.5 narrate=off).
The driver takes its collaborators as callables (DriverDeps) so the whole window logic runs in memory with fakes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional
from uuid import UUID

from core_pipeline.eval import keys as K
from core_pipeline.eval.b2 import decide_b2
from core_pipeline.eval.policy_events import EventStore, PolicyEvent
from core_pipeline.policy.baselines import decide_baseline, decide_e_initial, evidence_for, signals_from_snapshot
from core_pipeline.policy.decide_action import Decision, decide_action
from core_pipeline.policy.initial_snapshot import SnapshotError
from core_pipeline.policy.policy_input import PolicyInput
from core_pipeline.policy.signals_adapter import signals_dict, signals_from_envelopes
from core_pipeline.skills.replay_client import scoped_query

DECISION_RECORDED_POLICIES = ("E",)          # policy_decisions stays the CMMS-binding record (ST4-50 D4); B1/B2 never, B3 pending Stage 5


class DriverError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def assert_fastmcp_update_check_off() -> None:
    """fastmcp's PyPI update check (settings.check_for_updates, env FASTMCP_CHECK_FOR_UPDATES) must be off for replay
    runs. This disables that one request only; it is not an offline guarantee for model downloads or anything else."""
    v = (os.environ.get("FASTMCP_CHECK_FOR_UPDATES") or "").strip().lower()
    if v != "off":
        raise DriverError("UPDATE_CHECK_ON", f"export FASTMCP_CHECK_FOR_UPDATES=off before a replay run (now {v!r})")


def assert_no_llm_env() -> None:
    r = (os.environ.get("SGP_LLM_ROUTER") or "rule").strip().lower()
    n = (os.environ.get("SGP_LLM_NARRATOR") or "template").strip().lower()
    if r != "rule" or n != "template":
        raise DriverError("LLM_ENV_SET", f"replay runs narrate=off: unset SGP_LLM_ROUTER={r!r} / SGP_LLM_NARRATOR={n!r}")


@dataclass
class RunInfo:
    eval_run_id: str
    detector_version: str
    contract_sha256: str
    caller_role: str = "engineer"


@dataclass
class DriverDeps:
    skills: Any                                                   # gate_check / detect_diagnose / spend_query / retrieve (sync)
    events: EventStore
    freeze: Callable[[str, int], PolicyInput]                     # (episode_id, t) → snapshot (creates or returns existing)
    try_load: Callable[[str, int], Optional[PolicyInput]]         # (episode_id, t) → snapshot or None (never raises on missing)
    record_decision: Callable[..., Any]                           # (alarm_id, policy, signals, action, rule, evidence, decision_key) → decision_id
    decision_sha: Callable[[str], Optional[str]]                  # decision_id → policy_decisions.decision_sha256
    plan_write: Callable[..., Any]                                # (query_id, episode_id, t, thread_id, turn_id, initial_decision_id, kind)
    plan_mark: Callable[..., Any]                                 # (query_id, step, status, reason)
    b2_alarm: Callable[[str, int], bool]
    fault_map_path: str = "outputs/fault_to_guidance_map.json"


def _code(env: Optional[dict]) -> Optional[str]:
    return ((env or {}).get("reason") or {}).get("code")


def _ok(env: dict, what: str) -> dict:
    if env.get("status") != "OK":
        raise DriverError(_code(env) or "INTERNAL", f"{what} returned {env.get('status')}: {((env.get('reason') or {}).get('message'))}")
    return env


def k5_envelope(snap: PolicyInput) -> Optional[dict]:
    """The frozen k=5 grounding as a retrieve envelope (what signals_from_envelopes expects)."""
    if snap.k5 is None:
        return None
    k = snap.k5
    return {"status": "OK", "payload": {"alarm_id": snap.alarm_id, "k": 5, "mode": "scoped", "reused": True},
            "grounding": {"mode": "scoped", "tau": k.tau, "g_AB": k.g_ab, "g_C": k.g_c, "ev_AB": k.ev_ab, "ev_C": k.ev_c, "snapshot": k.snapshot},
            "trace": {"versions": dict(k.versions)}}


def detect_set(env: dict) -> list[int]:
    st = env["state"]
    return [int(x) for x in (st["expanded_set"] if st.get("expanded_set") is not None else st["pred_set"])]


class ReplayDriver:
    def __init__(self, deps: DriverDeps, run: RunInfo):
        self.deps, self.run = deps, run

    def _ctx(self, ep: str, t: int, step: str) -> dict:
        return K.ctx(self.run.eval_run_id, ep, t, step, self.run.caller_role)

    def _event(self, snap: Optional[PolicyInput], ep: str, t: int, policy: str, phase: str, sig, dec: Decision, evidence_stage: str,
               evidence: list, evidence_request_id: Optional[str], decision_id: Optional[str] = None, alarm_id: Optional[str] = None) -> PolicyEvent:
        return PolicyEvent(eval_run_id=self.run.eval_run_id, policy=policy, episode_id=str(ep), window_index=int(t), phase=phase,
                           detector_version=self.run.detector_version, contract_sha256=self.run.contract_sha256,
                           alarm=bool(sig.alarm), action=dec.action, rule=int(dec.rule), signals=signals_dict(sig),
                           evidence_stage=evidence_stage, evidence=list(evidence), snapshot_sha256=snap.content_sha256 if snap else None,
                           alarm_id=alarm_id, gate=sig.gate, reason=dec.reason, query_kind=dec.query_kind,
                           evidence_k={"k5": 5, "k10": 10}.get(evidence_stage), evidence_request_id=evidence_request_id,
                           decision_id=decision_id, decision_sha256=self.deps.decision_sha(decision_id) if decision_id else None)

    # ------------------------------------------------------------------------------------------------- first observation
    def _observe(self, ep: str, t: int) -> PolicyInput:
        """gate → detect → k=5 (on the FIRST pred_set) → freeze. Only when no snapshot exists yet."""
        keys = K.step_keys(t)
        g = _ok(self.deps.skills.gate_check(self._ctx(ep, t, keys["gate"]), ep, t), "gate_check")
        if (g.get("payload") or {}).get("gate") != "BLOCK":
            d = _ok(self.deps.skills.detect_diagnose(self._ctx(ep, t, keys["detect_initial"]), ep, t), "detect_diagnose")
            st = d["state"]
            if st.get("expanded_set") is not None or st.get("q_used"):
                raise DriverError("STATE_ALREADY_EXPANDED", f"window {t} is expanded/spent but has no snapshot: the first observation is lost")
            if st.get("alarm") is True:
                fs = [int(x) for x in st["pred_set"]]
                aid = st.get("alarm_id") or (d.get("payload") or {}).get("alarm_id")
                _ok(self.deps.skills.retrieve(self._ctx(ep, t, keys["retrieve_k5"]), scoped_query(fs, self.deps.fault_map_path), "scoped", fs, 5,
                                              alarm_id=aid), "retrieve k=5")
        return self.deps.freeze(ep, t)

    # ------------------------------------------------------------------------------------------------- E query path
    def _e_query(self, ep: str, t: int, snap: PolicyInput, initial: PolicyEvent) -> PolicyEvent:
        keys = K.step_keys(t)
        kind = initial.query_kind or "both"
        planned_expand, planned_rerank = kind in ("expand", "both"), kind in ("rerank", "both")
        sp = self.deps.skills.spend_query(self._ctx(ep, t, keys["spend"]), ep, t)
        if sp.get("status") != "OK":
            if _code(sp) == "QUERY_USED":
                raise DriverError("QUERY_OWNED_ELSEWHERE", f"window {t} query is owned by another request: state of a different run, not reused")
            raise DriverError(_code(sp) or "INTERNAL", f"spend_query returned {sp.get('status')}")
        query_id = str(sp["payload"]["query_id"])
        expand_allowed = bool(sp["payload"].get("expand_allowed"))
        self.deps.plan_write(query_id, ep, t, K.thread_id(self.run.eval_run_id), str(K.turn_id(self.run.eval_run_id, ep, t)), initial.decision_id, kind)
        if planned_expand:
            if t == 0 or not expand_allowed:
                self.deps.plan_mark(query_id, "expand", "skipped", "NO_EXPANSION")
            else:
                dx = self.deps.skills.detect_diagnose(self._ctx(ep, t, keys["detect_expand"]), ep, t, expand=2, query_id=query_id)
                self.deps.plan_mark(query_id, "expand", "done" if dx.get("status") == "OK" else "failed", None if dx.get("status") == "OK" else _code(dx))
        da = _ok(self.deps.skills.detect_diagnose(self._ctx(ep, t, keys["detect_after"]), ep, t), "detect_diagnose after_query")
        rk = None
        if planned_rerank:
            fs = detect_set(da)
            rk = self.deps.skills.retrieve(self._ctx(ep, t, keys["retrieve_k10"]), scoped_query(fs, self.deps.fault_map_path), "scoped", fs, 10,
                                           alarm_id=da["state"].get("alarm_id"), query_id=query_id)
            self.deps.plan_mark(query_id, "rerank", "done" if rk.get("status") == "OK" else "failed", None if rk.get("status") == "OK" else _code(rk))
        if rk is not None and rk.get("status") == "OK":
            retr, stage, rid = rk, "k10", (rk.get("trace") or {}).get("request_id") or self._ctx(ep, t, keys["retrieve_k10"])["request_id"]
        else:
            retr, stage, rid = k5_envelope(snap), ("k5" if snap.k5 else "none"), (snap.k5.request_id if snap.k5 else None)
        sig = signals_from_envelopes(da, retr)
        dec = decide_action(sig, "E")
        evidence = list(((retr or {}).get("grounding") or {}).get("snapshot") or [])
        aid = da["state"].get("alarm_id")
        did = None
        if sig.alarm and aid:
            did = str(self.deps.record_decision(aid, "E", signals_dict(sig), dec.action, dec.rule, evidence, K.decision_key(self.run.eval_run_id, "E", ep, t, "after_query")))
        ev = self._event(snap, ep, t, "E", "after_query", sig, dec, stage, evidence, rid, did, aid)
        self.deps.events.record(ev)
        return ev

    # ------------------------------------------------------------------------------------------------- one window
    def run_window(self, ep: str, t: int) -> dict:
        ep = str(UUID(str(ep)))
        have = self.deps.events.load(self.run.eval_run_id, ep, t)
        snap = self.deps.try_load(ep, t)
        if snap is None:
            snap = self._observe(ep, t)
        sig = signals_from_snapshot(snap)
        out: dict = {"window": t, "gate": snap.gate, "alarm": sig.alarm}
        for pol in ("B1", "B3"):
            if (pol, "initial") not in have:
                dec = decide_baseline(snap, pol)
                use_k5 = pol == "B3" and snap.k5 is not None
                have[(pol, "initial")] = self._event(snap, ep, t, pol, "initial", sig, dec, "k5" if use_k5 else "none",
                                                     evidence_for(snap) if use_k5 else [], snap.k5.request_id if use_k5 else None, None, snap.alarm_id)
                self.deps.events.record(have[(pol, "initial")])
            out[pol] = have[(pol, "initial")].action
        if ("E", "initial") not in have:
            dec = decide_e_initial(snap)
            did = None
            if sig.alarm and snap.alarm_id and "E" in DECISION_RECORDED_POLICIES:
                did = str(self.deps.record_decision(snap.alarm_id, "E", signals_dict(sig), dec.action, dec.rule, evidence_for(snap),
                                                    K.decision_key(self.run.eval_run_id, "E", ep, t, "initial")))
            have[("E", "initial")] = self._event(snap, ep, t, "E", "initial", sig, dec, "k5" if snap.k5 else "none", evidence_for(snap),
                                                 snap.k5.request_id if snap.k5 else None, did, snap.alarm_id)
            self.deps.events.record(have[("E", "initial")])
        e_init = have[("E", "initial")]
        out["E_initial"] = e_init.action
        if e_init.action == "query_more":
            if ("E", "after_query") not in have:
                have[("E", "after_query")] = self._e_query(ep, t, snap, e_init)
            out["E_after"] = have[("E", "after_query")].action
        if ("B2", "initial") not in have:
            alarm = self.deps.b2_alarm(ep, t)
            dec = decide_b2(alarm)
            from core_pipeline.eval.b2 import b2_signals
            have[("B2", "initial")] = self._event(None, ep, t, "B2", "initial", b2_signals(alarm), dec, "none", [], None, None, None)
            self.deps.events.record(have[("B2", "initial")])
        out["B2"] = have[("B2", "initial")].action
        return out

    def run_episode(self, ep: str, t_max: int = 47, on_window: Optional[Callable[[int, dict], None]] = None) -> list[dict]:
        rows = []
        for t in range(t_max + 1):
            r = self.run_window(ep, t)
            rows.append(r)
            if on_window:
                on_window(t, r)
        return rows


# ----------------------------------------------------------------------------------------------------- real wiring
def make_db_deps(dsn: str, skills: Any, run: RunInfo, processed_dir: str, zscore_dir: str, fault_map_path: str = "outputs/fault_to_guidance_map.json",
                 expected_k5_versions: Optional[dict] = None) -> DriverDeps:
    """Bind the driver to the test / evaluation database. Imports of DB-writing helpers stay inside this factory."""
    import psycopg

    from core_pipeline.agent.plans import plan_mark, plan_write
    from core_pipeline.eval.b2 import B2Runner
    from core_pipeline.eval.policy_events import DbEventStore
    from core_pipeline.policy.initial_snapshot import freeze_policy_input, load_policy_input
    from core_pipeline.skills.work_order import record_decision

    b2 = B2Runner(dsn, processed_dir, zscore_dir)

    def freeze(ep: str, t: int) -> PolicyInput:
        return freeze_policy_input(dsn, ep, t, run.detector_version, run.contract_sha256, run.caller_role, run.eval_run_id, expected_k5_versions)

    def try_load(ep: str, t: int) -> Optional[PolicyInput]:
        try:
            return load_policy_input(dsn, ep, t, run.detector_version, run.contract_sha256, run.caller_role, expected_k5_versions)
        except SnapshotError as e:
            if e.code == "SNAPSHOT_MISSING":
                return None
            raise

    def rec(alarm_id, policy, signals, action, rule, evidence, key):
        return record_decision(dsn, alarm_id, policy, signals, action, rule, evidence=evidence, decision_key=key)

    def dsha(decision_id: str) -> Optional[str]:
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT decision_sha256 FROM mart.policy_decisions WHERE decision_id=%s", (UUID(decision_id),))
            row = cur.fetchone()
        return row[0] if row else None

    def pw(query_id, ep, t, thread_id, turn_id, initial_decision_id, kind):
        return plan_write(dsn, query_id, ep, t, run.detector_version, run.contract_sha256, thread_id, turn_id, initial_decision_id, kind)

    def pm(query_id, step, status, reason):
        return plan_mark(dsn, query_id, step, status, reason)

    return DriverDeps(skills=skills, events=DbEventStore(dsn), freeze=freeze, try_load=try_load, record_decision=rec, decision_sha=dsha,
                      plan_write=pw, plan_mark=pm, b2_alarm=b2.alarm, fault_map_path=fault_map_path)
