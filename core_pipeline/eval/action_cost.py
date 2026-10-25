"""core_pipeline/eval/action_cost.py — H1 replay action cost from the policy_events export (prereg §4.4 / §5.1,
eval spec v3 §1, ST4-51). One output JSON is the H1 decision; it is never overturned by hand (prereg §5.1).

Input is the archived ledger file (core_pipeline/eval/archive.py: {"eval_run": ..., "policy_events": [...]}) plus an
episode manifest (episode_id -> run_id, fault_scenario, engine_id). The database is not read by the cost path, so the
result is reproducible from two files whose sha256 go into the output.

Definitions (eval spec v3 §1; registered, not chosen here):
  final action of a window   initial.action unless it is query_more; then the E after_query action (missing -> incomplete)
  c_fp     1.0  per final propose in a non-fault window (normal episode: all 48; fault episode: windows 0..7)
  c_fn     0.3  per window of the active span 8..47 before T; T = first window in 8..47 whose final action is propose;
                no T -> 40 x 0.3 = 12.0. A propose in 0..7 is not T. BLOCK / no_alarm / hold / query_more windows before T
                all accumulate (registered consequence a: an active no_alarm window before T accumulates delay)
  c_hold   0.1  per final hold (only alarm windows can hold; B2 never holds)
  c_query  0.05 per E initial query_more (expand + re-retrieve = one query, at most one per window);
                query_more -> hold costs both (0.05 + 0.1)
  after T  further proposes cost 0 and are counted [report]; undecidable costs 0 directly but never pauses the delay
Registered consequences pinned by tests: (b) an E after_query propose sets T at that window, (c) a normal episode's
B2 threshold-exceedance window is a B2 alarm window and B2 always proposes there -> c_fp only.
Arithmetic is exact (fractions.Fraction); floats appear only in the output and in the bootstrap.

  python -m core_pipeline.eval.action_cost manifest --dsn-env SGP_EVAL_PG_DSN --export <policy_events.json> --out <episode_manifest.json>
  python -m core_pipeline.eval.action_cost h1 --export <policy_events.json> --episodes <episode_manifest.json> --out <h1_result.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Optional

from core_pipeline import prereg_constants as P
from core_pipeline.eval import block_bootstrap as BB

SCHEMA = "h1_result/1"
MANIFEST_SCHEMA = "episode_manifest/1"
N_WINDOWS = 48
FIRST_ACTIVE = 8
N_ACTIVE = N_WINDOWS - FIRST_ACTIVE
POLICIES = ("E", "B1", "B2", "B3")
COMPARISONS = ("B1", "B2", "B3")
SPLIT_CODE = 1
FAULTS = tuple(range(0, 21))
N_RUN_IDS_ALLOWED = (60, 40)

# Single source = core_pipeline/prereg_constants.py (COST §4.4, COST_SENSITIVITY_C_FP). Floats are converted through their
# shortest repr (0.3 -> "0.3" -> 3/10), so the exact values are the registered decimals, not binary approximations.
def _exact(x: float) -> Fraction:
    return Fraction(repr(float(x)))


C_FP = _exact(P.COST["c_fp"])
C_FN = _exact(P.COST["c_fn"])
C_HOLD = _exact(P.COST["c_hold"])
C_QUERY = _exact(P.COST["c_query"])
C_FP_GRID = tuple(_exact(x) for x in P.COST_SENSITIVITY_C_FP)

ALLOWED_INITIAL = {"E": {"undecidable", "no_alarm", "hold", "propose", "query_more"},
                   "B1": {"undecidable", "no_alarm", "propose"},
                   "B3": {"undecidable", "no_alarm", "hold", "propose"},
                   "B2": {"no_alarm", "propose"}}
ALLOWED_AFTER_QUERY = {"hold", "propose"}


class CostInputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def coefficients() -> dict:
    return {"source": "core_pipeline/prereg_constants.py COST, COST_SENSITIVITY_C_FP", "c_fp": str(C_FP), "c_fn_per_window": str(C_FN), "c_fn_no_proposal": str(C_FN * N_ACTIVE), "c_hold": str(C_HOLD),
            "c_query": str(C_QUERY), "c_fp_grid": [str(x) for x in C_FP_GRID], "first_active_window": FIRST_ACTIVE, "n_windows": N_WINDOWS}


# ----------------------------------------------------------------------------------------------------- inputs
def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_export(path: Path) -> tuple[dict, list[dict]]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or "eval_run" not in doc or "policy_events" not in doc:
        raise CostInputError("EXPORT_FORMAT", "expected {eval_run, policy_events} (archive.py export)")
    return doc["eval_run"], list(doc["policy_events"])


@dataclass(frozen=True)
class EpisodeMeta:
    episode_id: str
    run_id: int
    fault_scenario: int
    engine_id: int

    @property
    def is_fault(self) -> bool:
        return self.fault_scenario > 0


def decode_engine(engine_id: int) -> tuple[int, int, int]:
    """engine_id = 100000*split_code + 1000*fault_scenario + run_id (outputs/stage0_episode_split.json engine_id_rule)."""
    e = int(engine_id)
    return e // 100000, (e // 1000) % 100, e % 1000


def parse_manifest(doc: dict) -> dict[str, EpisodeMeta]:
    if doc.get("schema") != MANIFEST_SCHEMA:
        raise CostInputError("MANIFEST_FORMAT", f"schema must be {MANIFEST_SCHEMA!r}")
    out: dict[str, EpisodeMeta] = {}
    for r in doc.get("episodes") or []:
        m = EpisodeMeta(str(r["episode_id"]), int(r["run_id"]), int(r["fault_scenario"]), int(r["engine_id"]))
        split, fault, run = decode_engine(m.engine_id)
        if (split, fault, run) != (SPLIT_CODE, m.fault_scenario, m.run_id):
            raise CostInputError("MANIFEST_ENGINE_RULE", f"{m.episode_id}: engine_id {m.engine_id} != rule(split {SPLIT_CODE}, fault {m.fault_scenario}, run {m.run_id})")
        if m.fault_scenario not in FAULTS:
            raise CostInputError("MANIFEST_FAULT", f"{m.episode_id}: fault_scenario {m.fault_scenario}")
        if m.episode_id in out:
            raise CostInputError("MANIFEST_DUPLICATE", m.episode_id)
        out[m.episode_id] = m
    if not out:
        raise CostInputError("MANIFEST_EMPTY", "no episodes")
    return out


def blocks_of(meta: dict[str, EpisodeMeta], pool_order: Optional[list[int]], allow_nonconformant: bool) -> tuple[list[int], list[str], bool]:
    """Run-id blocks in evaluation-pool order. Conformant = n in (60, 40), run ids = pool prefix, 21 episodes per run id
    covering faults 0..20. A non-conformant input is refused unless explicitly allowed (development smoke only)."""
    by_run: dict[int, list[EpisodeMeta]] = defaultdict(list)
    for m in meta.values():
        by_run[m.run_id].append(m)
    problems = []
    run_ids = sorted(by_run)
    if pool_order is not None:
        n = len(by_run)
        if n not in N_RUN_IDS_ALLOWED:
            problems.append(f"n_run_ids {n} not in {N_RUN_IDS_ALLOWED}")
        if sorted(pool_order[:n]) != run_ids:
            problems.append("run ids are not the evaluation-pool prefix of the same length")
        else:
            run_ids = list(pool_order[:n])
    else:
        problems.append("no split file: pool order unknown")
    for r in run_ids:
        faults = sorted(m.fault_scenario for m in by_run[r])
        if faults != list(FAULTS):
            problems.append(f"run id {r}: faults {faults[:3]}..{faults[-3:]} ({len(faults)} episodes), expected 0..20")
            break
    conformant = not problems
    if not conformant and not allow_nonconformant:
        raise CostInputError("NOT_PREREG_CONFORMANT", "; ".join(problems))
    return run_ids, problems, conformant


# ----------------------------------------------------------------------------------------------------- final actions
@dataclass(frozen=True)
class WindowOutcome:
    t: int
    alarm: bool
    gate: Optional[str]
    initial: str
    after: Optional[str] = None

    @property
    def final(self) -> str:
        return self.after if self.initial == "query_more" else self.initial


def _check_event(ev: dict) -> None:
    pol, ph, act, t = ev.get("policy"), ev.get("phase"), ev.get("action"), ev.get("window_index")
    if pol not in POLICIES or ph not in ("initial", "after_query") or not isinstance(t, int) or not 0 <= t < N_WINDOWS:
        raise CostInputError("EVENT_FIELDS", f"{pol}/{ph}/{t}")
    alarm, gate = bool(ev.get("alarm")), ev.get("gate")
    allowed = ALLOWED_INITIAL[pol] if ph == "initial" else ALLOWED_AFTER_QUERY
    if ph == "after_query" and pol != "E":
        raise CostInputError("INVARIANT_AFTER_QUERY_POLICY", f"{pol} after_query")
    if act not in allowed:
        raise CostInputError("INVARIANT_ACTION", f"{pol}/{ph} action {act!r} at window {t}")
    if act == "no_alarm" and alarm:
        raise CostInputError("INVARIANT_ALARM", f"{pol} no_alarm with alarm=true at window {t}")
    if act in ("propose", "hold", "query_more") and not alarm:
        raise CostInputError("INVARIANT_ALARM", f"{pol} {act} with alarm=false at window {t}")
    if act == "undecidable" and gate != "BLOCK":
        raise CostInputError("INVARIANT_BLOCK", f"{pol} undecidable with gate {gate!r} at window {t}")


def outcomes_by_episode(events: Iterable[dict], planned: list[str]) -> dict[tuple[str, str], list[WindowOutcome]]:
    """(policy, episode) -> 48 outcomes. Refuses anything short of complete (eval spec v3 §1: every planned
    (episode, window, policy) initial row AND an after_query row for every E initial query_more, nothing else)."""
    planned_set = set(planned)
    rows: dict[tuple, dict] = {}
    for ev in events:
        _check_event(ev)
        ep = str(ev["episode_id"])
        if ep not in planned_set:
            raise CostInputError("EVENT_NOT_PLANNED", ep)
        k = (ev["policy"], ep, int(ev["window_index"]), ev["phase"])
        if k in rows:
            raise CostInputError("EVENT_DUPLICATE", str(k))
        rows[k] = ev
    out: dict[tuple[str, str], list[WindowOutcome]] = {}
    missing_initial, missing_after, stray_after = [], [], []
    for ep in planned:
        for pol in POLICIES:
            seq = []
            for t in range(N_WINDOWS):
                ini = rows.get((pol, ep, t, "initial"))
                aft = rows.get((pol, ep, t, "after_query"))
                if ini is None:
                    missing_initial.append((pol, ep, t))
                    continue
                if ini["action"] == "query_more" and aft is None:
                    missing_after.append((ep, t))
                    continue
                if ini["action"] != "query_more" and aft is not None:
                    stray_after.append((ep, t))
                    continue
                seq.append(WindowOutcome(t, bool(ini["alarm"]), ini.get("gate"), ini["action"], aft["action"] if aft else None))
            out[(pol, ep)] = seq
    if missing_initial or missing_after or stray_after:
        raise CostInputError("INCOMPLETE", f"missing initial {len(missing_initial)}, missing after_query {len(missing_after)}, "
                                           f"after_query without query_more {len(stray_after)}; first: {(missing_initial + missing_after + stray_after)[:3]}")
    for ep in planned:                                   # B1 / B3 / E-initial read the same snapshot (ST4-50)
        for t in range(N_WINDOWS):
            seen = {(out[(p, ep)][t].alarm, out[(p, ep)][t].gate) for p in ("E", "B1", "B3")}
            if len(seen) != 1:
                raise CostInputError("INVARIANT_SHARED_SNAPSHOT", f"{ep} window {t}: alarm/gate differ across E/B1/B3 {sorted(seen, key=str)}")
    return out


# ----------------------------------------------------------------------------------------------------- one episode
@dataclass(frozen=True)
class EpisodeCost:
    n_fp: int                       # final proposes in non-fault windows
    n_fn_windows: int               # delay windows (T - 8, or 40 without T); 0 for a normal episode
    n_hold: int
    n_query: int
    T: Optional[int]
    n_after_T: int                  # proposes in the active span after T [report]
    n_block: int
    n_propose_active: int           # proposes in fault-active windows (oracle label: approve) [report]
    n_propose_nonfault: int         # = n_fp (oracle label: reject) [report]

    def total(self, c_fp: Fraction = C_FP) -> Fraction:
        return c_fp * self.n_fp + C_FN * self.n_fn_windows + C_HOLD * self.n_hold + C_QUERY * self.n_query

    def components(self, c_fp: Fraction = C_FP) -> dict:
        return {"fp": c_fp * self.n_fp, "fn": C_FN * self.n_fn_windows, "hold": C_HOLD * self.n_hold, "query": C_QUERY * self.n_query}


def episode_cost(outcomes: list[WindowOutcome], is_fault: bool) -> EpisodeCost:
    if [o.t for o in outcomes] != list(range(N_WINDOWS)):
        raise CostInputError("EPISODE_WINDOWS", "expected windows 0..47 in order")
    T = None
    if is_fault:
        T = next((o.t for o in outcomes if o.t >= FIRST_ACTIVE and o.final == "propose"), None)
    n_fp = sum(1 for o in outcomes if o.final == "propose" and (not is_fault or o.t < FIRST_ACTIVE))
    n_fn = 0 if not is_fault else (N_ACTIVE if T is None else T - FIRST_ACTIVE)
    n_active_prop = sum(1 for o in outcomes if is_fault and o.t >= FIRST_ACTIVE and o.final == "propose")
    return EpisodeCost(n_fp=n_fp, n_fn_windows=n_fn, n_hold=sum(1 for o in outcomes if o.final == "hold"),
                       n_query=sum(1 for o in outcomes if o.initial == "query_more"), T=T,
                       n_after_T=max(0, n_active_prop - 1) if T is not None else 0,
                       n_block=sum(1 for o in outcomes if o.final == "undecidable"),
                       n_propose_active=n_active_prop, n_propose_nonfault=n_fp)


# ----------------------------------------------------------------------------------------------------- whole run
def _f(x: Fraction) -> float:
    return float(x)


def _mean(xs: list[Fraction]) -> Fraction:
    return sum(xs, Fraction(0)) / len(xs)


def compute(eval_run: dict, events: list[dict], meta: dict[str, EpisodeMeta], pool_order: Optional[list[int]],
            allow_nonconformant: bool = False) -> dict:
    status = eval_run.get("status")
    if status != "complete":
        raise CostInputError("RUN_NOT_COMPLETE", f"eval_run status {status!r}: no cost or verdict for an incomplete run")
    planned = [str(e) for e in eval_run.get("planned_episodes") or []]
    if len(set(planned)) != len(planned) or set(planned) != set(meta):
        raise CostInputError("MANIFEST_MISMATCH", f"planned {len(planned)} episodes vs manifest {len(meta)} (sets must be equal)")
    run_ids, problems, conformant = blocks_of(meta, pool_order, allow_nonconformant)
    outs = outcomes_by_episode(events, planned)
    costs = {(p, ep): episode_cost(outs[(p, ep)], meta[ep].is_fault) for p in POLICIES for ep in planned}

    by_run: dict[int, list[str]] = defaultdict(list)
    for ep in planned:
        by_run[meta[ep].run_id].append(ep)
    ordered_eps = [ep for r in run_ids for ep in sorted(by_run[r], key=lambda e: meta[e].fault_scenario)]
    idx = BB.draw_block_indices(len(run_ids))

    def delta_blocks(j: str, c_fp: Fraction, eps_filter=None) -> list[list[float]]:
        return [[_f(costs[(j, ep)].total(c_fp) - costs[("E", ep)].total(c_fp))
                 for ep in sorted(by_run[r], key=lambda e: meta[e].fault_scenario) if eps_filter is None or eps_filter(ep)] for r in run_ids]

    def intervals(c_fp: Fraction, eps_filter=None) -> dict[str, BB.Interval]:
        return {j: BB.block_interval(delta_blocks(j, c_fp, eps_filter), idx) for j in COMPARISONS}

    primary = intervals(C_FP)
    verdict = BB.all_lower_gt_0(primary)
    sensitivity = {str(c): {j: iv.as_dict() for j, iv in intervals(c).items()} for c in C_FP_GRID}
    by_fault = {}
    for f in FAULTS:
        name = "normal" if f == 0 else str(f)
        if any(not any(meta[ep].fault_scenario == f for ep in by_run[r]) for r in run_ids):
            by_fault[name] = None                         # only possible for a non-conformant input (a block lacks this fault)
            continue
        ivs = intervals(C_FP, lambda ep, f=f: meta[ep].fault_scenario == f)
        by_fault[name] = {j: iv.as_dict() for j, iv in ivs.items()}

    per_policy = {}
    for p in POLICIES:
        cs = [costs[(p, ep)] for ep in ordered_eps]
        comp = {k: _f(_mean([c.components()[k] for c in cs])) for k in ("fp", "fn", "hold", "query")}
        fault_cs = [costs[(p, ep)] for ep in ordered_eps if meta[ep].is_fault]
        per_policy[p] = {"mean_cost": _f(_mean([c.total() for c in cs])), "mean_components": comp,
                         "counts": {"fp_proposals": sum(c.n_fp for c in cs), "delay_windows": sum(c.n_fn_windows for c in cs),
                                    "holds": sum(c.n_hold for c in cs), "queries": sum(c.n_query for c in cs),
                                    "proposals_after_T": sum(c.n_after_T for c in cs), "undecidable_windows": sum(c.n_block for c in cs),
                                    "fault_episodes_without_T": sum(1 for c in fault_cs if c.T is None)},
                         "oracle_labels": {"approve_active_window_proposals": sum(c.n_propose_active for c in cs),
                                           "reject_nonfault_window_proposals": sum(c.n_propose_nonfault for c in cs)}}
    n_windows_total = len(planned) * N_WINDOWS
    block_ratio = sum(costs[("E", ep)].n_block for ep in planned) / n_windows_total

    per_episode = [{"episode_id": ep, "run_id": meta[ep].run_id, "fault_scenario": meta[ep].fault_scenario,
                    "cost": {p: str(costs[(p, ep)].total()) for p in POLICIES},
                    "T": {p: costs[(p, ep)].T for p in POLICIES},
                    "n": {p: [costs[(p, ep)].n_fp, costs[(p, ep)].n_fn_windows, costs[(p, ep)].n_hold, costs[(p, ep)].n_query] for p in POLICIES}}
                   for ep in ordered_eps]
    return {
        "prereg_conformant": conformant,
        "conformance_problems": problems,
        "n_run_ids": len(run_ids),
        "n_episodes": len(planned),
        "run_ids_in_block_order": run_ids,
        "coefficients": coefficients(),
        "bootstrap": BB.procedure(),
        "primary": {j: iv.as_dict() for j, iv in primary.items()},
        "h1_pass": verdict if conformant else None,
        "h1_failed_comparisons": [j for j, iv in primary.items() if not iv.lower_gt_0] if conformant else None,
        "sensitivity_c_fp": sensitivity,
        "by_fault": by_fault,
        "per_policy": per_policy,
        "block_window_ratio_gate_policies": block_ratio,
        "per_episode_n_columns": ["fp_proposals", "delay_windows", "holds", "queries"],
        "per_episode": per_episode,
    }


def h1_result(export_path: Path, manifest_path: Path, split_path: Optional[Path], allow_nonconformant: bool = False,
              expect_export_sha256: Optional[str] = None) -> dict:
    export_sha = file_sha256(export_path)
    if expect_export_sha256 and expect_export_sha256 != export_sha:
        raise CostInputError("EXPORT_SHA_MISMATCH", f"file {export_sha[:12]} != expected {expect_export_sha256[:12]}")
    eval_run, events = load_export(export_path)
    meta = parse_manifest(json.loads(Path(manifest_path).read_text(encoding="utf-8")))
    pool = None
    split_sha = None
    if split_path is not None:
        split_sha = file_sha256(split_path)
        pool = [int(x) for x in json.loads(Path(split_path).read_text(encoding="utf-8"))["groups"]["evaluation_pool"]["run_ids"]]
    body = compute(eval_run, events, meta, pool, allow_nonconformant)
    prov = {"export_path": str(export_path), "export_sha256": export_sha, "export_sha256_verified_against": expect_export_sha256,
            "episode_manifest_sha256": file_sha256(manifest_path), "split_sha256": split_sha,
            "eval_run_id": eval_run.get("eval_run_id"), "eval_run_code_commit": eval_run.get("code_commit"),
            "eval_run_planned_sha256": eval_run.get("planned_sha256"), "eval_run_prereg_tag": eval_run.get("prereg_tag"),
            "n_events": len(events), "calculator": {"action_cost_sha256": file_sha256(Path(__file__)),
                                                    "block_bootstrap_sha256": file_sha256(Path(BB.__file__)),
                                                    "python": platform.python_version()}}
    note = None if body["prereg_conformant"] else "development observation: not prereg-conformant, no H1 verdict"
    return {"schema": SCHEMA, "provenance": prov, "note": note, **body}


# ----------------------------------------------------------------------------------------------------- CLI
def build_manifest(dsn: str, episodes: list[str]) -> dict:
    """episode_id -> engine key via the Skill-server helper; fault and run id decoded from the engine_id rule and
    cross-checked with tep_loader.decode_engine_id. DB access stays inside this function."""
    from uuid import UUID

    from core_pipeline.data.tep_loader import decode_engine_id
    from core_pipeline.skills.servers.common import engine_id_of

    rows = []
    for ep in episodes:
        eid = engine_id_of(dsn, UUID(str(ep)))
        if eid is None:
            raise CostInputError("MANIFEST_ENGINE_MISSING", str(ep))
        split, fault, run = decode_engine(int(eid))
        if int(decode_engine_id(int(eid))[1]) != fault:
            raise CostInputError("MANIFEST_DECODE_MISMATCH", f"{ep}: engine {eid}")
        rows.append({"episode_id": str(ep), "engine_id": int(eid), "run_id": run, "fault_scenario": fault})
    return {"schema": MANIFEST_SCHEMA, "episodes": rows}


def _write_new(path: Path, doc: dict, force: bool) -> str:
    path = Path(path)
    if path.exists() and not force:
        raise SystemExit(f"{path} exists (use --force to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=False)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m core_pipeline.eval.action_cost")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--dsn-env", required=True)
    m.add_argument("--export", required=True, type=Path)
    m.add_argument("--out", required=True, type=Path)
    m.add_argument("--force", action="store_true")
    h = sub.add_parser("h1")
    h.add_argument("--export", required=True, type=Path)
    h.add_argument("--episodes", required=True, type=Path)
    h.add_argument("--split", type=Path, default=Path("outputs/stage0_episode_split.json"))
    h.add_argument("--out", required=True, type=Path)
    h.add_argument("--expect-export-sha256")
    h.add_argument("--allow-nonconformant", action="store_true")
    h.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "manifest":
        dsn = os.environ.get(a.dsn_env)
        if not dsn:
            print(f"{a.dsn_env} is not set")
            return 2
        eval_run, _ = load_export(a.export)
        sha = _write_new(a.out, build_manifest(dsn, [str(e) for e in eval_run["planned_episodes"]]), a.force)
        print(f"manifest {a.out} sha256 {sha}")
        return 0
    try:
        doc = h1_result(a.export, a.episodes, a.split, a.allow_nonconformant, a.expect_export_sha256)
    except CostInputError as e:
        print(f"refused: {e}")
        return 3
    sha = _write_new(a.out, doc, a.force)
    print(f"h1_pass {doc['h1_pass']} conformant {doc['prereg_conformant']} -> {a.out} sha256 {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
