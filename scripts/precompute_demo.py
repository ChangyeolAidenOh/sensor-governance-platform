"""scripts/precompute_demo.py — S4a-03 (ST4-14): pre-compute the 10 demo presets on the PROJECT DB through the two Skill
servers, windows 0..47, and write the agent-facing manifest.

    SGP_PG_DSN=<project DSN> python -m scripts.precompute_demo \\
        [--presets 1,2,...] [--manifest outputs/demo_presets.json] [--truth outputs/demo_presets_truth.json] \\
        [--goldens tests/regression/golden] [--thread demo-precompute]

Per preset: replay_episode(mode="replay") → display window = FIRST prefix-BH alarm window in 0..47 (no range cut:
the operator sees the first alarm; a false alarm in a normal episode is kept as such); no alarm → display window 47
("all 48 windows observed, no alarm"). On an alarm window a scoped k=5 retrieve is computed once (stored by the server,
reused by every chat session) and decide_action(E) is COMPUTED BUT NOT RECORDED (decide_preview = observation at
pre-compute time; the chat records its own decisions). Idempotent: no cleaning on the project DB, already observed
windows come back reused=true from the servers. No --reset (clean_episode would also delete approvals/work orders).
Truth labels (fault scenario) go to the separate truth file only; the manifest note is observational.
--goldens compares the 48 window rows + model versions of presets 1/2/8/9 with G1b/G1/G4/G2 (never the selected
window, request ids or timestamps — those follow different rules).
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import psycopg

from core_pipeline import prereg_constants as P
from core_pipeline.policy.decide_action import decide_action
from core_pipeline.policy.signals_adapter import signals_dict, signals_from_envelopes
from core_pipeline.skills.client import SkillClient
from core_pipeline.skills.replay_client import episode_for_preset, replay_episode, scoped_query
from core_pipeline.skills.servers.common import new_ctx

GOLDEN_OF = {1: "G1b", 2: "G1", 8: "G4", 9: "G2"}
MANIFEST_SCHEMA = "demo_presets/1"


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _register_thread(dsn: str, thread_id: str, owner: str) -> None:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO mart.threads (thread_id, owner) VALUES (%s,%s) ON CONFLICT (thread_id) DO NOTHING", (thread_id, owner))
        conn.commit()


def _strip(row: dict) -> dict:
    """Comparable window row: drop per-run flags (stored / reused) that legitimately differ between first and later runs."""
    r = json.loads(json.dumps(row, default=str))
    gp = r.get("gate_payload")
    if isinstance(gp, dict):
        gp.pop("stored", None)
    return r


async def run_preset(sk: SkillClient, dsn: str, preset: int, thread: str) -> tuple[dict, dict]:
    ep = episode_for_preset(dsn, preset)
    run_uuid = uuid.uuid4()
    last: dict = {}
    def keep(t, g, d):
        last.update(d or g)
    rows = await replay_episode(sk, ep, thread, run_uuid=run_uuid, on_window=keep, mode="replay")
    alarm_rows = [r for r in rows if r["alarm"] is True]
    versions = dict(last["trace"]["versions"])
    entry = {"preset": preset, "episode_id": str(ep), "n_windows": len(rows), "n_block": sum(1 for r in rows if r["gate"] == "BLOCK"),
             "alarm_windows": [r["t"] for r in alarm_rows], "versions": versions}
    if alarm_rows:
        sel = alarm_rows[0]
        t = sel["t"]
        det = await sk.detect_diagnose(new_ctx(thread, run_uuid, mode="replay"), str(ep), t)
        assert det["status"] == "OK", det
        fs = list(det["state"]["expanded_set"] if det["state"].get("expanded_set") is not None else det["state"]["pred_set"])
        retr = await sk.retrieve(new_ctx(thread, run_uuid, mode="replay"), scoped_query(fs), "scoped", fs, 5, alarm_id=sel["alarm_id"])
        assert retr["status"] == "OK", retr
        sig = signals_from_envelopes(det, retr)
        dec = decide_action(sig, "E")
        entry.update({"kind": "alarm", "display_window": t, "alarm_id": sel["alarm_id"], "pred_set": sel["pred_set"], "top1": sel["top1"],
                      "note": f"창 {t}에서 첫 prefix-BH 알람 (0..47 기준), 전체 알람 창 {len(alarm_rows)}개",
                      "retrieve_k5": {"query": scoped_query(fs), "reused": bool(retr["payload"].get("reused")), "grounding": retr["grounding"],
                                      "top_keys": [c["clause_key"] for c in retr["payload"]["clauses"]], "scope_size": retr["payload"]["scope_size"]},
                      "decide_preview": {"action": dec.action, "rule": dec.rule, "query_kind": dec.query_kind, "signals": signals_dict(sig),
                                         "observed_at": "precompute", "stale_note": "사전 계산 시점 관측; 대화는 현재 세션의 실제 결정을 표시"}})
        entry["versions"].update({f"retrieve.{k}": v for k, v in retr["trace"]["versions"].items()})
    else:
        entry.update({"kind": "no_alarm", "display_window": P.N_WINDOWS - 1, "alarm_id": None, "pred_set": [], "top1": None,
                      "note": f"전체 {len(rows)}창 관측, prefix-BH 알람 미발생", "retrieve_k5": None, "decide_preview": None})
    return entry, {"preset": preset, "episode_id": str(ep), "windows": rows}


def compare_golden(golden_dir: Path, preset: int, rows: list[dict], versions: dict) -> dict:
    case = GOLDEN_OF.get(preset)
    if not case:
        return {}
    gp = golden_dir / f"{case}.json"
    if not gp.exists():
        return {"golden": case, "status": "missing"}
    g = json.loads(gp.read_text(encoding="utf-8"))
    grows = [_strip(r) for r in g["windows"]]
    mine = [_strip(r) for r in rows]
    diffs = [r["t"] for r, m in zip(grows, mine) if r != m] + ([f"len {len(grows)} vs {len(mine)}"] if len(grows) != len(mine) else [])
    gv = {k: v for k, v in g.get("versions", {}).items() if not k.startswith("retrieve.")}
    mv = {k: v for k, v in versions.items() if not k.startswith("retrieve.")}
    vdiff = {k: (gv.get(k), mv.get(k)) for k in set(gv) | set(mv) if gv.get(k) != mv.get(k)}
    return {"golden": case, "status": "same" if not diffs and not vdiff else "DIFF", "window_diffs": diffs, "version_diffs": vdiff}


async def main_async(a) -> int:
    dsn = os.environ.get("SGP_PG_DSN", "")
    if not dsn:
        sys.exit("SGP_PG_DSN is required (project DB; S4a-03 pre-computes on the project DB, ST4-14)")
    presets = [int(x) for x in a.presets.split(",")] if a.presets else [p for p, _, _ in P.DEMO_PRESETS]
    _register_thread(dsn, a.thread, "system")
    out_entries, raw, checks = [], {}, {}
    async with SkillClient(env={"SGP_PG_DSN": dsn}) as sk:
        for preset in presets:
            entry, rows = await run_preset(sk, dsn, preset, a.thread)
            out_entries.append(entry)
            raw[preset] = rows
            if a.goldens:
                checks[preset] = compare_golden(Path(a.goldens), preset, rows["windows"], entry["versions"])
            print(f"preset {preset}: kind={entry['kind']} display_window={entry['display_window']} alarms={len(entry['alarm_windows'])} "
                  f"preview={(entry.get('decide_preview') or {}).get('action')} golden={checks.get(preset, {}).get('status', '-')}")
    manifest = {"schema": MANIFEST_SCHEMA, "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "code_commit": _git_sha(),
                "contract_version": "1.0", "n_windows": P.N_WINDOWS, "display_rule": "first prefix-BH alarm window in 0..47; no alarm → window 47",
                "presets": out_entries, "golden_checks": checks}
    truth = {"schema": "demo_presets_truth/1", "note": "post-hoc reveal only — never read by the graph",
             "presets": [{"preset": p, "fault_scenario": f, "run_id": r, "episode_id": next((e["episode_id"] for e in out_entries if e["preset"] == p), None),
                          "reveal": ("정상 에피소드" if f == 0 else f"결함 {f}") + (" — 탐지 한계 사례(알람 미발생)" if f != 0 and next((e for e in out_entries if e["preset"] == p), {}).get("kind") == "no_alarm" else "")}
                         for p, f, r in P.DEMO_PRESETS if p in presets]}
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(a.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    Path(a.truth).write_text(json.dumps(truth, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if a.raw:
        Path(a.raw).write_text(json.dumps(raw, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    bad = [p for p, c in checks.items() if c.get("status") == "DIFF"]
    print(f"manifest {a.manifest} ({len(out_entries)} presets), truth {a.truth}" + (f"; GOLDEN DIFF: {bad}" if bad else ""))
    return 1 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default="")
    ap.add_argument("--manifest", default="outputs/demo_presets.json")
    ap.add_argument("--truth", default="outputs/demo_presets_truth.json")
    ap.add_argument("--raw", default="outputs/demo_presets_windows.json")
    ap.add_argument("--goldens", default="tests/regression/golden")
    ap.add_argument("--thread", default="demo-precompute")
    a = ap.parse_args()
    sys.exit(asyncio.run(main_async(a)))


if __name__ == "__main__":
    main()
