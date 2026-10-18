"""Fix the TEP episode split (Stage 0-B, decision ST0-09 / ST0-11).

Unit of assignment is the shared-seed group (split, run_id): fault-free run r and
faulty runs r of the same split share their pre-onset trajectory, so a run id is
always assigned to exactly one group with all 21 of its episodes.

Only the TESTING split is partitioned (960-sample runs, 160 pre-onset samples,
used for calibration / evaluation / demo). The TRAINING split is model-fitting
data in its entirety and is not partitioned here.

Groups over testing run_id 1..500:
    dev               run_id 1..20   development-exposed runs (ST0-11); never used
                                     for calibration, evaluation or demo
    calibration       100 run ids    conformal calibration uses the FAULT-FREE run of
                                     each id only; the faulty runs of these ids are unused
    evaluation_pool   360 run ids    ordered; the pre-registered n takes the first n
    demo_pool         20 run ids     public demo presets are chosen from these ids;
                                     the 8 demo fault types are fixed at pre-registration

Deterministic: a fixed seed shuffles run_id 21..500 once. Re-running must
reproduce the existing file byte-for-byte; a mismatch is an error unless --force.

Usage (repo root, after tep_loader has written data/processed/TEP):
    python scripts/make_episode_split.py --out outputs/stage0_episode_split.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from core_pipeline.data.tep_loader import (
    MANIFEST_FILE, TEP_N_RUNS, TEP_N_FAULTS, TEP_ONSET, TEP_RUN_LENGTH,
    load_divergence_meta, make_engine_id,
)

SEED = 20261018
SPLIT = "testing"
DEV_RUN_IDS = list(range(1, 21))
N_CALIBRATION = 100
N_EVALUATION_POOL = 360
N_DEMO_POOL = 20

EXPOSURE_RECORD = [
    {"what": "scripts/smoke_tep_v1_layers.py", "runs": "fault-free testing run 1-10 (gate + IF fit), run 11 (held-out score)",
     "faulty": "fault 1 run 1, fault 4 run 1 (IF scoring)"},
    {"what": "scripts/check_tep_onset.py", "runs": "testing and training runs 1-3, all 20 faults (read-only paired comparison)"},
    {"what": "scripts/inspect_tep_raw.py", "runs": "head rows of every file (read-only)"},
]


def build_split(seed: int = SEED) -> dict:
    rest = np.arange(DEV_RUN_IDS[-1] + 1, TEP_N_RUNS + 1)
    rng = np.random.default_rng(seed)
    order = rest[rng.permutation(len(rest))]
    n_total = N_CALIBRATION + N_EVALUATION_POOL + N_DEMO_POOL
    if len(order) != n_total:
        raise ValueError(f"group sizes {n_total} != available run ids {len(order)}")
    cal = order[:N_CALIBRATION]
    ev = order[N_CALIBRATION:N_CALIBRATION + N_EVALUATION_POOL]
    demo = order[N_CALIBRATION + N_EVALUATION_POOL:]
    groups = {
        "dev": {"run_ids": DEV_RUN_IDS, "ordered": False},
        "calibration": {"run_ids": sorted(int(r) for r in cal), "ordered": False},
        "evaluation_pool": {"run_ids": [int(r) for r in ev], "ordered": True},
        "demo_pool": {"run_ids": sorted(int(r) for r in demo), "ordered": False},
    }
    # integrity: disjoint and covering 1..500
    all_ids = [r for g in groups.values() for r in g["run_ids"]]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(range(1, TEP_N_RUNS + 1)):
        raise ValueError("groups are not a disjoint cover of run_id 1..500")
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed/TEP")
    ap.add_argument("--out", default="outputs/stage0_episode_split.json")
    ap.add_argument("--force", action="store_true", help="overwrite an existing file that differs")
    args = ap.parse_args()

    # Data presence: manifest + every testing run id present in the divergence metadata.
    manifest = json.loads((Path(args.processed_dir) / MANIFEST_FILE).read_text(encoding="utf-8"))
    meta = load_divergence_meta(args.processed_dir)
    present = set(meta.loc[meta["split"] == SPLIT, "run_id"].astype(int).unique().tolist())
    if present != set(range(1, TEP_N_RUNS + 1)):
        raise ValueError(f"testing run ids in metadata != 1..{TEP_N_RUNS}")
    faults_present = sorted(meta.loc[meta["split"] == SPLIT, "fault_scenario"].astype(int).unique().tolist())
    if faults_present != list(range(1, TEP_N_FAULTS + 1)):
        raise ValueError(f"testing faults in metadata {faults_present} != 1..{TEP_N_FAULTS}")

    groups = build_split()

    def engine_ids_for(run_ids, faults):
        return [int(make_engine_id(SPLIT, f, r)) for r in run_ids for f in faults]

    payload = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "split": SPLIT,
        "run_length": TEP_RUN_LENGTH[SPLIT],
        "spec_onset_cycle": TEP_ONSET[SPLIT],
        "loader_version": manifest["loader_version"],
        "assignment_unit": "(split, run_id) shared-seed group; all 21 episodes of a run id share one group (ST0-09)",
        "engine_id_rule": "engine_id = 100000*split_code + 1000*fault_scenario + run_id; split_code testing=1",
        "groups": {
            name: {
                "n_run_ids": len(g["run_ids"]),
                "ordered": g["ordered"],
                "run_ids": g["run_ids"],
                "n_episodes": len(g["run_ids"]) * (TEP_N_FAULTS + 1),
            }
            for name, g in groups.items()
        },
        "usage_rules": {
            "dev": "development only (smoke tests, debugging); excluded from every reported number",
            "calibration": "conformal calibration uses fault_scenario=0 episodes of these run ids only",
            "evaluation_pool": "pre-registered n takes run_ids[:n]; each selected run id contributes 20 fault episodes + 1 normal episode",
            "demo_pool": "public demo presets (8 fault + 2 normal) drawn from these run ids; fault types fixed at pre-registration",
        },
        "calibration_engine_ids_fault_free": engine_ids_for(groups["calibration"]["run_ids"], [0]),
        "development_exposure": EXPOSURE_RECORD,
        "decisions": ["ST0-08", "ST0-09", "ST0-11"],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        old = json.loads(out.read_text(encoding="utf-8"))
        new = json.loads(text)
        for k in ("created_at",):
            old.pop(k, None); new.pop(k, None)
        if old != new:
            raise SystemExit(f"{out} exists and differs from the deterministic split; use --force to overwrite")
        print(f"[ok] {out} already matches the deterministic split; not rewritten")
        return
    out.write_text(text, encoding="utf-8")
    sizes = {k: v["n_run_ids"] for k, v in payload["groups"].items()}
    print(f"[done] wrote {out}: seed={SEED} groups={sizes} episodes="
          f"{ {k: v['n_episodes'] for k, v in payload['groups'].items()} }")
    print("[eval pool head]", payload["groups"]["evaluation_pool"]["run_ids"][:10])


if __name__ == "__main__":
    main()
