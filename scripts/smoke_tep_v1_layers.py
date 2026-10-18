"""0-B smoke test: does TEP contract data flow through v1 Layer 1 / 2A unchanged?

Usage (from repo root, after `python -m core_pipeline.data.tep_loader`):
    python scripts/smoke_tep_v1_layers.py --output-dir data/processed/TEP --n-runs 10

What it checks (no v1 code is modified):
  1. near-constant process columns on fault-free testing runs (candidates for a
     TEP_LOW_VARIANCE_COLS constant; the gate's correlation metric is undefined on them)
  2. QualityGate.run_governed_pipeline on fault-free testing runs with the full
     52-column list and with low-variance columns removed. Pass criteria:
     evaluated sensor set == the list passed in, every score finite (no NaN/inf
     in any sensor_scores column), output rows/episodes preserved. The
     PASS/FLAG/BLOCK split is reported but is NOT a pass criterion (recalibration
     is a Stage 1 matter).
  3. Isolation Forest trained on fault-free runs (feature_cols passed explicitly),
     then scored on one fault-free and two faulty testing runs -> score summary
     before/after the spec onset (161)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from core_pipeline.data.tep_loader import (
    TEP_PROCESS_COLS, TEP_ONSET, make_engine_id, load_processed,
)
from core_pipeline.governance.quality_gate import QualityGate, run_governed_pipeline
from core_pipeline.anomaly.isolation_forest import train_isolation_forest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="data/processed/TEP")
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--std-floor", type=float, default=1e-6, help="std below this = near-constant")
    args = ap.parse_args()

    split = "testing"
    runs = [int(make_engine_id(split, 0, r)) for r in range(1, args.n_runs + 1)]

    # 1. fault-free runs, near-constant columns
    ff = load_processed(args.output_dir, split, faulty=False, engine_ids=runs)
    print(f"[load] fault-free {split}: {len(ff):,} rows, {ff['engine_id'].nunique()} runs")
    std = ff[TEP_PROCESS_COLS].std()
    low_var = std[std < args.std_floor].index.tolist()
    print(f"[std] min={std.min():.3e} median={std.median():.3e}; near-constant (< {args.std_floor}): {low_var}")
    print("[std] 10 lowest:")
    print(std.sort_values().head(10).to_string())

    # 2. quality gate on fault-free runs
    for label, cols in (("all 52", TEP_PROCESS_COLS),
                        ("minus low-variance", [c for c in TEP_PROCESS_COLS if c not in low_var])):
        gate = QualityGate()
        out = run_governed_pipeline(ff, cols, reference_size=30, window_size=20, gate=gate)
        hist = gate.get_history_summary()
        print(f"\n[gate] sensor_cols={label} ({len(cols)}): windows={len(hist)}, "
              f"output rows={len(out):,} (input {len(ff):,}), episodes={out['engine_id'].nunique()} (input {ff['engine_id'].nunique()})")
        statuses = pd.Series([r.status.name for r in gate.history])
        print("[gate] status counts (reported, not a criterion):", statuses.value_counts().to_dict())
        scores = pd.concat([r.sensor_scores for r in gate.history], ignore_index=True)
        evaluated = set(scores["sensor"].unique())
        num = scores.select_dtypes(include="number")
        n_nan = int(num.isna().sum().sum())
        n_inf = int(np.isinf(num.to_numpy(dtype=float)).sum())
        print(f"[gate] evaluated sensors == passed list: {evaluated == set(cols)} "
              f"(evaluated {len(evaluated)}, passed {len(cols)}, missing {sorted(set(cols) - evaluated)[:5]})")
        print(f"[gate] score cells: {num.size:,}; NaN={n_nan}; inf={n_inf}; columns={list(num.columns)}")
        if n_nan or n_inf:
            print("[gate] NaN/inf by column:", num.isna().sum().to_dict())
            print("[gate] sensors with any NaN:", sorted(scores.loc[num.isna().any(axis=1), "sensor"].unique())[:20])
        print("[gate] composite summary:", scores["composite"].describe().round(2).to_dict())

    # 3. Isolation Forest on fault-free, scored on fault-free + faulty runs
    cols = [c for c in TEP_PROCESS_COLS if c not in low_var]
    result = train_isolation_forest(ff, feature_cols=cols)
    model, feat = result["model"], result["feature_cols"]
    print(f"\n[IF] trained on {len(ff):,} fault-free rows, {len(feat)} features")

    targets = [(0, 11), (1, 1), (4, 1)]  # (fault_scenario, run_id): one held-out normal, two faulty
    onset = TEP_ONSET[split]
    for fault, run in targets:
        eid = int(make_engine_id(split, fault, run))
        df = load_processed(args.output_dir, split, faulty=(fault > 0), engine_ids=[eid])
        df = df.sort_values("cycle")
        score = -model.decision_function(df[feat].to_numpy())  # higher = more anomalous
        pre = score[df["cycle"].to_numpy() < onset]
        post = score[df["cycle"].to_numpy() >= onset]
        print(f"[IF] fault={fault:2d} run={run}: rows={len(df)} "
              f"score pre-onset mean={pre.mean():.3f} sd={pre.std():.3f} | "
              f"post-onset mean={post.mean():.3f} sd={post.std():.3f} | "
              f"post>pre95pct: {(post > np.percentile(pre, 95)).mean():.2f}")


if __name__ == "__main__":
    main()
