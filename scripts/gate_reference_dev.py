"""
scripts/gate_reference_dev.py — S1-04 (c): FLAG-100% diagnosis on DEV run ids ONLY (ST0-11 (3)). Never evaluation.
Applies v1 default thresholds (flag 70 / block 30) with the external reference and reports the composite and its
three components per window, so Stage 2 can see WHERE the score collapses (expected: PSI of a 20-sample window
against 10 reference bins is intrinsically large → stability ≈ 0). No threshold is changed here.
Writes outputs/gate_reference_dev.json (tracked? no — local) and prints a summary.
Usage: python scripts/gate_reference_dev.py --dsn $SGP_PG_DSN --processed data/processed/TEP [--run-ids 1 2]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import psycopg
from core_pipeline import prereg_constants as P
from core_pipeline.data.tep_loader import load_processed
from core_pipeline.governance.gate_adapter import GateReference, evaluate_window, gate_status_str
from core_pipeline.governance.quality_gate import QualityGate
from core_pipeline.stream.window_view import PROCESS_COLS, WindowView
from uuid import UUID


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--processed", default="data/processed/TEP")
    ap.add_argument("--ref", default="data/processed/gate_reference_v1.parquet")
    ap.add_argument("--run-ids", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--faults", type=int, nargs="+", default=[0, 1, 4])
    ap.add_argument("--out", default="outputs/gate_reference_dev.json")
    a = ap.parse_args()
    with psycopg.connect(a.dsn) as conn, conn.cursor() as cur:
        cur.execute("""SELECT episode_id, engine_id, run_id, fault_scenario FROM staging.episodes
                       WHERE "group"='dev' AND run_id = ANY(%s) AND fault_scenario = ANY(%s) ORDER BY run_id, fault_scenario""",
                    (a.run_ids, a.faults))
        rows = cur.fetchall()
    assert rows and all(r in range(1, 21) for _, _, r, _ in rows), "dev run ids only (1–20)"
    ref = GateReference.load(Path(a.ref))
    gate = QualityGate()                                   # v1 defaults, unchanged
    recs = []
    for episode_id, engine_id, run_id, fault in rows:
        df = load_processed(a.processed, P.SPLIT, faulty=fault > 0, columns=PROCESS_COLS, engine_ids=[engine_id])
        for t in range(P.N_WINDOWS):
            view = WindowView.build(UUID(str(episode_id)), t, df)
            res = evaluate_window(gate, view, ref)
            sc = res.sensor_scores
            recs.append({"run_id": run_id, "fault": fault, "window": t, "status": gate_status_str(res),
                         "composite_mean": float(sc["composite"].mean()), "composite_min": float(sc["composite"].min()),
                         "completeness_mean": float(sc["completeness"].mean()), "stability_mean": float(sc["stability"].mean()),
                         "consistency_mean": float(sc["consistency"].mean()), "psi_median": float(sc["psi"].median()),
                         "corr_dev_median": float(sc["corr_deviation"].median()), "n_flag": len(res.flagged_sensors),
                         "n_block": len(res.blocked_sensors), "nan": int(sc[["composite", "psi", "corr_deviation"]].isna().sum().sum())})
    d = pd.DataFrame(recs)
    summary = {"ref": ref.meta, "n_windows": len(d), "status_counts": d["status"].value_counts().to_dict(),
               "nan_cells": int(d["nan"].sum()),
               "by_fault": {int(f): {"composite_mean": round(float(g["composite_mean"].mean()), 2),
                                     "stability_mean": round(float(g["stability_mean"].mean()), 2),
                                     "consistency_mean": round(float(g["consistency_mean"].mean()), 2),
                                     "completeness_mean": round(float(g["completeness_mean"].mean()), 2),
                                     "psi_median": round(float(g["psi_median"].median()), 4),
                                     "corr_dev_median": round(float(g["corr_dev_median"].median()), 4),
                                     "status": g["status"].value_counts().to_dict()}
                            for f, g in d.groupby("fault")}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summary, "windows": recs}, indent=1))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
