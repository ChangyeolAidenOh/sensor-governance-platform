"""
scripts/detector_dev_check.py — S2-01 dev-only check (dev run ids 1–20, never evaluation). Reports, with no
success criterion: (1) empirical CDF of p on dev fault-free windows (null_p_ecdf: observed P(p<=t), a diagnostic, not an exchangeability check); (2) per-window alarm rate at
alpha (marginal, no BH) on fault-free vs. faulty active windows, per fault; (3) B2 Z-score alarm rates likewise.
These are development diagnostics; H2/H1 are evaluated only in Stage 5 with prefix-BH on the evaluation pool.
Usage: python scripts/detector_dev_check.py --dsn $SGP_PG_DSN --processed data/processed/TEP
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, psycopg
from core_pipeline import prereg_constants as P
from core_pipeline.data.tep_loader import load_processed
from core_pipeline.features.window_features import features_frame
from core_pipeline.anomaly.conformal import ConformalCalibration
from core_pipeline.anomaly.if_window import WindowIF
from core_pipeline.anomaly.zscore_l0 import ZScoreL0
from core_pipeline.stream.window_view import PROCESS_COLS
from scripts.fit_detector import windows_of


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--processed", default="data/processed/TEP")
    ap.add_argument("--models", default="models/detector_v1")
    ap.add_argument("--run-ids", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--out", default="outputs/detector_dev_check.json")
    a = ap.parse_args()
    assert all(1 <= r <= 20 for r in a.run_ids), "dev run ids only"
    m = WindowIF.load(Path(a.models)); conf = ConformalCalibration.load(Path(a.models) / "conformal"); z = ZScoreL0.load(Path(a.models))
    with psycopg.connect(a.dsn) as conn, conn.cursor() as cur:
        cur.execute("""SELECT engine_id, fault_scenario FROM staging.episodes WHERE "group"='dev' AND run_id = ANY(%s)
                       ORDER BY fault_scenario, engine_id""", (a.run_ids,))
        rows = cur.fetchall()
    by_fault = {}
    for f in range(0, P.N_FAULTS + 1):
        ids = [e for e, ff in rows if ff == f]
        df = load_processed(a.processed, P.SPLIT, faulty=f > 0, columns=PROCESS_COLS, engine_ids=ids)
        wins = windows_of(df, P.RUN_LENGTH)
        p = conf.p_values(m.score(features_frame(wins)))
        zal = np.array([z.alarm(w) for w in wins])
        idx = np.arange(len(wins)) % P.N_WINDOWS
        active = (idx >= P.ONSET_WINDOW) if f > 0 else np.zeros(len(wins), bool)
        rec = {"n_windows": len(wins)}
        if f == 0:
            rec["null_p_ecdf"] = {str(t): round(float((p <= t).mean()), 4) for t in (0.01, 0.05, 0.1, 0.25, 0.5)}
            rec["alarm_rate_alpha"] = round(float((p <= P.ALPHA).mean()), 4)
            rec["b2_alarm_rate"] = round(float(zal.mean()), 4)
        else:
            rec["pre_onset"] = {"alarm_rate_alpha": round(float((p[~active] <= P.ALPHA).mean()), 4), "b2": round(float(zal[~active].mean()), 4)}
            rec["active"] = {"alarm_rate_alpha": round(float((p[active] <= P.ALPHA).mean()), 4), "b2": round(float(zal[active].mean()), 4),
                             "p_median": round(float(np.median(p[active])), 4)}
        by_fault[f] = rec
    summary = {"models": m.meta.get("version"), "n_cal": conf.n_cal, "run_ids": a.run_ids, "alpha": P.ALPHA,
               "window": [P.WINDOW_SIZE, P.STRIDE], "by_fault": by_fault}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
