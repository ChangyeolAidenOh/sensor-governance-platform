"""
scripts/fit_detector_at.py — S2-07: Anomaly Transformer (v1 code) + split conformal, on the same windows as the IF
detector (training fault-free 12,500 for fitting; calibration fault-free 4,800 for conformal). Writes
models/detector_at_v1/. Gated by the input contract. ~minutes on M2 (MPS); use --epochs to shorten a smoke run,
but the reported model must use the recorded epochs.
Usage: python scripts/fit_detector_at.py --dsn $SGP_PG_DSN --processed data/processed/TEP [--epochs 30]
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, psycopg
from core_pipeline import prereg_constants as P
from core_pipeline.anomaly.at_window import WindowAT
from core_pipeline.anomaly.conformal import ConformalCalibration
from core_pipeline.data.tep_loader import load_processed
from core_pipeline.stream.window_view import PROCESS_COLS
from scripts.check_mart_inputs import verify_inputs


def raw_windows(df: pd.DataFrame, run_length: int) -> np.ndarray:
    """(n_windows, 20, 52) raw windows on the pre-registered grid, episode by episode (sorted by engine_id)."""
    out = []
    for _, g in df.groupby("engine_id", sort=True):
        x = g.sort_values("cycle")[PROCESS_COLS].to_numpy(dtype=np.float64)
        for k in range((run_length - P.WINDOW_SIZE) // P.STRIDE + 1):
            out.append(x[k * P.STRIDE: k * P.STRIDE + P.WINDOW_SIZE])
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--processed", default="data/processed/TEP")
    ap.add_argument("--split", default="outputs/stage0_episode_split.json")
    ap.add_argument("--out", default="models/detector_at_v1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sign", type=int, choices=[1, -1], default=1,
                    help="score orientation (+1: higher discrepancy = more anomalous). A dev-time decision (ST2-06) given "
                         "explicitly here so at_meta.json, conformal.json and meta.json always agree; never changed after the fact.")
    a = ap.parse_args()
    t0 = time.perf_counter()
    chk = verify_inputs(a.dsn, a.split, str(Path(a.processed) / "_manifest.json"))
    if not chk["ok"]:
        for b in chk["problems"]: print("FAIL", b)
        return 2
    tr = load_processed(a.processed, "training", faulty=False, columns=PROCESS_COLS)
    Wtr = raw_windows(tr, 500); del tr
    print(f"train windows: {Wtr.shape} ({time.perf_counter()-t0:.0f}s)")
    with psycopg.connect(a.dsn) as conn, conn.cursor() as cur:
        cur.execute("""SELECT engine_id FROM staging.episodes WHERE "group"='calibration' AND fault_scenario=0 ORDER BY engine_id""")
        cal_ids = [r[0] for r in cur.fetchall()]
    split_doc = json.loads(Path(a.split).read_text())
    assert cal_ids == sorted(P.engine_id(P.SPLIT, 0, r) for r in split_doc["groups"]["calibration"]["run_ids"])
    cal = load_processed(a.processed, P.SPLIT, faulty=False, columns=PROCESS_COLS, engine_ids=cal_ids)
    Wcal = raw_windows(cal, P.RUN_LENGTH); del cal
    print(f"calibration windows: {Wcal.shape} ({time.perf_counter()-t0:.0f}s)")

    meta = {"manifest_sha256": chk["manifest_sha256"], "input_contract_sha256": chk["input_contract_sha256"],
            "load_id": chk["load_id"], "prereg_commit": P.PREREG_COMMIT, "role": "Should comparison (prereg §4.1); IF remains primary",
            "train_data": "training fault-free, all runs (same windows as detector_v1)",
            "calibration_data": "calibration fault-free 100 runs (same windows as detector_v1)",
            "calibration_engine_ids_sha256": hashlib.sha256(",".join(map(str, cal_ids)).encode()).hexdigest()}
    at = WindowAT.fit(Wtr, n_epochs=a.epochs, device=a.device, meta=meta)
    at.meta["sign"] = a.sign                                   # single source of orientation
    cal_scores = at.score(Wcal)                                # oriented
    conf = ConformalCalibration.from_scores(cal_scores, {"detector": "at_window_v1", "sign": a.sign})
    out = Path(a.out); at.save(out); conf.save(out / "conformal")
    summary = {**at.meta, "n_cal": conf.n_cal,
               "cal_score_quantiles": {q: float(np.quantile(cal_scores, q)) for q in (0.5, 0.9, 0.95, 0.99)},
               "fit_seconds": round(time.perf_counter() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
