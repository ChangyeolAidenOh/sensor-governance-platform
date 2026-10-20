"""
scripts/fit_detector.py — S2-01/02: fit the window IF on TRAINING fault-free windows, calibrate conformal on
CALIBRATION fault-free windows (testing split, from staging), fit Z-score L0 stats on calibration rows.
Writes models/detector_v1/{if_model.joblib, if_meta.json, conformal.npy, conformal.json, zscore_l0.npz, zscore_l0.json,
meta.json}. Data usage follows prereg §3.0 exactly; nothing from dev/evaluation/demo is read.
Usage: python scripts/fit_detector.py --dsn $SGP_PG_DSN --processed data/processed/TEP
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
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
from scripts.check_mart_inputs import verify_inputs

DETECTOR_VERSION = "detector_v1"


def windows_of(df: pd.DataFrame, run_length: int) -> list[pd.DataFrame]:
    out = []
    for _, g in df.groupby("engine_id", sort=True):
        g = g.sort_values("cycle")
        for k in range((run_length - P.WINDOW_SIZE) // P.STRIDE + 1):
            s = k * P.STRIDE + 1
            out.append(g[(g["cycle"] >= s) & (g["cycle"] <= s + P.WINDOW_SIZE - 1)])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--processed", default="data/processed/TEP")
    ap.add_argument("--out", default="models/detector_v1")
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--split", default="outputs/stage0_episode_split.json")
    a = ap.parse_args()
    t0 = time.perf_counter()
    manifest = Path(a.processed) / "_manifest.json"

    # 0) Stage 1 input contract: refuse to train on inputs that differ from the loaded mart (ST1-01/03)
    chk = verify_inputs(a.dsn, a.split, str(manifest))
    if not chk["ok"]:
        for b in chk["problems"]: print("FAIL", b)
        print("abort: input contract check failed — fix the mart/inputs before fitting"); return 2
    msha = chk["manifest_sha256"]

    # 1) training split, fault-free (all 500 runs × 500 cycles → 25 windows each)
    tr = load_processed(a.processed, "training", faulty=False, columns=PROCESS_COLS)
    Xtr = features_frame(windows_of(tr, 500))
    print(f"train windows: {len(Xtr)} from {tr['engine_id'].nunique()} runs ({time.perf_counter()-t0:.0f}s)")

    # 2) calibration group fault-free (testing, from staging — never dev/eval/demo)
    with psycopg.connect(a.dsn) as conn, conn.cursor() as cur:
        cur.execute("""SELECT engine_id FROM staging.episodes WHERE "group"='calibration' AND fault_scenario=0 ORDER BY engine_id""")
        cal_ids = [r[0] for r in cur.fetchall()]
    # the ids must be exactly the frozen calibration group (fault 0), not just 100 of something
    split_doc = json.loads(Path(a.split).read_text())
    expected = sorted(P.engine_id(P.SPLIT, 0, r) for r in split_doc["groups"]["calibration"]["run_ids"])
    assert cal_ids == expected, "calibration engine ids differ from the frozen split"
    cal = load_processed(a.processed, P.SPLIT, faulty=False, columns=PROCESS_COLS, engine_ids=cal_ids)
    Xcal = features_frame(windows_of(cal, P.RUN_LENGTH))
    print(f"calibration windows: {len(Xcal)} ({time.perf_counter()-t0:.0f}s)")

    meta = {"version": DETECTOR_VERSION, "manifest_sha256": msha, "prereg_commit": P.PREREG_COMMIT,
            "input_contract_sha256": chk["input_contract_sha256"], "load_id": chk["load_id"],
            "train": "training fault-free, all runs", "calibration": "calibration group fault-free (100 testing runs)",
            "calibration_engine_ids_sha256": hashlib.sha256(",".join(map(str, cal_ids)).encode()).hexdigest(),
            "window": [P.WINDOW_SIZE, P.STRIDE]}
    model = WindowIF.fit(Xtr, n_estimators=a.n_estimators, meta=meta)
    cal_scores = model.score(Xcal)
    conf = ConformalCalibration.from_scores(cal_scores, meta)
    z = ZScoreL0.fit(cal, meta)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model.save(out); conf.save(out / "conformal"); z.save(out)
    summary = {**meta, "n_train_windows": int(len(Xtr)), "n_cal": conf.n_cal,
               "cal_score_quantiles": {q: float(np.quantile(cal_scores, q)) for q in (0.5, 0.9, 0.95, 0.99)},
               "fit_seconds": round(time.perf_counter() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
