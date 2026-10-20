"""
scripts/fit_diagnoser.py — S2-03: fit the LightGBM fault classifier on TRAINING faulty active windows and calibrate
the conformal sets on CALIBRATION faulty episodes' active windows — 'w20' (windows 8..47) and 'w40' (backward pairs
(k-1,k), k = 9..47, both active). Gated by the Stage 1 input contract. Writes models/diagnoser_v1/.
Usage: python scripts/fit_diagnoser.py --dsn $SGP_PG_DSN --processed data/processed/TEP
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, psycopg
from core_pipeline import prereg_constants as P
from core_pipeline.data.tep_loader import load_processed
from core_pipeline.features.window_features import episode_window_features, expanded_window_features
from core_pipeline.diagnosis.classifier import FaultDiagnoser
from core_pipeline.stream.window_view import PROCESS_COLS
from scripts.check_mart_inputs import verify_inputs

TRAIN_LEN, TRAIN_ONSET_WINDOW = 500, 1          # training split: 500 cycles, onset cycle 21 → window 1 fully active


def episodes(df: pd.DataFrame):
    for eid, g in df.groupby("engine_id", sort=True):
        g = g.sort_values("cycle")
        f = (int(eid) % 100_000) // 1000                       # fault encoded in engine_id (ST0-11); loader side only
        if "fault_scenario" in g.columns:
            assert int(g["fault_scenario"].iloc[0]) == f, eid
        yield int(eid), f, g[PROCESS_COLS].to_numpy(dtype=np.float64)


def build(df: pd.DataFrame, run_len: int, onset_w: int, want40: bool):
    X20, y20, X40, y40 = [], [], [], []
    last = (run_len - P.WINDOW_SIZE) // P.STRIDE
    for _, f, x in episodes(df):
        X20.append(episode_window_features(x, P.WINDOW_SIZE, P.STRIDE, onset_w, last)); y20.append(np.full(last - onset_w + 1, f))
        if want40:
            ks = np.arange(onset_w + 1, last + 1)
            X40.append(expanded_window_features(x, P.WINDOW_SIZE, P.STRIDE, ks)); y40.append(np.full(len(ks), f))
    out = (np.vstack(X20), np.concatenate(y20))
    return out + ((np.vstack(X40), np.concatenate(y40)) if want40 else (None, None))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--processed", default="data/processed/TEP")
    ap.add_argument("--split", default="outputs/stage0_episode_split.json")
    ap.add_argument("--out", default="models/diagnoser_v1")
    ap.add_argument("--rounds", type=int, default=300)
    a = ap.parse_args()
    t0 = time.perf_counter()
    chk = verify_inputs(a.dsn, a.split, str(Path(a.processed) / "_manifest.json"))
    if not chk["ok"]:
        for b in chk["problems"]: print("FAIL", b)
        print("abort: input contract check failed"); return 2

    tr = load_processed(a.processed, "training", faulty=True, columns=PROCESS_COLS)
    Xtr, ytr, _, _ = build(tr, TRAIN_LEN, TRAIN_ONSET_WINDOW, want40=False)
    print(f"train: {Xtr.shape} ({time.perf_counter()-t0:.0f}s)")
    del tr

    split_doc = json.loads(Path(a.split).read_text())
    cal_runs = split_doc["groups"]["calibration"]["run_ids"]
    with psycopg.connect(a.dsn) as conn, conn.cursor() as cur:
        cur.execute("""SELECT engine_id FROM staging.episodes WHERE "group"='calibration' AND fault_scenario>0 ORDER BY engine_id""")
        cal_ids = [r[0] for r in cur.fetchall()]
    expected = sorted(P.engine_id(P.SPLIT, f, r) for r in cal_runs for f in range(1, P.N_FAULTS + 1))
    assert cal_ids == expected, "calibration faulty engine ids differ from the frozen split"
    cal = load_processed(a.processed, P.SPLIT, faulty=True, columns=PROCESS_COLS, engine_ids=cal_ids)
    Xc20, yc20, Xc40, yc40 = build(cal, P.RUN_LENGTH, P.ONSET_WINDOW, want40=True)
    print(f"calibration: w20 {Xc20.shape}, w40 {Xc40.shape} ({time.perf_counter()-t0:.0f}s)")
    del cal

    meta = {"version": "diagnoser_v1", "manifest_sha256": chk["manifest_sha256"], "input_contract_sha256": chk["input_contract_sha256"],
            "load_id": chk["load_id"], "prereg_commit": P.PREREG_COMMIT,
            "train_data": "training faulty, active windows (window >= 1 of 25)", "calibration_data": "calibration faulty 2000 episodes, active windows",
            "calibration_engine_ids_sha256": hashlib.sha256(",".join(map(str, cal_ids)).encode()).hexdigest()}
    d = FaultDiagnoser.fit(Xtr, ytr, num_boost_round=a.rounds, meta=meta)
    print(f"fit done ({time.perf_counter()-t0:.0f}s)")
    d.calibrate("w20", Xc20, yc20); d.calibrate("w40", Xc40, yc40)
    # calibration-set self-report (not a test metric): top1 accuracy and set size on the calibration windows
    for key, X, y in (("w20", Xc20, yc20), ("w40", Xc40, yc40)):
        sets, pr = d.predict_sets(X, key)
        d.meta["calibration"][key].update({"top1_acc_on_cal": round(float((pr.argmax(1) + 1 == y).mean()), 4),
                                           "set_size_mean_on_cal": round(float(np.mean([len(s) for s in sets])), 3)})
    d.meta["fit_seconds"] = round(time.perf_counter() - t0, 1)
    d.save(Path(a.out))
    print(json.dumps(d.meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
