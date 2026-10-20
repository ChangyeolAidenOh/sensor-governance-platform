"""
scripts/diagnoser_dev_check.py — S2-03 dev-only (run ids 1–10): on dev faulty ACTIVE windows report coverage, set-size
distribution, s=0 rate, top-1 accuracy, per fault, for w20 and w40, plus Mondrian (exploratory). These are development
diagnostics; H4 is evaluated in Stage 5 on the evaluation pool (w20 primary windows only).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, psycopg
from core_pipeline import prereg_constants as P
from core_pipeline.data.tep_loader import load_processed
from core_pipeline.diagnosis.classifier import FaultDiagnoser, FAULTS
from core_pipeline.stream.window_view import PROCESS_COLS
from scripts.fit_diagnoser import build, episodes
from core_pipeline.features.window_features import expanded_window_features


def summarize(d, key, X, y, mondrian=False):
    pr = d.proba(X)
    sets = d.calib[key].sets_mondrian(pr) if mondrian else d.calib[key].sets(pr)
    hit = np.array([y[i] in sets[i] for i in range(len(y))]); size = np.array([len(s) for s in sets])
    per_f = {int(f): {"coverage": round(float(hit[y == f].mean()), 3), "size_mean": round(float(size[y == f].mean()), 2),
                      "top1": round(float((pr[y == f].argmax(1) + 1 == f).mean()), 3)} for f in FAULTS}
    return {"coverage": round(float(hit.mean()), 4), "set_size_mean": round(float(size.mean()), 3),
            "size_hist": {str(k): int((size == k).sum()) for k in range(0, 8)}, "size_gt3": round(float((size > 3).mean()), 4),
            "size_0": round(float((size == 0).mean()), 4), "valid_1to3": round(float(((size >= 1) & (size <= 3)).mean()), 4),
            "top1_acc": round(float((pr.argmax(1) + 1 == y).mean()), 4), "min_class_coverage": min(v["coverage"] for v in per_f.values()),
            "per_fault": per_f}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--processed", default="data/processed/TEP")
    ap.add_argument("--models", default="models/diagnoser_v1")
    ap.add_argument("--run-ids", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--out", default="outputs/diagnoser_dev_check.json")
    a = ap.parse_args()
    assert all(1 <= r <= 20 for r in a.run_ids), "dev run ids only"
    d = FaultDiagnoser.load(Path(a.models))
    with psycopg.connect(a.dsn) as conn, conn.cursor() as cur:
        cur.execute("""SELECT engine_id FROM staging.episodes WHERE "group"='dev' AND run_id = ANY(%s) AND fault_scenario>0 ORDER BY engine_id""", (a.run_ids,))
        ids = [r[0] for r in cur.fetchall()]
    df = load_processed(a.processed, P.SPLIT, faulty=True, columns=PROCESS_COLS, engine_ids=ids)
    X20, y20, X40, y40 = build(df, P.RUN_LENGTH, P.ONSET_WINDOW, want40=True)
    # window-8 expansion (7 ∪ 8): a normal/fault MIXED window not present in the w40 calibration — reported separately
    X8 = np.vstack([expanded_window_features(x, P.WINDOW_SIZE, P.STRIDE, np.array([P.ONSET_WINDOW])) for _, _, x in episodes(df)])
    y8 = np.array([f for _, f, _ in episodes(df)])
    out = {"models": d.meta["version"], "run_ids": a.run_ids, "n_windows": {"w20": int(len(y20)), "w40_k_ge_9": int(len(y40)), "w40_k8_mixed": int(len(y8))},
           "q": {k: v.q for k, v in d.calib.items()},
           "w20": summarize(d, "w20", X20, y20), "w40_k_ge_9": summarize(d, "w40", X40, y40), "w40_k8_mixed": summarize(d, "w40", X8, y8),
           "note": "all expansion candidates; the coverage of expansions ACTUALLY invoked by the policy is a Stage 5 replay-log item"}
    if d.calib["w20"].q_by_class:
        out["w20_mondrian_exploratory"] = summarize(d, "w20", X20, y20, mondrian=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, indent=1))
    slim = {k: (v if k not in ("w20", "w40_k_ge_9", "w40_k8_mixed", "w20_mondrian_exploratory") else {kk: vv for kk, vv in v.items() if kk != "per_fault"}) for k, v in out.items()}
    print(json.dumps(slim, indent=1))
    print("per-fault (w20):", json.dumps({f: (v["coverage"], v["size_mean"], v["top1"]) for f, v in out["w20"]["per_fault"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
