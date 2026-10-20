"""
scripts/detector_at_dev_check.py — S2-07 dev-only (run ids 1–10): AT + conformal alarm rates at fixed alpha on dev
fault-free and faulty active windows, side by side with detector_v1 (IF) from outputs/detector_dev_check.json.
Also reports the orientation check (how many faults score above fault-free). It never rewrites the model: the sign is
an explicit input of scripts/fit_detector_at.py --sign, so all metadata stays consistent. Before comparing it verifies
(1) the Stage 1 input contract on disk == the model's recorded contract and (2) the IF dev-check JSON was produced for
the same run ids, alpha, grid and n_cal; otherwise the IF column is omitted. Dev only; never the evaluation pool.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, psycopg
from core_pipeline import prereg_constants as P
from core_pipeline.anomaly.at_window import WindowAT
from core_pipeline.anomaly.conformal import ConformalCalibration
from core_pipeline.data.tep_loader import load_processed
from core_pipeline.stream.window_view import PROCESS_COLS
from scripts.fit_detector_at import raw_windows
from scripts.check_mart_inputs import verify_inputs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--processed", default="data/processed/TEP")
    ap.add_argument("--models", default="models/detector_at_v1")
    ap.add_argument("--if-check", default="outputs/detector_dev_check.json")
    ap.add_argument("--run-ids", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--split", default="outputs/stage0_episode_split.json")
    ap.add_argument("--out", default="outputs/detector_at_dev_check.json")
    a = ap.parse_args()
    assert all(1 <= r <= 20 for r in a.run_ids), "dev run ids only"
    at = WindowAT.load(Path(a.models))
    conf = ConformalCalibration.load(Path(a.models) / "conformal")
    assert at.meta.get("sign", 1) == conf.meta.get("sign", 1), "sign mismatch between at_meta and conformal — re-run fit_detector_at.py --sign"
    # (1) model contract == inputs on disk
    chk = verify_inputs(a.dsn, a.split, str(Path(a.processed) / "_manifest.json"))
    if not chk["ok"] or chk["input_contract_sha256"] != at.meta.get("input_contract_sha256"):
        for b in chk["problems"]: print("FAIL", b)
        print("abort: inputs on disk differ from the contract this model was fitted under"); return 2
    # (2) IF comparison only if produced for the same run ids / alpha / grid / n_cal
    if_res, if_note = {}, "IF column omitted"
    if Path(a.if_check).exists():
        ifj = json.loads(Path(a.if_check).read_text())
        same = (ifj.get("run_ids") == a.run_ids and ifj.get("n_cal") == conf.n_cal
                and ifj.get("alpha", P.ALPHA) == P.ALPHA and ifj.get("window", [P.WINDOW_SIZE, P.STRIDE]) == [P.WINDOW_SIZE, P.STRIDE])
        if same:
            if_res, if_note = ifj["by_fault"], f"IF column from {a.if_check} (same run ids, alpha, grid, n_cal)"
        else:
            if_note = f"IF column omitted: {a.if_check} was produced for run_ids={ifj.get('run_ids')} (this run: {a.run_ids})"
    with psycopg.connect(a.dsn) as conn, conn.cursor() as cur:
        cur.execute("""SELECT engine_id, fault_scenario FROM staging.episodes WHERE "group"='dev' AND run_id = ANY(%s)
                       ORDER BY fault_scenario, engine_id""", (a.run_ids,))
        rows = cur.fetchall()
    by_fault, normal_median = {}, None
    for f in range(0, P.N_FAULTS + 1):
        ids = [e for e, ff in rows if ff == f]
        df = load_processed(a.processed, P.SPLIT, faulty=f > 0, columns=PROCESS_COLS, engine_ids=ids)
        W = raw_windows(df, P.RUN_LENGTH)
        s = at.score(W); p = conf.p_values(s)
        idx = np.arange(len(W)) % P.N_WINDOWS
        active = (idx >= P.ONSET_WINDOW) if f > 0 else np.zeros(len(W), bool)
        rec = {"n_windows": int(len(W))}
        if f == 0:
            normal_median = float(np.median(s))
            rec["null_p_ecdf"] = {str(t): round(float((p <= t).mean()), 4) for t in (0.01, 0.05, 0.1, 0.25, 0.5)}
            rec["alarm_rate_alpha"] = round(float((p <= P.ALPHA).mean()), 4)
            rec["score_median"] = normal_median
        else:
            rec["active"] = {"alarm_rate_alpha": round(float((p[active] <= P.ALPHA).mean()), 4),
                             "p_median": round(float(np.median(p[active])), 4),
                             "score_median": float(np.median(s[active])),
                             "score_above_normal": bool(np.median(s[active]) > normal_median)}
            rec["pre_onset"] = {"alarm_rate_alpha": round(float((p[~active] <= P.ALPHA).mean()), 4)}
            if str(f) in if_res:
                rec["if_active_alarm_rate_alpha"] = if_res[str(f)]["active"]["alarm_rate_alpha"]
        by_fault[f] = rec
    n_above = sum(1 for f, r in by_fault.items() if f > 0 and r["active"]["score_above_normal"])
    summary = {"models": at.meta["version"], "sign": at.meta.get("sign", 1), "n_cal": conf.n_cal, "run_ids": a.run_ids,
               "alpha": P.ALPHA, "window": [P.WINDOW_SIZE, P.STRIDE], "input_contract_sha256": chk["input_contract_sha256"],
               "if_comparison": if_note,
               "orientation_check": {"faults_with_active_score_above_normal": n_above, "of": P.N_FAULTS,
                                     "note": "orientation is fixed by fit_detector_at.py --sign (dev decision, recorded in the decision log); this script only reports"},
               "by_fault": by_fault}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
