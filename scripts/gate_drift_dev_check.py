"""
scripts/gate_drift_dev_check.py — S2-04/05 dev-only (run ids 1–10): gate PASS/FLAG/BLOCK (with reasons) and the
drift indicator with the calibrated thresholds, using REAL detector alarms (prefix-BH) so the non-rejected history is
what the policy would see. Drift shares are computed among ASSESSED windows only (denominators reported); on active
windows the indicator's reaction is split by detected-so-far vs not — a report, not a success criterion.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, psycopg
from uuid import uuid4
from core_pipeline import prereg_constants as P
from core_pipeline.anomaly.registry import load_detector
from core_pipeline.data.tep_loader import load_processed
from core_pipeline.governance.gate_adapter import GateReference
from core_pipeline.governance.gate_v2 import GateCalibration, gate_check
from core_pipeline.governance.quality_gate import QualityGate
from core_pipeline.stream.state import EpisodeState
from core_pipeline.stream.window_view import PROCESS_COLS, WindowView


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--processed", default="data/processed/TEP")
    ap.add_argument("--ref", default="data/processed/gate_reference_v1.parquet")
    ap.add_argument("--calib", default="models/gate_v2/gate_calibration.json")
    ap.add_argument("--detector", default="models/detector_v1")
    ap.add_argument("--run-ids", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--faults", type=int, nargs="+", default=[0, 1, 4, 6, 13])
    ap.add_argument("--out", default="outputs/gate_drift_dev_check.json")
    a = ap.parse_args()
    assert all(1 <= r <= 20 for r in a.run_ids)
    ref, calib, gate = GateReference.load(Path(a.ref)), GateCalibration.load(Path(a.calib)), QualityGate()
    det = load_detector(Path(a.detector)); conf = det.conformal
    with psycopg.connect(a.dsn) as conn, conn.cursor() as cur:
        cur.execute("""SELECT engine_id, run_id, fault_scenario FROM staging.episodes WHERE "group"='dev' AND run_id=ANY(%s) AND fault_scenario=ANY(%s)
                       ORDER BY fault_scenario, run_id""", (a.run_ids, a.faults))
        rows = cur.fetchall()
    recs = []
    for f in a.faults:
        ids = [e for e, _, ff in rows if ff == f]
        df = load_processed(a.processed, P.SPLIT, faulty=f > 0, columns=PROCESS_COLS, engine_ids=ids)
        for eid, g in df.groupby("engine_id", sort=True):
            g = g.sort_values("cycle").reset_index(drop=True)
            st = EpisodeState(uuid4(), P.ALPHA)
            for t in range(P.N_WINDOWS):
                v = WindowView.build(st.episode_id, t, g)
                gd = gate_check(v, ref, calib, st, gate)
                p = None
                if gd.gate != "BLOCK":
                    p = conf.p_value(det.score_rows(v.current()))
                dec = st.observe(t, gd.gate, p)
                any_alarm_so_far = any(st.decision_at(w).alarm for w in st.tested_windows() if w < t)
                recs.append({"fault": f, "t": t, "active": f > 0 and t >= P.ONSET_WINDOW, "gate": gd.gate, "drift": gd.drift,
                             "assessed": gd.drift_assessed, "drift_stat": gd.drift_stat, "thr": gd.drift_threshold,
                             "n_hist": gd.n_history, "h": gd.n_history // P.WINDOW_SIZE, "age": gd.history_age_windows, "alarm": None if dec is None else dec.alarm,
                             "detected_so_far": any_alarm_so_far, "cmin": gd.composite_min,
                             "block_reasons": gd.block_reasons, "flag_reasons": gd.flag_reasons})
    d = pd.DataFrame(recs)
    def dist(x): return x.value_counts(normalize=True).round(4).to_dict()
    def crit(x):   # critical share among ASSESSED windows only, with the denominator
        ax = x[x["assessed"]]
        return {"critical_rate": round(float((ax["drift"] == "critical").mean()), 4) if len(ax) else None, "n_assessed": int(len(ax)), "n_total": int(len(x))}
    out = {"flag_threshold": calib.flag_threshold, "n_windows": len(d),
           "by_fault": {int(f): {"gate": dist(g["gate"]), "flag_reasons": dist(g["flag_reasons"].astype(str)),
                                 "drift_all": crit(g),
                                 "drift_pre_onset": crit(g[~g["active"]]),
                                 # reaction of the indicator on active windows, split by whether the detector had already
                                 # alarmed in this episode (detected shift → excluded from history) or not (missed → enters)
                                 "drift_active_after_first_alarm": crit(g[g["active"] & g["detected_so_far"]]) if f > 0 else None,
                                 "drift_active_before_any_alarm": crit(g[g["active"] & ~g["detected_so_far"]]) if f > 0 else None,
                                 "alarm_rate_active": round(float(g[g["active"]]["alarm"].fillna(False).mean()), 4) if f > 0 else None,
                                 "drift_stat_median_by_t": {t: round(float(g[(g["t"] == t) & g["assessed"]]["drift_stat"].median()), 3) if (g[(g["t"] == t) & g["assessed"]]).shape[0] else None for t in (4, 8, 16, 32, 47)},
                                 "history_windows_median_by_t": {t: int(g[g["t"] == t]["h"].median()) for t in (4, 8, 16, 32, 47)},
                                 "evidence_age_median_by_t": {t: (None if g[g["t"] == t]["age"].isna().all() else int(g[g["t"] == t]["age"].median())) for t in (8, 16, 32, 47)}}
                        for f, g in d.groupby("fault")}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps({"summary": out, "windows": recs}, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
