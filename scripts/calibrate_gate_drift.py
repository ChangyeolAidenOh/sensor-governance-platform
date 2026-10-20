"""
scripts/calibrate_gate_drift.py — S2-04/05 (ST2-03/04): calibrate the gate FLAG threshold and the per-t drift
thresholds on the CALIBRATION fault-free episodes with the SAME pipeline the evaluation runs:
  gate_check (integrity BLOCK only) → detector score → conformal p (LEAVE-ONE-OUT, since these windows are the
  detector's own calibration set) → prefix-BH via EpisodeState → non-rejected history → D_t.
  flag_threshold      = 5th percentile of window-min composite over all 4,800 windows
  drift_thresholds[h] = 99th percentile of D over all (episode, window) pairs whose non-rejected history holds exactly
                        h windows (indexed by HISTORY SIZE, not window position; n_assessed[h] recorded)
Gated by the Stage 1 input contract. Writes models/gate_v2/gate_calibration.json.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, psycopg
from uuid import uuid4
from core_pipeline import prereg_constants as P
from core_pipeline.anomaly.registry import load_detector
from core_pipeline.data.tep_loader import load_processed
from core_pipeline.governance.gate_adapter import GateReference
from core_pipeline.governance.gate_v2 import (GATE_VERSION, MIN_HISTORY, PSI_CONST, GateCalibration, drift_statistic,
                                              gate_check, non_alarm_history)
from core_pipeline.governance.quality_gate import QualityGate
from core_pipeline.stream.state import EpisodeState
from core_pipeline.stream.window_view import PROCESS_COLS, WindowView
from scripts.check_mart_inputs import verify_inputs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--processed", default="data/processed/TEP")
    ap.add_argument("--ref", default="data/processed/gate_reference_v1.parquet")
    ap.add_argument("--detector", default="models/detector_v1")
    ap.add_argument("--split", default="outputs/stage0_episode_split.json")
    ap.add_argument("--out", default="models/gate_v2/gate_calibration.json")
    a = ap.parse_args()
    t0 = time.perf_counter()
    chk = verify_inputs(a.dsn, a.split, str(Path(a.processed) / "_manifest.json"))
    if not chk["ok"]:
        for b in chk["problems"]: print("FAIL", b)
        return 2
    ref = GateReference.load(Path(a.ref))
    det = load_detector(Path(a.detector)); conf = det.conformal
    with psycopg.connect(a.dsn) as conn, conn.cursor() as cur:
        cur.execute("""SELECT engine_id FROM staging.episodes WHERE "group"='calibration' AND fault_scenario=0 ORDER BY engine_id""")
        ids = [r[0] for r in cur.fetchall()]
    split_doc = json.loads(Path(a.split).read_text())
    assert ids == sorted(P.engine_id(P.SPLIT, 0, r) for r in split_doc["groups"]["calibration"]["run_ids"])
    df = load_processed(a.processed, P.SPLIT, faulty=False, columns=PROCESS_COLS, engine_ids=ids)
    gate = QualityGate()
    varying = [c for c in PROCESS_COLS if ref.frame[c].std() > 0]
    # provisional calibration: integrity-only BLOCK, no FLAG threshold, no drift thresholds (we are computing them)
    prov = GateCalibration(flag_threshold=-np.inf, drift_thresholds=[None] * P.N_WINDOWS, varying_sensors=varying)
    cmins, D = [], {h: [] for h in range(P.N_WINDOWS + 1)}
    alarms = 0
    for i, (_, g) in enumerate(df.groupby("engine_id", sort=True)):
        g = g.sort_values("cycle").reset_index(drop=True)
        st = EpisodeState(uuid4(), P.ALPHA)
        for t in range(P.N_WINDOWS):
            v = WindowView.build(st.episode_id, t, g)
            gd = gate_check(v, ref, prov, st, gate)
            cmins.append(gd.composite_min)
            hist = non_alarm_history(v, st)                 # decided from windows < t, same rule as evaluation
            if len(hist) >= MIN_HISTORY:
                D[len(hist) // P.WINDOW_SIZE].append(drift_statistic(hist, ref))
            p = None
            if gd.gate != "BLOCK":
                p = conf.p_value_loo(det.score_rows(v.current()))   # LOO: this window is a calibration member
            dec = st.observe(t, gd.gate, p)
            alarms += bool(dec and dec.alarm)
        if i % 10 == 9:
            print(f"  {i+1}/100 episodes ({time.perf_counter()-t0:.0f}s)")
    cmins = np.array([c for c in cmins if np.isfinite(c)])
    flag_thr = float(np.percentile(cmins, 5))
    thr = [float(np.percentile(D[h], 99)) if len(D[h]) >= 20 else None for h in range(P.N_WINDOWS + 1)]
    calib = GateCalibration(flag_thr, thr, varying, {
        "version": GATE_VERSION, "psi_const": PSI_CONST, "min_history": MIN_HISTORY, "n_windows": int(len(cmins)),
        "alarm_rate_loo_prefix_bh": round(alarms / (100 * P.N_WINDOWS), 4),
        "n_assessed_by_h": {h: len(D[h]) for h in range(P.N_WINDOWS + 1)},
        "threshold_index": "h = number of non-rejected windows in the history",
        "composite_min_quantiles": {q: float(np.percentile(cmins, q)) for q in (1, 5, 25, 50, 75)},
        "composite_min_at_floor_share": round(float((cmins <= 30.0 + 1e-9).mean()), 4),   # share of windows at the 30 floor
        "n_assessed_note": "pairs (episode, window) — a frozen history contributes the same D repeatedly; NOT independent episodes",
        "drift_stat_median_by_h": {h: round(float(np.median(D[h])), 3) for h in (2, 8, 24, 47) if D[h]},
        "pipeline": "gate_check(integrity) → detector → conformal LOO p → prefix-BH → non-rejected history → D_t",
        "ref_source_sha256": ref.sha256, "detector_version": det.version, "detector_path": a.detector,
        "input_contract_sha256": chk["input_contract_sha256"], "load_id": chk["load_id"],
        "prereg_commit": P.PREREG_COMMIT, "seconds": round(time.perf_counter() - t0, 1)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); calib.save(Path(a.out))
    print(json.dumps({"flag_threshold": flag_thr, "drift_thresholds_sample": {h: thr[h] for h in (2, 8, 24, 47)},
                      **{k: v for k, v in calib.meta.items() if k != "n_assessed_by_h"},
                      "n_assessed_sample": {h: len(D[h]) for h in (2, 8, 24, 47)}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
