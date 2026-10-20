"""
core_pipeline/governance/gate_v2.py — S2-04/05: causal gate + drift with calibrated thresholds (decision ST2-03/04).

Gate (prereg §4.1 "gate ∈ {PASS, FLAG, BLOCK}"; BLOCK semantics narrowed — ST2-03, a policy-affecting amendment):
  BLOCK  = input-contract violation in the CURRENT window only: non-finite values or missing samples (fewer than
           WINDOW_SIZE rows). BLOCK ⇒ detect_diagnose is not called (skill contract §1.1).
  FLAG   = (a) a flat-lined sensor (zero range inside the window for a sensor that varies in calibration) — a constant
           value alone cannot distinguish a sensor/communication failure from a held control input, saturation or a
           real stuck valve, so it is flagged, not blocked; or (b) the v1 quality score (completeness / PSI stability /
           correlation consistency vs the external GateReference, QualityGate.evaluate unchanged) below the calibrated
           threshold = 5th percentile of the window-minimum composite over CALIBRATION fault-free windows.
           FLAG is informational; detection still runs. The v1 stability score drops out-of-range samples
           (quality_metrics.compute_psi), so FLAG is not a trustworthy quality verdict and is not presented as one.
  PASS   otherwise.
  Rationale: PSI/correlation scores cannot separate sensor-quality problems from process faults (ST1-04); letting
  them BLOCK would silence detection exactly when a fault starts.

Drift (prereg §4.1 "drift ∈ {ok, critical}"; input selection, aggregation and threshold ALL revised — ST2-04, an
amendment to the pre-registration, recorded with the original text preserved):
  This is NOT a direct check of "model trust". It is a *reference-departure indicator of the non-rejected history*:
  D_t = fraction of sensors whose open-bin PSI(reference → non-rejected history) > PSI_CONST (v1 constant 0.2), where
  the non-rejected history = samples of windows 0..t-1 that were tested and NOT alarmed under prefix-BH. Because the
  detector chooses that subset, a shift the detector catches is excluded (D_t stays low) and a shift it misses enters
  the history (D_t rises); after sustained alarms only old windows remain. These properties are reported, not tuned.
  critical iff D_t > the 99th percentile of D over calibration fault-free episodes with the SAME NUMBER OF
  NON-REJECTED WINDOWS h (threshold indexed by history size, NOT by window position t): PSI's small-sample bias is a
  function of how many samples the history holds, and once alarms persist the history stops growing while t keeps
  advancing — a t-indexed threshold then compares a frozen 160-sample history with a null computed for 900-sample
  histories (found on dev: critical 76–89% after the first alarm with D frozen at 0.33 — recorded in ST2-04). The
  calibration run uses the SAME detector, leave-one-out conformal p, prefix-BH and history selection
  (scripts/calibrate_gate_drift.py). Fewer than MIN_HISTORY non-rejected samples → not assessed (reported
  separately from 'ok', never folded into the ok share).
  The v1 composite FLAG is kept for completeness but carries no information at 20-sample windows: the window-min
  composite equals 30.0 on 100% of calibration windows (stability 0, consistency 0 for at least one sensor), so the
  5th-percentile threshold is 30.0 and never fires. FLAG in practice = flatline only.
All inputs are causal: current window + past windows' recorded decisions + external calibration only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core_pipeline import prereg_constants as P
from core_pipeline.governance.gate_adapter import GateReference
from core_pipeline.governance.quality_gate import QualityGate
from core_pipeline.stream.state import EpisodeState
from core_pipeline.stream.window_view import PROCESS_COLS, WindowView

PSI_CONST = 0.2                 # v1 per-sensor PSI "significant change" constant (quality_metrics docstring)
PSI_BINS = 10
PSI_EPS = 1e-4
MIN_HISTORY = 40                # samples of non-alarm history needed before drift is assessed (2 windows)
GATE_VERSION = "gate_v2"
CALIB_SCHEMA = 2                                   # 2 = thresholds indexed by non-rejected window count (rev 3)
THRESHOLD_INDEX = "non_rejected_window_count"


@dataclass(frozen=True)
class GateDecision:
    gate: str                   # PASS | FLAG | BLOCK
    drift: str                  # ok | critical   ('ok' also when not assessed; see drift_assessed)
    composite_min: float
    composite_mean: float
    block_reasons: list[str]    # non_finite | missing_samples
    flag_reasons: list[str]     # flatline:<sensors> | composite_below_threshold
    drift_assessed: bool
    drift_stat: Optional[float]
    drift_threshold: Optional[float]
    n_history: int              # non-rejected samples behind the drift decision
    history_age_windows: Optional[int] = None   # t − index of the last non-rejected window (staleness of the evidence)


@dataclass
class GateCalibration:
    flag_threshold: float                       # 5th percentile of window-min composite on calibration windows
    drift_thresholds: list[Optional[float]]     # indexed by h = number of non-rejected windows in the history (None = n/a)
    varying_sensors: list[str]                  # sensors with nonzero variance in calibration (flatline test applies)
    meta: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"schema_version": CALIB_SCHEMA, "threshold_index": THRESHOLD_INDEX,
                                          "flag_threshold": self.flag_threshold, "drift_thresholds": self.drift_thresholds,
                                          "varying_sensors": self.varying_sensors, "meta": self.meta}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "GateCalibration":
        """Refuses calibration files written for another threshold indexing (e.g. the rev-2 per-t file at the same
        path): a silent reinterpretation of t-indexed thresholds as h-indexed would reintroduce the frozen-history bug."""
        d = json.loads(Path(path).read_text())
        if d.get("schema_version") != CALIB_SCHEMA or d.get("threshold_index") != THRESHOLD_INDEX:
            raise ValueError(f"{path}: calibration schema {d.get('schema_version')!r}/{d.get('threshold_index')!r} "
                             f"!= required {CALIB_SCHEMA}/{THRESHOLD_INDEX!r} — re-run scripts/calibrate_gate_drift.py")
        return cls(d["flag_threshold"], d["drift_thresholds"], d["varying_sensors"], d["meta"])


def integrity_block_reasons(rows: pd.DataFrame, n_expected: int = P.WINDOW_SIZE) -> list[str]:
    """Input-contract violations only → BLOCK."""
    reasons = []
    x = rows[PROCESS_COLS].to_numpy(dtype=np.float64)
    if len(rows) < n_expected:
        reasons.append(f"missing_samples:{len(rows)}/{n_expected}")
    if not np.isfinite(x).all():
        reasons.append("non_finite")
    return reasons


def flatline_sensors(rows: pd.DataFrame, varying: list[str]) -> list[str]:
    """Constant value inside the window (exact range test) for sensors that vary in calibration → FLAG, not BLOCK."""
    return [c for c in varying if np.ptp(rows[c].to_numpy(dtype=np.float64)) == 0]


def psi_open_bins(reference: np.ndarray, current: np.ndarray, n_bins: int = PSI_BINS, eps: float = PSI_EPS) -> float:
    """PSI with the two outer bins open (-inf, +inf). v1 quality_metrics.compute_psi builds bins from reference
    percentiles and np.histogram DROPS samples outside [min, max] of the reference — a large shift then yields PSI ≈ 0
    (found by test_drift_uses_only_non_alarm_past_windows). Drift must count out-of-range mass, so it uses this version;
    the v1 function is left unchanged (its limitation is recorded in the decision log)."""
    ref = reference[np.isfinite(reference)]; cur = current[np.isfinite(current)]
    if len(ref) < 2 or len(cur) < 2:
        return 1.0
    inner = np.unique(np.percentile(ref, np.linspace(0, 100, n_bins + 1)[1:-1]))
    edges = np.concatenate([[-np.inf], inner, [np.inf]])
    rp = np.histogram(ref, bins=edges)[0] / len(ref) + eps
    cp = np.histogram(cur, bins=edges)[0] / len(cur) + eps
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def drift_statistic(history: pd.DataFrame, ref: GateReference) -> float:
    """Fraction of sensors with open-bin PSI(reference → history) > PSI_CONST."""
    n_over = sum(1 for c in PROCESS_COLS if psi_open_bins(ref.frame[c].to_numpy(), history[c].to_numpy()) > PSI_CONST)
    return n_over / len(PROCESS_COLS)


def non_rejected_windows(view: WindowView, state: EpisodeState) -> list[int]:
    return [w for w in state.tested_windows() if w < view.window_index and not state.decision_at(w).alarm]


def non_alarm_history(view: WindowView, state: EpisodeState) -> pd.DataFrame:
    """Non-rejected history: samples of past windows (< current) that were tested and not alarmed. Causal."""
    keep = non_rejected_windows(view, state)
    if not keep:
        return view.history().iloc[0:0]
    parts = [view.cycles(w * P.STRIDE + 1, w * P.STRIDE + P.WINDOW_SIZE) for w in keep]
    return pd.concat(parts, ignore_index=True)


def gate_check(view: WindowView, ref: GateReference, calib: GateCalibration, state: EpisodeState,
               gate: Optional[QualityGate] = None) -> GateDecision:
    cur = view.current()
    reasons = integrity_block_reasons(cur)
    if reasons:
        return GateDecision("BLOCK", "ok", float("nan"), float("nan"), reasons, [], False, None, None, 0)
    res = (gate or QualityGate()).evaluate(cur, ref.frame, PROCESS_COLS)
    sc = res.sensor_scores
    cmin, cmean = float(sc["composite"].min()), float(sc["composite"].mean())
    flags = []
    flat = flatline_sensors(cur, calib.varying_sensors)
    if flat:
        flags.append("flatline:" + ",".join(flat[:5]))
    if cmin < calib.flag_threshold:
        flags.append("composite_below_threshold")
    g = "FLAG" if flags else "PASS"
    keep = non_rejected_windows(view, state)
    hist = non_alarm_history(view, state)
    h = len(hist) // P.WINDOW_SIZE                                     # history size in windows
    age = (view.window_index - max(keep)) if keep else None
    thr = calib.drift_thresholds[h] if h < len(calib.drift_thresholds) else None
    if len(hist) < MIN_HISTORY or thr is None:
        return GateDecision(g, "ok", cmin, cmean, [], flags, False, None, thr, len(hist), age)
    d = drift_statistic(hist, ref)
    return GateDecision(g, "critical" if d > thr else "ok", cmin, cmean, [], flags, True, d, thr, len(hist), age)
