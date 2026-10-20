"""
core_pipeline/skills/detect_diagnose.py — S2-06: the detect_diagnose function contract (skill contract v0.1 §2.2).
MCP wrapping is Stage 4; this is the pure function the server will call. Skills return numbers; decide_action decides.

Enforced here (H0 tests):
  * REFUSED(GATE_BLOCK)        — gate_decision.gate == 'BLOCK': no detection is run at all
  * expand=1 (primary, 20 samples): detection score + conformal p (20-sample calibration) + diagnosis (w20 set)
  * expand=2 (backward 40 samples, only from the graph's query_more branch = win_state.q_used):
      DIAGNOSIS ONLY — w40 set + contributions. The detector's conformal calibration is for 20-sample windows, so no
      detection score/p is produced for the expanded input; the primary window's p and prefix-BH decision stand and
      the caller must NOT feed the expanded result into EpisodeState.observe() (prereg §4.5: expansion re-runs
      diagnosis, not detection).
      REFUSED(EXPAND_NOT_ALLOWED) without q_used; REFUSED(NO_EXPANSION) at window 0;
      REFUSED(EXPANSION_INVALID) if the previous window was BLOCK or the expanded rows violate the input contract.
  * causal — all inputs come from the WindowView (cycles ≤ end_cycle) and the episode's recorded decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from core_pipeline import prereg_constants as P
from core_pipeline.anomaly.conformal import ConformalCalibration
from core_pipeline.anomaly.if_window import WindowIF
from core_pipeline.diagnosis.classifier import FaultDiagnoser
from core_pipeline.features.window_features import window_features, FEATURE_NAMES
from core_pipeline.governance.gate_v2 import GateDecision, integrity_block_reasons
from core_pipeline.stream.state import EpisodeState, WindowState
from core_pipeline.stream.window_view import WindowView


@dataclass(frozen=True)
class DetectResult:
    status: str                              # OK | REFUSED
    reason: Optional[str] = None
    detection_run: bool = False              # False for expand=2 (diagnosis only)
    score: Optional[float] = None
    p_value: Optional[float] = None
    pred_set: list[int] = field(default_factory=list)
    set_size: int = 0
    valid: bool = False
    top1: Optional[int] = None
    top1_prob: Optional[float] = None
    calibration_window: Optional[int] = None  # 20 | 40
    contributions: list[dict] = field(default_factory=list)
    n_cal: Optional[int] = None


@dataclass
class DetectDiagnoseSkill:
    detector: WindowIF
    conformal: ConformalCalibration
    diagnoser: FaultDiagnoser

    def __call__(self, view: WindowView, gate: GateDecision, win_state: WindowState, expand: int = 1,
                 episode_state: Optional[EpisodeState] = None) -> DetectResult:
        if gate.gate == "BLOCK":
            return DetectResult("REFUSED", "GATE_BLOCK")
        if expand not in (1, 2):
            return DetectResult("REFUSED", "BAD_EXPAND")
        if expand == 1:
            feats = window_features(view.current())
            X = pd.DataFrame(feats[None, :], columns=FEATURE_NAMES)
            score = float(self.detector.score(X)[0])
            p = float(self.conformal.p_value(score))
            return self._diagnose(feats, "w20", 20, detection=(score, p))
        # ---- expand == 2: diagnosis only ----
        if not win_state.q_used:
            return DetectResult("REFUSED", "EXPAND_NOT_ALLOWED")
        rows = view.expanded()
        if rows is None:
            return DetectResult("REFUSED", "NO_EXPANSION")
        if episode_state is not None and (view.window_index - 1) in episode_state.undecidable_windows():
            return DetectResult("REFUSED", "EXPANSION_INVALID")            # previous window was BLOCK
        if integrity_block_reasons(rows, 2 * P.WINDOW_SIZE):
            return DetectResult("REFUSED", "EXPANSION_INVALID")
        return self._diagnose(window_features(rows), "w40", 40, detection=None)

    def _diagnose(self, feats: np.ndarray, key: str, cal_win: int, detection) -> DetectResult:
        sets, pr = self.diagnoser.predict_sets(feats[None, :], key)
        s = [int(f) for f in sets[0]]
        top1 = int(pr[0].argmax() + 1)
        contrib = self.diagnoser.contributions(feats[None, :])[0]
        score, p = detection if detection else (None, None)
        return DetectResult("OK", None, detection is not None, score, p, s, len(s), 1 <= len(s) <= P.SET_SIZE_MAX,
                            top1, float(pr[0].max()), cal_win, contrib, self.conformal.n_cal if detection else None)
