"""
core_pipeline/diagnosis/classifier.py — S2-03: fault-type classifier (LightGBM, 20 classes = faults 1..20) on
window-statistic features, plus split-conformal prediction sets (prereg §4.2).

  calibration scores  s_i = 1 - p_hat(y_i | x_i)   (true class of each calibration window)
  k = ceil((n+1)(1-alpha)),  q_hat = s_(k)  — the k-th smallest CALIBRATION SCORE (a threshold on the score scale;
  k/n is only the quantile level, never compared with scores directly),  alpha = 1 - COVERAGE_TARGET
  set(x) = { y : 1 - p_hat(y|x) <= q_hat }     (may be empty → set_size 0 → "indeterminate", prereg §4.2)

Two calibrations are kept: 'w20' (20-sample windows) and 'w40' (backward-expanded 40-sample windows); detect_diagnose
uses the one matching its input length (prereg §4.2). Mondrian (class-conditional q) is exploratory reporting only.
Contributions (narration input, never a policy input) — contract:
  * LightGBM built-in TreeSHAP (`pred_contrib=True`, raw-score space), per class; the trailing expected-value column
    is dropped before any aggregation.
  * explained class = the predicted top-1 class by default, or an explicit `for_class` (e.g. each conformal candidate).
    NEVER the true label.
  * aggregation per sensor over its 5 statistics: `shap_sum` = signed sum (direction), `shap_abs` = sum of absolute
    values. Ranking uses shap_abs (a signed sum can cancel across statistics of one sensor); the sign of shap_sum is
    kept as the direction.
  * wording: "sensors that contributed to the model score for fault X" — not a causal "sensor that caused the fault";
    with several candidates, a top-1 explanation is not a confirmed diagnosis.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from core_pipeline import prereg_constants as P
from core_pipeline.features.window_features import FEATURE_NAMES
from core_pipeline.stream.window_view import PROCESS_COLS

FAULTS = np.arange(1, P.N_FAULTS + 1)
DIAG_VERSION = "diagnoser_v1"
LGB_PARAMS = dict(objective="multiclass", num_class=P.N_FAULTS, learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=50, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, seed=P.SEED, num_threads=0)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


@dataclass
class ConformalSet:
    q: float
    n_cal: int
    q_by_class: dict | None = None          # Mondrian (exploratory)

    def sets(self, proba: np.ndarray) -> list[np.ndarray]:
        return [FAULTS[(1 - row) <= self.q] for row in proba]

    def sets_mondrian(self, proba: np.ndarray) -> list[np.ndarray]:
        qs = np.array([self.q_by_class[int(f)] for f in FAULTS])
        return [FAULTS[(1 - row) <= qs] for row in proba]


@dataclass
class FaultDiagnoser:
    booster: lgb.Booster
    calib: dict[str, ConformalSet]           # 'w20', 'w40'
    meta: dict

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, num_boost_round: int = 300, meta: dict | None = None) -> "FaultDiagnoser":
        assert X.shape[1] == len(FEATURE_NAMES) and set(np.unique(y)) <= set(FAULTS.tolist())
        ds = lgb.Dataset(X, label=y - 1, feature_name=FEATURE_NAMES, free_raw_data=False)
        booster = lgb.train(LGB_PARAMS, ds, num_boost_round=num_boost_round)
        m = {**(meta or {}), "version": DIAG_VERSION, "n_train": int(len(y)), "num_boost_round": num_boost_round,
             "params": {k: v for k, v in LGB_PARAMS.items() if k != "num_threads"}, "coverage_target": P.COVERAGE_TARGET}
        return cls(booster, {}, m)

    def proba(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(X)

    def calibrate(self, key: str, X_cal: np.ndarray, y_cal: np.ndarray) -> ConformalSet:
        assert key in ("w20", "w40")
        pr = self.proba(X_cal)
        s = 1 - pr[np.arange(len(y_cal)), y_cal - 1]
        alpha = 1 - P.COVERAGE_TARGET
        q_by = {int(f): conformal_quantile(s[y_cal == f], alpha) for f in FAULTS if (y_cal == f).sum() >= 20}
        cs = ConformalSet(conformal_quantile(s, alpha), int(len(s)), q_by if len(q_by) == P.N_FAULTS else None)
        self.calib[key] = cs
        self.meta.setdefault("calibration", {})[key] = {"n_cal": cs.n_cal, "q": cs.q}
        return cs

    def predict_sets(self, X: np.ndarray, key: str) -> tuple[list[np.ndarray], np.ndarray]:
        pr = self.proba(X)
        return self.calib[key].sets(pr), pr

    def contributions(self, X: np.ndarray, top: int = 5, for_class: np.ndarray | None = None) -> list[list[dict]]:
        """Top-`top` sensors by sum of |TreeSHAP| over the sensor's 5 statistics, for the explained class per row
        (`for_class`: array of fault ids 1..20; default = predicted top-1). Each item: sensor, shap_abs, shap_sum,
        direction (+1/-1), explained_class."""
        contrib = self.booster.predict(X, pred_contrib=True)            # (m, num_class*(n_feat+1))
        nf = len(FEATURE_NAMES)
        contrib = contrib.reshape(len(X), P.N_FAULTS, nf + 1)[:, :, :nf]   # drop the expected-value column
        cls = (self.proba(X).argmax(axis=1) + 1) if for_class is None else np.asarray(for_class)
        out = []
        for i in range(len(X)):
            c = contrib[i, int(cls[i]) - 1].reshape(5, len(PROCESS_COLS))   # (stat, sensor) for the explained class
            s_abs, s_sum = np.abs(c).sum(axis=0), c.sum(axis=0)
            top_idx = np.argsort(s_abs)[::-1][:top]
            out.append([{"sensor": PROCESS_COLS[j], "shap_abs": float(s_abs[j]), "shap_sum": float(s_sum[j]),
                         "direction": int(np.sign(s_sum[j])) or 0, "explained_class": int(cls[i])} for j in top_idx])
        return out

    def save(self, dir_: Path) -> None:
        dir_ = Path(dir_); dir_.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(dir_ / "lgb_model.txt"))
        joblib.dump({k: v.__dict__ for k, v in self.calib.items()}, dir_ / "diag_conformal.joblib")
        (dir_ / "diag_meta.json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, dir_: Path) -> "FaultDiagnoser":
        dir_ = Path(dir_)
        booster = lgb.Booster(model_file=str(dir_ / "lgb_model.txt"))
        calib = {k: ConformalSet(**v) for k, v in joblib.load(dir_ / "diag_conformal.joblib").items()}
        return cls(booster, calib, json.loads((dir_ / "diag_meta.json").read_text()))
