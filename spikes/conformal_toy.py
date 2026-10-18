"""0-E spike: conformal p-values with BH for FDR, and conformal prediction sets.

Synthetic data only (not TEP): the question is whether the machinery behaves as
the theory says on this machine, before it is wired to real scores in Stage 2.

Part A - detection (H2 mechanics)
    calibration: n_cal normal scores ~ lognormal(0, 1)
    batch: 180 normal windows + 20 anomalous windows (anomalous = normal + shift in log space)
    conformal p-value  p_i = (1 + #{s_cal >= s_i}) / (n_cal + 1)
    BH at alpha over the batch -> discoveries; realized FDR = false discoveries / max(1, discoveries)
    repeated R times with fresh calibration each time; FDR averaged.
    Exchangeability break: normal test windows drawn with log-scale multiplied by `shift`
    (shift = 1.0 is the exchangeable case). Expect mean FDR <= alpha at 1.0 and growth beyond.

Part B - diagnosis (conformal prediction sets)
    K classes; a synthetic classifier gives softmax probabilities with true-class signal
    `signal` + Gaussian noise. Split conformal: nonconformity 1 - p_hat[y]; q = ceil((n+1)(1-alpha))/n
    quantile on calibration; set(x) = {k : 1 - p_hat[k] <= q}. Report marginal coverage and set size.
    Mondrian: one quantile per class (class-conditional). Hard classes (low signal) show why.
    Label shift: test drawn only from the hard classes -> marginal coverage of plain split drops.

Usage:  python spikes/conformal_toy.py            (report -> outputs/stage0_conformal_toy.json)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPORT = Path("outputs/stage0_conformal_toy.json")
SEED = 20261018


# ----------------------------------------------------------------------------- Part A

def conformal_pvalues(scores_cal: np.ndarray, scores_test: np.ndarray) -> np.ndarray:
    """p_i = (1 + #{cal >= test_i}) / (n_cal + 1); larger score = more anomalous."""
    cal_sorted = np.sort(scores_cal)
    n = len(cal_sorted)
    # number of calibration scores >= s  ==  n - (number strictly less than s)
    ge = n - np.searchsorted(cal_sorted, scores_test, side="left")
    return (1.0 + ge) / (n + 1.0)


def bh_reject(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg: reject the m largest-ranked p <= k*alpha/m."""
    m = len(pvals)
    order = np.argsort(pvals)
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = pvals[order] <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(m, dtype=bool)
    rej[order[:k]] = True
    return rej


def detection_experiment(rng, n_cal=500, n_norm=180, n_anom=20, alpha=0.10, shift=1.0,
                         anom_shift=2.0, reps=500) -> dict:
    fdrs, powers, n_disc = [], [], []
    for _ in range(reps):
        cal = np.exp(rng.normal(0.0, 1.0, n_cal))
        normal = np.exp(rng.normal(0.0, 1.0, n_norm) * shift)         # shift=1 -> exchangeable
        anom = np.exp(rng.normal(0.0, 1.0, n_anom) + anom_shift)
        test = np.concatenate([normal, anom])
        truth = np.concatenate([np.zeros(n_norm, bool), np.ones(n_anom, bool)])
        p = conformal_pvalues(cal, test)
        rej = bh_reject(p, alpha)
        disc = rej.sum()
        fd = (rej & ~truth).sum()
        fdrs.append(fd / max(1, disc))
        powers.append((rej & truth).sum() / n_anom)
        n_disc.append(int(disc))
    fdrs = np.array(fdrs)
    return {"shift": shift, "alpha": alpha, "reps": reps, "fdr_mean": round(float(fdrs.mean()), 4),
            "fdr_se": round(float(fdrs.std(ddof=1) / math.sqrt(reps)), 4),
            "batches_fdr_gt_alpha": round(float((fdrs > alpha).mean()), 3),
            "power_mean": round(float(np.mean(powers)), 4), "discoveries_mean": round(float(np.mean(n_disc)), 2)}


def pvalue_uniformity(rng, n_cal=500, n_test=20000, shift=1.0) -> dict:
    """Under exchangeability conformal p-values of normals are super-uniform: P(p <= t) <= t."""
    cal = np.exp(rng.normal(0.0, 1.0, n_cal))
    normal = np.exp(rng.normal(0.0, 1.0, n_test) * shift)
    p = conformal_pvalues(cal, normal)
    grid = [0.01, 0.05, 0.10, 0.25, 0.50]
    return {"shift": shift, "P(p<=t)": {str(t): round(float((p <= t).mean()), 4) for t in grid}}


# ----------------------------------------------------------------------------- Part B

def synth_classifier(rng, n, K, signal_per_class):
    y = rng.integers(0, K, n)
    logits = rng.normal(0.0, 1.0, (n, K))
    logits[np.arange(n), y] += signal_per_class[y]
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs, y


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


def prediction_sets(probs: np.ndarray, q: float | np.ndarray) -> np.ndarray:
    """q scalar (split) or per-class vector (Mondrian). Returns boolean (n, K)."""
    return (1.0 - probs) <= q


def diagnosis_experiment(rng, K=20, n_cal=2000, n_test=5000, alpha=0.10, hard_classes=None,
                         test_only_hard=False) -> dict:
    signal = np.full(K, 3.0)
    hard = list(hard_classes or [])
    signal[hard] = 0.8                                    # weakly separable classes
    p_cal, y_cal = synth_classifier(rng, n_cal, K, signal)
    p_te, y_te = synth_classifier(rng, n_test, K, signal)
    if test_only_hard and hard:
        keep = np.isin(y_te, hard)
        p_te, y_te = p_te[keep], y_te[keep]
    nc_cal = 1.0 - p_cal[np.arange(n_cal), y_cal]

    # split conformal (one quantile)
    q = conformal_quantile(nc_cal, alpha)
    S = prediction_sets(p_te, q)
    cov = S[np.arange(len(y_te)), y_te].mean()
    size = S.sum(axis=1).mean()
    per_class_cov = {int(k): round(float(S[y_te == k, k].mean()), 3) for k in range(K) if (y_te == k).any()}

    # Mondrian (per-class quantile)
    qk = np.array([conformal_quantile(nc_cal[y_cal == k], alpha) if (y_cal == k).sum() > 10 else q for k in range(K)])
    Sm = prediction_sets(p_te, qk[None, :])
    cov_m = Sm[np.arange(len(y_te)), y_te].mean()
    size_m = Sm.sum(axis=1).mean()
    per_class_cov_m = {int(k): round(float(Sm[y_te == k, k].mean()), 3) for k in range(K) if (y_te == k).any()}

    return {"K": K, "alpha": alpha, "hard_classes": hard, "test_only_hard": test_only_hard,
            "split": {"coverage": round(float(cov), 4), "set_size_mean": round(float(size), 3),
                      "min_class_coverage": min(per_class_cov.values()), "q": round(q, 4)},
            "mondrian": {"coverage": round(float(cov_m), 4), "set_size_mean": round(float(size_m), 3),
                         "min_class_coverage": min(per_class_cov_m.values())},
            "top1_accuracy": round(float((p_te.argmax(axis=1) == y_te).mean()), 4)}


# ----------------------------------------------------------------------------- main

def main() -> None:
    rng = np.random.default_rng(SEED)
    alpha = 0.10

    print("== Part A: conformal p-values + BH, realized FDR by exchangeability break (shift = log-scale factor)")
    curve = []
    for shift in (1.0, 1.1, 1.25, 1.5, 2.0):
        r = detection_experiment(rng, alpha=alpha, shift=shift)
        curve.append(r)
        print(f"  shift {shift:>4}: FDR {r['fdr_mean']:.3f} ± {r['fdr_se']:.3f}  batches>alpha {r['batches_fdr_gt_alpha']:.2f}  "
              f"power {r['power_mean']:.3f}  discoveries {r['discoveries_mean']:.1f}")
    unif = [pvalue_uniformity(rng, shift=s) for s in (1.0, 1.5)]
    for u in unif:
        print(f"  p-value super-uniformity at shift {u['shift']}: {u['P(p<=t)']}")

    print("\n== Part B: conformal prediction sets for diagnosis (K=20)")
    hard = [3, 9, 15]                                     # by analogy with the TEP 'hard' faults
    b_plain = diagnosis_experiment(rng, alpha=alpha, hard_classes=hard)
    b_shift = diagnosis_experiment(rng, alpha=alpha, hard_classes=hard, test_only_hard=True)
    for name, b in (("iid test", b_plain), ("label shift: hard classes only", b_shift)):
        print(f"  {name}: split cov {b['split']['coverage']:.3f} size {b['split']['set_size_mean']:.2f} "
              f"min-class {b['split']['min_class_coverage']:.2f} | mondrian cov {b['mondrian']['coverage']:.3f} "
              f"size {b['mondrian']['set_size_mean']:.2f} min-class {b['mondrian']['min_class_coverage']:.2f} | top1 {b['top1_accuracy']:.3f}")

    base = curve[0]
    checks = {
        "A_fdr_controlled_when_exchangeable": base["fdr_mean"] <= alpha + 2 * base["fdr_se"],
        "A_pvalues_superuniform_when_exchangeable": all(v <= t * 1.15 + 0.005 for t, v in ((float(k), v) for k, v in unif[0]["P(p<=t)"].items())),
        "A_fdr_collapses_under_shift": curve[-1]["fdr_mean"] > alpha and curve[-1]["fdr_mean"] > base["fdr_mean"] + 3 * base["fdr_se"],
        "A_fdr_monotone_in_shift": all(curve[i]["fdr_mean"] <= curve[i + 1]["fdr_mean"] + 0.01 for i in range(len(curve) - 1)),
        "B_split_marginal_coverage_ok": b_plain["split"]["coverage"] >= 1 - alpha - 0.02,
        "B_split_min_class_coverage_below_target": b_plain["split"]["min_class_coverage"] < 1 - alpha - 0.02,
        # ~100 calibration points per class leave visible finite-sample slack on class-conditional
        # coverage; the claim tested is the improvement over the single-quantile set, not a fixed floor
        "B_mondrian_lifts_min_class_coverage": b_plain["mondrian"]["min_class_coverage"]
                                                >= b_plain["split"]["min_class_coverage"] + 0.15,
        "B_split_coverage_drops_under_label_shift": b_shift["split"]["coverage"] < 1 - alpha - 0.02,
        "B_mondrian_robust_to_label_shift": b_shift["mondrian"]["coverage"] >= 1 - alpha - 0.05,
    }
    report = {"seed": SEED, "alpha": alpha, "part_a": {"curve": curve, "uniformity": unif},
              "part_b": {"iid": b_plain, "label_shift": b_shift}, "checks": checks,
              "go": checks["A_fdr_controlled_when_exchangeable"] and checks["B_split_marginal_coverage_ok"]}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\n[checks]")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"[go] {report['go']}  (Go criterion: toy FDR <= alpha when exchangeable, and marginal coverage of the diagnosis set)")
    print(f"[report] {REPORT}")
    sys.exit(0 if report["go"] else 1)


if __name__ == "__main__":
    main()
