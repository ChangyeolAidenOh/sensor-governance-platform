"""S2-03: vectorised features == per-window features; conformal set construction and marginal coverage; w20/w40 separate."""
import numpy as np, pandas as pd, pytest
from core_pipeline import prereg_constants as P
from core_pipeline.features.window_features import window_features, episode_window_features, expanded_window_features, FEATURE_NAMES
from core_pipeline.diagnosis.classifier import FaultDiagnoser, conformal_quantile, FAULTS
from core_pipeline.stream.window_view import PROCESS_COLS


def test_vectorised_features_match_per_window():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(960, 52))
    df = pd.DataFrame(x, columns=PROCESS_COLS); df.insert(0, "cycle", range(1, 961))
    vec = episode_window_features(x, 20, 20)
    assert vec.shape == (48, 260)
    for k in (0, 8, 47):
        assert np.allclose(vec[k], window_features(df.iloc[k * 20:(k + 1) * 20]))
    exp = expanded_window_features(x, 20, 20, np.array([9, 20]))
    assert np.allclose(exp[0], window_features(df.iloc[8 * 20:10 * 20]))
    assert np.allclose(exp[1], window_features(df.iloc[19 * 20:21 * 20]))


def test_conformal_quantile_rule():
    s = np.arange(1, 101) / 100          # n=100
    assert conformal_quantile(s, 0.10) == 0.91     # ceil(101*0.9)=91 → 91st smallest
    assert conformal_quantile(s, 0.0) == 1.0


def synth(n_per_class, seed, sep=2.0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(20, 260)) * sep
    X = np.vstack([centers[f - 1] + rng.normal(size=(n_per_class, 260)) for f in FAULTS])
    y = np.repeat(FAULTS, n_per_class)
    return X, y


def test_sets_coverage_and_separate_calibrations(tmp_path):
    Xtr, ytr = synth(150, 1, sep=0.8)
    d = FaultDiagnoser.fit(Xtr, ytr, num_boost_round=60, meta={"train_data": "synthetic", "calibration_data": "synthetic"})  # descriptive meta must not collide with calibrate()
    Xc, yc = synth(100, 2, sep=0.8)
    d.calibrate("w20", Xc, yc)
    Xc40, yc40 = synth(100, 3, sep=1.2)                 # a different distribution → different q
    d.calibrate("w40", Xc40, yc40)
    assert d.calib["w20"].q != d.calib["w40"].q
    Xt, yt = synth(100, 4, sep=0.8)
    sets, pr = d.predict_sets(Xt, "w20")
    cov = np.mean([yt[i] in sets[i] for i in range(len(yt))])
    assert 0.85 <= cov <= 0.95                           # marginal coverage near 0.90 on exchangeable data
    sizes = np.array([len(s) for s in sets])
    assert sizes.max() <= 20 and (sizes == 0).mean() < 0.2
    contrib = d.contributions(Xt[:3])
    assert len(contrib) == 3 and len(contrib[0]) == 5 and contrib[0][0]["sensor"] in PROCESS_COLS
    assert {"shap_abs", "shap_sum", "direction", "explained_class"} <= set(contrib[0][0])
    assert contrib[0][0]["explained_class"] == int(pr[0].argmax() + 1)          # default = predicted top-1
    assert contrib[0][0]["shap_abs"] >= contrib[0][1]["shap_abs"]               # ranked by |.|
    alt = d.contributions(Xt[:1], for_class=np.array([7]))
    assert alt[0][0]["explained_class"] == 7
    d.save(tmp_path); d2 = FaultDiagnoser.load(tmp_path)
    s2, _ = d2.predict_sets(Xt[:10], "w20")
    assert all(np.array_equal(a, b) for a, b in zip(sets[:10], s2))
    assert d2.calib["w40"].q == d.calib["w40"].q
