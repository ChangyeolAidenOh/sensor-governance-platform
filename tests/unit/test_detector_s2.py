"""S2-01/02: window features, conformal p-values, window IF orientation, Z-score L0 — synthetic, no parquet/DB."""
import numpy as np, pandas as pd, pytest
from core_pipeline import prereg_constants as P
from core_pipeline.features.window_features import window_features, features_frame, FEATURE_NAMES
from core_pipeline.anomaly.conformal import ConformalCalibration
from core_pipeline.anomaly.if_window import WindowIF
from core_pipeline.anomaly.zscore_l0 import ZScoreL0
from core_pipeline.stream.window_view import PROCESS_COLS

rng = np.random.default_rng(20261018)


def rows(n=20, shift=0.0, seed=None):
    r = np.random.default_rng(seed) if seed is not None else rng
    f = r.normal(size=(n, 1))
    df = pd.DataFrame(f + 0.3 * r.normal(size=(n, len(PROCESS_COLS))) + shift, columns=PROCESS_COLS)
    df.insert(0, "cycle", np.arange(1, n + 1)); return df


def test_window_features_shape_and_determinism():
    w = rows(seed=1)
    a, b = window_features(w), window_features(w)
    assert a.shape == (260,) and np.array_equal(a, b) and len(FEATURE_NAMES) == 260
    assert window_features(rows(40, seed=2)).shape == (260,)             # expanded window, same function
    with pytest.raises(ValueError):
        window_features(rows(1, seed=3))


def test_conformal_p_formula():
    cal = ConformalCalibration.from_scores(np.array([0.1, 0.2, 0.3, 0.4, 0.5] * 4))   # n=20, ties
    assert cal.p_value(0.6) == 1 / 21                                      # nothing >= 0.6
    assert cal.p_value(0.5) == (1 + 4) / 21                                # ties count as >=
    assert cal.p_value(0.0) == 1.0
    with pytest.raises(ValueError):
        cal.p_value(float("nan"))
    with pytest.raises(ValueError):
        cal.p_values(np.array([0.1, np.inf]))


def test_conformal_fixed_calibration_regression():
    """Numerical regression on one fixed-seed calibration set. This estimates P(p<=t | C), NOT the marginal
    guarantee; kept only to catch implementation regressions."""
    r = np.random.default_rng(1)
    calib = ConformalCalibration.from_scores(r.normal(size=4800))
    p = calib.p_values(r.normal(size=20000))
    for t in (0.01, 0.05, 0.1, 0.25):
        assert abs((p <= t).mean() - t) < 0.02


def test_conformal_marginal_superuniformity():
    """Marginal P(p<=t) <= t: calibration AND test score are redrawn every repetition (exchangeable, iid normal)."""
    r = np.random.default_rng(2)
    R, n_cal = 4000, 50
    ps = np.empty(R)
    for i in range(R):
        cal = ConformalCalibration.from_scores(r.normal(size=n_cal))
        ps[i] = cal.p_value(float(r.normal()))
    for t in (0.05, 0.1, 0.25, 0.5):
        emp = (ps <= t).mean()
        se = np.sqrt(t * (1 - t) / R)
        assert emp <= t + 3 * se, (t, emp)
    # exact finite-sample value for exchangeable continuous scores: P(p <= k/(n+1)) = k/(n+1)
    assert abs((ps <= 5 / (n_cal + 1)).mean() - 5 / (n_cal + 1)) < 3 * np.sqrt(0.1 * 0.9 / R)


def test_window_if_orientation_and_persistence(tmp_path):
    Xtr = features_frame([rows(seed=100 + i) for i in range(300)])
    m = WindowIF.fit(Xtr, n_estimators=50)
    normal = m.score(features_frame([rows(seed=500 + i) for i in range(100)]))
    shifted = m.score(features_frame([rows(shift=3.0, seed=700 + i) for i in range(100)]))
    assert np.median(shifted) > np.median(normal)                           # higher = more anomalous
    m.save(tmp_path); m2 = WindowIF.load(tmp_path)
    assert np.allclose(m2.score(Xtr.iloc[:5]), m.score(Xtr.iloc[:5]))


def test_detector_end_to_end_alarm_rate():
    Xtr = features_frame([rows(seed=1000 + i) for i in range(400)])
    m = WindowIF.fit(Xtr, n_estimators=50)
    cal = ConformalCalibration.from_scores(m.score(features_frame([rows(seed=2000 + i) for i in range(500)])))
    p_norm = cal.p_values(m.score(features_frame([rows(seed=3000 + i) for i in range(300)])))
    p_shift = cal.p_values(m.score(features_frame([rows(shift=3.0, seed=4000 + i) for i in range(100)])))
    assert (p_norm <= P.ALPHA).mean() <= P.ALPHA + 0.05
    assert (p_shift <= P.ALPHA).mean() > 0.9


def test_zscore_l0(tmp_path):
    n = len(PROCESS_COLS)
    # known calibration: mean 0, std 1 for all sensors except the last, which is constant (std 0)
    r = np.random.default_rng(5)
    cal_rows = pd.DataFrame(r.normal(size=(100000, n)), columns=PROCESS_COLS)
    cal_rows[PROCESS_COLS[-1]] = 7.0
    z = ZScoreL0.fit(cal_rows)
    assert z.meta["excluded_constant_sensors"] == [PROCESS_COLS[-1]]
    mu, sd = z.mean[0], z.std[0]
    base = pd.DataFrame(np.zeros((20, n)) + z.mean, columns=PROCESS_COLS); base.insert(0, "cycle", range(1, 21))
    exactly3 = base.copy(); exactly3.loc[0, PROCESS_COLS[0]] = mu + 2.999 * sd   # just below the boundary (float-safe)
    assert not z.alarm(exactly3)
    over3 = base.copy(); over3.loc[0, PROCESS_COLS[0]] = mu + 3.001 * sd
    assert z.alarm(over3)
    const_dev = base.copy(); const_dev.loc[0, PROCESS_COLS[-1]] = 1000.0   # constant sensor deviates
    assert not z.alarm(const_dev)                                           # excluded from the test by design
    nan_rows = base.copy(); nan_rows.loc[0, PROCESS_COLS[1]] = np.nan
    with pytest.raises(ValueError):
        z.alarm(nan_rows)
    z.save(tmp_path); z2 = ZScoreL0.load(tmp_path)
    assert np.allclose(z2.mean, z.mean) and z2.meta["z_threshold"] == 3.0 and z2.alarm(over3)
