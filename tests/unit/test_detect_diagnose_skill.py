"""S2-06: detect_diagnose contract — BLOCK refusal, expand gating, causality, calibration-window routing."""
import numpy as np, pandas as pd, pytest
from uuid import uuid4
from core_pipeline import prereg_constants as P
from core_pipeline.anomaly.conformal import ConformalCalibration
from core_pipeline.anomaly.if_window import WindowIF
from core_pipeline.diagnosis.classifier import FaultDiagnoser, FAULTS
from core_pipeline.features.window_features import features_frame, episode_window_features
from core_pipeline.governance.gate_v2 import GateDecision
from core_pipeline.skills.detect_diagnose import DetectDiagnoseSkill
from core_pipeline.stream.state import EpisodeState, WindowState
from core_pipeline.stream.window_view import PROCESS_COLS, WindowView

PASS = GateDecision("PASS", "ok", 80.0, 90.0, [], [], False, None, None, 0)
BLOCK = GateDecision("BLOCK", "ok", float("nan"), float("nan"), ["non_finite"], [], False, None, None, 0)


def rows(n=20, seed=0, shift=0.0):
    r = np.random.default_rng(seed)
    df = pd.DataFrame(r.normal(size=(n, 1)) + 0.3 * r.normal(size=(n, len(PROCESS_COLS))) + shift, columns=PROCESS_COLS)
    df.insert(0, "cycle", np.arange(1, n + 1)); return df


@pytest.fixture(scope="module")
def skill():
    r = np.random.default_rng(7)
    det = WindowIF.fit(features_frame([rows(seed=i) for i in range(200)]), n_estimators=30)
    conf = ConformalCalibration.from_scores(det.score(features_frame([rows(seed=1000 + i) for i in range(300)])))
    centers = r.normal(size=(20, 260))
    X = np.vstack([centers[f - 1] + r.normal(size=(60, 260)) for f in FAULTS]); y = np.repeat(FAULTS, 60)
    diag = FaultDiagnoser.fit(X, y, num_boost_round=30, meta={"train_data": "synthetic", "calibration_data": "synthetic"})
    Xc = np.vstack([centers[f - 1] + r.normal(size=(40, 260)) for f in FAULTS]); yc = np.repeat(FAULTS, 40)
    diag.calibrate("w20", Xc, yc); diag.calibrate("w40", Xc * 1.1, yc)
    return DetectDiagnoseSkill(det, conf, diag)


def test_block_refuses_before_any_detection(skill):
    v = WindowView.build(uuid4(), 3, rows(960, seed=1))
    r = skill(v, BLOCK, WindowState(3))
    assert r.status == "REFUSED" and r.reason == "GATE_BLOCK" and r.p_value is None and r.pred_set == []


def test_expand_is_diagnosis_only_and_gated(skill):
    ep = rows(960, seed=1)
    v = WindowView.build(uuid4(), 3, ep)
    assert skill(v, PASS, WindowState(3), expand=2).reason == "EXPAND_NOT_ALLOWED"
    ok = skill(v, PASS, WindowState(3, q_used=True), expand=2)
    assert ok.status == "OK" and ok.calibration_window == 40
    assert not ok.detection_run and ok.score is None and ok.p_value is None   # no detection on the 40-sample input
    assert ok.set_size == len(ok.pred_set) and ok.top1 in range(1, 21)
    v0 = WindowView.build(uuid4(), 0, ep)
    assert skill(v0, PASS, WindowState(0, q_used=True), expand=2).reason == "NO_EXPANSION"
    assert skill(v, PASS, WindowState(3), expand=3).reason == "BAD_EXPAND"
    # previous window was BLOCK → the expanded span is invalid
    st = EpisodeState(v.episode_id, P.ALPHA)
    for t, g in enumerate(["PASS", "PASS", "BLOCK"]):
        st.observe(t, g, None if g == "BLOCK" else 0.5)
    assert skill(v, PASS, WindowState(3, q_used=True), expand=2, episode_state=st).reason == "EXPANSION_INVALID"
    # NaN inside the previous window's samples → invalid even if that window had been PASS somehow
    bad = ep.copy(); bad.loc[45, PROCESS_COLS[0]] = np.nan                      # cycle 46 ∈ window 2
    assert skill(WindowView.build(uuid4(), 3, bad), PASS, WindowState(3, q_used=True), expand=2).reason == "EXPANSION_INVALID"


def test_ok_result_fields_and_causality(skill):
    ep = rows(960, seed=2)
    v = WindowView.build(uuid4(), 5, ep)
    r = skill(v, PASS, WindowState(5))
    assert r.status == "OK" and r.detection_run and 0 < r.p_value <= 1 and r.calibration_window == 20 and r.n_cal == 300
    assert r.set_size == len(r.pred_set) and r.valid == (1 <= r.set_size <= 3)
    assert r.top1 in range(1, 21) and len(r.contributions) == 5 and "explained_class" in r.contributions[0]
    fut = ep.copy(); fut.loc[fut["cycle"] > 120, PROCESS_COLS] += 100.0        # future change → identical result
    r2 = skill(WindowView.build(uuid4(), 5, fut), PASS, WindowState(5))
    assert (r.score, r.p_value, r.pred_set, r.top1) == (r2.score, r2.p_value, r2.pred_set, r2.top1)


def test_skill_feeds_prefix_bh_through_episode_state(skill):
    ep = rows(960, seed=3); st = EpisodeState(uuid4(), P.ALPHA)
    for t in range(6):
        v = WindowView.build(st.episode_id, t, ep)
        g = BLOCK if t == 2 else PASS
        r = skill(v, g, WindowState(t))
        dec = st.observe(t, g.gate, None if r.status == "REFUSED" else r.p_value)
        assert (dec is None) == (t == 2)
    assert st.m_t == 5 and st.undecidable_windows() == [2]
