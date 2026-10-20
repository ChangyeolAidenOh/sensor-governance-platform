"""S2-04/05: gate v2 integrity BLOCK, calibrated FLAG, causal drift on non-alarm history."""
import numpy as np, pandas as pd, pytest
from uuid import uuid4
from core_pipeline import prereg_constants as P
from core_pipeline.governance.gate_adapter import GateReference
from core_pipeline.governance.gate_v2 import GateCalibration, gate_check, drift_statistic, non_alarm_history, MIN_HISTORY
from core_pipeline.stream.state import EpisodeState
from core_pipeline.stream.window_view import PROCESS_COLS, WindowView


def frame(seed, n=P.RUN_LENGTH, shift_after=None, shift=0.0):
    r = np.random.default_rng(seed)
    x = r.normal(size=(n, 1)) + 0.3 * r.normal(size=(n, len(PROCESS_COLS)))
    if shift_after is not None:
        x[shift_after:] += shift
    df = pd.DataFrame(x, columns=PROCESS_COLS); df.insert(0, "cycle", np.arange(1, n + 1)); return df


@pytest.fixture(scope="module")
def ref_and_calib():
    ref = GateReference.from_calibration([frame(100 + i) for i in range(3)], ["a", "b", "c"])
    calib = GateCalibration(flag_threshold=0.0, drift_thresholds=[0.5] * P.N_WINDOWS,
                            varying_sensors=PROCESS_COLS, meta={})
    return ref, calib


def test_block_only_on_integrity(ref_and_calib):
    ref, calib = ref_and_calib
    st = EpisodeState(uuid4(), P.ALPHA)
    ok = gate_check(WindowView.build(st.episode_id, 3, frame(1)), ref, calib, st)
    assert ok.gate in ("PASS", "FLAG") and not ok.block_reasons
    bad = frame(1); bad.loc[65, PROCESS_COLS[2]] = np.nan                     # NaN inside window 3 (cycles 61..80)
    d = gate_check(WindowView.build(st.episode_id, 3, bad), ref, calib, st)
    assert d.gate == "BLOCK" and "non_finite" in d.block_reasons
    flat = frame(1); flat.loc[60:79, PROCESS_COLS[0]] = 1.234                  # flat-lined sensor in window 3
    d = gate_check(WindowView.build(st.episode_id, 3, flat), ref, calib, st)
    assert d.gate == "FLAG" and not d.block_reasons and d.flag_reasons[0].startswith("flatline:" + PROCESS_COLS[0])
    short = frame(1).iloc[:70]                                                  # window 3 has only 10 of 20 samples
    with pytest.raises(ValueError):
        WindowView.build(st.episode_id, 3, short)                               # the view itself refuses short episodes
    # a huge process shift is NOT a BLOCK (faults must not silence detection)
    shifted = frame(1, shift_after=0, shift=50.0)
    d = gate_check(WindowView.build(st.episode_id, 3, shifted), ref, calib, st)
    assert d.gate in ("PASS", "FLAG")


def test_flag_threshold_is_calibrated_value(ref_and_calib):
    ref, _ = ref_and_calib
    st = EpisodeState(uuid4(), P.ALPHA)
    v = WindowView.build(st.episode_id, 2, frame(2))
    lo = GateCalibration(-1.0, [None] * P.N_WINDOWS, PROCESS_COLS)
    hi = GateCalibration(101.0, [None] * P.N_WINDOWS, PROCESS_COLS)
    assert gate_check(v, ref, lo, st).gate == "PASS" and gate_check(v, ref, hi, st).gate == "FLAG"


def test_drift_uses_only_non_alarm_past_windows(ref_and_calib):
    ref, calib = ref_and_calib
    st = EpisodeState(uuid4(), P.ALPHA)
    f = frame(3, shift_after=100, shift=30.0)                                   # shift from window 5 on
    # windows 0..4 normal (not alarmed), 5..7 alarmed (recorded by observe), window 8 under check
    for t in range(8):
        st.observe(t, "PASS", 0.9 if t < 5 else 0.0001)
    v = WindowView.build(st.episode_id, 8, f)
    hist = non_alarm_history(v, st)
    assert len(hist) == 5 * P.WINDOW_SIZE and hist["cycle"].max() == 100        # alarmed windows excluded
    d = gate_check(v, ref, calib, st)
    assert d.drift == "ok" and d.drift_assessed and d.n_history == 100         # shift lives only in alarmed windows
    # same episode, but the shifted windows were NOT alarmed → they enter the history → critical
    st2 = EpisodeState(uuid4(), P.ALPHA)
    for t in range(8):
        st2.observe(t, "PASS", 0.9)
    d2 = gate_check(WindowView.build(st2.episode_id, 8, f), ref, calib, st2)
    assert d2.drift == "critical" and d2.drift_stat > 0.5
    # not assessed with too little history
    st3 = EpisodeState(uuid4(), P.ALPHA); st3.observe(0, "PASS", 0.9)
    d3 = gate_check(WindowView.build(st3.episode_id, 1, f), ref, calib, st3)
    assert d3.drift == "ok" and not d3.drift_assessed and d3.drift_stat is None and d3.n_history < MIN_HISTORY


def test_drift_threshold_indexed_by_history_size_not_t(ref_and_calib):
    ref, _ = ref_and_calib
    thr = [None] * (P.N_WINDOWS + 1); thr[2] = 0.0; thr[10] = 1.0          # h=2 strict, h=10 lenient
    calib = GateCalibration(0.0, thr, PROCESS_COLS)
    f = frame(6, shift_after=0, shift=5.0)                                 # whole episode shifted vs reference
    st = EpisodeState(uuid4(), P.ALPHA)
    for t in range(10): st.observe(t, "PASS", 0.9 if t < 2 else 0.0001)   # 2 non-rejected windows, 8 alarmed
    d = gate_check(WindowView.build(st.episode_id, 10, f), ref, calib, st)
    assert d.n_history == 40 and d.drift_threshold == 0.0 and d.drift == "critical"   # uses thr[h=2], not thr[t=10]


def test_elapsed_time_alone_never_strengthens_drift(ref_and_calib):
    """Reviewer criterion: on the same (frozen) history, advancing t must not change D, the threshold or the verdict;
    only the reported staleness grows."""
    ref, _ = ref_and_calib
    thr = [None] + [0.3 + 0.01 * h for h in range(1, P.N_WINDOWS + 1)]    # any h-indexed table
    calib = GateCalibration(0.0, thr, PROCESS_COLS)
    f = frame(8, shift_after=0, shift=2.0)
    st = EpisodeState(uuid4(), P.ALPHA)
    for t in range(6): st.observe(t, "PASS", 0.9)                        # 6 non-rejected windows
    seen = []
    for t in range(6, P.N_WINDOWS):                                       # then alarms forever: history frozen
        d = gate_check(WindowView.build(st.episode_id, t, f), ref, calib, st)
        seen.append((d.drift, d.drift_stat, d.drift_threshold, d.n_history))
        assert d.history_age_windows == t - 5
        st.observe(t, "PASS", 0.0001)
    assert len(set(seen)) == 1                                            # identical decision at every later t


def test_drift_is_causal(ref_and_calib):
    ref, calib = ref_and_calib
    base = frame(4)
    st = EpisodeState(uuid4(), P.ALPHA)
    for t in range(10): st.observe(t, "PASS", 0.9)
    d0 = gate_check(WindowView.build(st.episode_id, 10, base), ref, calib, st)
    fut = base.copy(); fut.loc[fut["cycle"] > 220, PROCESS_COLS] += 100.0
    d1 = gate_check(WindowView.build(st.episode_id, 10, fut), ref, calib, st)
    assert (d0.gate, d0.drift, d0.drift_stat) == (d1.gate, d1.drift, d1.drift_stat)


def test_loo_p_is_uniform_on_calibration_members():
    from core_pipeline.anomaly.conformal import ConformalCalibration
    r = np.random.default_rng(11)
    sc = r.normal(size=4800)
    cal = ConformalCalibration.from_scores(sc)
    p = np.array([cal.p_value_loo(float(x)) for x in sc])
    assert np.allclose(np.sort(p), np.arange(1, 4801) / 4800)                   # uniform grid (no ties here)
    assert abs((p <= P.ALPHA).mean() - P.ALPHA) < 1e-9
    with pytest.raises(ValueError):
        cal.p_value_loo(4.5)                                                     # not a member (would have passed the old check)
    tied = ConformalCalibration.from_scores(np.array([0.0, 1.0, 1.0, 1.0, 2.0] * 4))   # ties: conservative by tie count
    assert tied.p_value_loo(1.0) == (1 + (4 + 12 - 1)) / 20                       # 4 twos + 12 ones, minus one self copy
    assert tied.p_value_loo(2.0) == (1 + 3) / 20


def test_calibration_loader_refuses_other_schema(tmp_path):
    import json
    from core_pipeline.governance.gate_v2 import GateCalibration
    good = GateCalibration(30.0, [None, 0.9] + [0.1] * 47, PROCESS_COLS, {})
    good.save(tmp_path / "c.json"); GateCalibration.load(tmp_path / "c.json")
    old = json.loads((tmp_path / "c.json").read_text()); old.pop("schema_version"); old.pop("threshold_index")
    (tmp_path / "old.json").write_text(json.dumps(old))                            # a rev-2 (per-t) style file
    with pytest.raises(ValueError):
        GateCalibration.load(tmp_path / "old.json")


def test_open_bin_psi_counts_out_of_range_mass():
    from core_pipeline.governance.gate_v2 import psi_open_bins
    from core_pipeline.governance.quality_metrics import compute_psi
    r = np.random.default_rng(9)
    ref = r.normal(size=5000)
    cur = np.concatenate([r.normal(size=500), r.normal(size=300) + 30.0])   # 37.5% of samples far outside the range
    assert compute_psi(ref, cur) < 0.1                                  # v1 silently drops the out-of-range mass
    assert psi_open_bins(ref, cur) > 0.5                                # open bins: the shift is counted
    assert compute_psi(ref, r.normal(size=500) + 30.0) == 1.0           # v1 only reacts when ALL mass is out of range
    assert psi_open_bins(ref, r.normal(size=5000)) < 0.05               # same distribution → small
