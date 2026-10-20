"""S1-04: the gate reference is external, deterministic, and evaluate_window is causal (uses current window only)."""
import numpy as np, pandas as pd, pytest
from uuid import uuid4
from core_pipeline import prereg_constants as P
from core_pipeline.governance.gate_adapter import GateReference, evaluate_window, N_REF_ROWS
from core_pipeline.governance.quality_gate import QualityGate
from core_pipeline.stream.window_view import PROCESS_COLS, WindowView


def frame(seed, n=P.RUN_LENGTH):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 1))
    df = pd.DataFrame(base + 0.3 * rng.normal(size=(n, len(PROCESS_COLS))), columns=PROCESS_COLS)
    df.insert(0, "cycle", np.arange(1, n + 1))
    return df


def test_reference_deterministic_and_sized():
    cal = [frame(100 + i) for i in range(5)]
    r1 = GateReference.from_calibration(cal, ["a", "b", "c", "d", "e"])
    r2 = GateReference.from_calibration(cal, ["a", "b", "c", "d", "e"])
    assert r1.frame.equals(r2.frame) and r1.meta == r2.meta and len(r1.frame) == N_REF_ROWS
    assert list(r1.frame.columns) == PROCESS_COLS


def test_evaluate_window_is_causal_and_finite(tmp_path):
    ref = GateReference.from_calibration([frame(1), frame(2)], ["a", "b"])
    ref.save(tmp_path / "ref.parquet"); ref = GateReference.load(tmp_path / "ref.parquet")
    gate = QualityGate()
    ep = frame(7)
    res0 = evaluate_window(gate, WindowView.build(uuid4(), 10, ep), ref)
    ep2 = ep.copy(); ep2.loc[ep2["cycle"] > 220, PROCESS_COLS] += 50.0     # future change
    res1 = evaluate_window(QualityGate(), WindowView.build(uuid4(), 10, ep2), ref)
    assert res0.status == res1.status and res0.overall_score == res1.overall_score
    sc = res0.sensor_scores
    assert np.isfinite(sc[["composite", "psi", "corr_deviation"]].to_numpy()).all()
