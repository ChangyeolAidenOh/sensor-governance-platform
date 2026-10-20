"""S2-07: fixed normalisation, score shape/orientation sign, persistence. Tiny model, few epochs (seconds on CPU)."""
import numpy as np, pytest
pytest.importorskip("torch")
from core_pipeline import prereg_constants as P
from core_pipeline.anomaly.at_window import WindowAT, FixedNormWindowDataset
from core_pipeline.stream.window_view import PROCESS_COLS


def windows(n, seed, shift=0.0):
    r = np.random.default_rng(seed)
    base = r.normal(size=(n, P.WINDOW_SIZE, 1))
    return base + 0.3 * r.normal(size=(n, P.WINDOW_SIZE, len(PROCESS_COLS))) + shift


def test_fixed_normalisation_not_per_dataset():
    W = windows(50, 1)
    mean, std = W.reshape(-1, 52).mean(0), W.reshape(-1, 52).std(0)
    ds = FixedNormWindowDataset(windows(10, 2, shift=5.0), mean, std)       # shifted data, training stats
    assert abs(ds.windows.mean()) > 1.0                                     # NOT re-centred on itself


def test_fit_score_save_load(tmp_path):
    at = WindowAT.fit(windows(64, 3), n_epochs=1, batch_size=16, d_model=16, n_heads=2, n_layers=1, d_ff=32, device="cpu")
    s = at.score(windows(8, 4))
    assert s.shape == (8,) and np.isfinite(s).all()
    at.save(tmp_path); at2 = WindowAT.load(tmp_path)
    assert np.allclose(at2.score(windows(8, 4)), s, atol=1e-5)
    at2.meta["sign"] = -1
    assert np.allclose(at2.score(windows(8, 4)), -s, atol=1e-5)
