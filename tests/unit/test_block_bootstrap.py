"""tests/unit/test_block_bootstrap.py — pinned run-id block bootstrap (prereg §5.1). Pure numpy, no DB."""
import numpy as np
import pytest

from core_pipeline.eval import block_bootstrap as BB


def test_constants_match_prereg():
    assert (BB.SEED, BB.N_BOOT, BB.LEVEL, BB.QUANTILE_METHOD) == (20261018, 2000, 0.95, "linear")


def test_index_matrix_is_pinned_generator_stream():
    idx = BB.draw_block_indices(60)
    assert idx.shape == (2000, 60)
    assert idx[0, :10].tolist() == [41, 52, 50, 23, 34, 2, 42, 44, 1, 51]
    assert idx[-1, -5:].tolist() == [13, 19, 43, 12, 47]
    assert int(idx.sum()) == 3533999
    assert np.array_equal(idx, BB.draw_block_indices(60))
    assert BB.draw_block_indices(40)[0, :10].tolist() == [27, 34, 33, 15, 23, 1, 28, 29, 1, 34]


def test_constant_deltas_give_a_degenerate_interval():
    blocks = [[0.4] * 21 for _ in range(60)]
    iv = BB.block_interval(blocks, BB.draw_block_indices(60))
    assert iv.point == pytest.approx(0.4) and iv.lower == pytest.approx(0.4) and iv.upper == pytest.approx(0.4)
    assert iv.n_blocks == 60 and iv.n_units == 1260


def test_resampling_unit_is_the_block_not_the_episode():
    blocks = [[0.0, 2.0], [2.0, 0.0], [1.0, 1.0]] * 10
    iv = BB.block_interval(blocks, BB.draw_block_indices(len(blocks)))
    assert (iv.lower, iv.upper) == (pytest.approx(1.0), pytest.approx(1.0))
    flat = [[x] for b in blocks for x in b]
    ep = BB.block_interval(flat, BB.draw_block_indices(len(flat)))
    assert ep.upper - ep.lower > 0.1


def test_statistic_is_ratio_of_sums_for_unequal_blocks():
    blocks = [[1.0], [0.0, 0.0, 0.0]]
    idx = np.array([[0, 0], [1, 1], [0, 1]])
    assert BB.resampled_means(blocks, idx).tolist() == [1.0, 0.0, 0.25]
    assert BB.block_interval(blocks, idx).point == 0.25


def test_percentile_is_linear_interpolation():
    stats = np.array([5.0, 1.0, 3.0, 2.0, 4.0, 10.0, 7.0])
    s = np.sort(stats)

    def manual(q):
        h = (len(s) - 1) * q
        lo = int(np.floor(h))
        return s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])

    lo, hi = BB.percentile_interval(stats)
    assert lo == pytest.approx(manual(0.025)) and hi == pytest.approx(manual(0.975))


def test_decision_is_strict_and_needs_every_comparison():
    iv = lambda lo: BB.Interval(point=1.0, lower=lo, upper=2.0, n_blocks=60, n_units=1260)
    assert BB.all_lower_gt_0({"B1": iv(0.01), "B2": iv(0.2), "B3": iv(1e-9)}) is True
    assert BB.all_lower_gt_0({"B1": iv(0.01), "B2": iv(0.0), "B3": iv(0.3)}) is False
    assert BB.all_lower_gt_0({"B1": iv(-0.1), "B2": iv(0.2), "B3": iv(0.3)}) is False


def test_shape_and_empty_block_are_refused():
    with pytest.raises(ValueError):
        BB.resampled_means([[1.0], [2.0]], np.zeros((5, 3), dtype=int))
    with pytest.raises(ValueError):
        BB.block_interval([[1.0], []], BB.draw_block_indices(2))
    with pytest.raises(ValueError):
        BB.draw_block_indices(0)
