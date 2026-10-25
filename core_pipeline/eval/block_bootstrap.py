"""core_pipeline/eval/block_bootstrap.py — run-id block bootstrap for the H1 decision (prereg §5.1, eval spec v3 §1).

Pure numpy. The resampling unit is a BLOCK (one run id = all of its episodes, ST0-09): blocks are drawn with
replacement, every episode of a drawn block enters the resample, and the statistic is the mean over the resampled
episodes (sum of block sums / sum of block sizes). With 21 episodes per block this equals the mean of block means.

Pinned procedure (all of it goes into the H1 output JSON):
  * generator      numpy.random.Generator(PCG64(seed)), seed 20261018
  * draws          ONE index matrix of shape (B, n_blocks) from rng.integers(0, n_blocks), B = 2000; the same matrix is
                   used for every comparison (B1/B2/B3, c_fp grid, per-fault reports) so all intervals are paired
  * interval       percentile, two-sided 95%: numpy.quantile(..., method="linear") at 0.025 / 0.975
  * point          the plain mean over all episodes (not the bootstrap mean)
  * decision       lower bound > 0, strictly (a lower bound of exactly 0 is not a pass)
Block order is the caller's (H1 uses the evaluation-pool order of the planned run ids), because the index matrix
refers to positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

SEED = 20261018
N_BOOT = 2000
LEVEL = 0.95
QUANTILE_METHOD = "linear"
GENERATOR = "numpy.random.Generator(PCG64)"


def procedure() -> dict:
    return {"seed": SEED, "n_boot": N_BOOT, "level": LEVEL, "interval": "percentile, two-sided", "quantile_method": QUANTILE_METHOD,
            "generator": GENERATOR, "draws": "rng.integers(0, n_blocks, size=(n_boot, n_blocks)), one matrix shared by all comparisons",
            "statistic": "mean over resampled episodes = sum(block sums) / sum(block sizes)", "decision": "lower > 0 (strict)",
            "numpy": np.__version__}


def draw_block_indices(n_blocks: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    if n_blocks < 1 or n_boot < 1:
        raise ValueError(f"n_blocks and n_boot must be >= 1 (got {n_blocks}, {n_boot})")
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.integers(0, n_blocks, size=(n_boot, n_blocks))


def _block_arrays(blocks: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    if not blocks:
        raise ValueError("no blocks")
    sizes = np.array([len(b) for b in blocks], dtype=np.float64)
    if (sizes == 0).any():
        raise ValueError("empty block")
    sums = np.array([float(np.sum(np.asarray(b, dtype=np.float64))) for b in blocks], dtype=np.float64)
    return sums, sizes


def resampled_means(blocks: Sequence[Sequence[float]], idx: np.ndarray) -> np.ndarray:
    sums, sizes = _block_arrays(blocks)
    if idx.ndim != 2 or idx.shape[1] != len(blocks):
        raise ValueError(f"index matrix {idx.shape} does not match {len(blocks)} blocks")
    return sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)


def percentile_interval(stats: np.ndarray, level: float = LEVEL) -> tuple[float, float]:
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(np.asarray(stats, dtype=np.float64), [a, 1.0 - a], method=QUANTILE_METHOD)
    return float(lo), float(hi)


@dataclass(frozen=True)
class Interval:
    point: float
    lower: float
    upper: float
    n_blocks: int
    n_units: int

    @property
    def lower_gt_0(self) -> bool:
        return self.lower > 0.0

    def as_dict(self) -> dict:
        return {"mean": self.point, "lower": self.lower, "upper": self.upper, "lower_gt_0": self.lower_gt_0,
                "n_blocks": self.n_blocks, "n_units": self.n_units}


def block_interval(blocks: Sequence[Sequence[float]], idx: np.ndarray, level: float = LEVEL) -> Interval:
    sums, sizes = _block_arrays(blocks)
    point = float(sums.sum() / sizes.sum())
    lo, hi = percentile_interval(resampled_means(blocks, idx), level)
    return Interval(point, lo, hi, len(blocks), int(sizes.sum()))


def all_lower_gt_0(intervals: dict[str, Interval]) -> bool:
    """Intersection-union decision of H1: every pairwise interval must clear zero; no multiplicity correction."""
    if not intervals:
        raise ValueError("no intervals")
    return all(iv.lower_gt_0 for iv in intervals.values())
