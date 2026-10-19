"""
core_pipeline/anomaly/prefix_bh.py — causal prefix-BH alarm rule (prereg-v1 rev 3 §4.1).

At window t the tested (non-BLOCK) p-values p_0..p_t are BH-adjusted at level alpha; window t is an alarm iff its own
adjusted value <= alpha. The decision for t is taken once, at time t, and never revised by later windows. This is a
causal heuristic: no FDR-control claim for the cumulative rejection set (prereg §4.1, counterexample recorded there).
"""
from __future__ import annotations

from typing import Sequence


def bh_adjusted(pvalues: Sequence[float]) -> list[float]:
    """Standard Benjamini–Hochberg step-up adjusted p-values (monotone), same order as input."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):           # from largest p downwards
        i = order[rank - 1]
        running = min(running, pvalues[i] * m / rank)
        adj[i] = min(1.0, running)
    return adj


def prefix_bh_last(pvalues_prefix: Sequence[float], alpha: float) -> tuple[bool, float, int]:
    """(alarm_t, p_adj_t, m_t) for the LAST element of the tested prefix. m_t = number of tested windows."""
    if not pvalues_prefix:
        raise ValueError("empty prefix")
    adj = bh_adjusted(pvalues_prefix)
    return adj[-1] <= alpha, adj[-1], len(pvalues_prefix)
