"""
core_pipeline/stream/state.py — replay state with three lifetimes (S1-03, reviewer 2026-10-19).

  EpisodeState  — lives for one episode, POLICY side. Holds only what the policy may know: tested-window p-values,
                  m_t, confirmed alarms (immutable once set), proposal history (which windows the policy proposed on).
                  No labels, no engine_id, no fault.
  WindowState   — lives for one window: q_used, expanded. Reset every window.
  Evaluator state (labels, first active-region proposal, c_fn) lives OUTSIDE the replayer: core_pipeline/eval joins
                  mart.window_labels via staging.episode_keys. The replayer never computes it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from core_pipeline.anomaly.prefix_bh import prefix_bh_last


@dataclass(frozen=True)
class AlarmDecision:
    window_index: int
    alarm: bool
    p_value: float
    p_adj: float
    m_t: int


@dataclass
class WindowState:
    window_index: int
    q_used: bool = False
    expanded: bool = False


@dataclass
class EpisodeState:
    episode_id: UUID
    alpha: float
    _tested: list[tuple[int, float]] = field(default_factory=list)      # (window_index, p) for non-BLOCK windows
    _decisions: dict[int, AlarmDecision] = field(default_factory=dict)  # confirmed, never rewritten
    _undecidable: list[int] = field(default_factory=list)              # BLOCK windows (no test, no p)
    proposals: list[int] = field(default_factory=list)                  # windows on which the policy proposed
    _last_seen: int = -1

    @property
    def m_t(self) -> int:
        return len(self._tested)

    def observe(self, window_index: int, gate: str, p_value: Optional[float]) -> Optional[AlarmDecision]:
        """Record window t. Windows must arrive in order; each window exactly once.
        BLOCK: no detection call, no p-value appended, m_t unchanged, returns None.
        Otherwise: prefix-BH on tested windows 0..t, decision for t fixed here and never revised."""
        if window_index != self._last_seen + 1:
            raise ValueError(f"window {window_index} out of order (last seen {self._last_seen})")
        self._last_seen = window_index
        if gate == "BLOCK":
            if p_value is not None:
                raise ValueError("BLOCK window must not carry a p-value (detect must not have been called)")
            self._undecidable.append(window_index)
            return None
        if p_value is None:
            raise ValueError("non-BLOCK window needs a p-value")
        self._tested.append((window_index, float(p_value)))
        alarm, p_adj, m_t = prefix_bh_last([p for _, p in self._tested], self.alpha)
        d = AlarmDecision(window_index, alarm, float(p_value), p_adj, m_t)
        self._decisions[window_index] = d
        return d

    def decision_at(self, window_index: int) -> Optional[AlarmDecision]:
        return self._decisions.get(window_index)

    def tested_windows(self) -> list[int]:
        return [w for w, _ in self._tested]

    def undecidable_windows(self) -> list[int]:
        return list(self._undecidable)

    def record_proposal(self, window_index: int) -> None:
        self.proposals.append(window_index)
