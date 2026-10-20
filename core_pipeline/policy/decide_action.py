"""
core_pipeline/policy/decide_action.py — S4a-02: the pre-registered action policy as a PURE function
(prereg-v1 rev 3 §4.5 + amendment A-3; skill contract §1.7 "Skills return numbers, the policy decides").

Inputs are the signals of ONE window, all causal and all produced by the Skills / EpisodeState:
  gate     PASS | FLAG | BLOCK | None   (None = policy without a gate, i.e. B2)
  drift    ok | critical
  alarm    prefix-BH decision for this window (B2: fixed-threshold exceedance)
  s        diagnosis set size (0 = empty → indeterminate; valid iff 1 <= s <= SET_SIZE_MAX)
  g_AB     best reranker score among A/B chunks in SCOPED mode, or None (no A/B chunk / open mode / retrieve not run)
  g_C      best C score or None
  q_used   query_more already spent on this window (window-scoped, never reset by re-asking)
  tau      fixed threshold (models/retrieval_v1/tau.json); None = not fixed → ev_AB is None → never a proposal basis
  retrieve_mode  'scoped' | 'open' | None — only SCOPED results can ground (ev_AB is False for open/None even with a
           high g_AB). The adapter that builds Signals from Skill results must copy the mode from the retrieve result
           it actually used; the policy never trusts a bare score.
Proposal EVENTS (for the pre-registered cost accounting) are what this function returns; the mock CMMS draft/issue
procedure is a separate, L1-only path.
Rules are evaluated top-down, first match wins; a query_more re-entry re-evaluates from rule 3 with q_used=True
(so rule 5 is closed but rule 4 can still propose). Baselines B1/B2/B3 get their own pre-registered rows — E's
conditions are not applied to them wholesale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from core_pipeline import prereg_constants as P

Policy = Literal["E", "B1", "B2", "B3"]
Action = Literal["undecidable", "no_alarm", "hold", "propose", "query_more"]


@dataclass(frozen=True)
class Signals:
    gate: Optional[str]
    drift: str
    alarm: bool
    s: int
    g_AB: Optional[float]
    g_C: Optional[float]
    q_used: bool
    tau: Optional[float]
    retrieve_mode: Optional[str] = None            # must be set explicitly by the adapter; missing → cannot ground

    @property
    def valid_s(self) -> bool:
        return 1 <= self.s <= P.SET_SIZE_MAX

    @property
    def ev_AB(self) -> Optional[bool]:
        """None when tau is not fixed or no A/B score exists in a way that cannot be evaluated; False/True otherwise."""
        if self.tau is None:
            return None
        if self.retrieve_mode != "scoped":
            return False                                   # open-mode evidence never grounds a proposal
        return (self.g_AB is not None) and (self.g_AB >= self.tau)

    @property
    def ev_C(self) -> Optional[bool]:
        if self.tau is None:
            return None
        return (self.g_C is not None) and (self.g_C >= self.tau)


@dataclass(frozen=True)
class Decision:
    action: Action
    rule: int                       # which row of the pre-registered table fired (1..6)
    reason: str
    query_kind: Optional[str] = None   # for query_more: 'expand' | 'rerank' | 'both'


def decide_action(sig: Signals, policy: Policy) -> Decision:
    if policy not in ("E", "B1", "B2", "B3"):
        raise ValueError(policy)
    # rule 1 — BLOCK (policies that use the gate)
    if policy != "B2" and sig.gate == "BLOCK":
        return Decision("undecidable", 1, "gate BLOCK: detection not called")
    # rule 2 — no alarm
    if not sig.alarm:
        return Decision("no_alarm", 2, "no alarm at this window")
    if policy == "B1":
        return Decision("propose", 4, "B1: always propose on alarm")
    if policy == "B2":
        return Decision("propose", 4, "B2: fixed playbook on threshold exceedance")
    if policy == "B3":
        if sig.ev_AB is True:
            return Decision("propose", 4, "B3: alarm ∧ ev_AB")
        return Decision("hold", 6, "B3: alarm without A/B grounding" + (" (tau not fixed)" if sig.ev_AB is None else ""))
    # ---- E ----
    # rule 3 — drift critical → hold (model_trust_low); evaluated only on alarm windows
    if sig.drift == "critical":
        return Decision("hold", 3, "drift critical: model_trust_low")
    # rule 4 — valid set ∧ A/B grounding
    if sig.valid_s and sig.ev_AB is True:
        return Decision("propose", 4, "valid set ∧ ev_AB")
    # rule 5 — one query_more per window; both remedies at once when both conditions fail
    if not sig.q_used:
        need_expand = not sig.valid_s
        need_rerank = sig.ev_AB is not True          # includes ev_AB None (tau unfixed) and False
        kind = "both" if (need_expand and need_rerank) else ("expand" if need_expand else "rerank")
        return Decision("query_more", 5, f"query_more ({kind}); re-enter at rule 3 with q_used", kind)
    # rule 6 — hold
    why = []
    if not sig.valid_s:
        why.append("diagnosis indeterminate" + (" (empty set)" if sig.s == 0 else f" (set size {sig.s})"))
    if sig.ev_AB is not True:
        why.append("no A/B grounding" + (" (tau not fixed)" if sig.ev_AB is None else "") + (" — C-only reference" if sig.ev_C else ""))
    return Decision("hold", 6, "; ".join(why) or "hold")
