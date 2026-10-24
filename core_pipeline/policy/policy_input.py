"""core_pipeline/policy/policy_input.py — the FIRST-observation policy input (ST4-50, option C), pure data.

A PolicyInput is what B1, B3 and E-initial decide from. By construction it holds no expanded_set, q_used, k=10 result
or any other query_more output: those fields do not exist here, so a baseline cannot read them. This module imports no
DB / Skill / state-store code — baselines.py depends only on it and on decide_action.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class K5Grounding:
    """The first OK scoped k=5 retrieve envelope of an alarm window (grounding block + its own bound input)."""
    request_id: str
    fault_set: list[int]
    mode: str
    tau: Optional[float]
    g_ab: Optional[float]
    g_c: Optional[float]
    ev_ab: Optional[bool]
    ev_c: Optional[bool]
    snapshot: list[dict]                 # [{clause_key, grade, score}] — what record_decision(evidence=) receives
    versions: dict                       # retrieve trace.versions (embedder, reranker, tau, fault_map_sha256, k_*, contract_sha256, detector)


@dataclass(frozen=True)
class PolicyInput:
    episode_id: str
    window_index: int
    detector_version: str
    contract_sha256: str
    caller_role: str
    gate: str
    drift: str
    drift_assessed: bool
    p_value: Optional[float]
    p_adj: Optional[float]
    m_t: int
    alarm: Optional[bool]
    alarm_id: Optional[str]
    pred_set: list[int]
    set_size: int
    top1: Optional[int]
    k5: Optional[K5Grounding] = None
    snapshot_id: Optional[str] = None
    content_sha256: str = field(default="")

    def detection_block(self) -> dict:
        return {"gate": self.gate, "drift": self.drift, "drift_assessed": bool(self.drift_assessed), "p_value": self.p_value,
                "p_adj": self.p_adj, "m_t": int(self.m_t), "alarm": self.alarm, "alarm_id": self.alarm_id,
                "pred_set": [int(x) for x in self.pred_set], "set_size": int(self.set_size), "top1": self.top1}

    def k5_block(self) -> Optional[dict]:
        if self.k5 is None:
            return None
        k = self.k5
        return {"request_id": k.request_id, "fault_set": [int(x) for x in k.fault_set], "mode": k.mode, "tau": k.tau, "g_ab": k.g_ab,
                "g_c": k.g_c, "ev_ab": k.ev_ab, "ev_c": k.ev_c, "snapshot": k.snapshot, "versions": k.versions}

    def content_hash(self) -> str:
        return sha256_json({"key": [self.episode_id, int(self.window_index), self.detector_version, self.contract_sha256, self.caller_role],
                            "detection": self.detection_block(), "k5": self.k5_block()})


# version fields that must agree between a stored k=5 envelope and the evaluation run's expectation
K5_VERSION_KEYS = ("embedder", "reranker", "tau", "fault_map_sha256", "k_primary", "contract_sha256", "detector")


def versions_mismatch(stored: dict, expected: dict) -> list[str]:
    """Keys of K5_VERSION_KEYS whose values differ (missing on either side counts as a difference)."""
    return [k for k in K5_VERSION_KEYS if str(stored.get(k)) != str(expected.get(k))]
