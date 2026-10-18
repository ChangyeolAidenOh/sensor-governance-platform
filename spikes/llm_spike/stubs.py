"""Deterministic Skill stubs for the 0-F LLM selection spike.

These are NOT the real Skills: they return fixed, run-id-keyed values so that the only
thing under test is the model's tool-calling behavior (format, chaining, approval
resume, refusal). Argument schemas mirror the draft Skill contract (docs/skill_contract.md,
0-H) and are validated by LangGraph's ToolNode; a schema violation comes back to the
model as an error ToolMessage, which the runner counts as a format error.

Run ids used by scenarios are from the TEP testing dev group (ST0-13: run_id 1-20).
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

# run_id -> fixed world state (gate status, detection, diagnosis set)
RUNS = {
    3:  {"gate": "PASS",  "flagged": [],                  "p_value": 0.004, "bh_pass": True,  "fault_set": [14],    "top_sensors": ["xmv_10", "xmeas_9", "xmeas_21"]},
    7:  {"gate": "PASS",  "flagged": [],                  "p_value": 0.012, "bh_pass": True,  "fault_set": [4, 11], "top_sensors": ["xmeas_9", "xmeas_21", "xmv_10"]},
    12: {"gate": "BLOCK", "flagged": ["xmeas_7"],         "p_value": None,  "bh_pass": None,  "fault_set": [],      "top_sensors": []},
    15: {"gate": "PASS",  "flagged": [],                  "p_value": 0.41,  "bh_pass": False, "fault_set": [],      "top_sensors": []},
    18: {"gate": "FLAG",  "flagged": ["xmeas_37"],        "p_value": 0.03,  "bh_pass": True,  "fault_set": [17],    "top_sensors": ["xmeas_38", "xmeas_12"]},
}
DEFAULT_RUN = {"gate": "PASS", "flagged": [], "p_value": 0.5, "bh_pass": False, "fault_set": [], "top_sensors": []}

CANNED_CLAUSES = {   # fallback when outputs/fault_guidance_candidates.json is absent
    14: [("P-77-2011:0:7.3", "정비, 검사 및 테스트"), ("P-77-2011:0:7.2", "비상차단시스템의 실패요인"), ("P-129-2013:0:5.2.2", "점검방법")],
    4:  [("P-53-2012:0:6", "방호대책"), ("P-53-2012:0:6.7", "긴급배출 및 급랭"), ("P-65-2012:0:4.1", "일반 특성")],
    11: [("P-53-2012:0:6", "방호대책"), ("P-38-2012:0:5.3", "취득정보 해석")],
}
CROSS_CUTTING = [("P-116-2012:0:5.2.2", "경보의 선정 및 우선순위 부여"), ("P-129-2013:0:4.3", "계측기 점검 목적 및 대상")]
CANDIDATES_PATH = Path("outputs/fault_guidance_candidates.json")


def _clauses_for(fault_id: int) -> list[tuple[str, str]]:
    if CANDIDATES_PATH.exists():
        try:
            data = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
            for f in data.get("faults", []):
                if f.get("fault") == fault_id and f.get("primary_candidates"):
                    return [(c["clause_key"], c["title"]) for c in f["primary_candidates"][:3]]
        except (OSError, ValueError):
            pass
    return CANNED_CLAUSES.get(fault_id, [])


class WindowArgs(BaseModel):
    run_id: int = Field(description="TEP episode run id")
    window_start: int = Field(ge=1, description="first cycle of the window (1-based)")
    window_end: int = Field(ge=1, description="last cycle of the window (inclusive)")


class RetrieveArgs(BaseModel):
    fault_ids: list[int] = Field(description="candidate fault ids from detect_diagnose; empty list = generic alarm-handling clauses")
    query: str = Field(min_length=3, description="what guidance is needed, in natural language")


class WorkOrderArgs(BaseModel):
    run_id: int
    fault_id: int = Field(description="the fault the work order addresses")
    clause_keys: list[str] = Field(min_length=1, description="cited clause keys from retrieve; at least one is required")
    action: str = Field(min_length=5, description="the maintenance action to request")


@tool("gate_check", args_schema=WindowArgs)
def gate_check(run_id: int, window_start: int, window_end: int) -> dict:
    """Layer-1 data quality gate for one window. Returns PASS/FLAG/BLOCK and flagged sensors.
    BLOCK means the window must not be passed to detection."""
    r = RUNS.get(run_id, DEFAULT_RUN)
    return {"run_id": run_id, "window": [window_start, window_end], "status": r["gate"],
            "flagged_sensors": r["flagged"], "overall_score": {"PASS": 82.0, "FLAG": 55.0, "BLOCK": 20.0}[r["gate"]]}


@tool("detect_diagnose", args_schema=WindowArgs)
def detect_diagnose(run_id: int, window_start: int, window_end: int) -> dict:
    """Conformal anomaly p-value (with BH decision) and a conformal fault set for one window.
    Only call this after gate_check returned PASS or FLAG."""
    r = RUNS.get(run_id, DEFAULT_RUN)
    if r["gate"] == "BLOCK":
        return {"error": "window is BLOCKed by the quality gate; detection not available"}
    return {"run_id": run_id, "window": [window_start, window_end], "p_value": r["p_value"], "bh_pass": r["bh_pass"],
            "fault_set": r["fault_set"], "set_size": len(r["fault_set"]), "top_sensors": r["top_sensors"]}


@tool("retrieve", args_schema=RetrieveArgs)
def retrieve(fault_ids: list[int], query: str) -> dict:
    """Retrieve KOSHA GUIDE clauses that support an action for the given fault ids.
    Returns clause keys and titles; an action may only be proposed with at least one clause."""
    clauses = []
    for f in fault_ids:
        clauses.extend(_clauses_for(f))
    clauses.extend(CROSS_CUTTING)
    seen, out = set(), []
    for k, t in clauses:
        if k not in seen:
            seen.add(k); out.append({"clause_key": k, "title": t})
    return {"query": query, "fault_ids": fault_ids, "clauses": out[:6]}


@tool("propose_work_order", args_schema=WorkOrderArgs)
def propose_work_order(run_id: int, fault_id: int, clause_keys: list[str], action: str) -> dict:
    """Propose a maintenance work order. This pauses for engineer approval and returns the decision.
    A work order is only issued when the decision is 'approve'."""
    decision = interrupt({"run_id": run_id, "fault_id": fault_id, "clause_keys": clause_keys, "action": action,
                          "question": "approve this work order?"})
    d = str(decision.get("decision", "")) if isinstance(decision, dict) else str(decision)
    if d == "approve":
        return {"status": "issued", "work_order_id": f"WO-{run_id}-{fault_id}", "action": action, "clause_keys": clause_keys}
    return {"status": "held", "reason": f"engineer decision: {d or 'none'}"}


TOOLS = [gate_check, detect_diagnose, retrieve, propose_work_order]
