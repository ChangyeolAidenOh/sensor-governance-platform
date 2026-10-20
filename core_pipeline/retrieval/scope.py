"""
core_pipeline/retrieval/scope.py — S3-02: the fault→guidance mapping as a HARD filter in scoped mode (prereg §4.6,
skill contract §1.6/§1.8): scope = union over the PREDICTED set's A∪B∪C clauses (empty set → C only). Each returned
chunk gets a grade A / B / C relative to the predicted set; in open mode anything outside the scope is 'OUT'.
A clause can be A for one fault and B for another — the grade is the best over the set's faults.
"""
from __future__ import annotations

import json
from pathlib import Path

GRADE_RANK = {"A": 0, "B": 1, "C": 2, "OUT": 3}


class FaultScope:
    def __init__(self, fault_map_path: Path):
        fm = json.loads(Path(fault_map_path).read_text(encoding="utf-8"))
        self.common = set(fm["common_clauses"].keys())
        self.by_fault = {int(f): {"A": set(v["clauses"]["A"]), "B": set(v["clauses"]["B"])} for f, v in fm["faults"].items()}
        self.meta = {"frozen_at": fm.get("frozen_at"), "decisions": fm.get("decisions")}

    def grade(self, clause_key: str, pred_set: list[int]) -> str:
        best = "OUT"
        if clause_key in self.common:
            best = "C"
        for f in pred_set:
            g = self.by_fault.get(int(f))
            if not g:
                continue
            if clause_key in g["A"]:
                return "A"
            if clause_key in g["B"]:
                best = "B"
        return best

    def scope_keys(self, pred_set: list[int]) -> set[str]:
        keys = set(self.common)
        for f in pred_set:
            g = self.by_fault.get(int(f))
            if g:
                keys |= g["A"] | g["B"]
        return keys
