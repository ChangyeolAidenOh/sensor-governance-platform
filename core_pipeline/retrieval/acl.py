"""
core_pipeline/retrieval/acl.py — S3-02: access-control pre-filter (skill contract §1.6: ACL layer BEFORE the mapping
scope filter and before any search). Roles map to guide allowlists. The mechanism is what matters for the contract
("no clause outside the caller's ACL ever enters the candidate set"); the role table is a placeholder policy.
  engineer   — all 28 guides (A 16 + B 10 + distractors P-44, P-22; distractors stay retrievable so the RAG trap
               questions can actually be failed)
  restricted — guides mapped as primary/secondary to at least one fault (no distractors)
"""
from __future__ import annotations

import json
from pathlib import Path

DISTRACTORS = {"P-44", "P-22"}


def load_roles(fault_map_path: Path, all_guides: set[str]) -> dict[str, set[str]]:
    fm = json.loads(Path(fault_map_path).read_text(encoding="utf-8"))
    mapped = set()
    for f in fm["faults"].values():
        mapped |= set(f["primary_guides"]) | set(f["secondary_guides"])
    for key in fm["common_clauses"]:
        mapped.add(key.split(":")[0])
    return {"engineer": set(all_guides), "restricted": {g for g in all_guides if g in mapped}}


def allowed(role: str, roles: dict[str, set[str]], guide_no: str) -> bool:
    if role not in roles:
        raise KeyError(f"unknown role {role!r}")
    return guide_no in roles[role]
