"""core_pipeline/agent/keys.py — deterministic identifiers of the graph (ST4-16).

Every Skill call the graph issues carries request_id = uuid5(NS, thread|run_uuid|step_key). With the server's bound-input
idempotency (ST4-13) a re-executed node (interrupt re-run, crash + recovery, lagging executor) REPLAYS the stored
envelope instead of executing side effects twice. The step_key names the logical step, so two calls that must be
distinct (initial vs after_query detect, draft v1 vs v2) get distinct ids while retries of one step share one id.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

NS = uuid.uuid5(uuid.NAMESPACE_URL, "sgp/agent/request-id/v1")


def request_id(thread_id: str, run_uuid: str, step_key: str) -> str:
    return str(uuid.uuid5(NS, f"{thread_id}|{run_uuid}|{step_key}"))


def ctx_for(thread_id: str, run_uuid: str, step_key: str, caller_role: str = "engineer", mode: str = "chat") -> dict:
    return {"request_id": request_id(thread_id, run_uuid, step_key), "thread_id": thread_id, "run_uuid": str(run_uuid),
            "caller_role": caller_role, "mode": mode}


def decision_key(thread_id: str, turn_id: str, phase: str) -> str:
    assert phase in ("initial", "after_query"), phase
    return f"{thread_id}:{turn_id}:{phase}"


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")).hexdigest()


# step keys of one turn (window t = the thread's display window)
def k_gate(t: int) -> str: return f"gate:t{t}"
def k_detect(t: int, expand: int, phase: str) -> str: return f"detect:t{t}:x{expand}:{phase}"
def k_spend(t: int) -> str: return f"spend:t{t}"
def k_retrieve(mode: str, k: int) -> str: return f"retrieve:{mode}:k{k}"
def k_draft(version: int) -> str: return f"draft:v{version}"
def k_approve(version: int) -> str: return f"approve:v{version}"
def k_issue(version: int) -> str: return f"issue:v{version}"
