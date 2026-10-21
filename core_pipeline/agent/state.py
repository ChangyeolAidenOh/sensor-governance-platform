"""core_pipeline/agent/state.py — the LangGraph State of one chat thread (S4a-04).

Thread-level fields are set by load_context from mart.threads + the demo manifest; per-turn fields are RESET by
load_context at every turn (LangGraph keeps non-reducer fields across turns otherwise). Nothing secret lives here
(session secret: contextvar, see session.py). Tool arguments are read from these fields only (contract §3.4).
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


def _append(a: list, b: list) -> list:
    return list(a or []) + list(b or [])


class AgentState(TypedDict, total=False):
    # thread (stable)
    thread_id: str
    owner: str
    preset: int
    episode_id: str
    display_window: int
    alarm_id: Optional[str]                  # the display window's alarm (None for no_alarm presets)
    kind: str                                # alarm | no_alarm (manifest)
    caller_role: str
    messages: Annotated[list[dict], _append]     # {role, content, turn_id}
    user_claims: Annotated[list[int], _append]   # fault numbers the user asserted (contract §1.5), never a Skill input
    refs_discarded: Annotated[int, operator.add]  # cumulative count of user references that mismatched the state
    # turn (reset by load_context)
    turn_id: str
    run_uuid: str
    message: str
    category: Optional[str]
    route: Optional[dict]
    turn_refs_discarded: int
    turn_refs_note: Optional[str]
    refusal: Optional[dict]                  # {code, message} → narrate refuses, no further tool call
    gate: Optional[dict]
    detect: Optional[dict]
    retrieve: Optional[dict]                 # scoped k=5 (action) or open k=5 (evidence)
    decision: Optional[dict]                 # phase initial
    query: Optional[dict]                    # {query_id, kind, status, expand_status, rerank_status, owned}
    detect_x: Optional[dict]                 # expand=2 envelope
    detect_after: Optional[dict]             # expand=1 re-read after the query (latest state)
    retrieve_k10: Optional[dict]
    decision_after: Optional[dict]           # phase after_query
    draft: Optional[dict]
    draft_version: int
    modify_round: int
    approval_input: Optional[dict]
    approval: Optional[dict]
    work_order: Optional[dict]
    history: Optional[dict]
    answer: Optional[str]
    trace: list[dict]                        # per-turn step log (tool, step_key, status, reason, replayed)
