"""core_pipeline/eval/keys.py — deterministic identities of the replay driver (ST4-50 S4b-02).

Every Skill call of an evaluation run carries request_id = uuid5(NS, eval_run | episode | window | step_key), so a
resumed run re-sends the SAME request ids and gets the stored envelopes back (ST4-13 replay): one logical spend per
window, never a second spend under a new id. The first detect (x1:initial) and the post-query re-read (x1:after_query)
are different steps on purpose. decision_key / turn_id follow the same rule.
"""
from __future__ import annotations

import uuid
from uuid import UUID

NS = uuid.UUID("8f1f1a2e-4b1c-4c3e-9a44-2f0f6e4e5b02")
REPLAY_MODE = "replay"


def step_keys(t: int) -> dict:
    return {"gate": f"gate:t{t}", "detect_initial": f"detect:t{t}:x1:initial", "detect_expand": f"detect:t{t}:x2:initial",
            "detect_after": f"detect:t{t}:x1:after_query", "spend": f"spend:t{t}", "retrieve_k5": f"retrieve:scoped:k5:t{t}",
            "retrieve_k10": f"retrieve:scoped:k10:t{t}"}


def request_id(eval_run_id: UUID | str, episode_id: UUID | str, t: int, step_key: str) -> UUID:
    return uuid.uuid5(NS, f"{UUID(str(eval_run_id))}|{UUID(str(episode_id))}|{int(t)}|{step_key}")


def thread_id(eval_run_id: UUID | str) -> str:
    return f"eval:{UUID(str(eval_run_id))}"


def turn_id(eval_run_id: UUID | str, episode_id: UUID | str, t: int) -> UUID:
    return uuid.uuid5(NS, f"turn|{UUID(str(eval_run_id))}|{UUID(str(episode_id))}|{int(t)}")


def decision_key(eval_run_id: UUID | str, policy: str, episode_id: UUID | str, t: int, phase: str) -> str:
    return f"eval:{UUID(str(eval_run_id))}:{policy}:{UUID(str(episode_id))}:{int(t)}:{phase}"


def ctx(eval_run_id: UUID | str, episode_id: UUID | str, t: int, step_key: str, caller_role: str) -> dict:
    return {"request_id": str(request_id(eval_run_id, episode_id, t, step_key)), "thread_id": thread_id(eval_run_id),
            "run_uuid": str(UUID(str(eval_run_id))), "caller_role": caller_role, "mode": REPLAY_MODE}
