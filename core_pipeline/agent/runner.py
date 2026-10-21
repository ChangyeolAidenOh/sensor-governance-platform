"""core_pipeline/agent/runner.py — the public entry points of the agent (ST4-16): AgentSession.invoke / resume.

Both verify the session secret against mart.threads BEFORE touching the checkpointer, register the request as a turn
(client_request_id = turn_id), take the execution lease and only then run the graph inside a session_scope. Handling
of a resend of the same turn (reviewer 2026-10-21):
    registered (never ran)              → claim and start
    running, lease valid                → no second execution; {"status": "running"}
    running, lease expired              → reclaim; continue from the checkpoint if the stored State belongs to THIS
                                          turn (State.turn_id == turn_id): completed checkpoint whose DB record is
                                          missing → recover the result WITHOUT re-running; pending interrupt →
                                          record waiting_approval; otherwise invoke(None) continues; a State of an
                                          earlier turn → run the message
    waiting_approval                    → return that turn's interrupt payload
    done / failed                       → return that turn's stored result (never the thread's latest checkpoint)
A resume is its own turn (kind='resume', parent = the waiting message turn); resume done + parent transition are ONE
transaction (TurnStore.finish_resume): the parent closes when the graph completes and stays waiting with the NEW
interrupt when a modify round interrupts again. A resent finished resume returns its own result and nothing else.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from langgraph.types import Command

from core_pipeline.agent.graph import GraphDeps, build_graph
from core_pipeline.agent.leased_checkpointer import LeasedCheckpointer
from core_pipeline.agent.session import SessionAuth, session_scope, verify_secret
from core_pipeline.agent.turns import LostLease, TurnBusy, TurnConflict, TurnStore, message_sha

log = logging.getLogger("sgp.agent.runner")

RESULT_FIELDS = ("category", "decision", "decision_after", "query", "approval", "work_order", "trace", "answer", "refusal", "turn_refs_discarded")


def _pending_interrupt(state_snapshot) -> Optional[dict]:
    for task in getattr(state_snapshot, "tasks", ()) or ():
        for intr in getattr(task, "interrupts", ()) or ():
            v = getattr(intr, "value", intr)
            return v if isinstance(v, dict) else {"value": v}
    return None


def _checkpoint_id(state_snapshot) -> Optional[str]:
    try:
        return state_snapshot.config["configurable"].get("checkpoint_id")
    except Exception:
        return None


def compose_result(values: dict, turn_id: str) -> dict:
    out = {"status": "done", "turn_id": turn_id}
    for k in RESULT_FIELDS:
        if k in values:
            out[k] = values[k]
    d = (values.get("draft") or {})
    if d:
        out["draft"] = {"status": d.get("status"), "payload": d.get("payload"), "reason": d.get("reason")}
    return out


class AgentSession:
    def __init__(self, deps: GraphDeps, checkpointer):
        self.deps = deps
        self.turns: TurnStore = deps.turns
        self.checkpointer = checkpointer if isinstance(checkpointer, LeasedCheckpointer) else LeasedCheckpointer(checkpointer, self.turns)
        self.graph = build_graph(deps, self.checkpointer)

    def pending_turn(self, thread_id: str, secret: str) -> Optional[dict]:
        """The turn the UI must resend (registered / running) or resume (waiting_approval) before a new one is accepted."""
        verify_secret(self.deps.dsn, thread_id, secret)
        t = self.turns.pending(thread_id)
        return None if t is None else {"turn_id": t.turn_id, "kind": t.kind, "status": t.status, "lease_valid": t.lease_valid, "message": t.message}

    # ------------------------------------------------------------------------------------------------- public
    def invoke(self, thread_id: str, secret: str, message: str, client_request_id: str) -> dict:
        row = verify_secret(self.deps.dsn, thread_id, secret)
        turn = self.turns.register(thread_id, client_request_id, "message", {"message": message}, str(uuid.uuid4()))
        if turn.status in ("done", "failed"):
            return dict(turn.result or {"status": turn.status, "turn_id": turn.turn_id})
        if turn.status == "waiting_approval":
            return {"status": "waiting_approval", "turn_id": turn.turn_id, "interrupt": turn.result}
        if turn.status == "running" and turn.lease_valid:
            return {"status": "running", "turn_id": turn.turn_id}
        executor = str(uuid.uuid4())
        claimed = self.turns.claim(thread_id, client_request_id, executor)
        if claimed is None:
            return {"status": "running", "turn_id": client_request_id}
        auth = SessionAuth(thread_id, row.owner, secret, executor, client_request_id)
        config = {"configurable": {"thread_id": thread_id}}
        with session_scope(auth):
            try:
                snap = self.graph.get_state(config)
                vals = dict(snap.values or {})
                if vals.get("turn_id") == client_request_id:
                    if not snap.next:                                       # completed checkpoint, DB record missing → recover
                        result = compose_result(vals, client_request_id)
                        self.turns.finish(thread_id, client_request_id, executor, "done", _checkpoint_id(snap), result)
                        return result
                    pend = _pending_interrupt(snap)
                    if pend is not None:
                        self.turns.finish(thread_id, client_request_id, executor, "waiting_approval", _checkpoint_id(snap), pend)
                        return {"status": "waiting_approval", "turn_id": client_request_id, "interrupt": pend}
                    out = self.graph.invoke(None, config)                   # continue from the checkpoint
                else:
                    out = self.graph.invoke({"thread_id": thread_id, "turn_id": client_request_id, "run_uuid": claimed.run_uuid,
                                             "message": message, "caller_role": self.deps.caller_role}, config)
                return self._settle(thread_id, client_request_id, executor, out, config)
            except LostLease:
                log.warning("turn %s: lease lost, executor %s stops", client_request_id, executor)
                return {"status": "lost_lease", "turn_id": client_request_id}
            except Exception as e:
                log.exception("turn %s failed", client_request_id)
                try:
                    self.turns.finish(thread_id, client_request_id, executor, "failed", None, {"status": "failed", "turn_id": client_request_id, "error": str(e)[:500]})
                except LostLease:
                    pass
                raise

    def resume(self, thread_id: str, secret: str, parent_turn_id: str, decision: str, client_request_id: str,
               modifications: Optional[dict] = None) -> dict:
        row = verify_secret(self.deps.dsn, thread_id, secret)
        payload = {"parent_turn_id": parent_turn_id, "decision": decision, "modifications": modifications}
        prior = self.turns.get(thread_id, client_request_id)
        if prior is not None and prior.status in ("done", "failed"):          # resend of a finished resume → its stored result
            if prior.message_sha256 != message_sha(payload):
                raise TurnConflict(f"resume {client_request_id} already registered with a different payload")
            # its own stored result only — never a parent "repair": finish_resume() closes resume + parent atomically, and a
            # resent OLD modify must not rewind a parent that has since moved to a newer draft (reviewer 2026-10-21)
            return dict(prior.result or {"status": prior.status, "turn_id": prior.turn_id})
        parent = self.turns.get(thread_id, parent_turn_id)
        if parent is None or parent.status != "waiting_approval":
            raise TurnBusy(f"turn {parent_turn_id} is not waiting for approval ({parent.status if parent else 'unknown'})")
        turn = self.turns.register(thread_id, client_request_id, "resume", payload, parent.run_uuid, parent_turn_id=parent_turn_id)
        if turn.status in ("done", "failed"):
            return dict(turn.result or {"status": turn.status, "turn_id": turn.turn_id})
        if turn.status == "running" and turn.lease_valid:
            return {"status": "running", "turn_id": turn.turn_id}
        executor = str(uuid.uuid4())
        claimed = self.turns.claim(thread_id, client_request_id, executor)
        if claimed is None:
            return {"status": "running", "turn_id": client_request_id}
        auth = SessionAuth(thread_id, row.owner, secret, executor, parent_turn_id, lease_turn_id=client_request_id)   # State = parent turn, lease = resume turn
        config = {"configurable": {"thread_id": thread_id}}
        with session_scope(auth):
            try:
                snap = self.graph.get_state(config)
                vals = dict(snap.values or {})
                if vals.get("turn_id") != parent_turn_id:
                    raise RuntimeError("checkpoint does not belong to the waiting turn")
                if not snap.next:                                             # resume already completed, records missing
                    return self._close(thread_id, parent_turn_id, client_request_id, executor, snap, vals)
                self.turns.touch(thread_id, client_request_id, executor)
                out = self.graph.invoke(Command(resume={"decision": decision, "modifications": modifications}), config)
                return self._settle(thread_id, parent_turn_id, executor, out, config, resume_turn_id=client_request_id)
            except LostLease:
                return {"status": "lost_lease", "turn_id": client_request_id}
            except Exception as e:
                log.exception("resume %s failed", client_request_id)
                try:
                    self.turns.finish(thread_id, client_request_id, executor, "failed", None, {"status": "failed", "turn_id": client_request_id, "error": str(e)[:500]})
                except LostLease:
                    pass
                raise

    # ------------------------------------------------------------------------------------------------- internals
    def _settle(self, thread_id: str, turn_id: str, executor: str, out: dict, config: dict, resume_turn_id: Optional[str] = None) -> dict:
        snap = self.graph.get_state(config)
        ck = _checkpoint_id(snap)
        if "__interrupt__" in out:
            pend = _pending_interrupt(snap) or {"value": [getattr(i, "value", i) for i in out["__interrupt__"]]}
            res = {"status": "waiting_approval", "turn_id": turn_id, "interrupt": pend}
            if resume_turn_id:                                             # a modify round: resume done + parent stays waiting, ONE transaction
                self.turns.finish_resume(thread_id, resume_turn_id, executor, turn_id, "waiting_approval", ck, res, pend)
            else:
                self.turns.finish(thread_id, turn_id, executor, "waiting_approval", ck, pend)
            return res
        result = compose_result(dict(snap.values or {}), turn_id)
        if resume_turn_id:                                                 # resume done + parent done, ONE transaction
            self.turns.finish_resume(thread_id, resume_turn_id, executor, turn_id, "done", ck, result, result)
        else:
            self.turns.finish(thread_id, turn_id, executor, "done", ck, result)
        return result

    def _close(self, thread_id: str, parent_turn_id: str, resume_turn_id: str, executor: str, snap, vals: dict) -> dict:
        result = compose_result(vals, parent_turn_id)
        self.turns.finish_resume(thread_id, resume_turn_id, executor, parent_turn_id, "done", _checkpoint_id(snap), result, result)
        return result
