"""core_pipeline/agent/turns.py — turn registry + execution lease of one chat thread (ST4-16, reviewer 2026-10-21).

A turn is (thread_id, turn_id) where turn_id is the client_request_id the UI/test issues per user message or resume.
    same turn_id + same message  → a resend: never a second execution; handled by status (see AgentSession)
    same turn_id + other message → TurnConflict
    new turn while another turn of the thread is registered / running (lease valid) / waiting_approval → TurnBusy
Execution lease: claim() sets status=running with an executor token; EVERY later write by that executor
(touch / mark / finish) is conditioned on `executor = token AND status = 'running'`, so once a lease expired and
another executor reclaimed the turn, the old executor's next write raises LostLease instead of overwriting. A slow
executor that is still alive therefore cannot continue past its next node boundary (graph nodes call touch()).
Skill-level side effects are idempotent by step key (keys.py), so a lagging executor can never diverge DB state.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

import psycopg

LEASE_SECONDS = float(os.environ.get("SGP_TURN_LEASE_SECONDS", "900"))
STATUSES = ("registered", "running", "waiting_approval", "done", "failed")


class TurnConflict(RuntimeError):
    pass


class TurnBusy(RuntimeError):
    pass


class LostLease(RuntimeError):
    pass


@dataclass(frozen=True)
class Turn:
    thread_id: str
    turn_id: str
    kind: str
    message_sha256: str
    message: Any
    run_uuid: str
    status: str
    executor: Optional[str]
    lease_valid: bool
    checkpoint_id: Optional[str]
    result: Optional[dict]


_COLS = "thread_id, turn_id, kind, message_sha256, message, run_uuid, status, executor, (lease_until IS NOT NULL AND lease_until > now()), checkpoint_id, result"


def _row(r) -> Turn:
    msg = r[4] if not isinstance(r[4], str) else json.loads(r[4])
    res = r[10] if r[10] is None or isinstance(r[10], dict) else json.loads(r[10])
    return Turn(r[0], str(r[1]), r[2], r[3], msg, str(r[5]), r[6], (str(r[7]) if r[7] else None), bool(r[8]), r[9], res)


def message_sha(message: Any) -> str:
    return hashlib.sha256(json.dumps(message, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


class TurnStore:
    def __init__(self, dsn: str, lease_seconds: float = LEASE_SECONDS):
        self.dsn, self.lease_seconds = dsn, lease_seconds

    @staticmethod
    def _lock(cur, thread_id: str) -> None:
        key = int.from_bytes(hashlib.sha256(f"turn|{thread_id}".encode()).digest()[:8], "big", signed=True)
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (key,))

    def get(self, thread_id: str, turn_id: str) -> Optional[Turn]:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT {_COLS} FROM mart.graph_turns WHERE thread_id=%s AND turn_id=%s", (thread_id, UUID(turn_id)))
            r = cur.fetchone()
        return _row(r) if r else None

    def active_other(self, cur, thread_id: str, turn_id: str, parent_turn_id: Optional[str] = None) -> Optional[Turn]:
        """An open turn of the thread other than turn_id (and other than a parent that is waiting for this resume)."""
        # 'running' counts as open WHATEVER its lease (reviewer 2026-10-21): an expired turn must be recovered (resent)
        # before the thread accepts a new turn, otherwise two executors could touch the same checkpoint chain
        cur.execute(f"""SELECT {_COLS} FROM mart.graph_turns WHERE thread_id=%s AND turn_id<>%s
                        AND status IN ('registered','running','waiting_approval') ORDER BY created_at""", (thread_id, UUID(turn_id)))
        for r in cur.fetchall():
            t = _row(r)
            if parent_turn_id and t.turn_id == parent_turn_id and t.status == "waiting_approval":
                continue
            return t
        return None

    def register(self, thread_id: str, turn_id: str, kind: str, message: Any, run_uuid: str, parent_turn_id: Optional[str] = None) -> Turn:
        """Idempotent registration. Serialises per thread: a new turn is refused while another turn is still open
        (a resume may be registered while its parent message turn is waiting_approval)."""
        sha = message_sha(message)
        with psycopg.connect(self.dsn) as conn:
            with conn.transaction(), conn.cursor() as cur:
                self._lock(cur, thread_id)
                cur.execute(f"SELECT {_COLS} FROM mart.graph_turns WHERE thread_id=%s AND turn_id=%s", (thread_id, UUID(turn_id)))
                r = cur.fetchone()
                if r:
                    t = _row(r)
                    if t.message_sha256 != sha or t.kind != kind:
                        raise TurnConflict(f"turn {turn_id} already registered with a different message")
                    return t
                other = self.active_other(cur, thread_id, turn_id, parent_turn_id)
                if other is not None:
                    hint = "resend it to recover" if (other.status == "running" and not other.lease_valid) or other.status == "registered" else "finish or resume it first"
                    raise TurnBusy(f"thread has an open turn {other.turn_id} ({other.status}{'' if other.lease_valid or other.status != 'running' else ', lease expired'}); {hint}")
                cur.execute("""INSERT INTO mart.graph_turns (thread_id, turn_id, kind, message_sha256, message, run_uuid, status)
                               VALUES (%s,%s,%s,%s,%s,%s,'registered')""",
                            (thread_id, UUID(turn_id), kind, sha, json.dumps(message, ensure_ascii=False, default=str), UUID(run_uuid)))
                cur.execute(f"SELECT {_COLS} FROM mart.graph_turns WHERE thread_id=%s AND turn_id=%s", (thread_id, UUID(turn_id)))
                return _row(cur.fetchone())

    def claim(self, thread_id: str, turn_id: str, executor: str, from_statuses: tuple[str, ...] = ("registered",)) -> Optional[Turn]:
        """Take the execution lease. Succeeds from one of `from_statuses`, or from 'running' with an EXPIRED lease."""
        with psycopg.connect(self.dsn) as conn:
            with conn.transaction(), conn.cursor() as cur:
                self._lock(cur, thread_id)
                cur.execute(f"""UPDATE mart.graph_turns SET status='running', executor=%s,
                                  lease_until=now() + make_interval(secs => %s), updated_at=now()
                                WHERE thread_id=%s AND turn_id=%s
                                  AND (status = ANY(%s) OR (status='running' AND (lease_until IS NULL OR lease_until <= now())))
                                RETURNING {_COLS}""",
                            (UUID(executor), self.lease_seconds, thread_id, UUID(turn_id), list(from_statuses)))
                r = cur.fetchone()
                return _row(r) if r else None

    def touch(self, thread_id: str, turn_id: str, executor: str) -> None:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("""UPDATE mart.graph_turns SET lease_until=now() + make_interval(secs => %s), updated_at=now()
                           WHERE thread_id=%s AND turn_id=%s AND executor=%s AND status='running'""",
                        (self.lease_seconds, thread_id, UUID(turn_id), UUID(executor)))
            if cur.rowcount != 1:
                conn.rollback()
                raise LostLease(f"turn {turn_id}: lease lost (reclaimed by another executor or already finished)")
            conn.commit()

    def finish(self, thread_id: str, turn_id: str, executor: str, status: str, checkpoint_id: Optional[str], result: Optional[dict]) -> None:
        assert status in ("waiting_approval", "done", "failed"), status
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("""UPDATE mart.graph_turns SET status=%s, checkpoint_id=%s, result=%s, lease_until=NULL, updated_at=now()
                           WHERE thread_id=%s AND turn_id=%s AND executor=%s AND status='running'""",
                        (status, checkpoint_id, json.dumps(result, ensure_ascii=False, default=str) if result is not None else None,
                         thread_id, UUID(turn_id), UUID(executor)))
            if cur.rowcount != 1:
                conn.rollback()
                raise LostLease(f"turn {turn_id}: cannot finish, lease lost")
            conn.commit()

    def pending(self, thread_id: str) -> Optional[Turn]:
        """The turn the UI must resend/resume before a new one is accepted (None if the thread is free)."""
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(f"""SELECT {_COLS} FROM mart.graph_turns WHERE thread_id=%s AND status IN ('registered','running','waiting_approval')
                            ORDER BY created_at LIMIT 1""", (thread_id,))
            r = cur.fetchone()
        return _row(r) if r else None

    def finish_resume(self, thread_id: str, resume_turn_id: str, executor: str, parent_turn_id: str, parent_status: str,
                      checkpoint_id: Optional[str], result: Optional[dict], parent_payload: Optional[dict]) -> None:
        """ONE transaction (reviewer 2026-10-21 point 4): resume turn → done (conditioned on the executor lease) AND the
        parent turn → `parent_status` ('done' with the result, or stays 'waiting_approval' with the new interrupt payload)."""
        assert parent_status in ("done", "waiting_approval"), parent_status
        with psycopg.connect(self.dsn) as conn:
            with conn.transaction(), conn.cursor() as cur:
                self._lock(cur, thread_id)
                cur.execute("""UPDATE mart.graph_turns SET status='done', checkpoint_id=%s, result=%s, lease_until=NULL, updated_at=now()
                               WHERE thread_id=%s AND turn_id=%s AND executor=%s AND status='running'""",
                            (checkpoint_id, json.dumps(result, ensure_ascii=False, default=str) if result is not None else None,
                             thread_id, UUID(resume_turn_id), UUID(executor)))
                if cur.rowcount != 1:
                    raise LostLease(f"resume {resume_turn_id}: cannot finish, lease lost")
                cur.execute("""UPDATE mart.graph_turns SET status=%s, checkpoint_id=%s, result=%s, updated_at=now()
                               WHERE thread_id=%s AND turn_id=%s AND status='waiting_approval'""",
                            (parent_status, checkpoint_id, json.dumps(parent_payload, ensure_ascii=False, default=str), thread_id, UUID(parent_turn_id)))

    def set_waiting(self, thread_id: str, turn_id: str, checkpoint_id: Optional[str], interrupt: Optional[dict]) -> None:
        """A modify round interrupted again: the parent stays waiting with the NEW interrupt payload."""
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("""UPDATE mart.graph_turns SET checkpoint_id=%s, result=%s, updated_at=now()
                           WHERE thread_id=%s AND turn_id=%s AND status='waiting_approval'""",
                        (checkpoint_id, json.dumps(interrupt, ensure_ascii=False, default=str), thread_id, UUID(turn_id)))
            conn.commit()

    def close_waiting(self, thread_id: str, turn_id: str, checkpoint_id: Optional[str], result: dict) -> None:
        """The resume completed: parent waiting_approval → done (conditioned on the waiting status, idempotent)."""
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("""UPDATE mart.graph_turns SET status='done', checkpoint_id=%s, result=%s, updated_at=now()
                           WHERE thread_id=%s AND turn_id=%s AND status='waiting_approval'""",
                        (checkpoint_id, json.dumps(result, ensure_ascii=False, default=str), thread_id, UUID(turn_id)))
            conn.commit()

    def list_thread(self, thread_id: str) -> list[Turn]:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT {_COLS} FROM mart.graph_turns WHERE thread_id=%s ORDER BY created_at", (thread_id,))
            return [_row(r) for r in cur.fetchall()]
