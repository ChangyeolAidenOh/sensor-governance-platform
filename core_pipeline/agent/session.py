"""core_pipeline/agent/session.py — session identity for the local demo (ST4-16, reviewer 2026-10-21 point 5).

start_session() registers mart.threads(thread_id, owner, episode_id, window_index, preset, owner_secret_sha256) and
returns (thread_id, secret). The secret is the caller's proof for every later invoke/resume; only its sha256 is stored.
The secret never enters graph State, config or traces: the public wrapper verifies it against the DB BEFORE touching
the checkpointer, then hands it to the nodes through a process-local ContextVar (reset in a finally), and the
approval_resume node verifies it AGAINST THE DB AGAIN before any work_order_approve / work_order_issue call.
The approver passed to the server is the DB owner of the thread, never a value from the resume payload.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets as _secrets
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional

import psycopg


class AuthError(PermissionError):
    pass


@dataclass(frozen=True)
class ThreadRow:
    thread_id: str
    owner: str
    episode_id: Optional[str]
    window_index: Optional[int]
    preset: Optional[int]


@dataclass(frozen=True)
class SessionAuth:
    thread_id: str
    owner: str
    secret: str
    executor: str                  # execution lease token (turns.py)
    turn_id: str                   # the turn the graph State belongs to (a resume keeps its parent's turn_id)
    lease_turn_id: Optional[str] = None   # the turn whose lease this executor holds (defaults to turn_id)

    @property
    def lease_turn(self) -> str:
        return self.lease_turn_id or self.turn_id


_AUTH: ContextVar[Optional[SessionAuth]] = ContextVar("sgp_session_auth", default=None)


def _sha(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def start_session(dsn: str, owner: str, preset: int, episode_id: str, display_window: int) -> tuple[str, str]:
    thread_id = f"s-{uuid.uuid4()}"
    secret = _secrets.token_urlsafe(32)
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO mart.threads (thread_id, owner, episode_id, window_index, preset, owner_secret_sha256)
                       VALUES (%s,%s,%s,%s,%s,%s)""", (thread_id, owner, uuid.UUID(str(episode_id)), int(display_window), int(preset), _sha(secret)))
        conn.commit()
    return thread_id, secret


def verify_secret(dsn: str, thread_id: str, secret: str) -> ThreadRow:
    """DB check of the session secret. Raises AuthError on unknown thread, missing secret or mismatch."""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT owner, episode_id, window_index, preset, owner_secret_sha256 FROM mart.threads WHERE thread_id=%s", (thread_id,))
        r = cur.fetchone()
    if not r or not r[4] or not secret:
        raise AuthError("unknown thread or no session secret registered")
    if not hmac.compare_digest(_sha(secret), r[4]):
        raise AuthError("session secret does not match the thread owner")
    return ThreadRow(thread_id, r[0], (str(r[1]) if r[1] else None), r[2], r[3])


def current_auth() -> SessionAuth:
    a = _AUTH.get()
    if a is None:
        raise AuthError("no verified session in this execution context")
    return a


@contextmanager
def session_scope(auth: SessionAuth) -> Iterator[SessionAuth]:
    token = _AUTH.set(auth)
    try:
        yield auth
    finally:
        _AUTH.reset(token)
