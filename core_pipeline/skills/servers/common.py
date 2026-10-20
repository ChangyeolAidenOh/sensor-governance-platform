"""core_pipeline/skills/servers/common.py — shared plumbing for the two FastMCP Skill servers (Stage 4a, S4a-00/01).

Skill contract v1.0 (docs/skill_contract.md):
  * one envelope for every tool: {status, payload, grounding, state, reason, trace}
    status ∈ {OK, REFUSED, ERROR}; grounding only from retrieve; state = the COMMITTED mart.episode_windows row
  * every call carries a Ctx (request_id, thread_id, run_uuid, caller_role, mode) and is written to mart.agent_steps
    (input, output, latency, violation flag); a repeated request_id with the same input replays the stored output,
    a repeated request_id with a different input is refused (REQUEST_ID_REUSED)
  * stdio transport: NOTHING may be written to stdout by a server process (it is the MCP channel) — log to stderr.

No torch / lightgbm import here: this module is imported by both servers and by the client.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import psycopg

SKILL_CONTRACT_VERSION = "1.0"
STATUSES = ("OK", "REFUSED", "ERROR")
ROLES = ("engineer", "restricted")
MODES = ("chat", "event", "replay")

REASON_CODES = (
    "GATE_BLOCK", "GATE_NOT_RUN", "DETECT_NOT_RUN", "WINDOW_ORDER", "WINDOW_OUT_OF_RANGE", "DATA_MISSING",
    "NO_EXPANSION", "EXPANSION_INVALID", "QUERY_USED", "ARG_NOT_IN_STATE", "ACL_DENIED", "NO_ALARM",
    "EVENT_MISMATCH", "DECISION_MISMATCH", "NO_PROPOSE_DECISION", "STALE_DECISION", "NO_EVIDENCE",
    "FORBIDDEN_ACTION", "THREAD_MISMATCH", "SUPERSEDED", "CONTENT_MISMATCH", "APPROVAL_MISMATCH", "TOKEN_USED",
    "NO_APPROVAL", "UNKNOWN_PROPOSAL", "MODEL_UNAVAILABLE", "INDEX_MISMATCH", "CMMS_DOWN", "DEMO_LIMIT",
    "BAD_REQUEST", "SCHEMA_INVALID", "REQUEST_ID_REUSED", "REQUEST_IN_PROGRESS", "QUERY_REQUIRED", "APPROVER_MISMATCH", "INTERNAL",
)
PENDING_STALE_SECONDS = float(os.environ.get("SGP_REQUEST_STALE_SECONDS", "300"))

logging.basicConfig(stream=sys.stderr, level=os.environ.get("SGP_LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("sgp.skills")


def env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    v = os.environ.get(name, default)
    if required and not v:
        raise RuntimeError(f"environment variable {name} is required")
    return v


def is_test_dsn(dsn: str) -> bool:
    from urllib.parse import urlparse
    name = (urlparse(dsn).path or "").lstrip("/").split("?")[0] if "://" in dsn else dsn
    return "test" in name.lower()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()


def jsonable(obj: Any) -> Any:
    """Make dataclasses / numpy / UUID / Path JSON-serialisable (recursively)."""
    import dataclasses
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (UUID, Path)):
        return str(obj)
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):     # numpy scalar
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    return obj


# ----------------------------------------------------------------------------------------------------- context
@dataclass(frozen=True)
class Ctx:
    request_id: UUID
    thread_id: str
    run_uuid: UUID
    caller_role: str
    mode: str = "chat"

    @classmethod
    def parse(cls, d: dict) -> "Ctx":
        try:
            rid, run = UUID(str(d["request_id"])), UUID(str(d["run_uuid"]))
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"BAD_REQUEST: ctx.request_id / ctx.run_uuid must be UUIDs ({e})")
        role, mode = str(d.get("caller_role", "")), str(d.get("mode", "chat"))
        if role not in ROLES:
            raise ValueError(f"ACL_DENIED: unknown caller_role {role!r}")
        if mode not in MODES:
            raise ValueError(f"BAD_REQUEST: mode must be one of {MODES}")
        tid = str(d.get("thread_id", ""))
        if not tid:
            raise ValueError("BAD_REQUEST: ctx.thread_id is required")
        return cls(rid, tid, run, role, mode)

    def as_dict(self) -> dict:
        return {"request_id": str(self.request_id), "thread_id": self.thread_id, "run_uuid": str(self.run_uuid),
                "caller_role": self.caller_role, "mode": self.mode}


# ----------------------------------------------------------------------------------------------------- envelope
def envelope(status: str, *, payload: Optional[dict] = None, reason: Optional[tuple[str, str]] = None,
             grounding: Optional[dict] = None, state: Optional[dict] = None, trace: Optional[dict] = None) -> dict:
    assert status in STATUSES, status
    if status != "OK":
        assert reason is not None and reason[0] in REASON_CODES, reason
    return {
        "status": status,
        "payload": jsonable(payload) if payload is not None else None,
        "grounding": jsonable(grounding) if grounding is not None else None,
        "state": jsonable(state) if state is not None else None,
        "reason": {"code": reason[0], "message": reason[1]} if reason else None,
        "trace": jsonable(trace or {}),
    }


def code_of(exc: BaseException, default: str = "INTERNAL") -> tuple[str, str]:
    """Skill modules raise PermissionError / ValueError with messages 'CODE: detail'. Map to (code, message)."""
    msg = str(exc)
    head = msg.split(":", 1)[0].strip()
    if head in REASON_CODES:
        return head, msg
    # work_order.draft policy refusal and other non-coded messages
    if isinstance(exc, PermissionError):
        return "FORBIDDEN_ACTION" if "action" in msg.lower() else "BAD_REQUEST", msg
    if isinstance(exc, ValueError):
        if "out of order" in msg:
            return "WINDOW_ORDER", msg
        return "BAD_REQUEST", msg
    return default, msg


# ----------------------------------------------------------------------------------------------------- step log
class StepLog:
    """mart.agent_runs / mart.agent_steps writer with REQUEST OWNERSHIP (ST4-13, reviewer 2026-10-21 point 2):
      claim   — INSERT the request as PENDING before any work (ON CONFLICT DO NOTHING). The inserting process owns the
                execution; a concurrent duplicate sees the PENDING row and is refused (REQUEST_IN_PROGRESS) instead of
                executing the side effects a second time.
      replay  — a completed row with the SAME bound input (ctx incl. thread/role/mode + skill + tool + contract + args)
                is returned as-is; the same request_id with a different bound input is REQUEST_ID_REUSED.
      finish  — the FINAL envelope (after output validation) is written; if that write fails the caller gets
                ERROR(INTERNAL) and the row stays PENDING until it goes stale (PENDING_STALE_SECONDS) and is reclaimed.
    """

    def __init__(self, dsn: str, skill: str, versions: dict):
        self.dsn, self.skill, self.versions = dsn, skill, versions

    def bound_input(self, ctx: Ctx, tool: str, inp: dict) -> dict:
        return {"ctx": ctx.as_dict(), "skill": self.skill, "tool": tool, "contract_version": SKILL_CONTRACT_VERSION, "args": jsonable(inp)}

    def claim(self, ctx: Ctx, bound: dict) -> tuple[str, Optional[dict]]:
        """→ ('owned', None) | ('replay', envelope) | ('reused', None) | ('in_progress', None)"""
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO mart.agent_runs (run_uuid, thread_id, mode) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                        (ctx.run_uuid, ctx.thread_id, ctx.mode))
            cur.execute("SELECT coalesce(max(seq),0)+1 FROM mart.agent_steps WHERE run_uuid=%s", (ctx.run_uuid,))
            seq = cur.fetchone()[0]
            cur.execute("""INSERT INTO mart.agent_steps (run_uuid, request_id, seq, skill, status, input)
                           VALUES (%s,%s,%s,%s,'PENDING',%s) ON CONFLICT (request_id) DO NOTHING RETURNING step_id""",
                        (ctx.run_uuid, ctx.request_id, seq, self.skill, json.dumps(bound, ensure_ascii=False)))
            if cur.fetchone():
                conn.commit()
                return "owned", None
            cur.execute("SELECT input, output, status, created_at < now() - make_interval(secs => %s) FROM mart.agent_steps WHERE request_id=%s",
                        (PENDING_STALE_SECONDS, ctx.request_id))
            row = cur.fetchone()
            stored_in = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            if sha256_json(stored_in) != sha256_json(bound):
                conn.commit()
                return "reused", None
            if row[2] != "PENDING" and row[1] is not None:
                out = row[1] if isinstance(row[1], dict) else json.loads(row[1])
                out.setdefault("trace", {})["replayed"] = True
                conn.commit()
                return "replay", out
            if row[3]:                                               # stale PENDING: reclaim the execution
                cur.execute("UPDATE mart.agent_steps SET created_at=now() WHERE request_id=%s AND status='PENDING'", (ctx.request_id,))
                conn.commit()
                return "owned", None
            conn.commit()
            return "in_progress", None

    def trace(self, ctx: Ctx, latency_ms: int, replayed: bool = False, **extra) -> dict:
        t = {"request_id": str(ctx.request_id), "thread_id": ctx.thread_id, "run_uuid": str(ctx.run_uuid),
             "skill": self.skill, "contract_version": SKILL_CONTRACT_VERSION, "latency_ms": int(latency_ms),
             "versions": self.versions, "replayed": replayed}
        t.update(extra)
        return t

    def finish(self, ctx: Ctx, out: dict, latency_ms: int, violation: bool) -> None:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("""UPDATE mart.agent_steps SET status=%s, reason_code=%s, output=%s, latency_ms=%s, violation=%s
                           WHERE request_id=%s AND status='PENDING'""",
                        (out["status"], (out.get("reason") or {}).get("code"), json.dumps(out, ensure_ascii=False), int(latency_ms), violation, ctx.request_id))
            if cur.rowcount != 1:
                raise RuntimeError("INTERNAL: request row is not PENDING any more (lost ownership)")
            conn.commit()


def run_tool(steplog: StepLog, tool: str, raw_ctx: dict, inp: dict, fn, violation_codes: tuple[str, ...] = (),
             validate=None) -> dict:
    """Common wrapper: parse ctx → claim request → fn(ctx) → envelope → validate → finish. `validate(out) -> [errors]`
    runs on the FINAL envelope before it is stored, so a replay returns exactly what the caller got."""
    t0 = time.perf_counter()
    base_trace = {"skill": steplog.skill, "tool": tool, "contract_version": SKILL_CONTRACT_VERSION}
    try:
        ctx = Ctx.parse(raw_ctx)
    except ValueError as e:
        code, msg = code_of(e, "BAD_REQUEST")
        return envelope("REFUSED" if code == "ACL_DENIED" else "ERROR", reason=(code, msg), trace=base_trace)
    bound = steplog.bound_input(ctx, tool, inp)
    try:
        kind, prior = steplog.claim(ctx, bound)
    except Exception as e:
        log.exception("claim failed")
        return envelope("ERROR", reason=("INTERNAL", f"request could not be recorded: {e}"), trace=base_trace)
    if kind == "replay":
        return prior
    if kind == "reused":
        return envelope("REFUSED", reason=("REQUEST_ID_REUSED", "request_id already used with a different input/context"),
                        trace=steplog.trace(ctx, 0, tool=tool))
    if kind == "in_progress":
        return envelope("REFUSED", reason=("REQUEST_IN_PROGRESS", "the same request is being executed; retry later"),
                        trace=steplog.trace(ctx, 0, tool=tool))
    try:
        out = fn(ctx)
    except Exception as e:                                   # never let a stack trace cross the MCP boundary silently
        log.exception("%s failed", tool)
        code, msg = code_of(e)
        out = envelope("ERROR" if code in ("INTERNAL", "MODEL_UNAVAILABLE", "DATA_MISSING", "CMMS_DOWN", "INDEX_MISMATCH") else "REFUSED",
                       reason=(code, msg))
    latency = int((time.perf_counter() - t0) * 1000)
    out["trace"] = steplog.trace(ctx, latency, tool=tool)
    if validate is not None:
        errs = validate(out)
        if errs:
            log.error("output schema violation in %s: %s", tool, errs)
            out = envelope("ERROR", reason=("SCHEMA_INVALID", "; ".join(errs)[:800]), trace=out["trace"])
    violation = out["status"] != "OK" and (out.get("reason") or {}).get("code") in violation_codes
    try:
        steplog.finish(ctx, out, latency, violation)
    except Exception as e:
        log.exception("agent_steps finish failed")
        return envelope("ERROR", reason=("INTERNAL", f"result not durably recorded ({e}); retry with the same request_id"),
                        trace=out["trace"])
    return out


# ----------------------------------------------------------------------------------------------------- DB helpers
def alarm_id_of(episode_id: UUID, window_index: int, detector_version: str) -> str:
    return f"{episode_id}:{window_index}:{detector_version}"


def window_bounds(dsn: str, episode_id: UUID, window_index: int) -> Optional[tuple[int, int, str, Optional[int]]]:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute('SELECT start_cycle, end_cycle, "group", demo_preset FROM mart.v_windows_agent WHERE episode_id=%s AND window_index=%s',
                    (episode_id, window_index))
        return cur.fetchone()


def engine_id_of(dsn: str, episode_id: UUID) -> Optional[int]:
    """Loader-side key map (staging). The detect server is a trusted backend process; the LLM never sees engine_id."""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT engine_id FROM staging.episode_keys WHERE episode_id=%s", (episode_id,))
        row = cur.fetchone()
        return row[0] if row else None


def new_ctx(thread_id: str, run_uuid: Optional[UUID] = None, caller_role: str = "engineer", mode: str = "chat") -> dict:
    return {"request_id": str(uuid.uuid4()), "thread_id": thread_id, "run_uuid": str(run_uuid or uuid.uuid4()),
            "caller_role": caller_role, "mode": mode}
