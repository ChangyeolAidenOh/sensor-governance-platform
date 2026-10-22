"""core_pipeline/agent/llm_ledger.py — reserve-before-call / settle-after ledger on mart.agent_steps (ST4-21).

Rows: skill='llm'; tool, stage key and the reservation live in `input` (the table has no tool column); the result lives in
`output`. request_id = uuid5(LLM_NS, thread|run_uuid|stage_key|a<attempt>) — its own namespace, never a Skill request id.

Every path that changes cost or blocking state — reserve, settle, note_no_response, manual settle/void, settle_usage,
resolve_exceeded — runs ONE transaction whose first statement is pg_advisory_xact_lock('sgp_llm_ledger'), so the
"no unresolved blocking row / SUM + reserved ≤ cap" check in reserve cannot interleave with a settlement elsewhere.

reserve(...) returns a Reservation whose status is:
  owned               a new PENDING row with cost_usd = reserved estimate; the caller may make exactly one request
  replay              the same key with the same bound input already finished: output returned, no call
  in_progress         the same key is PENDING (unknown outcome or a concurrent executor): no call, caller falls back
  mismatch            the same key with a DIFFERENT bound input: refused, no call (mirrors REQUEST_ID_REUSED)
  exceeded_unresolved an earlier settlement exceeded its reservation or has unknown usage and nobody resolved it:
                      no new call until it is resolved with evidence (scripts/llm_ledger_resolve.py)
  budget_refused      SGP_LLM_BUDGET_USD unset, or SUM(cost_usd of all llm rows incl. PENDING) + reserved > cap: no row

settle(...) is a conditional UPDATE of the PENDING row at the ACTUAL usage (never truncated to the reservation); the
reservation itself (input.reserve, incl. the FULL tiered price table) is immutable; every settlement — automatic,
manual_settle, settle_usage — prices the actual usage from that stored table (tier chosen by the ACTUAL input tokens),
so a prompt that turns out to be above the tier boundary is settled at the higher rates even though the estimate was not. Two distinct blocking events are kept apart:
  * reservation_exceeded (actual cost known and > reserved): cleared only by resolve_exceeded(evidence) after the
    price table / formula were re-checked;
  * usage_missing (response received, usage not reported): settled at the reserved amount and blocked; cleared only
    by settle_usage(input_tokens, output_tokens, evidence), which records the real cost — if that real cost exceeds the
    reservation, reservation_exceeded is raised as a separate event and still needs resolve_exceeded.
Discovery of an overrun is never its resolution: no path sets exceeded_resolved=true without an explicit resolution.
The ledger never touches checkpoints or mart.graph_turns: a turn result already returned is never changed here.

The reserved amount is a conservative estimate of the input cost plus the max_tokens output bound. It controls the sum
of concurrent reservations; it is NOT a proven upper bound of the vendor's bill.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Protocol
from uuid import UUID

from core_pipeline.agent.budget import LLM_LEDGER_SKILL, budget_usd
from core_pipeline.agent.llm_client import LLMRequest, LLMResponse, cost_from_table, tier_for

LLM_NS = uuid.uuid5(uuid.NAMESPACE_URL, "sgp/agent/llm_ledger")
LOCK_KEY = "sgp_llm_ledger"
RESERVATION_STATUSES = ("owned", "replay", "in_progress", "mismatch", "exceeded_unresolved", "budget_refused")
SQL_LOCK = "SELECT pg_advisory_xact_lock(hashtext(%s))"
SQL_BLOCKED = """SELECT count(*) FROM mart.agent_steps
                 WHERE skill=%s AND status<>'PENDING'
                   AND (coalesce((output->>'usage_missing')::boolean, false)
                        OR (coalesce((output->>'reservation_exceeded')::boolean, false)
                            AND NOT coalesce((output->>'exceeded_resolved')::boolean, false)))"""


def request_id(thread_id: str, run_uuid: str | UUID, stage_key: str, attempt: int) -> UUID:
    return uuid.uuid5(LLM_NS, f"{thread_id}|{run_uuid}|{stage_key}|a{int(attempt)}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(v) -> dict:
    if v is None:
        return {}
    return v if isinstance(v, dict) else json.loads(v)


def _stored_table(inp: dict) -> dict:
    """The price table preserved in the reservation row (never the current global table). Rows written before the
    tiered table (rev2) hold only `prices_per_mtok`: they are settled at that pair, as recorded at the time."""
    res = inp.get("reserve") or {}
    if res.get("price_table"):
        return res["price_table"]
    pair = res.get("prices_per_mtok")
    if not pair or len(pair) != 2:
        raise RuntimeError("reservation row has no preserved price table")
    return {"version": res.get("price_table_version", "legacy"), "boundary_input_tokens": 10**12, "tiers": [list(pair), list(pair)]}


def _stored_cost(inp: dict, input_tokens: int, output_tokens: int) -> float:
    return cost_from_table(_stored_table(inp), input_tokens, output_tokens)


def _check_tokens(input_tokens: int, output_tokens: int) -> None:
    if int(input_tokens) < 0 or int(output_tokens) < 0:
        raise ValueError("token counts must be non-negative")


def _check_evidence(evidence: str) -> None:
    if not (evidence or "").strip():
        raise ValueError("evidence is required")


@dataclass(frozen=True)
class Reservation:
    status: str
    request_id: UUID
    reserved_usd: float = 0.0
    output: Optional[dict] = None             # replay only
    reason: str = ""


class LlmLedger(Protocol):
    def reserve(self, *, thread_id: str, run_uuid: str, stage_key: str, attempt: int, tool: str, req: LLMRequest) -> Reservation: ...
    def settle(self, rid: UUID, resp: Optional[LLMResponse], *, req: LLMRequest, output: dict, violation: bool = False,
               reason_code: Optional[str] = None) -> dict: ...
    def note_no_response(self, rid: UUID, note: str) -> None: ...


def _settlement(req: LLMRequest, resp: Optional[LLMResponse], output: dict, reason_code: Optional[str]) -> tuple[str, float, Optional[int], Optional[int], dict, Optional[str]]:
    """→ (status, cost_usd, input_tokens, output_tokens, output_json, reason_code). resp=None: documented rejection, nothing generated."""
    reserved = req.reserved_usd()
    out = dict(output)
    if resp is None:
        out.update({"cost_usd": 0.0, "reservation_exceeded": False, "usage_missing": False})
        return "ERROR", 0.0, None, None, out, reason_code or "LLM_REQUEST_REJECTED"
    if resp.usage_known:
        table = req.reserve_record()["price_table"]                  # identical to the table stored in the row at reserve time
        actual = cost_from_table(table, resp.input_tokens, resp.output_tokens)
        exceeded = actual > reserved
        calib = {"estimated_input_tokens": req.estimated_input_tokens(), "actual_input_tokens": resp.input_tokens,
                 "input_ratio": (round(resp.input_tokens / req.estimated_input_tokens(), 4) if req.estimated_input_tokens() else None),
                 "actual_output_tokens": resp.output_tokens, "max_tokens": int(req.max_tokens),
                 "reserved_usd": reserved, "actual_usd": actual, "formula_version": req.reserve_record()["formula_version"],
                 "price_tier_estimated": tier_for(table, req.estimated_input_tokens()), "price_tier_actual": tier_for(table, resp.input_tokens)}
        out.update({"text": resp.text, "usage": dict(resp.usage), "stop_reason": resp.stop_reason, "model": resp.model,
                    "cost_usd": actual, "calibration": calib, "reservation_exceeded": exceeded, "usage_missing": False})
        if exceeded:
            out["exceeded_resolved"] = False
        return "OK", actual, resp.input_tokens, resp.output_tokens, out, reason_code
    out.update({"text": resp.text, "usage": dict(resp.usage), "stop_reason": resp.stop_reason, "model": resp.model,
                "cost_usd": reserved, "usage_missing": True, "reservation_exceeded": False})
    return "OK", reserved, None, None, out, reason_code or "LLM_USAGE_MISSING"


# ----------------------------------------------------------------------------------------------------- postgres
class PostgresLedger:
    def __init__(self, dsn: str, mode: str = "chat") -> None:
        self.dsn, self.mode = dsn, mode

    def _input(self, *, thread_id: str, run_uuid: str, stage_key: str, attempt: int, tool: str, req: LLMRequest) -> dict:
        return {"skill": LLM_LEDGER_SKILL, "tool": tool, "stage_key": stage_key, "attempt": int(attempt), "bound": req.bound_input(),
                "reserve": req.reserve_record(), "ctx": {"thread_id": thread_id, "run_uuid": str(run_uuid), "mode": self.mode}}

    def _tx(self):
        import psycopg
        return psycopg.connect(self.dsn)

    def reserve(self, *, thread_id: str, run_uuid: str, stage_key: str, attempt: int, tool: str, req: LLMRequest) -> Reservation:
        rid = request_id(thread_id, run_uuid, stage_key, attempt)
        inp = self._input(thread_id=thread_id, run_uuid=run_uuid, stage_key=stage_key, attempt=attempt, tool=tool, req=req)
        reserved = req.reserved_usd()
        with self._tx() as conn, conn.cursor() as cur:
            cur.execute(SQL_LOCK, (LOCK_KEY,))
            cur.execute("INSERT INTO mart.agent_runs (run_uuid, thread_id, mode) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                        (UUID(str(run_uuid)), thread_id, self.mode))
            cur.execute("SELECT status, input, output FROM mart.agent_steps WHERE request_id=%s AND skill=%s", (rid, LLM_LEDGER_SKILL))
            row = cur.fetchone()
            if row is not None:
                if (_json(row[1]).get("bound") or {}) != inp["bound"]:
                    conn.commit()
                    return Reservation("mismatch", rid, reason="same stage key with a different bound input")
                if row[0] == "PENDING" or row[2] is None:
                    conn.commit()
                    return Reservation("in_progress", rid, reason="reservation pending (unknown outcome or concurrent executor)")
                conn.commit()
                return Reservation("replay", rid, reserved_usd=0.0, output=_json(row[2]))
            cur.execute(SQL_BLOCKED, (LLM_LEDGER_SKILL,))
            if cur.fetchone()[0] > 0:
                conn.commit()
                return Reservation("exceeded_unresolved", rid, reason="an earlier settlement exceeded its reservation or has unknown usage and is unresolved")
            cap = budget_usd()
            if cap is None:
                conn.commit()
                return Reservation("budget_refused", rid, reason="SGP_LLM_BUDGET_USD is not set: paid LLM calls are refused (ST4-06)")
            cur.execute("SELECT coalesce(sum(cost_usd), 0) FROM mart.agent_steps WHERE skill=%s", (LLM_LEDGER_SKILL,))
            spent = float(cur.fetchone()[0])
            if spent + reserved > cap:
                conn.commit()
                return Reservation("budget_refused", rid, reason=f"spent {spent:.6f} + reserved {reserved:.6f} > cap {cap:.6f} USD")
            cur.execute("SELECT coalesce(max(seq),0)+1 FROM mart.agent_steps WHERE run_uuid=%s", (UUID(str(run_uuid)),))
            seq = cur.fetchone()[0]
            cur.execute("""INSERT INTO mart.agent_steps (run_uuid, request_id, seq, skill, status, input, cost_usd)
                           VALUES (%s,%s,%s,%s,'PENDING',%s,%s)""",
                        (UUID(str(run_uuid)), rid, seq, LLM_LEDGER_SKILL, json.dumps(inp, ensure_ascii=False), reserved))
            conn.commit()
        return Reservation("owned", rid, reserved_usd=reserved)

    def settle(self, rid: UUID, resp: Optional[LLMResponse], *, req: LLMRequest, output: dict, violation: bool = False,
               reason_code: Optional[str] = None) -> dict:
        status, cost, itok, otok, out, code = _settlement(req, resp, output, reason_code)
        out["settled_at"] = _now()
        with self._tx() as conn, conn.cursor() as cur:
            cur.execute(SQL_LOCK, (LOCK_KEY,))
            cur.execute("""UPDATE mart.agent_steps SET status=%s, reason_code=%s, output=%s, latency_ms=%s, input_tokens=%s,
                                  output_tokens=%s, cost_usd=%s, violation=%s
                           WHERE request_id=%s AND status='PENDING' AND skill=%s""",
                        (status, code, json.dumps(out, ensure_ascii=False), (resp.latency_ms if resp else None), itok, otok, cost,
                         bool(violation), rid, LLM_LEDGER_SKILL))
            if cur.rowcount != 1:
                raise RuntimeError("INTERNAL: llm ledger row is not PENDING any more (lost ownership)")
            conn.commit()
        return out

    def note_no_response(self, rid: UUID, note: str) -> None:
        """Audit note on a PENDING row that STAYS PENDING: no usable response, billing unknown."""
        with self._tx() as conn, conn.cursor() as cur:
            cur.execute(SQL_LOCK, (LOCK_KEY,))
            cur.execute("""UPDATE mart.agent_steps SET output = coalesce(output, '{}'::jsonb) || %s::jsonb
                           WHERE request_id=%s AND status='PENDING' AND skill=%s""",
                        (json.dumps({"no_response": {"note": note, "at": _now()}}, ensure_ascii=False), rid, LLM_LEDGER_SKILL))
            conn.commit()

    # ---- manual resolution (scripts/llm_ledger_resolve.py): conditional updates with evidence, never a re-call
    def manual_settle(self, rid: UUID, input_tokens: int, output_tokens: int, *, evidence: str, by: str) -> dict:
        """PENDING (lost response) → OK at the real usage proven by evidence. An overrun found here is recorded, not resolved."""
        _check_evidence(evidence)
        _check_tokens(input_tokens, output_tokens)
        with self._tx() as conn, conn.cursor() as cur:
            cur.execute(SQL_LOCK, (LOCK_KEY,))
            cur.execute("SELECT input, output FROM mart.agent_steps WHERE request_id=%s AND status='PENDING' AND skill=%s FOR UPDATE", (rid, LLM_LEDGER_SKILL))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("no PENDING llm row for this request_id")
            inp = _json(row[0])
            reserved = float(inp["reserve"]["reserved_usd"])
            actual = _stored_cost(inp, input_tokens, output_tokens)
            out = _json(row[1])
            exceeded = actual > reserved
            out.update({"cost_usd": actual, "usage_missing": False, "reservation_exceeded": exceeded,
                        "resolution": {"kind": "manual_settle", "by": by, "at": _now(), "evidence": evidence,
                                       "input_tokens": int(input_tokens), "output_tokens": int(output_tokens)}})
            if exceeded:
                out["exceeded_resolved"] = False
            cur.execute("""UPDATE mart.agent_steps SET status='OK', reason_code='LLM_MANUAL_SETTLE', output=%s, input_tokens=%s,
                                  output_tokens=%s, cost_usd=%s WHERE request_id=%s AND status='PENDING' AND skill=%s""",
                        (json.dumps(out, ensure_ascii=False), int(input_tokens), int(output_tokens), actual, rid, LLM_LEDGER_SKILL))
            conn.commit()
        return out

    def manual_void(self, rid: UUID, *, evidence: str, by: str) -> dict:
        """PENDING → ERROR at cost 0, only with evidence that the vendor never processed the request."""
        _check_evidence(evidence)
        with self._tx() as conn, conn.cursor() as cur:
            cur.execute(SQL_LOCK, (LOCK_KEY,))
            cur.execute("SELECT output FROM mart.agent_steps WHERE request_id=%s AND status='PENDING' AND skill=%s FOR UPDATE", (rid, LLM_LEDGER_SKILL))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("no PENDING llm row for this request_id")
            out = _json(row[0])
            out.update({"cost_usd": 0.0, "reservation_exceeded": False, "usage_missing": False,
                        "resolution": {"kind": "manual_void", "by": by, "at": _now(), "evidence": evidence}})
            cur.execute("""UPDATE mart.agent_steps SET status='ERROR', reason_code='LLM_VOID', output=%s, cost_usd=0
                           WHERE request_id=%s AND status='PENDING' AND skill=%s""", (json.dumps(out, ensure_ascii=False), rid, LLM_LEDGER_SKILL))
            conn.commit()
        return out

    def settle_usage(self, rid: UUID, input_tokens: int, output_tokens: int, *, evidence: str, by: str) -> dict:
        """A settled row with usage_missing=true gets its real usage (vendor console) recorded; this clears usage_missing
        and raises reservation_exceeded as a SEPARATE event if the real cost exceeds the reservation."""
        _check_evidence(evidence)
        _check_tokens(input_tokens, output_tokens)
        with self._tx() as conn, conn.cursor() as cur:
            cur.execute(SQL_LOCK, (LOCK_KEY,))
            cur.execute("""SELECT input, output FROM mart.agent_steps WHERE request_id=%s AND skill=%s AND status<>'PENDING'
                           AND coalesce((output->>'usage_missing')::boolean, false) FOR UPDATE""", (rid, LLM_LEDGER_SKILL))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("no settled llm row with usage_missing=true for this request_id")
            inp, out = _json(row[0]), _json(row[1])
            reserved = float(inp["reserve"]["reserved_usd"])
            actual = _stored_cost(inp, input_tokens, output_tokens)
            exceeded = actual > reserved
            out.update({"cost_usd": actual, "usage_missing": False, "reservation_exceeded": exceeded,
                        "usage_resolution": {"by": by, "at": _now(), "evidence": evidence, "input_tokens": int(input_tokens), "output_tokens": int(output_tokens)}})
            if exceeded:
                out["exceeded_resolved"] = False
            cur.execute("""UPDATE mart.agent_steps SET output=%s, input_tokens=%s, output_tokens=%s, cost_usd=%s
                           WHERE request_id=%s AND skill=%s""", (json.dumps(out, ensure_ascii=False), int(input_tokens), int(output_tokens), actual, rid, LLM_LEDGER_SKILL))
            conn.commit()
        return out

    def resolve_exceeded(self, rid: UUID, *, evidence: str, by: str) -> dict:
        """Clears a KNOWN overrun (actual cost recorded) after the price table / formula were re-checked. Refuses rows
        whose usage is still missing: record the usage first (settle_usage)."""
        _check_evidence(evidence)
        with self._tx() as conn, conn.cursor() as cur:
            cur.execute(SQL_LOCK, (LOCK_KEY,))
            cur.execute("""SELECT output FROM mart.agent_steps WHERE request_id=%s AND skill=%s AND status<>'PENDING'
                           AND coalesce((output->>'reservation_exceeded')::boolean, false)
                           AND NOT coalesce((output->>'exceeded_resolved')::boolean, false) FOR UPDATE""", (rid, LLM_LEDGER_SKILL))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("no unresolved exceeded settlement for this request_id")
            out = _json(row[0])
            if out.get("usage_missing"):
                raise RuntimeError("usage is still missing for this row: record it with settle_usage before resolving the overrun")
            out.update({"exceeded_resolved": True, "exceeded_resolution": {"by": by, "at": _now(), "evidence": evidence}})
            cur.execute("UPDATE mart.agent_steps SET output=%s WHERE request_id=%s AND skill=%s", (json.dumps(out, ensure_ascii=False), rid, LLM_LEDGER_SKILL))
            conn.commit()
        return out

    def list_rows(self, *, pending_only: bool = False, blocked_only: bool = False) -> list[dict]:
        where = ["skill=%s"]
        if pending_only:
            where.append("status='PENDING'")
        if blocked_only:
            where.append("""status<>'PENDING' AND (coalesce((output->>'usage_missing')::boolean, false)
                            OR (coalesce((output->>'reservation_exceeded')::boolean, false) AND NOT coalesce((output->>'exceeded_resolved')::boolean, false)))""")
        with self._tx() as conn, conn.cursor() as cur:
            cur.execute(f"""SELECT request_id, status, reason_code, cost_usd, input_tokens, output_tokens, input, output, created_at
                            FROM mart.agent_steps WHERE {' AND '.join(where)} ORDER BY created_at""", (LLM_LEDGER_SKILL,))
            rows = []
            for r in cur.fetchall():
                inp, out = _json(r[6]), _json(r[7])
                rows.append({"request_id": str(r[0]), "status": r[1], "reason_code": r[2], "cost_usd": float(r[3] or 0),
                             "input_tokens": r[4], "output_tokens": r[5], "tool": inp.get("tool"), "stage_key": inp.get("stage_key"),
                             "reserved_usd": (inp.get("reserve") or {}).get("reserved_usd"), "usage_missing": out.get("usage_missing"),
                             "reservation_exceeded": out.get("reservation_exceeded"), "exceeded_resolved": out.get("exceeded_resolved"),
                             "output": out, "created_at": str(r[8])})
            return rows

    def totals(self) -> dict:
        with self._tx() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*), coalesce(sum(cost_usd),0), count(*) FILTER (WHERE status='PENDING') FROM mart.agent_steps WHERE skill=%s",
                        (LLM_LEDGER_SKILL,))
            n, s, p = cur.fetchone()
            cur.execute(SQL_BLOCKED, (LLM_LEDGER_SKILL,))
            x = cur.fetchone()[0]
            return {"rows": n, "spent_usd": float(s), "pending": p, "blocked_unresolved": x, "cap_usd": budget_usd()}


# ----------------------------------------------------------------------------------------------------- in-memory
@dataclass
class _Row:
    status: str
    input: dict
    output: Optional[dict] = None
    cost_usd: float = 0.0
    violation: bool = False
    reason_code: Optional[str] = None


def _blocked(r: _Row) -> bool:
    o = r.output or {}
    return r.status != "PENDING" and (bool(o.get("usage_missing")) or (bool(o.get("reservation_exceeded")) and not o.get("exceeded_resolved")))


@dataclass
class MemoryLedger:
    """Same decision procedure as PostgresLedger (single process; no lock needed) for unit tests and dry runs.
    cap=None reproduces 'SGP_LLM_BUDGET_USD unset'."""
    cap: Optional[float] = None
    mode: str = "chat"
    rows: dict[UUID, _Row] = field(default_factory=dict)

    def reserve(self, *, thread_id: str, run_uuid: str, stage_key: str, attempt: int, tool: str, req: LLMRequest) -> Reservation:
        rid = request_id(thread_id, run_uuid, stage_key, attempt)
        inp = {"skill": LLM_LEDGER_SKILL, "tool": tool, "stage_key": stage_key, "attempt": int(attempt), "bound": req.bound_input(),
               "reserve": req.reserve_record(), "ctx": {"thread_id": thread_id, "run_uuid": str(run_uuid), "mode": self.mode}}
        reserved = req.reserved_usd()
        row = self.rows.get(rid)
        if row is not None:
            if row.input["bound"] != inp["bound"]:
                return Reservation("mismatch", rid, reason="same stage key with a different bound input")
            if row.status == "PENDING" or row.output is None:
                return Reservation("in_progress", rid, reason="reservation pending")
            return Reservation("replay", rid, output=row.output)
        if any(_blocked(r) for r in self.rows.values()):
            return Reservation("exceeded_unresolved", rid, reason="unresolved exceeded / usage-missing settlement")
        if self.cap is None:
            return Reservation("budget_refused", rid, reason="SGP_LLM_BUDGET_USD is not set")
        spent = sum(r.cost_usd for r in self.rows.values())
        if spent + reserved > self.cap:
            return Reservation("budget_refused", rid, reason=f"spent {spent:.6f} + reserved {reserved:.6f} > cap {self.cap:.6f}")
        self.rows[rid] = _Row("PENDING", inp, cost_usd=reserved)
        return Reservation("owned", rid, reserved_usd=reserved)

    def settle(self, rid: UUID, resp: Optional[LLMResponse], *, req: LLMRequest, output: dict, violation: bool = False,
               reason_code: Optional[str] = None) -> dict:
        row = self.rows[rid]
        if row.status != "PENDING":
            raise RuntimeError("INTERNAL: llm ledger row is not PENDING any more")
        status, cost, _, _, out, code = _settlement(req, resp, output, reason_code)
        out["settled_at"] = _now()
        row.status, row.output, row.cost_usd, row.violation, row.reason_code = status, out, cost, bool(violation), code
        return out

    def note_no_response(self, rid: UUID, note: str) -> None:
        row = self.rows[rid]
        if row.status == "PENDING":
            row.output = {**(row.output or {}), "no_response": {"note": note, "at": _now()}}

    def settle_usage(self, rid: UUID, input_tokens: int, output_tokens: int, *, evidence: str, by: str) -> dict:
        _check_evidence(evidence)
        _check_tokens(input_tokens, output_tokens)
        row = self.rows[rid]
        if row.status == "PENDING" or not (row.output or {}).get("usage_missing"):
            raise RuntimeError("no settled llm row with usage_missing=true for this request_id")
        reserved = float(row.input["reserve"]["reserved_usd"])
        actual = _stored_cost(row.input, input_tokens, output_tokens)
        row.output = {**row.output, "cost_usd": actual, "usage_missing": False, "reservation_exceeded": actual > reserved,
                      "usage_resolution": {"by": by, "at": _now(), "evidence": evidence}}
        if actual > reserved:
            row.output["exceeded_resolved"] = False
        row.cost_usd = actual
        return row.output

    def resolve_exceeded(self, rid: UUID, *, evidence: str, by: str) -> dict:
        _check_evidence(evidence)
        row = self.rows[rid]
        o = row.output or {}
        if row.status == "PENDING" or not o.get("reservation_exceeded") or o.get("exceeded_resolved"):
            raise RuntimeError("no unresolved exceeded settlement for this request_id")
        if o.get("usage_missing"):
            raise RuntimeError("usage is still missing for this row: record it with settle_usage before resolving the overrun")
        row.output = {**o, "exceeded_resolved": True, "exceeded_resolution": {"by": by, "at": _now(), "evidence": evidence}}
        return row.output

    def spent_usd(self) -> float:
        return sum(r.cost_usd for r in self.rows.values())
