"""S4a-05 minimum LLM smoke — one chat turn, at most TWO real HTTP calls (router 1 + narrate 1).

    python scripts/llm_smoke_min.py --preset 1                 # DRY RUN ($0): FakeLLM + MemoryLedger through the real graph
    python scripts/llm_smoke_min.py --preset 1 --paid          # REAL: AnthropicClient + PostgresLedger on SGP_PG_DSN

Dry run: runs the same turn on a separate thread (NOT auto-deleted: thread, checkpoints and Skill rows stay in the
DB) with scripted LLM output (router → "alarm"), captures the two FINAL prompts and prints their reservation estimates
(the narrate figure assumes the LLM classifies the message as alarm). With --paid the dry-run report is printed and
flushed BEFORE any real call.
Manifest: the validated demo_presets.json from S4a-03 (--manifest; searched in the usual places) — mart.alarms holds
alarm rows only, so a no-alarm preset cannot be rebuilt from it.
Completion (exit 0) means BOTH features were verified on this run: router row OK + usage known + output validated,
narrate row OK + usage known + auxiliary checks passed. Cap respected but something missing (e.g. the LLM chose a
category whose narration is fixed, so no narrate call) → "incomplete" (exit 5). Cap violated → exit 6.
Paid run: new thread, LLMRouter/LLMNarrator with max_attempts=1, a counting client that refuses a third call, mode
"llm" (narrate only on the LLM-routed turn). Why a first-call failure never triggers the second call:
  * invalid router output → rule result with matched=llm_fallback → the "llm" narrator makes no call
  * usage missing / actual > reserved on the first settlement → the second reserve() returns exceeded_unresolved
  * no response → the row stays PENDING; the narrator's reserve is a different key, so the ledger would allow it —
    the counting client therefore refuses any call after a first-call failure WITHOUT sending it (settled at cost 0).
Prints: client call count, ledger rows created by this run (reserved vs actual, tokens, tiers, calibration), totals
before/after, the turn result. Never resolves or voids anything.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv

from core_pipeline.agent import llm_narrator as LN
from core_pipeline.agent import llm_router as LR
from core_pipeline.agent.graph import GraphDeps
from core_pipeline.agent.llm_client import DEFAULT_MODEL, FakeLLMClient, LLMOutcomeUnknown, LLMRequest, LLMRequestRejected, LLMResponse
from core_pipeline.agent.llm_ledger import MemoryLedger, PostgresLedger
from core_pipeline.agent.llm_narrator import LLMNarrator
from core_pipeline.agent.llm_router import LLMRouter
from core_pipeline.agent.router import RuleRouter
from core_pipeline.agent.runner import AgentSession
from core_pipeline.agent.session import start_session
from core_pipeline.agent.turns import TurnStore
from core_pipeline.skills.client import SyncSkillClient

import psycopg

DEFAULT_MESSAGE = "지금 어떤 상황이야?"
MAX_REAL_CALLS = 2


class CountingClient:
    """Wraps the real client: counts calls, refuses the (MAX_REAL_CALLS+1)-th, and stops after a first-call failure."""

    def __init__(self, inner) -> None:
        self.inner, self.calls, self.failed = inner, [], False

    def complete(self, req: LLMRequest) -> LLMResponse:
        # Refusing here sends NO request, so cost 0 is certain: LLMRequestRejected settles the already-made reservation at 0.
        if len(self.calls) >= MAX_REAL_CALLS:
            raise LLMRequestRejected(f"smoke cap: {MAX_REAL_CALLS} real calls already made; request not sent")
        if self.failed:
            raise LLMRequestRejected("smoke stop: an earlier call failed; request not sent")
        self.calls.append({"max_tokens": req.max_tokens, "estimated_input_tokens": req.estimated_input_tokens(), "reserved_usd": req.reserved_usd()})
        try:
            return self.inner.complete(req)
        except (LLMOutcomeUnknown, LLMRequestRejected):
            self.failed = True
            raise


MANIFEST_CANDIDATES = ("outputs/demo_presets.json", "data/demo_presets.json", "demo_presets.json")


def load_manifest(path: str | None, preset: int) -> dict:
    cands = [path] if path else list(MANIFEST_CANDIDATES)
    for c in cands:
        if c and os.path.exists(c):
            m = json.load(open(c, encoding="utf-8"))
            entries = [e for e in m.get("presets", []) if int(e.get("preset", -1)) == preset]
            if not entries:
                raise SystemExit(f"preset {preset} not in {c}")
            return {"schema": m.get("schema", "demo_presets/1"), "presets": entries, "_path": c}
    raise SystemExit(f"demo_presets.json not found (tried {cands}); pass --manifest <path>")


def run_turn(dsn: str, manifest: dict, preset: int, message: str, router, narrator, owner: str) -> dict:
    from langgraph.checkpoint.postgres import PostgresSaver
    entry = manifest["presets"][0]
    env = {"SGP_PG_DSN": dsn}
    with SyncSkillClient(env=env) as sk:
        deps = GraphDeps(dsn=dsn, skills=sk, turns=TurnStore(dsn), manifest=manifest, router=router, narrator=narrator)
        with PostgresSaver.from_conn_string(dsn) as cp:
            cp.setup()
            tid, secret = start_session(dsn, owner, preset, entry["episode_id"], entry["display_window"])
            t0 = time.perf_counter()
            out = AgentSession(deps, cp).invoke(tid, secret, message, str(uuid.uuid4()))
            return {"thread_id": tid, "elapsed_s": round(time.perf_counter() - t0, 2), "result": out}


ANSWER_KEYS = ("answer", "narration", "text", "message", "reply")


def rows_for_thread(dsn: str, thread_id: str) -> list[dict]:
    """LLM ledger rows of exactly this thread (input.ctx.thread_id), not a time window."""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("""SELECT request_id, status, reason_code, cost_usd, input_tokens, output_tokens, violation, input, output, created_at
                       FROM mart.agent_steps WHERE skill='llm' AND input->'ctx'->>'thread_id' = %s ORDER BY created_at""", (thread_id,))
        rows = []
        for r in cur.fetchall():
            inp = r[7] if isinstance(r[7], dict) else json.loads(r[7])
            out = (r[8] if isinstance(r[8], dict) else json.loads(r[8])) if r[8] else {}
            rows.append({"request_id": str(r[0]), "tool": inp.get("tool"), "stage_key": inp.get("stage_key"), "status": r[1], "reason_code": r[2],
                         "reserved_usd": (inp.get("reserve") or {}).get("reserved_usd"), "cost_usd": float(r[3] or 0),
                         "input_tokens": r[4], "output_tokens": r[5], "violation": r[6],
                         "usage_missing": out.get("usage_missing"), "reservation_exceeded": out.get("reservation_exceeded"),
                         "calibration": out.get("calibration"), "validation_errors": out.get("validation_errors"), "check_errors": out.get("check_errors"),
                         "fallback_reason": out.get("fallback_reason"), "no_response": out.get("no_response"),
                         "has_merged": bool(out.get("merged")), "has_prose": bool(out.get("prose")), "created_at": str(r[9])})
        return rows


def answer_of(result: dict) -> tuple[str | None, str | None]:
    for k in ANSWER_KEYS:
        if isinstance((result or {}).get(k), str):
            return k, result[k]
    return None, None


def without_secrets(d: dict) -> dict:
    return {k: v for k, v in (d or {}).items() if "secret" not in k.lower()}


def emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str), flush=True)


def brief(out: dict) -> dict:
    keep = {}
    for k, v in (out or {}).items():
        if isinstance(v, (int, float, bool, type(None))) or (isinstance(v, str) and len(v) < 1500):
            keep[k] = v
        else:
            keep[k] = str(v)[:400] + "…"
    return keep


def main() -> int:
    load_dotenv(".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", type=int, default=1)
    ap.add_argument("--message", default=DEFAULT_MESSAGE)
    ap.add_argument("--paid", action="store_true", help="make real calls (needs a separate approval; default is the $0 dry run)")
    ap.add_argument("--dsn", default=os.environ.get("SGP_PG_DSN"))
    ap.add_argument("--owner", default="smoke")
    ap.add_argument("--manifest", default=None, help="validated demo_presets.json (default: searched in outputs/, data/, .)")
    ap.add_argument("--save", default=None, help="write the full report (answer in full, no secrets) to this JSON path")
    a = ap.parse_args()
    if not a.dsn:
        print("SGP_PG_DSN is required", file=sys.stderr)
        return 2

    rr = RuleRouter().route(a.message, {})
    if rr.matched != "default":
        print(json.dumps({"abort": "message is decided by RuleRouter (no LLM would be called)", "matched": rr.matched, "category": rr.category}, ensure_ascii=False))
        return 3
    manifest = load_manifest(a.manifest, a.preset)
    entry = manifest["presets"][0]
    report: dict = {"mode": "paid" if a.paid else "dry_run", "at": datetime.now(timezone.utc).isoformat(), "preset": entry, "manifest": manifest.get("_path"), "message": a.message,
                    "model": DEFAULT_MODEL, "max_attempts": 1, "narrator_mode": "llm", "max_real_calls": MAX_REAL_CALLS}

    # ---- dry run: final prompts and their reservations ($0)
    fake = FakeLLMClient([json.dumps({"category": "alarm", "user_claims": [], "ref_window": None, "ref_faults": [], "ref_proposal": None, "rationale": "smoke"}),
                          "요약 문장 (dry run)."])
    mem = MemoryLedger(cap=1.0)
    dry = run_turn(a.dsn, manifest, a.preset, a.message, LLMRouter(fake, mem, max_attempts=1), LLMNarrator(fake, mem, mode="llm", max_attempts=1), a.owner + "-dry")
    reqs = [{"tool": ("router" if i == 0 else "narrate"), "estimated_input_tokens": r.estimated_input_tokens(), "max_tokens": r.max_tokens,
             "reserved_usd": r.reserved_usd(), "prompt_bytes": len((r.system + r.user).encode("utf-8"))} for i, r in enumerate(fake.calls)]
    dk, dans = answer_of(dry["result"])
    report["dry_run"] = {"thread_id": dry["thread_id"], "elapsed_s": dry["elapsed_s"], "requests": reqs,
                         "reserved_total_usd": round(sum(r["reserved_usd"] for r in reqs), 6),
                         "note": "estimates (utf8 bytes+64, max_tokens bound); the narrate prompt assumes category=alarm; settled at actual usage",
                         "answer_key": dk, "answer_full": dans, "result": brief(without_secrets(dry["result"]))}
    emit({"phase": "dry_run", **report})                        # printed and flushed BEFORE any real call
    if not a.paid:
        if a.save:
            json.dump(report, open(a.save, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
        return 0

    # ---- paid run
    if not os.environ.get("SGP_LLM_BUDGET_USD"):
        print("SGP_LLM_BUDGET_USD is not set: the ledger would refuse every reservation", file=sys.stderr)
        return 2
    from core_pipeline.agent.llm_client import AnthropicClient
    ledger = PostgresLedger(a.dsn)
    before = ledger.totals()
    if before["blocked_unresolved"]:
        print(json.dumps({"abort": "ledger has unresolved blocking rows", "totals": before}, ensure_ascii=False, default=str))
        return 4
    client = CountingClient(AnthropicClient())
    paid = run_turn(a.dsn, manifest, a.preset, a.message, LLMRouter(client, ledger, max_attempts=1),
                    LLMNarrator(client, ledger, mode="llm", max_attempts=1), a.owner)
    rows = rows_for_thread(a.dsn, paid["thread_id"])
    after = ledger.totals()
    pk, pans = answer_of(paid["result"])
    router_rows = [r for r in rows if r["tool"] == "router"]
    narrate_rows = [r for r in rows if r["tool"] == "narrate"]
    def ok_router(r): return r["status"] == "OK" and r["usage_missing"] is False and not r["validation_errors"] and r["has_merged"]
    def ok_narrate(r): return r["status"] == "OK" and r["usage_missing"] is False and not r["check_errors"] and r["has_prose"]
    cap = {"client_calls_le_2": len(client.calls) <= MAX_REAL_CALLS, "rows_eq_calls": len(rows) == len(client.calls)}
    verify = {"router_called_once": len(router_rows) == 1, "router_verified": len(router_rows) == 1 and ok_router(router_rows[0]),
              "narrate_called_once": len(narrate_rows) == 1, "narrate_verified": len(narrate_rows) == 1 and ok_narrate(narrate_rows[0]),
              "no_pending": all(r["status"] != "PENDING" for r in rows), "no_block": after["blocked_unresolved"] == 0}
    info = {"actual_le_reserved": all(r["cost_usd"] <= (r["reserved_usd"] or 0) for r in rows if r["status"] == "OK"),
            "answer_has_llm_summary": bool(pans and "요약(LLM 생성)" in pans)}
    if not verify["narrate_called_once"] and len(router_rows) == 1 and ok_router(router_rows[0]):
        reason = "router verified but no narrate call: the LLM may have chosen a fixed-narration category (see result.category)"
    elif not rows:
        reason = "no ledger rows for this thread (budget refused or rule-decided?)"
    else:
        reason = None
    verdict = "cap_violated" if not all(cap.values()) else ("complete" if all(verify.values()) else "incomplete")
    report["paid_run"] = {"thread_id": paid["thread_id"], "elapsed_s": paid["elapsed_s"], "client_calls": len(client.calls),
                          "client_call_requests": client.calls, "ledger_rows_this_thread": rows,
                          "spent_this_thread_usd": round(sum(r["cost_usd"] for r in rows), 6),
                          "totals_before": before, "totals_after": after, "answer_key": pk, "answer_full": pans,
                          "result": brief(without_secrets(paid["result"]))}
    report["checks"] = {"cap": cap, "verification": verify, "info": info, "verdict": verdict, "incomplete_reason": reason}
    emit({"phase": "paid", **report})
    if a.save:
        json.dump(report, open(a.save, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    return {"complete": 0, "incomplete": 5, "cap_violated": 6}[verdict]


if __name__ == "__main__":
    sys.exit(main())
