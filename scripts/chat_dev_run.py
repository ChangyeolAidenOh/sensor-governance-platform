"""S4a-07 chat dev-set harness (ST4-33..35). TEST DB ONLY (dbname contains 'test', distinct from SGP_PG_DSN).

    python scripts/chat_dev_run.py reset --run-id b1                      # backup (allow-listed) → clean+replay 4 episodes → manifest → pre-check
    python scripts/chat_dev_run.py reset --run-id b1 --precheck-only      # pre-check only on an already replayed state (no backup/clean/replay)
    python scripts/chat_dev_run.py run   --run-id b1 --mode baseline      # rule/template, $0
    python scripts/chat_dev_run.py run   --run-id d1 --mode dry           # FakeLLM + MemoryLedger ($0): prompt sizes / reservation estimates
    python scripts/chat_dev_run.py run   --run-id p1 --mode paid --max-calls 20 --paid-approved "<approval text>"   # separate approval
    python scripts/chat_dev_run.py judge --run-id b1                      # write / refresh the human judgment sheet
    python scripts/chat_dev_run.py score --run-id b1 [--publish]          # re-score from turns.jsonl with the current code, merge judgments, optional copy to evals/results/

One run = one reset + the 10 items in id order (one thread per item, single turn, no approval resume; an interrupt
ends the item). `run` refuses a context that was already used by a run — reset again. Everything written under
outputs/chat_dev/<run-id>/ is untracked; `score --publish` copies the refined result to evals/results/.
Observation = the four GraphDeps injection points (router / narrator / skills / record_decision) + DB reads;
graph, runner and checkpointer are not touched or read. Rows copied out of the DB keep allow-listed columns only.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from core_pipeline.agent import chat_eval as CE

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path("outputs/chat_dev")
PROJECT_MANIFEST = "outputs/demo_presets.json"
OWNER = "chatdev"
COMMON_CRITERIA_KEY = "human_criteria_common"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str), flush=True)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def sha16(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def git_dirty() -> bool | None:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain", "--", "core_pipeline", "scripts", "evals"], text=True).strip())
    except Exception:
        return None


def views_roundtrip_fn():
    """(fn, views_sha16, import_error). demo/ is not an installed package (demo/app.py adds the repo root itself), so the
    root is put on sys.path first; an import failure is returned as text, never swallowed."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from demo.views import answer_blocks, answer_from_blocks
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"[:300]
    vp = ROOT / "demo" / "views.py"
    return (lambda ans: answer_from_blocks(answer_blocks(ans)) == ans), (sha16(str(vp)) if vp.exists() else None), None


def scorer_version() -> dict:
    """Scoring-code identity, separate from the run-time code commit (uncommitted patches are told apart by file hashes)."""
    return {"code_commit": git_commit(), "git_dirty": git_dirty(), "chat_eval_sha16": sha16(CE.__file__), "run_script_sha16": sha16(__file__)}


def out_dir(run_id: str) -> Path:
    d = OUT_ROOT / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def dump_json(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dsn", default=None, help="admin TEST DSN (default SGP_TEST_PG_DSN)")
    ap.add_argument("--dev-set", default=CE.DEV_SET_PATH)


def dsn_of(a) -> str:
    dsn = a.dsn or os.environ.get("SGP_TEST_PG_DSN")
    CE.assert_test_dsn(dsn, os.environ.get("SGP_PG_DSN"))
    return dsn


# ------------------------------------------------------------------------------------------------- DB helpers
def _conn(dsn: str):
    import psycopg
    from psycopg.rows import dict_row
    return psycopg.connect(dsn, row_factory=dict_row)


def _rows(dsn: str, sql: str, *args) -> list[dict]:
    with _conn(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql, args)
        return [dict(r) for r in cur.fetchall()]


def _listify(v):
    if v is None:
        return None
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return list(v)


def probe(dsn: str, episode_id: str, dw: int, detector: str | None, contract: str | None, manifest_alarm_id: str | None) -> dict:
    """State that the run must explain: the episode_windows row at the display window (key episode·window·detector_version·
    contract_sha256 — table columns only; `alarm_id` is an envelope field, not a column), the alarm row of that window in
    mart.alarms (episode_id·window_index·detector) compared with the manifest, and the decision rows of the CONFIRMED alarm.
    More than one window row, or more than one alarm row, is a mismatch (`problems`) — never a silent first-row pick."""
    cond, args = ["episode_id=%s", "window_index=%s"], [uuid.UUID(episode_id), int(dw)]
    if detector:
        cond.append("detector_version=%s"); args.append(detector)
    if contract:
        cond.append("contract_sha256=%s"); args.append(contract)
    rows = _rows(dsn, f"SELECT alarm, drift, gate, pred_set, expanded_set, q_used, query_request_id FROM mart.episode_windows WHERE {' AND '.join(cond)}", *args)
    acond, aargs = ["episode_id=%s", "window_index=%s"], [uuid.UUID(episode_id), int(dw)]
    if detector:
        acond.append("detector=%s"); aargs.append(detector)
    alarm_rows = _rows(dsn, f"SELECT alarm_id FROM mart.alarms WHERE {' AND '.join(acond)}", *aargs)
    problems: list[str] = []
    if len(rows) != 1:
        problems.append(f"episode_windows rows at window {dw}: {len(rows)} (expected 1)")
    if len(alarm_rows) > 1:
        problems.append(f"mart.alarms rows at window {dw}: {len(alarm_rows)} (expected 0 or 1)")
    alarm_id = alarm_rows[0]["alarm_id"] if len(alarm_rows) == 1 else None
    if (manifest_alarm_id or None) != (alarm_id or None):
        problems.append(f"alarm_id: manifest {manifest_alarm_id} vs mart.alarms {alarm_id}")
    r = rows[0] if len(rows) == 1 else {}
    if len(rows) == 1 and bool(r.get("alarm")) != bool(alarm_id):
        problems.append(f"episode_windows.alarm={r.get('alarm')} but alarm row {'present' if alarm_id else 'absent'}")
    dec = _rows(dsn, "SELECT decision_key, action, rule FROM mart.policy_decisions WHERE alarm_id=%s ORDER BY seq", alarm_id) if alarm_id else []
    return {"found": len(rows) == 1, "n_rows": len(rows), "n_alarm_rows": len(alarm_rows), "alarm": r.get("alarm"), "drift": r.get("drift"), "gate": r.get("gate"),
            "pred_set": _listify(r.get("pred_set")), "expanded_set": _listify(r.get("expanded_set")), "q_used": r.get("q_used"),
            "query_request_id": (str(r["query_request_id"]) if r.get("query_request_id") else None), "alarm_id": alarm_id,
            "n_decisions": len(dec), "decision_keys": [d["decision_key"] for d in dec], "problems": problems}


def probe_diff(a: dict, b: dict) -> dict:
    keys = ("n_rows", "n_alarm_rows", "alarm", "drift", "pred_set", "expanded_set", "q_used", "query_request_id", "alarm_id", "n_decisions")
    return {k: {"before": a.get(k), "now": b.get(k)} for k in keys if a.get(k) != b.get(k)}


def counts_for_proposal(dsn: str, proposal_id: str) -> dict:
    pid = uuid.UUID(proposal_id)
    a = _rows(dsn, "SELECT count(*) AS n FROM mart.approvals WHERE proposal_id=%s", pid)[0]["n"]
    w = _rows(dsn, "SELECT count(*) AS n FROM mart.work_orders WHERE proposal_id=%s", pid)[0]["n"]
    return {"approvals": a, "work_orders": w}


def counts_for_alarm(dsn: str, alarm_id: str) -> dict:
    props = [r["proposal_id"] for r in _rows(dsn, "SELECT proposal_id FROM mart.proposals WHERE alarm_id=%s", alarm_id)]
    if not props:
        return {"approvals": 0, "work_orders": 0, "proposals": 0}
    a = _rows(dsn, "SELECT count(*) AS n FROM mart.approvals WHERE proposal_id = ANY(%s)", props)[0]["n"]
    w = _rows(dsn, "SELECT count(*) AS n FROM mart.work_orders WHERE proposal_id = ANY(%s)", props)[0]["n"]
    return {"approvals": a, "work_orders": w, "proposals": len(props)}


def plan_view(dsn: str, query_request_id: str | None, q_used) -> dict | None:
    if not query_request_id:
        return None
    from core_pipeline.agent.plans import plan_get, plan_status
    plan = plan_get(dsn, query_request_id)
    return {"query_id": query_request_id, "status": plan_status(plan, bool(q_used)), "owned": None if plan is None else True,
            "kind": (plan or {}).get("kind"), "expand_status": (plan or {}).get("expand_status"), "rerank_status": (plan or {}).get("rerank_status"),
            "thread_id": (plan or {}).get("thread_id")}


def backup_episode(dsn: str, episode_id: str) -> dict:
    """Allow-listed copy of the Stage 4a rows clean_episode will delete (verification evidence, NOT a restore backup)."""
    ep = uuid.UUID(episode_id)
    alarms = [r["alarm_id"] for r in _rows(dsn, "SELECT alarm_id FROM mart.alarms WHERE episode_id=%s", ep)]
    out = {"episode_id": episode_id, "alarms": alarms, "proposals": [], "proposal_drafts": [], "approvals": [], "work_orders": [],
           "policy_decisions": [], "query_plans": []}
    if alarms:
        props = _rows(dsn, "SELECT proposal_id, alarm_id, policy, action, hold_reason, ev_ab, current_version, episode_id, window_index, thread_id, created_at "
                           "FROM mart.proposals WHERE alarm_id = ANY(%s)", alarms)
        out["proposals"] = [CE.strip_secret_columns(p) for p in props]
        pids = [p["proposal_id"] for p in props]
        if pids:
            # proposal_drafts: column list not confirmed → SELECT * then drop any token/secret-named column (strip_secret_columns)
            out["proposal_drafts"] = [CE.strip_secret_columns(r) for r in _rows(dsn, "SELECT * FROM mart.proposal_drafts WHERE proposal_id = ANY(%s)", pids)]
            # approvals: explicit columns — approval_token is NOT selected
            out["approvals"] = [CE.strip_secret_columns(r) for r in _rows(dsn, "SELECT approval_id, proposal_id, approver, decision, token_used, draft_version, content_sha256, thread_id, decided_at "
                                                                            "FROM mart.approvals WHERE proposal_id = ANY(%s)", pids)]
            out["work_orders"] = [CE.strip_secret_columns(r) for r in _rows(dsn, "SELECT work_order_id, proposal_id, approval_id, cmms_ref, status, draft_version, issued_at "
                                                                              "FROM mart.work_orders WHERE proposal_id = ANY(%s)", pids)]
        out["policy_decisions"] = [CE.strip_secret_columns(r) for r in _rows(dsn, "SELECT decision_id, alarm_id, policy, action, rule, decision_key, seq, decided_at FROM mart.policy_decisions WHERE alarm_id = ANY(%s)", alarms)]
    out["query_plans"] = [CE.strip_secret_columns(r) for r in _rows(dsn, "SELECT query_id, episode_id, window_index, detector_version, thread_id, turn_id, initial_decision_id, kind, planned_expand, planned_rerank, "
                                                                        "expand_status, expand_reason, rerank_status, rerank_reason, created_at, updated_at FROM mart.query_plans WHERE episode_id=%s", ep)]
    ws = _rows(dsn, "SELECT count(*) AS n, count(*) FILTER (WHERE q_used) AS n_q FROM mart.episode_windows WHERE episode_id=%s", ep)[0]
    out["episode_windows"] = {"n_rows": ws["n"], "n_q_used": ws["n_q"]}
    return out


# ------------------------------------------------------------------------------------------------- reset
def cmd_reset(a) -> int:
    dsn = dsn_of(a)
    ds = CE.load_dev_set(a.dev_set)
    errs = CE.validate_dev_set(ds)
    if errs:
        emit({"abort": "dev set invalid", "errors": errs}); return 2
    proj = load_json(Path(a.manifest))
    proj_by = {int(e["preset"]): e for e in proj["presets"]}
    presets = sorted({int(it["context"]["preset"]) for it in ds["items"]})
    d = out_dir(a.run_id)
    from core_pipeline.skills import replay_client as RC
    from core_pipeline.skills.client import SkillClient
    entries, replay_rows = {}, {}
    if a.precheck_only:
        # verification only: no backup, no clean_episode, no replay — uses the manifest the earlier reset wrote; the earlier backup file is left untouched
        mp = d / "manifest_test.json"
        if not mp.exists():
            emit({"abort": f"{mp} not found; run a full reset first"}); return 2
        if (d / "context.json").exists() and (load_json(d / "context.json").get("used_by_run")):
            emit({"abort": "this run-id was already used by a run; choose a new run-id and reset"}); return 2
        entries = {int(e["preset"]): e for e in load_json(mp)["presets"]}
        replay_rows = {"note": "precheck-only rerun: replay rows of the earlier reset were not recorded (probe fix)"}
    else:
        backups = {p: backup_episode(dsn, str(RC.episode_for_preset(dsn, p))) for p in presets}
        dump_json(d / "pre_reset_backup.json", {"at": now(), "note": "allow-listed rows deleted by clean_episode (verification evidence; not a restore backup)", "presets": backups})
        emit({"phase": "backup", "presets": {p: {k: (len(v) if isinstance(v, list) else v) for k, v in b.items() if k != "episode_id"} for p, b in backups.items()}})
        if a.backup_only:
            return 0

    async def replay():
        async with SkillClient(env={"SGP_PG_DSN": dsn}) as sk:
            for p in presets:
                ep = RC.episode_for_preset(dsn, p)
                t0 = time.perf_counter()
                RC.clean_episode(dsn, ep)
                rows = await RC.replay_episode(sk, ep, f"chatdev-reset-{a.run_id}-{uuid.uuid4().hex[:6]}", mode="replay")
                alarm = next((r for r in rows if r["alarm"] is True), None)
                entries[p] = {"preset": p, "episode_id": str(ep), "display_window": alarm["t"] if alarm else len(rows) - 1,
                              "kind": "alarm" if alarm else "no_alarm", "alarm_id": alarm["alarm_id"] if alarm else None,
                              "pred_set": alarm["pred_set"] if alarm else []}
                replay_rows[p] = {"elapsed_s": round(time.perf_counter() - t0, 1), "windows": [{k: r.get(k) for k in ("t", "gate", "alarm", "pred_set", "top1")} for r in rows]}
                emit({"phase": "replay", "preset": p, "entry": entries[p], "elapsed_s": replay_rows[p]["elapsed_s"]})
    if not a.precheck_only:
        asyncio.run(replay())
        manifest = {"schema": "demo_presets/1", "generated_for": f"chat dev-set run {a.run_id} on the test DB (ST4-33) — NOT a demo manifest", "presets": [entries[p] for p in presets]}
        dump_json(d / "manifest_test.json", manifest)
    # pre-check against the project manifest + DB probe + item requirements
    ctx = {"run_id": a.run_id, "created": now(), "dbname": CE.dbname_of(dsn), "code_commit": git_commit(), "dev_set_sha16": sha16(a.dev_set),
           "dev_set_revision": ds["_meta"].get("revision"), "project_manifest": a.manifest, "project_manifest_sha16": sha16(a.manifest),
           "used_by_run": None, "presets": {}, "items": {}, "replay": replay_rows}
    for p in presets:
        e, pe = entries[p], proj_by.get(p)
        reasons = []
        if pe is None:
            reasons.append("preset not in the project manifest")
        else:
            for k in ("display_window", "kind", "alarm_id"):
                if e.get(k) != pe.get(k):
                    reasons.append(f"{k}: test {e.get(k)} vs project {pe.get(k)}")
            if sorted(e.get("pred_set") or []) != sorted(pe.get("pred_set") or []):
                reasons.append(f"pred_set: test {e.get('pred_set')} vs project {pe.get('pred_set')}")
        vers = (pe or {}).get("versions") or {}
        detector, contract = vers.get("detector"), (vers.get("contract_sha256") or vers.get("retrieve.contract_sha256"))
        pr = probe(dsn, e["episode_id"], e["display_window"], detector, contract, e.get("alarm_id"))
        reasons += pr["problems"]
        if pr.get("q_used"):
            reasons.append("q_used is true after reset")
        if pr.get("n_decisions"):
            reasons.append(f"{pr['n_decisions']} decision rows after reset")
        ctx["presets"][str(p)] = {"entry": e, "project_entry": {k: pe.get(k) for k in ("display_window", "kind", "alarm_id", "pred_set")} if pe else None,
                                  "detector": detector, "contract": contract, "probe": pr, "prep": "failed" if reasons else "ok", "reasons": reasons}
    for it in ds["items"]:
        p = str(it["context"]["preset"])
        pc = ctx["presets"][p]
        reasons = list(pc["reasons"])
        req, e, pr = it["context"].get("requires") or {}, pc["entry"], pc["probe"]
        if req.get("kind") and e["kind"] != req["kind"]:
            reasons.append(f"kind {e['kind']} != {req['kind']}")
        if req.get("kind") == "alarm" and not pr.get("alarm_id"):
            reasons.append("no confirmed alarm row")
        if req.get("drift") and pr.get("drift") != req["drift"]:
            reasons.append(f"drift {pr.get('drift')} != {req['drift']}")
        if req.get("q_used") is False and pr.get("q_used"):
            reasons.append("q_used true")
        if req.get("claim_in_set") is not None:
            inset = all(c in (e.get("pred_set") or []) for c in it["expected"]["user_claims"])
            if inset != bool(req["claim_in_set"]):
                reasons.append(f"claim_in_set {inset} != {req['claim_in_set']}")
        if req.get("display_window_lt") is not None and not (e["display_window"] < int(req["display_window_lt"])):
            reasons.append(f"display_window {e['display_window']} not < {req['display_window_lt']}")
        ctx["items"][it["id"]] = {"preset": int(p), "prep": "failed" if reasons else "ok", "reasons": reasons}
    dump_json(d / "context.json", ctx)
    emit({"phase": "precheck", "presets": {p: {"prep": v["prep"], "reasons": v["reasons"]} for p, v in ctx["presets"].items()},
          "items": {i: v["prep"] for i, v in ctx["items"].items()}, "out": str(d)})
    return 0 if all(v["prep"] == "ok" for v in ctx["items"].values()) else 1


# ------------------------------------------------------------------------------------------------- run
class CountingClient:
    """Paid mode: refuses the (cap+1)-th request without sending it and stops after any billing-unknown outcome.
    Every request that IS sent is recorded with the item and tool (router / narrate) — the source of the LLM call counts."""

    def __init__(self, inner, cap: int, rec: CE.Recorder) -> None:
        self.inner, self.cap, self.rec, self.calls, self.stopped = inner, int(cap), rec, [], None

    def complete(self, req):
        from core_pipeline.agent import llm_router as LR
        from core_pipeline.agent.llm_client import LLMOutcomeUnknown, LLMRequestRejected
        if self.stopped:
            raise LLMRequestRejected(f"run stopped: {self.stopped}; request not sent")
        if len(self.calls) >= self.cap:
            raise LLMRequestRejected(f"cap: {self.cap} real calls already made; request not sent")
        self.calls.append({"n": len(self.calls) + 1, "item": self.rec.current, "tool": ("router" if req.system == LR.SYSTEM_PROMPT else "narrate"),
                           "max_tokens": req.max_tokens, "estimated_input_tokens": req.estimated_input_tokens(), "reserved_usd": req.reserved_usd()})
        try:
            return self.inner.complete(req)
        except LLMOutcomeUnknown as e:
            self.stopped = f"billing unknown after call {len(self.calls)}: {e}"[:200]
            raise


def ledger_rows_for_thread(dsn: str, thread_id: str) -> list[dict]:
    rows = _rows(dsn, """SELECT request_id, status, reason_code, cost_usd, input_tokens, output_tokens, violation, input, output
                         FROM mart.agent_steps WHERE skill='llm' AND input->'ctx'->>'thread_id' = %s ORDER BY created_at""", thread_id)
    out = []
    for r in rows:
        inp = r["input"] if isinstance(r["input"], dict) else json.loads(r["input"])
        o = (r["output"] if isinstance(r["output"], dict) else json.loads(r["output"])) if r["output"] else {}
        out.append({"request_id": str(r["request_id"]), "tool": inp.get("tool"), "stage_key": inp.get("stage_key"), "attempt": inp.get("attempt"),
                    "status": r["status"], "reason_code": r["reason_code"], "reserved_usd": (inp.get("reserve") or {}).get("reserved_usd"),
                    "cost_usd": float(r["cost_usd"] or 0), "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"], "violation": r["violation"],
                    "usage_missing": o.get("usage_missing"), "reservation_exceeded": o.get("reservation_exceeded"), "calibration": o.get("calibration"),
                    "llm_raw_text": o.get("text"), "rule_extract": o.get("rule_extract"), "llm_parsed": o.get("llm_parsed"), "merged": o.get("merged"),
                    "validation_errors": o.get("validation_errors"), "prose": o.get("prose"), "rejected_prose": o.get("rejected_prose"),
                    "check_errors": o.get("check_errors"), "fallback_reason": o.get("fallback_reason"), "no_response": o.get("no_response")})
    return out


def build_llm(mode: str, dsn: str, a, ds: dict, rec: CE.Recorder, prompt_log: list):
    """(router, narrator, client, ledger) for the mode. baseline → deterministic; dry → FakeLLM; paid → real (approved only)."""
    from core_pipeline.agent.narrator import TemplateNarrator
    from core_pipeline.agent.router import RuleRouter
    if mode == "baseline":
        return RuleRouter(), TemplateNarrator(), None, None
    from core_pipeline.agent import llm_router as LR
    from core_pipeline.agent.llm_client import FakeLLMClient, LLMResponse
    from core_pipeline.agent.llm_ledger import MemoryLedger, PostgresLedger
    from core_pipeline.agent.llm_narrator import LLMNarrator
    from core_pipeline.agent.llm_router import LLMRouter
    if mode == "dry":
        by_msg = {it["question"]: it for it in ds["items"]}

        def scripted(req):
            tool = "router" if req.system == LR.SYSTEM_PROMPT else "narrate"
            prompt_log.append({"item": rec.current, "tool": tool, "prompt_bytes": len((req.system + req.user).encode("utf-8")),
                               "estimated_input_tokens": req.estimated_input_tokens(), "max_tokens": req.max_tokens, "reserved_usd": req.reserved_usd(),
                               "note": "estimate under the assumption that the fake router returns the item's expected category"})
            if tool == "router":
                msg = req.user.split("[message]\n", 1)[-1]
                it = by_msg.get(msg)
                cat = it["expected"]["category"] if it else "alarm"
                text = json.dumps({"category": cat, "user_claims": [], "ref_window": None, "ref_faults": [], "ref_proposal": None, "rationale": "dry run"})
            else:
                text = "요약 문장입니다 (dry run)."
            return LLMResponse(text=text, input_tokens=100, output_tokens=50, latency_ms=1, model=req.model)
        client = FakeLLMClient([scripted] * 40)
        ledger = MemoryLedger(cap=1.0)
    else:
        if not a.paid_approved:
            raise SystemExit("paid mode needs --paid-approved \"<approval text>\" (dry-run → cost/HTTP-count approval → run)")
        if not os.environ.get("SGP_LLM_BUDGET_USD"):
            raise SystemExit("SGP_LLM_BUDGET_USD is not set: the ledger would refuse every reservation")
        from core_pipeline.agent.llm_client import AnthropicClient
        ledger = PostgresLedger(dsn)
        client = CountingClient(AnthropicClient(), a.max_calls, rec)
    return LLMRouter(client, ledger, max_attempts=1), LLMNarrator(client, ledger, mode="llm", max_attempts=1), client, ledger


def cmd_run(a) -> int:
    dsn = dsn_of(a)
    ds = CE.load_dev_set(a.dev_set)
    d = out_dir(a.run_id)
    ctx_path = d / "context.json"
    if not ctx_path.exists():
        emit({"abort": f"no context.json for run-id {a.run_id}: run `reset --run-id {a.run_id}` first"}); return 2
    ctx = load_json(ctx_path)
    if ctx.get("used_by_run"):
        emit({"abort": "this context was already used by a run; reset again (one reset = one run)", "used_by_run": ctx["used_by_run"]}); return 2
    if sha16(a.dev_set) != ctx.get("dev_set_sha16"):
        emit({"abort": "dev set changed since reset"}); return 2
    manifest = load_json(d / "manifest_test.json")
    entries = {int(e["preset"]): e for e in manifest["presets"]}
    # contamination check at run start: every preset probe must equal the reset probe
    expected_probe: dict[str, dict] = {}
    start_diffs = {}
    for p, pc in ctx["presets"].items():
        e = pc["entry"]
        pr = probe(dsn, e["episode_id"], e["display_window"], pc.get("detector"), pc.get("contract"), e.get("alarm_id"))
        diff = probe_diff(pc["probe"], pr)
        if diff:
            start_diffs[p] = diff
        expected_probe[e["episode_id"]] = pr
    if start_diffs:
        emit({"abort": "state changed since reset (reset again)", "diffs": start_diffs}); return 2
    rec = CE.Recorder()
    prompt_log: list = []
    router_inner, narrator_inner, client, ledger = build_llm(a.mode, dsn, a, ds, rec, prompt_log)
    totals_before = ledger.totals() if a.mode == "paid" else None
    if a.mode == "paid" and totals_before["blocked_unresolved"]:
        emit({"abort": "ledger has unresolved blocking rows", "totals": totals_before}); return 4
    from langgraph.checkpoint.postgres import PostgresSaver
    from core_pipeline.agent.graph import GraphDeps
    from core_pipeline.agent.runner import AgentSession
    from core_pipeline.agent.session import start_session
    from core_pipeline.agent.turns import TurnStore
    from core_pipeline.skills.client import SyncSkillClient
    from core_pipeline.skills.work_order import record_decision
    views_fn, views_sha, views_err = views_roundtrip_fn()
    if views_err:
        emit({"warning": "demo.views not importable; the display round-trip check will be unobserved", "error": views_err})
    ctx["used_by_run"] = {"run_id": a.run_id, "mode": a.mode, "started": now()}
    dump_json(ctx_path, ctx)
    run_threads: list[str] = []
    scored, turns_path = [], d / "turns.jsonl"
    turns_path.write_text("", encoding="utf-8")
    with SyncSkillClient(env={"SGP_PG_DSN": dsn}) as sk:
        deps = GraphDeps(dsn=dsn, skills=CE.RecordingSkills(sk, rec), turns=TurnStore(dsn), manifest=manifest,
                         router=CE.RecordingRouter(router_inner, rec), narrator=CE.RecordingNarrator(narrator_inner, rec),
                         record_decision=CE.recording_decision(record_decision, rec))
        with PostgresSaver.from_conn_string(dsn) as cp:
            cp.setup()
            sess = AgentSession(deps, cp)
            for it in sorted(ds["items"], key=lambda x: x["id"]):
                iid, p = it["id"], int(it["context"]["preset"])
                e = entries[p]
                dw = int(e["display_window"])
                prep = ctx["items"][iid]
                obs: dict = {"id": iid, "preset": p, "dw": dw, "episode_id": e["episode_id"], "alarm_id": e.get("alarm_id"), "question": it["question"],
                             "context": {"prep": prep["prep"], "reasons": prep["reasons"], "contaminated": False, "contamination": []}}
                if prep["prep"] == "failed":
                    obs["status"] = "not_executed"
                    sc = CE.score_item(it, obs)
                    scored.append(sc)
                    turns_path.open("a", encoding="utf-8").write(json.dumps(obs, ensure_ascii=False, default=str) + "\n")
                    emit({"item": iid, "status": "not_executed", "context_prep_failed": prep["reasons"]})
                    continue
                if client is not None and getattr(client, "stopped", None):
                    obs["status"] = "not_executed"; obs["error"] = f"run stopped: {client.stopped}"
                    scored.append(CE.score_item(it, obs))
                    turns_path.open("a", encoding="utf-8").write(json.dumps(obs, ensure_ascii=False, default=str) + "\n")
                    emit({"item": iid, "status": "not_executed", "reason": client.stopped})
                    continue
                detector, contract = ctx["presets"][str(p)].get("detector"), ctx["presets"][str(p)].get("contract")
                pre = probe(dsn, e["episode_id"], dw, detector, contract, e.get("alarm_id"))
                diff = probe_diff(expected_probe[e["episode_id"]], pre)
                if diff:
                    obs["context"]["contaminated"] = True
                    obs["context"]["contamination"].append({"when": "before_turn", "diff": diff})
                tid, secret = start_session(dsn, OWNER, p, e["episode_id"], dw)
                rec.begin(iid, tid, dw)
                run_threads.append(tid)
                turn_id = str(uuid.uuid4())
                t0 = time.perf_counter()
                try:
                    res = sess.invoke(tid, secret, it["question"], turn_id)
                    obs["status"] = res.get("status")
                    obs["error"] = res.get("error")
                except Exception as ex:
                    res = {"status": "error"}
                    obs["status"], obs["error"] = "error", f"{type(ex).__name__}: {ex}"[:500]
                obs["elapsed_s"] = round(time.perf_counter() - t0, 2)
                del secret
                r = rec.items[iid]
                obs.update({"thread_id": tid, "turn_id": turn_id, "route": r["route"], "skills": r["skills"], "decisions": r["decisions"], "narration": r["narration"],
                            "result": {k: res.get(k) for k in ("status", "turn_id", "category", "decision", "decision_after", "query", "trace", "answer", "refusal", "turn_refs_discarded", "error")},
                            "draft": (res.get("draft") or {}) and {"status": res["draft"].get("status"), "reason": res["draft"].get("reason"),
                                                                   "proposal_id": ((res["draft"].get("payload") or {}).get("proposal_id"))},
                            "interrupt": (res.get("interrupt") if res.get("status") == "waiting_approval" else None) and
                                         {k: res["interrupt"].get(k) for k in ("proposal_id", "draft_version", "content_sha256", "question")}})
                if obs["narration"]:
                    obs["views_roundtrip"] = views_fn(obs["narration"]["answer"]) if views_fn else None
                    obs["views_roundtrip_source"] = {"posthoc": False, "views_sha16": views_sha, "import_error": views_err}
                # DB reads after the turn
                post = probe(dsn, e["episode_id"], dw, detector, contract, e.get("alarm_id"))
                foreign = [k for k in post["decision_keys"] if not any(k.startswith(t + ":") for t in run_threads) and k not in expected_probe[e["episode_id"]]["decision_keys"]]
                if foreign:
                    obs["context"]["contaminated"] = True
                    obs["context"]["contamination"].append({"when": "after_turn", "foreign_decision_keys": foreign})
                expected_probe[e["episode_id"]] = post
                db = {"decision_rows": sum(1 for k in post["decision_keys"] if k.startswith(tid + ":")), "decision_rows_alarm_total": post["n_decisions"],
                      "q_used": post["q_used"], "plan": plan_view(dsn, post["query_request_id"], post["q_used"])}
                pid = (obs.get("interrupt") or {}).get("proposal_id") or (obs.get("draft") or {}).get("proposal_id")
                if pid:
                    db.update(counts_for_proposal(dsn, pid))
                elif e.get("alarm_id"):
                    db.update(counts_for_alarm(dsn, e["alarm_id"]))
                obs["db"] = db
                if a.mode == "paid":
                    obs["llm_calls"] = [c for c in client.calls if c.get("item") == iid]
                    rows = ledger_rows_for_thread(dsn, tid)
                    obs["ledger"] = rows
                    blocking = [r["request_id"] for r in rows if r.get("usage_missing") or r.get("reservation_exceeded") or r["status"] == "PENDING"]
                    if blocking and not client.stopped:                      # billing unknown / over-reservation: stop the run ourselves, not only the ledger
                        client.stopped = f"ledger row(s) need manual resolution after item {iid}: {blocking}"[:200]
                    obs["router_ledger"] = next(({k: r.get(k) for k in ("status", "reason_code", "rule_extract", "llm_parsed", "merged", "validation_errors", "llm_raw_text", "fallback_reason")} for r in rows if r["tool"] == "router"), None)
                    obs["narrate_ledger"] = next(({k: r.get(k) for k in ("status", "reason_code", "prose", "rejected_prose", "check_errors", "fallback_reason", "violation")} for r in rows if r["tool"] == "narrate"), None)
                sc = CE.score_item(it, obs)
                scored.append(sc)
                turns_path.open("a", encoding="utf-8").write(json.dumps(obs, ensure_ascii=False, default=str) + "\n")
                emit({"item": iid, "status": obs["status"], "category": (obs["route"] or {}).get("result", {}).get("category"), "matched": (obs["route"] or {}).get("result", {}).get("matched"),
                      "router_ok": sc.get("router", {}).get("ok"), "contract_ok": sc.get("contract", {}).get("ok"), "failure_kind": sc.get("contract", {}).get("failure_kind"),
                      "narration_ok": (sc.get("narration") or {}).get("ok"), "contaminated": obs["context"]["contaminated"], "elapsed_s": obs["elapsed_s"]})
    summary = CE.summarize(scored, a.mode)
    result = {"run_id": a.run_id, "mode": a.mode, "finished": now(), "code_commit": git_commit(), "dev_set_revision": ctx.get("dev_set_revision"), "dev_set_sha16": ctx.get("dev_set_sha16"),
              "project_manifest_sha16": ctx.get("project_manifest_sha16"), "dbname": ctx.get("dbname"), "threads": run_threads, "summary": summary, "items": scored}
    if a.mode == "dry":
        result["dry_run"] = {"prompts": prompt_log, "reserved_total_usd": round(sum(x["reserved_usd"] for x in prompt_log), 6),
                             "note": "estimates (utf8 bytes+64, max_tokens bound) under the assumption that the LLM router returns each item's expected category; not the paid run's final prompts or cost"}
    if a.mode == "paid":
        result["paid"] = {"approval": a.paid_approved, "max_calls": a.max_calls, "client_calls": len(client.calls), "client_call_requests": client.calls, "stopped": client.stopped,
                          "totals_before": totals_before, "totals_after": ledger.totals()}
        spent = 0.0
        for line in turns_path.read_text(encoding="utf-8").splitlines():
            o = json.loads(line)
            spent += sum(r["cost_usd"] for r in o.get("ledger") or [])
        result["paid"]["spent_this_run_usd"] = round(spent, 6)
    dump_json(d / "result.json", result)
    emit({"phase": "done", "summary": summary, "out": str(d / "result.json")})
    return 0


# ------------------------------------------------------------------------------------------------- judge / score
def cmd_judge(a) -> int:
    d = out_dir(a.run_id)
    res = load_json(d / "result.json")
    ds = CE.load_dev_set(a.dev_set)
    items = {it["id"]: it for it in ds["items"]}
    common = ds["_meta"].get(COMMON_CRITERIA_KEY, [])
    jp = d / "judgment.json"
    existing = load_json(jp) if jp.exists() else {"run_id": a.run_id, "judge": a.judge, "items": {}}
    turns = {json.loads(l)["id"]: json.loads(l) for l in (d / "turns.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    for sc in res["items"]:
        iid = sc["id"]
        if not sc.get("scored") or not (sc.get("narration") or {}).get("applicable"):
            existing["items"].setdefault(iid, {"applicable": False, "note": (sc.get("narration") or {}).get("note") or sc.get("note")})
            continue
        crit = list(common) + list(items[iid].get("human", []))
        cur = existing["items"].get(iid) or {"applicable": True, "verdicts": {}, "note": ""}
        cur.update({"applicable": True, "question": items[iid]["question"], "answer": (turns[iid].get("narration") or {}).get("answer"),
                    "flags": sc["narration"].get("flags"), "criteria": crit})
        cur["verdicts"] = {c: cur.get("verdicts", {}).get(c) for c in crit}
        existing["items"][iid] = cur
    dump_json(jp, existing)
    emit({"phase": "judge", "sheet": str(jp), "fill": "verdicts: pass | fail (+note); rerun score afterwards"})
    return 0


def rescore(d: Path, dev_set_path: str, allow_devset_change: bool = False) -> tuple[dict, list[dict], dict]:
    """Recompute every item from turns.jsonl (the raw observations) with the CURRENT scoring code; result.json keeps the
    run-time record. Same items and expectations only: a dev set whose sha differs from the one the run used is refused
    unless --allow-devset-change is given (and then recorded)."""
    res = load_json(d / "result.json")
    ds = CE.load_dev_set(dev_set_path)
    cur = sha16(dev_set_path)
    if cur != res.get("dev_set_sha16"):
        if not allow_devset_change:
            raise SystemExit(f"dev set changed since the run (run {res.get('dev_set_sha16')} vs now {cur}); re-scoring must use the same items/expectations — "
                             f"restore the dev set or pass --allow-devset-change (recorded)")
    items = {it["id"]: it for it in ds["items"]}
    obs_list = [json.loads(l) for l in (d / "turns.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    # post-hoc display round-trip on the STORED answers for items whose run-time check was unobserved (turns.jsonl itself is not rewritten)
    views_fn, views_sha, views_err = views_roundtrip_fn()
    posthoc = []
    for o in obs_list:
        if o.get("views_roundtrip") is None and (o.get("narration") or {}).get("answer") is not None and views_fn:
            o["views_roundtrip"] = views_fn(o["narration"]["answer"])
            o["views_roundtrip_source"] = {"posthoc": True, "views_sha16": views_sha, "at": now()}
            posthoc.append(o["id"])
    scored = [CE.score_item(items[o["id"]], o) for o in obs_list]
    res["items"] = scored
    res["summary"] = CE.summarize(scored, res["mode"])
    res["rescored"] = {"at": now(), "from": "turns.jsonl", "scorer": scorer_version(), "dev_set_sha16_now": cur,
                       "dev_set_changed_since_run": cur != res.get("dev_set_sha16"),
                       "views_roundtrip_posthoc": {"items": posthoc, "views_sha16": views_sha, "import_error": views_err,
                                                   "note": "string parse/reassemble preservation on stored answers (not a Streamlit render check)"}}
    return res, obs_list, ds


def cmd_score(a) -> int:
    d = out_dir(a.run_id)
    res, obs_list, ds = rescore(d, a.dev_set, a.allow_devset_change)
    jp = d / "judgment.json"
    j = load_json(jp) if jp.exists() else None
    human = CE.judgment_status(j, res["items"], ds)
    if j:
        for sc in res["items"]:
            v = (j.get("items") or {}).get(sc["id"]) or {}
            if v.get("applicable") and sc.get("narration"):
                sc["narration"]["human"] = v.get("verdicts")
                sc["narration"]["human_note"] = v.get("note")
    res["summary"]["human"] = human
    res["scored_at"] = now()
    dump_json(d / "scored.json", res)
    emit({"phase": "score", "summary": res["summary"], "rescored": res["rescored"], "out": str(d / "scored.json")})
    if a.publish:
        if not human["complete"]:
            emit({"abort": "human judgment incomplete; fill judgment.json (judge) before --publish", "human": human}); return 3
        ctx = load_json(d / "context.json") if (d / "context.json").exists() else {}
        run_meta = {k: res.get(k) for k in ("run_id", "mode", "finished", "code_commit", "dev_set_revision", "dev_set_sha16", "project_manifest_sha16", "dbname", "threads", "rescored", "paid", "dry_run")}
        run_meta["summary"] = res["summary"]
        refined = CE.refined_result(ds, ctx, obs_list, res["items"], j, run_meta)
        dst = Path("evals/results") / f"chat_dev_{a.run_id}_{res.get('code_commit', 'unknown')}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dump_json(dst, refined)
        emit({"published": str(dst), "items": len(refined["items"])})
    return 0


def main() -> int:
    load_dotenv(".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reset"); common_args(r)
    r.add_argument("--manifest", default=PROJECT_MANIFEST, help="project demo_presets.json (selection reference)")
    r.add_argument("--backup-only", action="store_true", help="write the allow-listed backup and stop (no deletion)")
    r.add_argument("--precheck-only", action="store_true", help="verify an already replayed state (no backup / clean / replay); keeps the earlier backup file")
    u = sub.add_parser("run"); common_args(u)
    u.add_argument("--mode", choices=("baseline", "dry", "paid"), required=True)
    u.add_argument("--max-calls", type=int, default=20)
    u.add_argument("--paid-approved", default=None)
    jd = sub.add_parser("judge"); common_args(jd)
    jd.add_argument("--judge", default="aiden")
    s = sub.add_parser("score"); common_args(s)
    s.add_argument("--publish", action="store_true", help="write the refined result to evals/results/ (requires a complete judgment sheet)")
    s.add_argument("--allow-devset-change", action="store_true", help="re-score even though evals/chat_dev_set.json changed since the run (recorded)")
    a = ap.parse_args()
    return {"reset": cmd_reset, "run": cmd_run, "judge": cmd_judge, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
