"""core_pipeline/eval/archive.py — export a run's policy_events ledger to a JSON file and mark the run archived.
Archiving is what frees the evaluation database for a new run (eval_runs.py: at most one active run)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import UUID

import psycopg

from core_pipeline.eval.eval_runs import RUN_SPACE_KEY, RunStateError


def archive_run(dsn: str, eval_run_id: UUID | str, out_dir: Path = Path("outputs/eval")) -> dict:
    rid = UUID(str(eval_run_id))
    with psycopg.connect(dsn) as conn:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (RUN_SPACE_KEY,))
            cur.execute("SELECT status FROM mart.eval_runs WHERE eval_run_id=%s FOR UPDATE", (rid,))
            row = cur.fetchone()
            if not row:
                raise RunStateError(f"unknown eval run {rid}")
            if row[0] == "running":
                raise RunStateError("a running run cannot be archived; finish it first")
            cur.execute("""SELECT row_to_json(e) FROM mart.policy_events e WHERE eval_run_id=%s
                           ORDER BY episode_id, window_index, policy, phase""", (rid,))
            events = [r[0] if isinstance(r[0], dict) else json.loads(r[0]) for r in cur.fetchall()]
            cur.execute("SELECT row_to_json(r) FROM mart.eval_runs r WHERE eval_run_id=%s", (rid,))
            run = cur.fetchone()[0]
            payload = json.dumps({"eval_run": run, "policy_events": events}, ensure_ascii=False, sort_keys=True, default=str, indent=1)
            sha = hashlib.sha256(payload.encode()).hexdigest()
            target = Path(out_dir) / str(rid)
            target.mkdir(parents=True, exist_ok=True)
            (target / "policy_events.json").write_text(payload, encoding="utf-8")
            cur.execute("UPDATE mart.eval_runs SET status='archived', events_export_sha256=%s, updated_at=now() WHERE eval_run_id=%s", (sha, rid))
    return {"eval_run_id": str(rid), "events": len(events), "path": str(target / "policy_events.json"), "sha256": sha}
