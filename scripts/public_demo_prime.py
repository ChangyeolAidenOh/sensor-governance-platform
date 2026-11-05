"""scripts/public_demo_prime.py: ask every public question once per preset through the REAL graph and servers (demo-public).

    python scripts/public_demo_prime.py --app-dsn <sgp_agent DSN of the LOCAL build DB> --skill-dsn <admin DSN of the same DB> \\
        --manifest outputs/demo_presets_public.json --questions outputs/public_demo_questions.json --synthetic-presets 11 \\
        --report build/public/prime_report.json

Why: the public app answers `retrieve` from stored rows only (core_pipeline.skills.public_skills.StoredRetrieve). The
scoped k=5 retrieval of every alarm is stored by precompute; this script adds what a visitor can still trigger: the
open-mode retrieval of the evidence question (keyed on the exact sentence), the query_more path of the action question
(spend -> expand=2 -> k=10) and, for the synthetic preset, the draft that opens the approval gate (rejected here so no
proposal stays pending). It runs the same DemoRuntime as the local demo with BOTH servers (the retrieve server loads
the bge models locally), LLM mode forced off, owner "public-prime". Never run it against the project or evaluation DB.
Exit 1 if any turn failed, any evidence retrieval was not OK, or the synthetic preset did not reach waiting_approval.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

from psycopg.conninfo import conninfo_to_dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OWNER = "public-prime"


def _trace_rows(res: dict) -> list[dict]:
    return [{"tool": t.get("tool"), "step": t.get("step"), "status": t.get("status"), "reason": t.get("reason")} for t in (res.get("trace") or [])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dsn", required=True)
    ap.add_argument("--skill-dsn", required=True)
    ap.add_argument("--manifest", default="outputs/demo_presets_public.json")
    ap.add_argument("--questions", default="outputs/public_demo_questions.json")
    ap.add_argument("--synthetic-presets", default="11")
    ap.add_argument("--report", default="build/public/prime_report.json")
    ap.add_argument("--presets", default="", help="comma-separated subset (default: every manifest preset)")
    a = ap.parse_args()
    for dsn in (a.app_dsn, a.skill_dsn):
        name = conninfo_to_dict(dsn).get("dbname") or ""
        if "eval" in name or name == "sgp_v2":
            print(f"refusing to prime database {name!r}", file=sys.stderr)
            return 2
    for k in list(os.environ):
        if k.startswith("SGP_LLM_"):
            os.environ.pop(k)                                   # rule router + template narrator only ($0)
    os.environ["SGP_DEMO_MANIFEST"] = a.manifest

    from core_pipeline.agent.session import start_session
    from demo.runtime import DemoConfig, build_runtime, check_dsns

    syn = tuple(sorted({int(x) for x in a.synthetic_presets.split(",") if x.strip()}))
    cfg = DemoConfig(app_dsn=a.app_dsn, skill_dsn=a.skill_dsn, manifest_path=a.manifest, synthetic=False,
                     dsn_facts=check_dsns(a.app_dsn, a.skill_dsn, False), public=False, synthetic_presets=syn)
    questions = [q for q in json.loads(Path(a.questions).read_text(encoding="utf-8"))["questions"] if q.get("text")]
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    only = {int(x) for x in a.presets.split(",") if x.strip()}
    entries = [e for e in manifest["presets"] if not only or int(e["preset"]) in only]

    rt = build_runtime(cfg)
    report, failures = [], []
    try:
        for e in entries:
            preset, kind = int(e["preset"]), e["kind"]
            tid, secret = start_session(a.app_dsn, OWNER, preset, e["episode_id"], int(e["display_window"]))
            rows = []
            for q in questions:
                if kind not in (q.get("kinds") or [kind]):
                    continue
                t0 = time.perf_counter()
                res = rt.invoke(tid, secret, q["text"].strip(), str(uuid.uuid4()))
                row = {"preset": preset, "kind": kind, "question": q["id"], "status": res.get("status"), "category": res.get("category"),
                       "refusal": (res.get("refusal") or {}).get("code"), "trace": _trace_rows(res), "ms": int((time.perf_counter() - t0) * 1000)}
                if res.get("status") == "waiting_approval":
                    rej = rt.resume(tid, secret, res["turn_id"], "reject", str(uuid.uuid4()))
                    row["resume_reject"] = rej.get("status")
                rows.append(row)
                bad_retrieve = [t for t in row["trace"] if t["tool"] == "retrieve" and t["status"] != "OK"]
                if res.get("status") not in ("done", "waiting_approval"):
                    failures.append(f"preset {preset} {q['id']}: status {res.get('status')} {res.get('error', '')}")
                if q["id"] == "evidence" and bad_retrieve:
                    failures.append(f"preset {preset} evidence: retrieve {bad_retrieve[0]['status']} {bad_retrieve[0]['reason']}")
                if preset in syn and q["id"] == "action" and res.get("status") != "waiting_approval":
                    failures.append(f"synthetic preset {preset}: action reached {res.get('status')} not waiting_approval (refusal={(res.get('refusal') or {}).get('code')})")
                print(f"preset {preset:>2} {q['id']:<9} {res.get('status'):<16} cat={res.get('category')} trace={[(t['tool'], t['status']) for t in row['trace']]}")
            report.extend(rows)
    finally:
        rt.close()
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps({"owner": OWNER, "manifest": a.manifest, "rows": report, "failures": failures}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"report {a.report}: {len(report)} turns, {len(failures)} failures")
    for f in failures:
        print("  FAIL", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
