"""0-F report: per-model metrics from runs.jsonl and the ST0-22 decision rule.

    python spikes/llm_spike/report.py --runs outputs/stage0_llm_spike/runs.jsonl

Decision rule (plan §4.1, ST0-22): among candidates with zero format errors over all runs,
pick the lowest mean cost per request; tie -> lowest mean latency per request. Candidates
with any exception (transport) are reported but not excluded on that basis alone; runs
flagged needs_judge are listed for the semantic judge (separate step, separate cost).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 4) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs/stage0_llm_spike/runs.jsonl")
    ap.add_argument("--out-json", default="outputs/stage0_llm_selection.json")
    ap.add_argument("--out-md", default="docs/llm_selection.md")
    ap.add_argument("--exclude", nargs="*", default=["fake"])
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.runs).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["model"] not in args.exclude]
    models = sorted({r["model"] for r in rows})
    per = {}
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        intr = [r for r in rs if r["type"] == "interrupt" or (r["type"] == "chain" and r.get("interrupt_ok") is not None)]
        oos = [r for r in rs if r["type"] == "oos"]
        per[m] = {
            "runs": len(rs), "exceptions": sum(1 for r in rs if r["exception"]),
            "pass_rate": round(sum(1 for r in rs if r["pass"]) / len(rs), 3),
            "format_error_runs": sum(1 for r in rs if r["n_format_errors"] > 0),
            "format_error_rate": round(sum(1 for r in rs if r["n_format_errors"] > 0) / len(rs), 3),
            "expected_tools_rate": round(sum(1 for r in rs if r["expected_tools_ok"]) / len(rs), 3),
            "forbidden_tool_runs": sum(1 for r in rs if r["forbidden_called"]),
            "interrupt_resume_rate": round(sum(1 for r in intr if r.get("interrupt_ok")) / len(intr), 3) if intr else None,
            "oos_refusal_rate": round(sum(1 for r in oos if r["final_ok"]) / len(oos), 3) if oos else None,
            "needs_judge_runs": [f"{r['scenario']}r{r['rep']}" for r in rs if r.get("needs_judge")],
            "cost_total_usd": round(sum(r["cost_usd"] for r in rs), 4),
            "cost_per_run_usd": mean([r["cost_usd"] for r in rs]),
            "cost_per_request_usd": round(sum(r["cost_usd"] for r in rs) / max(1, sum(r["requests"] for r in rs)), 6),
            "requests_per_run": mean([r["requests"] for r in rs]),
            "output_tokens_per_request": round(sum(r["output_tokens"] for r in rs) / max(1, sum(r["requests"] for r in rs)), 1),
            "reasoning_tokens_total": sum(r.get("reasoning_tokens", 0) for r in rs),
            "latency_per_request_mean": mean([r["latency_per_request_mean"] for r in rs]),
            "wall_per_run_mean": mean([r["wall_seconds"] for r in rs]),
        }
    eligible = [m for m in models if per[m]["format_error_runs"] == 0 and per[m]["exceptions"] == 0]
    ranked = sorted(eligible, key=lambda m: (per[m]["cost_per_request_usd"], per[m]["latency_per_request_mean"] or 1e9))
    selected = ranked[0] if ranked else None
    judge_candidates = [m for m in models if m != selected and m.split("-")[0] != (selected or "").split("-")[0]] if selected else []
    judge = min(judge_candidates, key=lambda m: per[m]["cost_per_request_usd"]) if judge_candidates else None
    decision = {"rule": "zero format errors and zero exceptions -> min cost per request -> min latency per request",
                "eligible": eligible, "ranking": ranked, "selected": selected, "judge": judge,
                "all_failed": not eligible, "note": "runs with needs_judge still require the semantic judge pass before the OOS metric is final"}
    out = {"per_model": per, "decision": decision, "n_runs": len(rows)}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    cols = ["runs", "pass_rate", "format_error_rate", "expected_tools_rate", "forbidden_tool_runs", "interrupt_resume_rate",
            "oos_refusal_rate", "exceptions", "cost_per_request_usd", "cost_per_run_usd", "requests_per_run",
            "output_tokens_per_request", "latency_per_request_mean"]
    md = ["# 0-F LLM selection (spike result)", "",
          f"Runs: {len(rows)} · rule: {decision['rule']}", "",
          "| model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for m in models:
        md.append(f"| {m} | " + " | ".join(str(per[m][c]) for c in cols) + " |")
    md += ["", f"**Eligible (0 format errors, 0 exceptions):** {eligible or 'none'}",
           f"**Ranking:** {ranked}", f"**Selected:** {selected}", f"**Judge (different vendor, cheapest):** {judge}", ""]
    for m in models:
        if per[m]["needs_judge_runs"]:
            md.append(f"- {m}: needs semantic judge for {per[m]['needs_judge_runs']}")
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=1, ensure_ascii=False))
    print(f"[report] {args.out_json}, {args.out_md}")


if __name__ == "__main__":
    main()
