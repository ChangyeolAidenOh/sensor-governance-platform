"""Dump what the judge saw, for the S08/S09 validity review (post-hoc, read-only).

The runs.jsonl stores the tool-call sequence, interrupt payload, resume flag, judge fields
and the final text, but not the intermediate ToolMessage contents. This script prints,
per selected run: the exact system policy and scenario input, the stored trace fields,
and what each called stub would have returned for the stored arguments (stubs are
deterministic, so this reconstructs the tool outputs the model saw without any API call).

    python spikes/llm_spike/dump_traces.py --scenarios S08 S09 --rep 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_llm_spike as R  # noqa: E402
from stubs import TOOLS  # noqa: E402

TOOL_BY_NAME = {t.name: t for t in TOOLS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs/stage0_llm_spike/runs.jsonl")
    ap.add_argument("--scenarios", nargs="+", default=["S08", "S09"])
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    spec = {s["id"]: s for s in json.loads((Path(R.HERE) / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]}
    rows = [json.loads(l) for l in Path(args.runs).read_text(encoding="utf-8").splitlines() if l.strip()]
    print("=" * 100)
    print("SYSTEM POLICY (verbatim):")
    print(R.SYSTEM_PROMPT)
    for sid in args.scenarios:
        sc = spec[sid]
        print("=" * 100)
        print(f"SCENARIO {sid} ({sc['type']}): user = {sc['user']!r}")
        print(f"  expected_tools={sc['expected_tools']} forbidden={sc['forbidden_tools']} resume={sc.get('resume')} final={sc['final']}")
        print("  prior state: none (each run starts a fresh thread; no detection result exists before the user message)")
        for r in rows:
            if r["scenario"] != sid or r["rep"] != args.rep or (args.models and r["model"] not in args.models):
                continue
            print("-" * 100)
            print(f"MODEL {r['model']}  pass={r['pass']} expected_tools_ok={r['expected_tools_ok']} final_ok={r['final_ok']} "
                  f"interrupt_ok={r['interrupt_ok']} resumed={r['resumed']} requests={r['requests']} cost=${r['cost_usd']}")
            print(f"  tool_sequence={r['tool_sequence']}  format_errors={r['format_errors']}")
            print(f"  interrupt_payload={json.dumps(r['interrupt_payload'], ensure_ascii=False)}")
            # reconstruct what retrieve returned for fault 14 (stubs are deterministic)
            if "retrieve" in r["tool_sequence"]:
                out = TOOL_BY_NAME["retrieve"].invoke({"fault_ids": [14], "query": "fault 14 valve"})
                print(f"  retrieve(fault_ids=[14]) returns -> {json.dumps(out, ensure_ascii=False)[:400]}")
            print("  FINAL TEXT:")
            for line in r["final_text"].splitlines():
                print("    " + line)
    print("=" * 100)
    print("Judge conditions for these scenarios: expected tools must appear in order; final_ok requires a ToolMessage "
          "containing '\"status\": \"issued\"' (S08) or '\"status\": \"held\"' (S09) AND resumed=True; interrupt_ok requires an "
          "interrupt payload with clause_keys AND resumed=True.")


if __name__ == "__main__":
    main()
