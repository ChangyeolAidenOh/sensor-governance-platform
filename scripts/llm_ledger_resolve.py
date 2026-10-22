"""Manual resolution of LLM ledger rows (ST4-21). Every change is a conditional UPDATE with evidence; nothing is re-called.

    python scripts/llm_ledger_resolve.py --list                      # all llm rows + totals
    python scripts/llm_ledger_resolve.py --list-pending
    python scripts/llm_ledger_resolve.py --list-blocked                 # usage-missing or unresolved-exceeded rows
    python scripts/llm_ledger_resolve.py --settle <request_id> --input-tokens N --output-tokens M --evidence "<vendor console ref>"
        PENDING (lost response) -> OK at the real usage; an overrun found here is recorded, not resolved
    python scripts/llm_ledger_resolve.py --void <request_id> --evidence "<vendor console shows no request>"
    python scripts/llm_ledger_resolve.py --settle-usage <request_id> --input-tokens N --output-tokens M --evidence "<vendor console ref>"
        settled row with usage_missing=true -> real usage recorded (clears usage_missing; may raise reservation_exceeded)
    python scripts/llm_ledger_resolve.py --resolve-exceeded <request_id> --evidence "<price/formula check ref>"
        known overrun -> cleared (refused while usage is still missing)

DSN from SGP_PG_DSN (or --dsn). --by defaults to $USER.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from uuid import UUID

from dotenv import load_dotenv

from core_pipeline.agent.llm_ledger import PostgresLedger


def main() -> int:
    load_dotenv(".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.environ.get("SGP_PG_DSN"))
    ap.add_argument("--by", default=os.environ.get("USER", "operator"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--list-pending", action="store_true")
    ap.add_argument("--list-blocked", action="store_true")
    ap.add_argument("--settle", metavar="REQUEST_ID")
    ap.add_argument("--void", metavar="REQUEST_ID")
    ap.add_argument("--settle-usage", metavar="REQUEST_ID")
    ap.add_argument("--resolve-exceeded", metavar="REQUEST_ID")
    ap.add_argument("--input-tokens", type=int)
    ap.add_argument("--output-tokens", type=int)
    ap.add_argument("--evidence", default="")
    a = ap.parse_args()
    if not a.dsn:
        print("SGP_PG_DSN is required", file=sys.stderr)
        return 2
    led = PostgresLedger(a.dsn)
    if a.list or a.list_pending or a.list_blocked:
        rows = led.list_rows(pending_only=a.list_pending, blocked_only=a.list_blocked)
        for r in rows:
            r.pop("output", None)
        print(json.dumps({"rows": rows, "totals": led.totals()}, ensure_ascii=False, indent=2, default=str))
        return 0
    if a.settle:
        if a.input_tokens is None or a.output_tokens is None:
            print("--settle requires --input-tokens and --output-tokens", file=sys.stderr)
            return 2
        out = led.manual_settle(UUID(a.settle), a.input_tokens, a.output_tokens, evidence=a.evidence, by=a.by)
    elif a.void:
        out = led.manual_void(UUID(a.void), evidence=a.evidence, by=a.by)
    elif a.settle_usage:
        if a.input_tokens is None or a.output_tokens is None:
            print("--settle-usage requires --input-tokens and --output-tokens", file=sys.stderr)
            return 2
        out = led.settle_usage(UUID(a.settle_usage), a.input_tokens, a.output_tokens, evidence=a.evidence, by=a.by)
    elif a.resolve_exceeded:
        out = led.resolve_exceeded(UUID(a.resolve_exceeded), evidence=a.evidence, by=a.by)
    else:
        ap.print_help()
        return 2
    print(json.dumps({"result": {k: out.get(k) for k in ("cost_usd", "usage_missing", "reservation_exceeded", "exceeded_resolved", "resolution", "usage_resolution", "exceeded_resolution")},
                      "totals": led.totals()}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
