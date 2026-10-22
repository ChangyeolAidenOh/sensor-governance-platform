"""Dump the request body the installed langchain_anthropic adapter would send for the router / narrator requests.
No network call. Checks (ST4-20 paid-call preconditions): no `tools` key, max_tokens as configured, how
reasoning_effort="medium" is translated (thinking budget / effort field), model id, installed versions.

    python scripts/llm_payload_check.py            # prints JSON report, exit 1 on a failed check
"""
from __future__ import annotations

import importlib.metadata as md
import json
import sys

from core_pipeline.agent.llm_client import ALLOWED_MODELS, DEFAULT_MODEL, NARRATE_MAX_TOKENS, ROUTER_MAX_TOKENS, AnthropicClient
from core_pipeline.agent.llm_narrator import build_request as narrate_request
from core_pipeline.agent.llm_router import build_request as router_request


def version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "not installed"


def main() -> int:
    client = AnthropicClient()
    state = {"display_window": 8, "kind": "alarm", "alarm_id": "x", "turn_id": "t", "thread_id": "th", "run_uuid": "r"}
    reqs = {"router": router_request("지금 상태 어때?", state, DEFAULT_MODEL), "narrate": narrate_request("창 8: gate=PASS", DEFAULT_MODEL)}
    report = {"versions": {p: version(p) for p in ("langchain-anthropic", "anthropic", "langchain-core")}, "allowed_models": ALLOWED_MODELS,
              "payloads": {}, "checks": []}
    ok = True
    for name, req in reqs.items():
        try:
            payload = client.request_payload(req)
        except Exception as e:
            report["payloads"][name] = {"error": f"{type(e).__name__}: {e}"}
            report["checks"].append({"request": name, "check": "payload captured", "ok": False})
            ok = False
            continue
        shown = {k: (v if k != "messages" else "<messages omitted>") for k, v in payload.items()}
        report["payloads"][name] = shown
        want_max = ROUTER_MAX_TOKENS if name == "router" else NARRATE_MAX_TOKENS
        checks = [("no tools in request", "tools" not in payload and "tool_choice" not in payload),
                  ("max_tokens as configured", payload.get("max_tokens") == want_max),
                  ("model id", payload.get("model") == req.model),
                  ("effort/thinking field present (manual read)", any(k in payload for k in ("thinking", "effort", "output_config", "reasoning", "reasoning_effort")))]
        for label, res in checks:
            report["checks"].append({"request": name, "check": label, "ok": bool(res)})
            ok = ok and bool(res)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
