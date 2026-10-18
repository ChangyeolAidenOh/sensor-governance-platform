"""0-F pre-step: can this account actually reach the candidate models? (no generation, no billing)

Uses each vendor's list-models endpoint only. Keys are read from .env and never printed.
Candidate name fragments come from the plan (§4.1 0-F); the real model ids are whatever the
vendor lists, so the output is the authoritative id to use later.

    ANTHROPIC_API_KEY  -> GET https://api.anthropic.com/v1/models
    OPENAI_API_KEY     -> GET https://api.openai.com/v1/models
    GOOGLE_API_KEY     -> GET https://generativelanguage.googleapis.com/v1beta/models

Usage:  python scripts/llm_access_probe.py   (writes outputs/stage0_llm_access.json)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(".env")
OUT = Path("outputs/stage0_llm_access.json")

CANDIDATES = {   # plan candidate -> (vendor, substrings that must all appear in the model id, lowercase)
    "Claude Sonnet 5.5": ("anthropic", ["sonnet", "5"]),
    "Claude Haiku 5.5": ("anthropic", ["haiku", "5"]),
    "GPT-6.1 Sol": ("openai", ["gpt-6"]),
    "Gemini 4 Argon": ("google", ["gemini-4"]),
}


def list_models(vendor: str) -> tuple[list[str], str]:
    try:
        if vendor == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                return [], "no ANTHROPIC_API_KEY in .env"
            r = requests.get("https://api.anthropic.com/v1/models?limit=100",
                             headers={"x-api-key": key, "anthropic-version": "2023-06-01"}, timeout=30)
            if r.status_code != 200:
                return [], f"HTTP {r.status_code}: {r.text[:200]}"
            return [m.get("id", "") for m in r.json().get("data", [])], "ok"
        if vendor == "openai":
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                return [], "no OPENAI_API_KEY in .env"
            r = requests.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
            if r.status_code != 200:
                return [], f"HTTP {r.status_code}: {r.text[:200]}"
            return [m.get("id", "") for m in r.json().get("data", [])], "ok"
        if vendor == "google":
            key = os.environ.get("GOOGLE_API_KEY")
            if not key:
                return [], "no GOOGLE_API_KEY in .env"
            r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                             params={"key": key, "pageSize": 200}, timeout=30)
            if r.status_code != 200:
                return [], f"HTTP {r.status_code}: {r.text[:200].replace(key, '<KEY>')}"
            return [m.get("name", "").replace("models/", "") for m in r.json().get("models", [])], "ok"
    except requests.RequestException as exc:
        return [], f"request failed: {exc}"
    return [], "unknown vendor"


def main() -> None:
    report = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "vendors": {}, "candidates": {}}
    cache: dict[str, tuple[list[str], str]] = {}
    for vendor in ("anthropic", "openai", "google"):
        models, status = list_models(vendor)
        cache[vendor] = (models, status)
        report["vendors"][vendor] = {"status": status, "n_models": len(models)}
        print(f"[{vendor}] {status}; {len(models)} models listed")
    for cand, (vendor, frags) in CANDIDATES.items():
        models, status = cache[vendor]
        hits = [m for m in models if all(f in m.lower() for f in frags)]
        report["candidates"][cand] = {"vendor": vendor, "accessible": bool(hits), "matching_ids": hits[:10],
                                      "vendor_status": status}
        print(f"  {cand:<18} {'ACCESSIBLE' if hits else 'NOT FOUND'}  {hits[:5]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[report] {OUT}  (no keys stored)")


if __name__ == "__main__":
    main()
