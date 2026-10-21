"""core_pipeline/agent/manifest.py — the agent-facing demo manifest written by scripts/precompute_demo.py (ST4-14).

outputs/demo_presets.json holds, per preset: episode_id, display_window, kind (alarm | no_alarm), alarm_id, pred_set,
versions and an OBSERVATIONAL note. It never holds the true fault label — that lives in outputs/demo_presets_truth.json,
which the graph never reads (post-hoc reveal only). decide_preview is what the precompute observed at that time; the
chat always shows the CURRENT session's real decision (shared state may have moved on).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

DEFAULT_PATH = os.environ.get("SGP_DEMO_MANIFEST", "outputs/demo_presets.json")


def load_manifest(path: Optional[str] = None) -> dict:
    p = Path(path or DEFAULT_PATH)
    if not p.exists():
        raise FileNotFoundError(f"demo manifest {p} not found — run scripts.precompute_demo first (S4a-03)")
    m = json.loads(p.read_text(encoding="utf-8"))
    if m.get("schema") != "demo_presets/1":
        raise ValueError(f"unexpected manifest schema {m.get('schema')!r}")
    return m


def entry_for_preset(manifest: dict, preset: int) -> dict:
    for e in manifest["presets"]:
        if int(e["preset"]) == int(preset):
            return e
    raise KeyError(f"preset {preset} not in manifest")


def entry_for_episode(manifest: dict, episode_id: str) -> Optional[dict]:
    for e in manifest["presets"]:
        if str(e["episode_id"]) == str(episode_id):
            return e
    return None
