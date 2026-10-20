"""tests/unit/test_client_root.py — the client resolves the repo root (reviewer 2026-10-21 point 1a, rejected: the file
lives at core_pipeline/skills/client.py, so parents[2] IS the root). Guards against the module moving."""
from pathlib import Path

from core_pipeline.skills.client import ROOT, SERVERS


def test_root_is_repo_root():
    assert (ROOT / "core_pipeline").is_dir() and ROOT.name != "core_pipeline"
    for p in SERVERS.values():
        assert p.exists(), p
