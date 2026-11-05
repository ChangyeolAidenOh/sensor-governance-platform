"""Unit tests of the public-demo additions (demo-public branch): no database, no servers.

    pytest tests/unit/test_public_demo.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "outputs" / "public_demo_questions.json"


def test_public_questions_route_to_their_category():
    from core_pipeline.agent.router import RuleRouter
    qs = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    r = RuleRouter()
    assert [q["id"] for q in qs] == ["status", "diagnosis", "evidence", "history", "method", "action", "oos"]
    for q in qs:
        assert r.route(q["text"], {}).category == q["category"], q["id"]
    assert all(q["text"] == q["text"].strip() for q in qs)        # the app strips messages; stored rows are keyed on the exact text


def test_public_limit_hit_reads_the_trace():
    from demo.public_guard import public_limit_hit
    assert public_limit_hit({"trace": [{"tool": "retrieve", "status": "REFUSED", "reason": "DEMO_LIMIT"}]})
    assert not public_limit_hit({"trace": [{"tool": "retrieve", "status": "OK", "reason": None}]})
    assert not public_limit_hit({"trace": [{"tool": "gate_check", "status": "REFUSED", "reason": "DEMO_LIMIT"}]})
    assert not public_limit_hit({})


def test_load_config_public_rules(tmp_path):
    from demo.runtime import DemoConfigError, load_config
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"schema": "demo_presets/1", "presets": []}), encoding="utf-8")
    base = {"SGP_AGENT_PG_DSN": "postgresql://sgp_agent:a@db.example:5432/sgp_v2", "SGP_PG_DSN": "postgresql://sgp_skill:b@db.example:5432/sgp_v2",
            "SGP_DEMO_MANIFEST": str(manifest)}
    cfg = load_config({**base, "SGP_PUBLIC_DEMO": "1", "SGP_DEMO_SYNTHETIC_PRESETS": "11"})
    assert cfg.public and cfg.synthetic_presets == (11,) and not cfg.synthetic
    cfg = load_config(base)
    assert not cfg.public and cfg.synthetic_presets == ()
    with pytest.raises(DemoConfigError):                                 # synthetic presets need public mode
        load_config({**base, "SGP_DEMO_SYNTHETIC_PRESETS": "11"})
    with pytest.raises(DemoConfigError):                                 # the test-DB flag and the public flag do not combine
        load_config({**base, "SGP_PUBLIC_DEMO": "1", "SGP_DEMO_SYNTHETIC": "1", "SGP_DEMO_SYNTHETIC_PRESETS": "11"})
    with pytest.raises(DemoConfigError):                                 # ST4-30 still applies in public mode
        load_config({**base, "SGP_PUBLIC_DEMO": "1", "SGP_PG_DSN": base["SGP_AGENT_PG_DSN"]})


def test_fingerprint_separates_public_from_local(tmp_path):
    from demo.runtime import fingerprint, load_config
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"schema": "demo_presets/1", "presets": []}), encoding="utf-8")
    base = {"SGP_AGENT_PG_DSN": "postgresql://sgp_agent:a@db.example:5432/sgp_v2", "SGP_PG_DSN": "postgresql://sgp_skill:b@db.example:5432/sgp_v2",
            "SGP_DEMO_MANIFEST": str(manifest)}
    a = fingerprint(load_config(base), base)
    b = fingerprint(load_config({**base, "SGP_PUBLIC_DEMO": "1"}), base)
    c = fingerprint(load_config({**base, "SGP_PUBLIC_DEMO": "1", "SGP_DEMO_SYNTHETIC_PRESETS": "11"}), base)
    assert len({a, b, c}) == 3


def test_stored_retrieve_refuses_unknown_open_query(monkeypatch):
    from core_pipeline.skills.public_skills import PUBLIC_LIMIT_MESSAGE, StoredRetrieve
    from core_pipeline.skills.servers.common import Ctx

    class Inner:
        pass
    sr = StoredRetrieve(Inner(), "postgresql://x:y@localhost:1/none")
    monkeypatch.setattr(sr, "_stored_open", lambda *a, **k: None)
    ctx = Ctx.parse({"request_id": "11111111-1111-4111-8111-111111111111", "thread_id": "s-x", "run_uuid": "22222222-2222-4222-8222-222222222222",
                     "caller_role": "engineer", "mode": "chat"})
    out = sr._answer(ctx, "이 문장은 저장되어 있지 않다", "open", [], 5, None)
    assert out["status"] == "REFUSED" and out["reason"]["code"] == "DEMO_LIMIT" and out["reason"]["message"] == PUBLIC_LIMIT_MESSAGE
    assert sr.misses and sr.misses[0]["mode"] == "open"


def test_manifest_finalizer_labels_synthetic(tmp_path):
    raw = {"schema": "demo_presets/1", "presets": [{"preset": 1, "kind": "alarm", "episode_id": "e1", "display_window": 8, "alarm_id": "a1", "versions": {}},
                                                   {"preset": 11, "kind": "alarm", "episode_id": "e11", "display_window": 9, "alarm_id": "a11", "versions": {}}]}
    src, out = tmp_path / "raw.json", tmp_path / "out.json"
    src.write_text(json.dumps(raw), encoding="utf-8")
    r = subprocess.run([sys.executable, "scripts/public_demo_manifest.py", "--raw", str(src), "--out", str(out), "--synthetic-presets", "11"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["public"] is True and doc["synthetic_presets"] == [11]
    kinds = {e["preset"]: (e["kind"], e["synthetic"]) for e in doc["presets"]}
    assert kinds == {1: ("alarm", False), 11: ("synthetic", True)}
    raw["presets"][1]["kind"] = "no_alarm"
    src.write_text(json.dumps(raw), encoding="utf-8")
    r = subprocess.run([sys.executable, "scripts/public_demo_manifest.py", "--raw", str(src), "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2
