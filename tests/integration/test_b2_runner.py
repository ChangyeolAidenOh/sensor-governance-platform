"""tests/integration/test_b2_runner.py — ST4-50 criterion 6: B2 reads raw samples + fixed calibration only. Needs the
processed TEP parquet, models/detector_v1/zscore_l0.* and the test DB (engine key lookup). No Skill servers."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import psycopg
import pytest

from core_pipeline.anomaly.zscore_l0 import ZScoreL0
from core_pipeline.eval.b2 import B2Runner, decide_b2
from core_pipeline.skills.replay_client import clean_episode, episode_for_preset

DSN = os.environ.get("SGP_TEST_PG_DSN")
ZDIR = Path(os.environ.get("SGP_DETECTOR_DIR", "models/detector_v1"))
PDIR = os.environ.get("SGP_PROCESSED_DIR", "data/processed/TEP")
pytestmark = pytest.mark.skipif(not DSN or not (ZDIR / "zscore_l0.npz").exists(), reason="test DB or zscore_l0 artefact missing")
ROOT = Path(__file__).resolve().parents[2]


def test_b2_module_imports_no_state_or_skill_code():
    tree = ast.parse((ROOT / "core_pipeline" / "eval" / "b2.py").read_text(encoding="utf-8"))
    mods = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not {m for m in mods if "state_store" in m or "skills.client" in m or "initial_snapshot" in m or "skills_detect" in m}, mods


def test_b2_matches_direct_zscore_and_ignores_db_state():
    ep = episode_for_preset(DSN, 9)
    runner = B2Runner(DSN, PDIR, ZDIR)
    model = ZScoreL0.load(ZDIR)
    direct = [bool(model.alarm(runner.view(ep, t).current())) for t in (0, 1, 8, 47)]
    before = [runner.alarm(ep, t) for t in (0, 1, 8, 47)]
    assert before == direct
    det = json.loads(Path("outputs/model_meta/detector_v1.json").read_text())
    clean_episode(DSN, ep)
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:                    # synthetic BLOCK + spent rows: B2 must not care
        cur.execute("""INSERT INTO mart.episode_windows (episode_id, window_index, detector_version, contract_sha256, gate, drift, drift_assessed,
                         p_value, p_adj, m_t, alarm, pred_set, set_size, top1, q_used)
                       VALUES (%s,0,%s,%s,'BLOCK','ok',true,NULL,NULL,0,NULL,'{}',0,NULL,false),
                              (%s,1,%s,%s,'PASS','ok',true,0.01,0.02,1,true,'{1}',1,1,true)""",
                    (ep, det["version"], det["input_contract_sha256"], ep, det["version"], det["input_contract_sha256"]))
        conn.commit()
    try:
        after = [runner.alarm(ep, t) for t in (0, 1, 8, 47)]
    finally:
        clean_episode(DSN, ep)
    assert after == before
    for a in before:
        assert decide_b2(a).action == ("propose" if a else "no_alarm")
    assert runner.max_abs_z(ep, 8) >= 0.0
