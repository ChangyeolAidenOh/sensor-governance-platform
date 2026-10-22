"""Prepare the TEST DB for the synthetic approval-UI check (ST4-29): replay one preset through the Skill servers (as the
test fixture `world` of tests/integration/test_graph_agent.py does) and write an untracked manifest for the demo.

    python scripts/demo_synthetic_prepare.py --preset 1
    SGP_DEMO_SYNTHETIC=1 SGP_AGENT_PG_DSN="$SGP_AGENT_TEST_PG_DSN" SGP_PG_DSN="$SGP_TEST_PG_DSN" \
        SGP_DEMO_MANIFEST=outputs/demo_presets_test.json streamlit run demo/app.py

Refuses any DSN whose dbname does not contain "test". Uses the replay helpers of core_pipeline.skills.replay_client
(episode_for_preset / clean_episode / replay_episode, as in the test fixture); if they live elsewhere the ImportError
names what is missing. Writes outputs/demo_presets_test.json (schema demo_presets/1, same fields as the test fixture).
Role grants are not touched; rerun scripts/apply_role_grants.py on the test DSN if a destructive test ran in between.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid

from dotenv import load_dotenv
from psycopg.conninfo import conninfo_to_dict

OUT = "outputs/demo_presets_test.json"


def main() -> int:
    load_dotenv(".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", type=int, default=1)
    ap.add_argument("--dsn", default=None, help="admin TEST DSN (default SGP_TEST_PG_DSN)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n-windows", type=int, default=48)
    a = ap.parse_args()
    dsn = a.dsn or os.environ.get("SGP_TEST_PG_DSN")
    if not dsn:
        print("SGP_TEST_PG_DSN is required", file=sys.stderr)
        return 2
    if "test" not in (conninfo_to_dict(dsn).get("dbname") or ""):
        print("refusing: the DSN must name a database containing 'test'", file=sys.stderr)
        return 2
    try:
        from core_pipeline.skills import replay_client as RC
        episode_for_preset, clean_episode, replay_episode = RC.episode_for_preset, RC.clean_episode, RC.replay_episode
    except (ImportError, AttributeError) as e:
        print(f"replay helpers not found in core_pipeline.skills.replay_client ({e}); send the import lines of tests/integration/test_graph_agent.py", file=sys.stderr)
        return 3
    from core_pipeline.skills.client import SkillClient

    env = {"SGP_PG_DSN": dsn}
    entry: dict = {}

    async def replay():
        async with SkillClient(env=env) as sk:
            ep = episode_for_preset(dsn, a.preset)
            clean_episode(dsn, ep)
            rows = await replay_episode(sk, ep, f"demo-synthetic-{uuid.uuid4().hex[:6]}", mode="replay")
            alarm = next((r for r in rows if r["alarm"] is True), None)
            entry.update({"preset": a.preset, "episode_id": str(ep), "display_window": alarm["t"] if alarm else a.n_windows - 1,
                          "kind": "alarm" if alarm else "no_alarm", "alarm_id": alarm["alarm_id"] if alarm else None,
                          "pred_set": alarm["pred_set"] if alarm else []})
    asyncio.run(replay())
    manifest = {"schema": "demo_presets/1", "generated_for": "synthetic approval-UI check on the test DB (ST4-29) — NOT a demo manifest",
                "presets": [entry]}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(manifest, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({"out": a.out, "entry": entry}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
