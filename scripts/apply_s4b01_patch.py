"""scripts/apply_s4b01_patch.py — ST4-50 S4b-01: wire sql/010 into replay_client.SQL_FILES and make clean_episode remove
the episode's policy_input_snapshots BEFORE alarms / episode_windows (both are FK targets of the snapshot table).
policy_events / eval_runs are run-scoped ledgers and are deliberately NOT touched by clean_episode.
Idempotent: exits 0 without changes when already applied. Run from the repo root: python scripts/apply_s4b01_patch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PATH = Path("core_pipeline/skills/replay_client.py")
OLD_FILES = 'SQL_FILES = ("sql/003_skill_state.sql", "sql/004_skill_state_rev2.sql", "sql/005_gate_checks.sql", "sql/007_graph_state.sql")'
NEW_FILES = ('SQL_FILES = ("sql/003_skill_state.sql", "sql/004_skill_state_rev2.sql", "sql/005_gate_checks.sql", "sql/007_graph_state.sql",\n'
             '             "sql/010_policy_isolation.sql")')
OLD_CLEAN = """        alarms = [r[0] for r in cur.fetchall()]
        if alarms:
            cur.execute("SELECT proposal_id FROM mart.proposals WHERE alarm_id = ANY(%s)", (alarms,))"""
NEW_CLEAN = """        alarms = [r[0] for r in cur.fetchall()]
        # first-observation snapshots reference alarms and episode_windows (schema v8, ST4-50); policy_events / eval_runs
        # are evaluation-run ledgers, never deleted here
        cur.execute("DELETE FROM mart.policy_input_snapshots WHERE episode_id=%s", (episode_id,))
        if alarms:
            cur.execute("SELECT proposal_id FROM mart.proposals WHERE alarm_id = ANY(%s)", (alarms,))"""


def main() -> int:
    src = PATH.read_text(encoding="utf-8")
    if NEW_FILES in src and NEW_CLEAN in src:
        print("already applied")
        return 0
    for old, new, name in ((OLD_FILES, NEW_FILES, "SQL_FILES"), (OLD_CLEAN, NEW_CLEAN, "clean_episode")):
        n = src.count(old)
        if n != 1:
            print(f"{name}: expected exactly 1 match, found {n}; nothing written")
            return 1
        src = src.replace(old, new)
    PATH.write_text(src, encoding="utf-8")
    print("patched core_pipeline/skills/replay_client.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
