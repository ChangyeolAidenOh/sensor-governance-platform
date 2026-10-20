"""core_pipeline/skills/replay_client.py — replay an episode window by window THROUGH the Skill boundary (MCP client),
the path S4a-03 (demo pre-computation), the goldens (S4a-01) and the Stage 5 default replay (ST4-01) share.
No models are imported here; everything goes through SkillClient. Also: test-DB cleanup helpers (destructive — the
callers must guard with is_test_dsn / _assert_disposable)."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Callable, Optional
from uuid import UUID

import psycopg

from core_pipeline import prereg_constants as P
from core_pipeline.skills.client import SkillClient
from core_pipeline.skills.servers.common import new_ctx

SQL_FILES = ("sql/003_skill_state.sql", "sql/004_skill_state_rev2.sql", "sql/005_gate_checks.sql")


def apply_stage4a_schema(dsn: str) -> None:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        for f in SQL_FILES:
            cur.execute(Path(f).read_text(encoding="utf-8"))
        conn.commit()


def episode_for_preset(dsn: str, preset: int) -> UUID:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT episode_id FROM staging.episodes WHERE demo_preset=%s", (preset,))
        row = cur.fetchone()
    if not row:
        raise RuntimeError(f"no episode with demo_preset={preset} in staging.episodes")
    return row[0]


def dev_episodes(dsn: str, n: int = 2) -> list[UUID]:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("""SELECT episode_id FROM mart.v_windows_agent WHERE "group"='dev' GROUP BY episode_id ORDER BY episode_id LIMIT %s""", (n,))
        return [r[0] for r in cur.fetchall()]


def clean_episode(dsn: str, episode_id: UUID) -> None:
    """Remove every Stage 4a execution row of one episode (FK order). Destructive: test DB only."""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT alarm_id FROM mart.alarms WHERE episode_id=%s", (episode_id,))
        alarms = [r[0] for r in cur.fetchall()]
        if alarms:
            cur.execute("SELECT proposal_id FROM mart.proposals WHERE alarm_id = ANY(%s)", (alarms,))
            props = [r[0] for r in cur.fetchall()]
            if props:
                cur.execute("DELETE FROM mart.work_orders WHERE proposal_id = ANY(%s)", (props,))
                cur.execute("DELETE FROM mart.approvals WHERE proposal_id = ANY(%s)", (props,))
                cur.execute("DELETE FROM mart.proposal_drafts WHERE proposal_id = ANY(%s)", (props,))
                cur.execute("DELETE FROM mart.proposals WHERE proposal_id = ANY(%s)", (props,))
            cur.execute("DELETE FROM mart.policy_decisions WHERE alarm_id = ANY(%s)", (alarms,))
            cur.execute("DELETE FROM mart.evidence WHERE alarm_id = ANY(%s)", (alarms,))
            cur.execute("DELETE FROM mart.diagnoses WHERE alarm_id = ANY(%s)", (alarms,))
            cur.execute("DELETE FROM mart.alarms WHERE alarm_id = ANY(%s)", (alarms,))
        cur.execute("DELETE FROM mart.episode_windows WHERE episode_id=%s", (episode_id,))
        cur.execute("DELETE FROM mart.gate_checks WHERE episode_id=%s", (episode_id,))
        # request records of this episode (ST4-13 bound inputs): otherwise a stored scoped retrieval would be replayed
        cur.execute("DELETE FROM mart.agent_steps WHERE input->'args'->>'episode_id'=%s OR input->'args'->>'alarm_id' = ANY(%s)",
                    (str(episode_id), alarms or [""]))
        conn.commit()


def window_row(env: dict) -> dict:
    """Compact per-window record from gate + detect envelopes (what the goldens freeze)."""
    st = env.get("state") or {}
    return {"t": st.get("window_index"), "gate": st.get("gate"), "drift": st.get("drift"), "p_value": st.get("p_value"),
            "p_adj": st.get("p_adj"), "alarm": st.get("alarm"), "m_t": st.get("m_t"), "pred_set": st.get("pred_set"),
            "top1": st.get("top1"), "alarm_id": st.get("alarm_id")}


async def replay_episode(sk: SkillClient, episode_id: UUID, thread_id: str, t_max: int = P.N_WINDOWS - 1,
                         run_uuid: Optional[UUID] = None, on_window: Optional[Callable[[int, dict, Optional[dict]], None]] = None) -> list[dict]:
    """gate_check → (non-BLOCK) detect_diagnose for windows 0..t_max. Returns one compact row per window."""
    run_uuid = run_uuid or uuid.uuid4()
    rows: list[dict] = []
    for t in range(t_max + 1):
        g = await sk.gate_check(new_ctx(thread_id, run_uuid), str(episode_id), t)
        if g["status"] != "OK":
            raise RuntimeError(f"gate_check({t}) {g['status']} {g['reason']}")
        d = None
        if g["payload"]["gate"] == "BLOCK":
            rows.append({**window_row(g), "gate_payload": g["payload"]})
        else:
            d = await sk.detect_diagnose(new_ctx(thread_id, run_uuid), str(episode_id), t)
            if d["status"] != "OK":
                raise RuntimeError(f"detect_diagnose({t}) {d['status']} {d['reason']}")
            rows.append({**window_row(d), "gate_payload": g["payload"], "diagnosis": d["payload"]["diagnosis"]})
        if on_window:
            on_window(t, g, d)
    return rows


def scoped_query(pred_set: list[int], fault_map_path: str = "outputs/fault_to_guidance_map.json") -> str:
    """Graph query template (contract §2.3): the Korean fault descriptions of the predicted set + the action question.
    Deterministic, no LLM. (rev 2 fix: `topic` is an English slug and gave the reranker nothing to match.)"""
    m = json.loads(Path(fault_map_path).read_text(encoding="utf-8"))
    descs = []
    for f in pred_set:
        e = m["faults"].get(str(f)) or m["faults"].get(f)
        if e and e.get("description") and e["description"] not in descs:
            descs.append(e["description"])
    head = "; ".join(descs) if descs else "공정 이상 경보"
    return f"{head} 이상 징후 발생 시 점검 및 조치 사항"
