"""demo/public_guard.py: usage limits of the PUBLIC demo (demo-public branch).

Counts come from the tables the app already writes (mart.threads, mart.graph_turns), so no new table or grant is needed
and the limits survive a container restart. Days are UTC. The limits protect the hosted Postgres and the free compute
quota; they are not a billing cap on the Streamlit websocket itself (docs/public_demo_deploy.md explains the rest).
"""
from __future__ import annotations

from typing import Optional

import psycopg

PUBLIC_OWNER = "public"
SESSION_MAX_TURNS = 15
SESSION_MAX_MINUTES = 20
DAY_MAX_TURNS = 300
DAY_MAX_MINUTES = 90


def session_usage(dsn: str, thread_id: str) -> dict:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("""SELECT count(g.turn_id), extract(epoch FROM (now() - t.created_at)) / 60.0
                       FROM mart.threads t LEFT JOIN mart.graph_turns g USING (thread_id)
                       WHERE t.thread_id=%s GROUP BY t.created_at""", (thread_id,))
        r = cur.fetchone()
    return {"turns": int(r[0]) if r else 0, "age_min": float(r[1]) if r else 0.0}


def day_usage(dsn: str) -> dict:
    """Turns and (capped) session minutes of today's public threads."""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("""SELECT count(g.turn_id),
                              least(extract(epoch FROM (coalesce(max(g.created_at), t.created_at) - t.created_at)) / 60.0, %s)
                       FROM mart.threads t LEFT JOIN mart.graph_turns g USING (thread_id)
                       WHERE t.owner=%s AND t.created_at >= date_trunc('day', now())
                       GROUP BY t.thread_id, t.created_at""", (SESSION_MAX_MINUTES, PUBLIC_OWNER))
        rows = cur.fetchall()
    return {"turns": int(sum(r[0] for r in rows)), "minutes": float(sum(r[1] for r in rows)), "sessions": len(rows)}


def check_new_session(dsn: str) -> Optional[str]:
    d = day_usage(dsn)
    if d["turns"] >= DAY_MAX_TURNS:
        return f"오늘의 공개 데모 턴 한도({DAY_MAX_TURNS}턴)에 도달했습니다. 내일(UTC 기준) 다시 열립니다."
    if d["minutes"] >= DAY_MAX_MINUTES:
        return f"오늘의 공개 데모 사용 시간 한도({DAY_MAX_MINUTES}분)에 도달했습니다. 내일(UTC 기준) 다시 열립니다."
    return None


def check_turn(dsn: str, thread_id: str) -> Optional[str]:
    s = session_usage(dsn, thread_id)
    if s["turns"] >= SESSION_MAX_TURNS:
        return f"이 세션의 턴 한도({SESSION_MAX_TURNS}턴)에 도달했습니다. 새 세션을 시작해 주세요."
    if s["age_min"] >= SESSION_MAX_MINUTES:
        return f"세션 시간 한도({SESSION_MAX_MINUTES}분)가 지났습니다. 새 세션을 시작해 주세요."
    return check_new_session(dsn)


def public_limit_hit(view: dict) -> bool:
    """True when a retrieve in this turn was refused because its result was not pre-computed (StoredRetrieve DEMO_LIMIT)."""
    for tr in view.get("trace") or []:
        if tr.get("tool") == "retrieve" and tr.get("reason") == "DEMO_LIMIT":
            return True
    return False
