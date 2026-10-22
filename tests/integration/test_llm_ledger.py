"""S4a-05 ledger integration (ST4-21) on the TEST database only (ST4-09): SGP_TEST_PG_DSN with 'test' in the DB name.
No LLM call is made; rows are written and removed per test (run_uuid scoped). Requires no pre-existing unresolved
exceeded llm row on the test DB (the check is global by design).

    SGP_TEST_PG_DSN=... python -m pytest tests/integration/test_llm_ledger.py -q
"""
from __future__ import annotations

import os
import threading
import uuid

import psycopg
import pytest

from core_pipeline.agent.llm_client import LLMRequest, LLMResponse, cost_usd
from core_pipeline.agent.llm_ledger import PostgresLedger, request_id
from core_pipeline.skills.servers.common import is_test_dsn

DSN = os.environ.get("SGP_TEST_PG_DSN", "")
if not DSN or not is_test_dsn(DSN):
    pytest.fail("SGP_TEST_PG_DSN must be set and contain 'test' in the database name (ST4-09)", pytrace=False)


def req(user: str = "u", max_tokens: int = 100) -> LLMRequest:
    return LLMRequest(model="claude-haiku-5-5", system="s", user=user, max_tokens=max_tokens, timeout_s=1)


def resp(i: int, o: int, known: bool = True) -> LLMResponse:
    return LLMResponse(text="{}", input_tokens=i, output_tokens=o, latency_ms=1, model="claude-haiku-5-5", usage_known=known)


@pytest.fixture
def run(monkeypatch):
    led = PostgresLedger(DSN)
    t = led.totals()
    if t["blocked_unresolved"]:
        pytest.fail("test DB has an unresolved blocking llm row; resolve it first (scripts/llm_ledger_resolve.py --list-blocked)")
    run_uuid, thread = str(uuid.uuid4()), f"t-{uuid.uuid4()}"
    yield led, thread, run_uuid, t["spent_usd"]
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM mart.agent_steps WHERE run_uuid=%s", (uuid.UUID(run_uuid),))
        cur.execute("DELETE FROM mart.agent_runs WHERE run_uuid=%s", (uuid.UUID(run_uuid),))
        conn.commit()


def row(rid):
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT status, cost_usd, input_tokens, output_tokens, violation, reason_code, output, input FROM mart.agent_steps WHERE request_id=%s", (rid,))
        return cur.fetchone()


def test_budget_unset_refuses_and_writes_nothing(run, monkeypatch):
    led, thread, run_uuid, _ = run
    monkeypatch.delenv("SGP_LLM_BUDGET_USD", raising=False)
    r = led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="llm:router:t1", attempt=1, tool="router", req=req())
    assert r.status == "budget_refused" and row(r.request_id) is None


def test_reserve_settle_replay_mismatch(run, monkeypatch):
    led, thread, run_uuid, spent0 = run
    monkeypatch.setenv("SGP_LLM_BUDGET_USD", str(spent0 + 1.0))
    q = req()
    r = led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="llm:router:t1", attempt=1, tool="router", req=q)
    assert r.status == "owned" and r.request_id == request_id(thread, run_uuid, "llm:router:t1", 1)
    st = row(r.request_id)
    assert st[0] == "PENDING" and float(st[1]) == pytest.approx(q.reserved_usd()) and st[7]["reserve"]["estimated_input_tokens"] == q.estimated_input_tokens()
    assert led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="llm:router:t1", attempt=1, tool="router", req=q).status == "in_progress"
    out = led.settle(r.request_id, resp(10, 5), req=q, output={"merged": {"category": "alarm"}}, violation=False)
    st = row(r.request_id)
    assert st[0] == "OK" and float(st[1]) == pytest.approx(cost_usd("claude-haiku-5-5", 10, 5)) and st[2] == 10 and st[3] == 5
    assert st[7]["reserve"]["reserved_usd"] == pytest.approx(q.reserved_usd())        # reservation immutable after settle
    assert out["calibration"]["actual_input_tokens"] == 10 and st[6]["reservation_exceeded"] is False
    rp = led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="llm:router:t1", attempt=1, tool="router", req=q)
    assert rp.status == "replay" and rp.output["merged"]["category"] == "alarm"
    assert led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="llm:router:t1", attempt=1, tool="router", req=req("other")).status == "mismatch"
    with pytest.raises(RuntimeError):
        led.settle(r.request_id, resp(1, 1), req=q, output={})                    # not PENDING any more


def test_cap_bounds_concurrent_reservations(run, monkeypatch):
    led, thread, run_uuid, spent0 = run
    q = req()
    monkeypatch.setenv("SGP_LLM_BUDGET_USD", str(spent0 + q.reserved_usd() * 1.5))
    a = led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="a", attempt=1, tool="router", req=q)
    b = led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="b", attempt=1, tool="router", req=q)
    assert a.status == "owned" and b.status == "budget_refused" and row(b.request_id) is None
    led.settle(a.request_id, resp(1, 1), req=q, output={})                        # actual far below the reservation
    assert led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="b", attempt=1, tool="router", req=q).status == "owned"


def test_reservation_exceeded_blocks_with_budget_headroom_until_resolved(run, monkeypatch):
    """Cap has plenty of headroom; the actual cost exceeded its own reservation → no new reservation until resolved."""
    led, thread, run_uuid, spent0 = run
    q = req(max_tokens=10)
    monkeypatch.setenv("SGP_LLM_BUDGET_USD", str(spent0 + 100.0))
    a = led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="a", attempt=1, tool="narrate", req=q)
    out = led.settle(a.request_id, resp(20_000, 20_000), req=q, output={})
    actual = cost_usd("claude-haiku-5-5", 20_000, 20_000)
    assert out["reservation_exceeded"] is True and out["exceeded_resolved"] is False and float(row(a.request_id)[1]) == pytest.approx(actual)
    assert actual > a.reserved_usd and led.totals()["spent_usd"] < spent0 + 100.0
    b = led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="b", attempt=1, tool="narrate", req=q)
    assert b.status == "exceeded_unresolved" and row(b.request_id) is None
    with pytest.raises(ValueError):
        led.resolve_exceeded(a.request_id, evidence="", by="test")
    with pytest.raises(RuntimeError):
        led.settle_usage(a.request_id, 1, 1, evidence="x", by="test")              # usage is known: not a usage-missing row
    led.resolve_exceeded(a.request_id, evidence="price table ST0-22 re-checked; formula v1 margin too small", by="test")
    with pytest.raises(RuntimeError):
        led.resolve_exceeded(a.request_id, evidence="again", by="test")              # already resolved
    assert row(a.request_id)[6]["exceeded_resolved"] is True
    assert led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="b", attempt=1, tool="narrate", req=q).status == "owned"


def test_unknown_usage_blocks_until_real_usage_is_recorded(run, monkeypatch):
    """usage_missing is cleared only by settle_usage (real tokens), never by resolve_exceeded's evidence string."""
    led, thread, run_uuid, spent0 = run
    q = req(max_tokens=10)
    monkeypatch.setenv("SGP_LLM_BUDGET_USD", str(spent0 + 100.0))
    a = led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="a", attempt=1, tool="router", req=q)
    out = led.settle(a.request_id, resp(0, 0, known=False), req=q, output={})
    assert out["usage_missing"] is True and out["reservation_exceeded"] is False
    assert float(row(a.request_id)[1]) == pytest.approx(q.reserved_usd()) and row(a.request_id)[5] == "LLM_USAGE_MISSING"
    assert led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="b", attempt=1, tool="router", req=q).status == "exceeded_unresolved"
    with pytest.raises(RuntimeError):
        led.resolve_exceeded(a.request_id, evidence="vendor console usage: 120/40 tokens", by="test")
    assert led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="b", attempt=1, tool="router", req=q).status == "exceeded_unresolved"
    with pytest.raises(ValueError):
        led.settle_usage(a.request_id, -5, 40, evidence="x", by="test")
    out = led.settle_usage(a.request_id, 30_000, 30_000, evidence="vendor console usage: 30000/30000", by="test")
    st = row(a.request_id)
    assert out["usage_missing"] is False and out["reservation_exceeded"] is True and out["exceeded_resolved"] is False
    assert st[2] == 30_000 and float(st[1]) == pytest.approx(cost_usd("claude-haiku-5-5", 30_000, 30_000))
    assert led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="b", attempt=1, tool="router", req=q).status == "exceeded_unresolved"
    led.resolve_exceeded(a.request_id, evidence="formula v1 margin re-checked", by="test")
    assert led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="b", attempt=1, tool="router", req=q).status == "owned"


def test_manual_settle_uses_stored_prices_and_does_not_auto_resolve(run, monkeypatch):
    led, thread, run_uuid, spent0 = run
    q = req(max_tokens=10)
    monkeypatch.setenv("SGP_LLM_BUDGET_USD", str(spent0 + 100.0))
    a = led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="a", attempt=1, tool="router", req=q)
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:                  # price pair preserved in the row, independent of the global table
        cur.execute("""UPDATE mart.agent_steps SET input = jsonb_set(input, '{reserve,prices_per_mtok}', '[1.0, 2.0]'::jsonb) WHERE request_id=%s""", (a.request_id,))
        conn.commit()
    with pytest.raises(ValueError):
        led.manual_settle(a.request_id, 100, -1, evidence="x", by="test")
    out = led.manual_settle(a.request_id, 1_000_000, 1_000_000, evidence="vendor console", by="test")
    assert float(row(a.request_id)[1]) == pytest.approx(3.0)                   # 1.0 + 2.0 at the stored prices, not 0.10/0.50
    assert out["reservation_exceeded"] is True and out["exceeded_resolved"] is False
    assert led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="b", attempt=1, tool="router", req=q).status == "exceeded_unresolved"
    led.resolve_exceeded(a.request_id, evidence="checked", by="test")


def test_no_response_keeps_pending_and_manual_resolution(run, monkeypatch):
    led, thread, run_uuid, spent0 = run
    q = req()
    monkeypatch.setenv("SGP_LLM_BUDGET_USD", str(spent0 + 100.0))
    a = led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="a", attempt=1, tool="router", req=q)
    led.note_no_response(a.request_id, "timeout after 30s")
    st = row(a.request_id)
    assert st[0] == "PENDING" and st[6]["no_response"]["note"] == "timeout after 30s" and float(st[1]) == pytest.approx(q.reserved_usd())
    assert led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="a", attempt=1, tool="router", req=q).status == "in_progress"
    assert led.totals()["pending"] >= 1
    with pytest.raises(ValueError):
        led.manual_void(a.request_id, evidence="", by="test")
    out = led.manual_settle(a.request_id, 120, 40, evidence="vendor console request abc", by="test")
    st = row(a.request_id)
    assert st[0] == "OK" and st[5] == "LLM_MANUAL_SETTLE" and float(st[1]) == pytest.approx(cost_usd("claude-haiku-5-5", 120, 40))
    assert out["resolution"]["kind"] == "manual_settle" and out["resolution"]["evidence"] and out["reservation_exceeded"] is False
    with pytest.raises(RuntimeError):
        led.manual_settle(a.request_id, 1, 1, evidence="x", by="test")            # no longer PENDING
    b = led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="b", attempt=1, tool="router", req=q)
    led.note_no_response(b.request_id, "connection reset")
    out = led.manual_void(b.request_id, evidence="vendor console shows no request", by="test")
    st = row(b.request_id)
    assert st[0] == "ERROR" and st[5] == "LLM_VOID" and float(st[1]) == 0.0 and out["resolution"]["kind"] == "manual_void"


def test_parallel_reservations_serialize_under_the_lock(run, monkeypatch):
    led, thread, run_uuid, spent0 = run
    q = req()
    monkeypatch.setenv("SGP_LLM_BUDGET_USD", str(spent0 + q.reserved_usd() * 2.5))
    results: list = []

    def go(i: int) -> None:
        results.append(led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key=f"p{i}", attempt=1, tool="router", req=q).status)

    ts = [threading.Thread(target=go, args=(i,)) for i in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sorted(results) == sorted(["owned", "owned"] + ["budget_refused"] * 4)
    assert led.totals()["spent_usd"] == pytest.approx(spent0 + 2 * q.reserved_usd())


def test_settlement_cannot_slip_between_reserve_checks(run, monkeypatch):
    """A settlement that creates a blocking row is serialized with reserve under the same advisory lock."""
    led, thread, run_uuid, spent0 = run
    q = req(max_tokens=10)
    monkeypatch.setenv("SGP_LLM_BUDGET_USD", str(spent0 + 100.0))
    a = led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="a", attempt=1, tool="router", req=q)
    with psycopg.connect(DSN) as holder, holder.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("sgp_llm_ledger",))
        done = {}

        def late_settle():
            led.settle(a.request_id, resp(20_000, 20_000), req=q, output={})
            done["t"] = True

        t = threading.Thread(target=late_settle)
        t.start()
        t.join(timeout=1.0)
        assert t.is_alive() and not done                                       # settle waits for the lock
        holder.rollback()                                                      # release
        t.join(timeout=10)
    assert done.get("t") and led.reserve(thread_id=thread, run_uuid=run_uuid, stage_key="b", attempt=1, tool="router", req=q).status == "exceeded_unresolved"
    led.resolve_exceeded(a.request_id, evidence="checked", by="test")
