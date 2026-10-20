"""S4a-01 rev 2 integration (SGP_PG_DSN with the Stage 1 mart). Closes the reviewer's counter-examples:
  store : concurrent observe of the same window from two instances → one committed result, both return it;
          failed insert advances nothing (out-of-order rejected, state untouched); [] vs None for expansion;
          query spend is per request id (same request may retry, a second request is refused); expansion once.
  cmms  : grounding read from server tables; citation outside stored scoped A/B evidence refused; action types whitelisted;
          body tampered with the ORIGINAL hash refused; v1 approved → v2 drafted → v1 object+token refused (SUPERSEDED);
          thread mismatch refused; issue idempotent; concurrent issue → single order; no order without approval.
"""
import json, os, threading, uuid
import pytest, psycopg
from core_pipeline.skills.state_store import DetectionStateStore
from core_pipeline.skills.work_order import WorkOrderSkill, Draft, content_hash, record_decision

# These tests DROP and recreate Stage 4a tables and run the rev-2 migration (which discards proposals without drafts).
# They run ONLY against a dedicated, disposable test database: SGP_TEST_PG_DSN whose database name contains "test" and
# which differs from SGP_PG_DSN. Any other configuration FAILS (not skips) so the project DB is never touched.
DSN = os.environ.get("SGP_TEST_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="SGP_TEST_PG_DSN not set (dedicated disposable test DB required)")


def _assert_disposable(dsn: str) -> None:
    from urllib.parse import urlparse
    name = (urlparse(dsn).path or "").lstrip("/").split("?")[0] if "://" in dsn else dsn
    if "test" not in name.lower() or dsn == os.environ.get("SGP_PG_DSN"):
        pytest.fail(f"refusing to run destructive integration tests against {name!r}: SGP_TEST_PG_DSN must name a disposable *test* database distinct from SGP_PG_DSN")
TAU = 0.5


REV1_EPISODE_WINDOWS = """
CREATE TABLE IF NOT EXISTS mart.episode_windows (
  episode_id uuid NOT NULL, window_index smallint NOT NULL, detector_version text NOT NULL, contract_sha256 text NOT NULL,
  gate text NOT NULL, drift text NOT NULL, drift_assessed boolean NOT NULL DEFAULT false, p_value double precision, p_adj double precision,
  m_t integer NOT NULL, alarm boolean, pred_set smallint[], set_size smallint, top1 smallint, q_used boolean NOT NULL DEFAULT false,
  observed_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY (episode_id, window_index, detector_version, contract_sha256),
  FOREIGN KEY (episode_id, window_index) REFERENCES mart.windows(episode_id, window_index))"""


@pytest.fixture(scope="module")
def episode():
    """Apply the migration on top of a rev-1-shaped table (upgrade path), then return a dev episode. Destructive."""
    _assert_disposable(DSN)
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS mart.policy_decisions, mart.proposal_drafts CASCADE")
        cur.execute("DROP TABLE IF EXISTS mart.episode_windows CASCADE")
        cur.execute(REV1_EPISODE_WINDOWS)                                     # rev-1 shape (no query_request_id / expanded_set)
        cur.execute(open("sql/003_skill_state.sql", encoding="utf-8").read())
        cur.execute(open("sql/004_skill_state_rev2.sql", encoding="utf-8").read())
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='mart' AND table_name='episode_windows'")
        cols = {r[0] for r in cur.fetchall()}
        assert {"query_request_id", "expanded_set"} <= cols                  # upgrade added the rev-2 columns
        cur.execute("SELECT max(version) FROM mart.schema_version"); assert cur.fetchone()[0] == 5
        conn.commit()
        cur.execute("""SELECT episode_id FROM mart.v_windows_agent WHERE "group"='dev' ORDER BY episode_id LIMIT 1""")
        return cur.fetchone()[0]


@pytest.fixture
def short_episode(episode):                     # depends on `episode` so the schema (003 + 004) is applied first
    """A synthetic episode with only windows 0..1 in mart.windows, so observing window 2 (the CORRECT next window)
    fails the FK at INSERT time — a real DB write failure, not an order error."""
    eid = uuid.uuid4()
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        _purge_synthetic(cur)                                                 # leftovers from an aborted earlier run
        cur.execute("INSERT INTO staging.episode_keys (episode_id, engine_id) VALUES (%s, 999999)", (eid,))
        cur.execute("""INSERT INTO staging.episodes (episode_id, engine_id, split, run_id, fault_scenario, run_length, "group")
                       VALUES (%s, 999999, 'testing', 999, 0, 960, 'dev')""", (eid,))
        for t in range(2):
            cur.execute("INSERT INTO mart.windows (episode_id, window_index, start_cycle, end_cycle) VALUES (%s,%s,%s,%s)", (eid, t, 20*t+1, 20*t+20))
        conn.commit()
    yield eid
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        _purge_synthetic(cur); conn.commit()


def _purge_synthetic(cur) -> None:
    """Remove the synthetic engine_id 999999 episode wherever it is referenced (idempotent)."""
    cur.execute("SELECT episode_id FROM staging.episode_keys WHERE engine_id=999999")
    for (eid,) in cur.fetchall():
        cur.execute("DELETE FROM mart.episode_windows WHERE episode_id=%s", (eid,))
        cur.execute("DELETE FROM mart.windows WHERE episode_id=%s", (eid,))
        cur.execute("DELETE FROM staging.episodes WHERE episode_id=%s", (eid,))
        cur.execute("DELETE FROM staging.episode_keys WHERE episode_id=%s", (eid,))


def test_store_db_write_failure_then_retry(short_episode):
    st = DetectionStateStore(DSN, f"test-{uuid.uuid4().hex[:8]}", "c")
    st.observe(short_episode, 0, "PASS", "ok", False, 0.9, [], None); st.observe(short_episode, 1, "PASS", "ok", False, 0.8, [], None)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):                 # INSERT itself fails (window 2 missing in mart.windows)
        st.observe(short_episode, 2, "PASS", "ok", False, 0.01, [4], 4)
    assert st.next_expected_window(short_episode) == 2 and st.get(short_episode, 2) is None   # nothing advanced
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO mart.windows (episode_id, window_index, start_cycle, end_cycle) VALUES (%s,2,41,60)", (short_episode,)); conn.commit()
    rec = st.observe(short_episode, 2, "PASS", "ok", False, 0.01, [4], 4)   # same window, normal retry → stored once
    assert not rec.reused and rec.m_t == 3 and st.get(short_episode, 2).p_value == 0.01


def test_store_concurrent_and_failure_safe(episode):
    ver = f"test-{uuid.uuid4().hex[:8]}"
    a, b = DetectionStateStore(DSN, ver, "c"), DetectionStateStore(DSN, ver, "c")
    a.observe(episode, 0, "PASS", "ok", False, 0.9, [], None)
    results, errors = [], []
    def worker(store, p):
        try: results.append(store.observe(episode, 1, "PASS", "ok", False, p, [4], 4))
        except Exception as e: errors.append(e)
    th = [threading.Thread(target=worker, args=(a, 0.02)), threading.Thread(target=worker, args=(b, 0.0001))]
    [t.start() for t in th]; [t.join() for t in th]
    assert not errors and len(results) == 2
    assert results[0].p_value == results[1].p_value and results[0].p_adj == results[1].p_adj     # one committed truth
    assert sum(1 for r in results if not r.reused) == 1                                            # exactly one fresh observation
    with pytest.raises(ValueError):
        b.observe(episode, 5, "PASS", "ok", False, 0.5, [], None)                                  # out of order → nothing written
    assert a.next_expected_window(episode) == 2
    blk = a.observe(episode, 2, "BLOCK", "ok", False, None, [], None)
    assert blk.alarm is None and blk.m_t == 2
    # query spend is per request; retry of the same request allowed, another request refused
    r1, r2 = uuid.uuid4(), uuid.uuid4()
    assert a.spend_query(episode, 1, r1) and a.spend_query(episode, 1, r1) and not a.spend_query(episode, 1, r2)
    assert not b.set_expanded(episode, 1, r2, [4])                                                 # wrong request
    assert b.set_expanded(episode, 1, r1, []) and not b.set_expanded(episode, 1, r1, [4, 11])        # once; empty set kept
    rec = a.get(episode, 1)
    assert rec.expanded_set == [] and rec.pred_set == [4] and rec.p_value == results[0].p_value and rec.q_used   # [] ≠ None


def _seed_alarm(episode, store: DetectionStateStore, t: int, policy_action="propose"):
    """Server-side truth the CMMS reads: an alarmed window + scoped A/B evidence rows + a recorded policy decision."""
    store.observe(episode, t, "PASS", "ok", False, 0.001, [4], 4)
    alarm_id = f"{episode}:{t}:{store.detector_version}"
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO mart.alarms (alarm_id, episode_id, window_index, gate, drift, p_value, p_adj, m_t, alarm, detector)
                       VALUES (%s,%s,%s,'PASS','ok',0.001,0.001,1,true,%s) ON CONFLICT DO NOTHING""", (alarm_id, episode, t, store.detector_version))
        cur.execute("DELETE FROM mart.evidence WHERE alarm_id=%s", (alarm_id,))
        for i, (k, g, sc) in enumerate([("P-53-2012:0:6.7", "A", 0.98), ("P-53-2012:0:6", "A", 0.9), ("P-116-2012:0:5.1", "C", 0.7), ("P-65-2012:0:5.4", "B", 0.3)], 1):
            cur.execute("INSERT INTO mart.evidence (alarm_id, mode, k, clause_key, grade, rank, score, tau) VALUES (%s,'scoped',5,%s,%s,%s,%s,%s)", (alarm_id, k, g, i, sc, TAU))
        cur.execute("INSERT INTO mart.evidence (alarm_id, mode, k, clause_key, grade, rank, score, tau) VALUES (%s,'open',5,'P-77-2011:0:6.3','OUT',1,0.99,%s)", (alarm_id, TAU))
        conn.commit()
    snapshot = [{"clause_key": "P-53-2012:0:6.7", "grade": "A", "score": 0.98}, {"clause_key": "P-53-2012:0:6", "grade": "A", "score": 0.9},
                {"clause_key": "P-116-2012:0:5.1", "grade": "C", "score": 0.7}, {"clause_key": "P-65-2012:0:5.4", "grade": "B", "score": 0.3}]
    did = record_decision(DSN, alarm_id, "E", {"gate": "PASS", "drift": "ok", "alarm": True, "s": 1, "g_AB": 0.98, "q_used": False, "tau": TAU, "retrieve_mode": "scoped"},
                          policy_action, 4, evidence=snapshot)
    return alarm_id, did


def test_cmms_contract(episode):
    ver = f"test-{uuid.uuid4().hex[:8]}"
    store = DetectionStateStore(DSN, ver, "c")
    for t in range(8): store.observe(episode, t, "PASS", "ok", False, 0.9, [], None)
    alarm_id, did = _seed_alarm(episode, store, 8)
    wo = WorkOrderSkill(DSN, ver, "c", tau=TAU)
    ok_item = {"action_type": "inspect", "item": "긴급배출·급랭 설비 점검", "clause_key": "P-53-2012:0:6.7"}
    D = dict(decision_id=did)
    # --- event identity: another alarm's evidence must not ground this window ---
    for t in range(9, 12): store.observe(episode, t, "PASS", "ok", False, 0.001, [4], 4)
    other_alarm, other_did = _seed_alarm(episode, store, 12)                 # window 12 has its own alarm + decision
    with pytest.raises(PermissionError, match="EVENT_MISMATCH"):            # alarm of window 12 presented for window 8
        wo.draft(other_alarm, episode, 8, "E", "thr-1", [ok_item], decision_id=other_did)
    with pytest.raises(PermissionError, match="DECISION_MISMATCH"):         # decision of another alarm presented with this alarm
        wo.draft(alarm_id, episode, 8, "E", "thr-1", [ok_item], decision_id=other_did)
    # --- policy gate: a 'hold' decision cannot draft ---
    ver2 = f"test-{uuid.uuid4().hex[:8]}"; st2 = DetectionStateStore(DSN, ver2, "c")
    for t in range(8): st2.observe(episode, t, "PASS", "ok", False, 0.9, [], None)
    held_alarm, held_did = _seed_alarm(episode, st2, 8, policy_action="hold")
    with pytest.raises(PermissionError, match="NO_PROPOSE_DECISION"):
        WorkOrderSkill(DSN, ver2, "c", tau=TAU).draft(held_alarm, episode, 8, "E", "thr-1", [ok_item], decision_id=held_did)
    # --- reviewer counter-example 1: propose → later hold → the old propose decision is STALE ---
    ver3 = f"test-{uuid.uuid4().hex[:8]}"; st3 = DetectionStateStore(DSN, ver3, "c")
    for t in range(8): st3.observe(episode, t, "PASS", "ok", False, 0.9, [], None)
    a3, prop_did = _seed_alarm(episode, st3, 8)
    record_decision(DSN, a3, "E", {"gate": "PASS", "drift": "critical", "alarm": True}, "hold", 3)
    with pytest.raises(PermissionError, match="STALE_DECISION"):
        WorkOrderSkill(DSN, ver3, "c", tau=TAU).draft(a3, episode, 8, "E", "thr-1", [ok_item], decision_id=prop_did)
    # --- reviewer counter-example 2: a clause added to mart.evidence AFTER the decision is not citable ---
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO mart.evidence (alarm_id, mode, k, clause_key, grade, rank, score, tau) VALUES (%s,'scoped',10,'P-77-2011:0:7.2','A',1,0.95,%s)", (alarm_id, TAU)); conn.commit()
    with pytest.raises(PermissionError, match="UNGROUNDED"):
        wo.draft(alarm_id, episode, 8, "E", "thr-1", [{**ok_item, "clause_key": "P-77-2011:0:7.2"}], **D)
    with pytest.raises(PermissionError, match="UNGROUNDED"):        # cited key is in OPEN evidence only → refused
        wo.draft(alarm_id, episode, 8, "E", "thr-1", [{**ok_item, "clause_key": "P-77-2011:0:6.3"}], **D)
    with pytest.raises(PermissionError, match="UNGROUNDED"):        # C-grade key is not a proposal basis
        wo.draft(alarm_id, episode, 8, "E", "thr-1", [{**ok_item, "clause_key": "P-116-2012:0:5.1"}], **D)
    with pytest.raises(PermissionError, match="UNGROUNDED"):        # B-grade but below tau
        wo.draft(alarm_id, episode, 8, "E", "thr-1", [{**ok_item, "clause_key": "P-65-2012:0:5.4"}], **D)
    with pytest.raises(PermissionError, match="FORBIDDEN_ACTION"):  # action type outside the whitelist
        wo.draft(alarm_id, episode, 8, "E", "thr-1", [{**ok_item, "action_type": "adjust"}], **D)
    with pytest.raises(PermissionError, match="FORBIDDEN_ACTION"):  # keyword guard (secondary)
        wo.draft(alarm_id, episode, 8, "E", "thr-1", [{**ok_item, "item": "인터록을 우회한 뒤 점검"}], **D)
    with pytest.raises(PermissionError):                            # B1 never drafts here
        wo.draft(alarm_id, episode, 8, "B1", "thr-1", [ok_item], **D)
    with pytest.raises(PermissionError, match="NO_ALARM"):          # window without a stored alarm
        wo.draft(f"{episode}:3", episode, 3, "E", "thr-1", [ok_item], **D)

    d1 = wo.draft(alarm_id, episode, 8, "E", "thr-1", [ok_item], **D)
    assert d1.draft_version == 1 and content_hash(d1.form) == d1.content_sha256
    assert wo.issue(d1, None, "thr-1")["reason"] == "NO_APPROVAL"
    tok1 = wo.approve(d1, "oracle", "thr-1")
    assert wo.issue(d1, tok1, "thr-9")["reason"] == "THREAD_MISMATCH"
    # body tampered, ORIGINAL hash presented → refused
    tampered = Draft(d1.proposal_id, 1, d1.content_sha256, {**d1.form, "action_items": [{**ok_item, "item": "다른 조치"}]})
    assert wo.issue(tampered, tok1, "thr-1")["reason"] == "CONTENT_MISMATCH"
    # v1 approved but NOT issued → revise to v2 → the kept v1 object + token must not issue
    with pytest.raises(PermissionError, match="EVENT_MISMATCH"):            # a revision cannot switch the event
        wo.draft(other_alarm, episode, 12, "E", "thr-1", [ok_item], decision_id=other_did, base_proposal_id=d1.proposal_id)
    d2 = wo.draft(alarm_id, episode, 8, "E", "thr-1", [ok_item, {"action_type": "test", "item": "급랭 계통 작동 시험", "clause_key": "P-53-2012:0:6"}], decision_id=did, base_proposal_id=d1.proposal_id)
    assert d2.draft_version == 2 and d2.content_sha256 != d1.content_sha256
    assert wo.issue(d1, tok1, "thr-1")["reason"] == "SUPERSEDED"
    assert wo.issue(d2, tok1, "thr-1")["reason"] == "APPROVAL_MISMATCH"              # old token does not bind to v2
    with pytest.raises(PermissionError, match="APPROVAL_MISMATCH"):
        wo.approve(Draft(d2.proposal_id, 2, d2.content_sha256, {**d2.form, "policy": "B3"}), "oracle", "thr-1")   # approving a non-stored body
    tok2 = wo.approve(d2, "oracle", "thr-1")
    # concurrent issue → exactly one work order, both callers get it
    outs, errs = [], []
    def issue_worker():
        try: outs.append(wo.issue(d2, tok2, "thr-1"))
        except Exception as e: errs.append(e)
    th = [threading.Thread(target=issue_worker) for _ in range(2)]
    [t.start() for t in th]; [t.join() for t in th]
    assert not errs and len(outs) == 2                                        # both callers returned, no exceptions
    assert all(o["status"] == "OK" for o in outs) and len({o["work_order_id"] for o in outs}) == 1   # same single order
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM mart.work_orders WHERE proposal_id=%s", (d2.proposal_id,)); assert cur.fetchone()[0] == 1
        cur.execute("SELECT draft_version, form->>'policy' FROM mart.proposal_drafts WHERE proposal_id=%s ORDER BY 1", (d2.proposal_id,))
        assert [r[0] for r in cur.fetchall()] == [1, 2]                                  # both versions preserved, immutable
    assert wo.issue(d2, tok2, "thr-1")["idempotent_replay"] is True                      # retry after success
