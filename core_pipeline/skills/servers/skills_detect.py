"""core_pipeline/skills/servers/skills_detect.py — FastMCP server 1 of 2 (ST4-01): gate_check · detect_diagnose ·
work_order_{draft,approve,issue,status}. Owns the episode detection state (ST4-02) and the mock CMMS (ST4-03).
Imports lightgbm (diagnoser); NEVER torch. The graph process imports neither server.

Contract v1.0 points enforced here (docs/skill_contract.md):
  §1.1  BLOCK ⇒ no detection: gate_check commits the BLOCK window row itself (p/p_adj/alarm NULL, m_t unchanged);
        detect_diagnose on a BLOCK window → REFUSED(GATE_BLOCK), logged as a violation, nothing written (H0-(d)).
  §1.3  work_order_issue without a token → REFUSED(NO_APPROVAL) (H0-(c)); tokens are one-shot (TOKEN_USED).
  §1.4  tool arguments are state references: episode_id/window must exist in the grid; detect reads the gate result the
        SERVER stored for the same (episode, window, contract, gate version, calibration sha) — GATE_NOT_RUN otherwise.
  §2.2  expand=2 is diagnosis only, allowed once per window per request_id (QUERY_USED), never at t=0 (NO_EXPANSION).
  §0    request ownership + idempotent replay by request_id (ST4-13); every call → mart.agent_steps.
  ST4-13 partial-commit completion: a stored BLOCK gate without its window row, or an alarm window without its
        alarms/diagnoses rows, is completed idempotently on the next call instead of being returned incomplete.
  ST4-13 query_more: `spend_query` spends the window's single query once and returns query_id; `detect_diagnose(expand=2)`
        and `retrieve(k=10)` must present that query_id (QUERY_REQUIRED / QUERY_USED).
  ST4-13 approval binding: `work_order_approve` accepts only the registered owner of ctx.thread_id (mart.threads) as
        approver (APPROVER_MISMATCH); exposing approve/issue to the LLM tool list is forbidden (graph nodes only, S4a-04).

Run:   python -m core_pipeline.skills.servers.skills_detect            (stdio; the client spawns it)
Env:   SGP_PG_DSN (required), SGP_GATE_REF (required: gate reference parquet, see README_bundle), SGP_DETECTOR_DIR,
       SGP_DIAGNOSER_DIR, SGP_GATE_CALIB, SGP_PROCESSED_DIR, SGP_TAU_PATH, SGP_TEST_BLOCK_WINDOWS (test DSN only).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional
from uuid import UUID

import numpy as np
import pandas as pd
import psycopg
from fastmcp import FastMCP

from core_pipeline import prereg_constants as P
from core_pipeline.anomaly.conformal import ConformalCalibration
from core_pipeline.anomaly.if_window import WindowIF
from core_pipeline.data.tep_loader import decode_engine_id, load_processed
from core_pipeline.diagnosis.classifier import DIAG_VERSION, FaultDiagnoser
from core_pipeline.governance.gate_adapter import GateReference
from core_pipeline.governance.gate_v2 import GATE_VERSION, GateCalibration, GateDecision, gate_check as gate_check_fn
from core_pipeline.skills.detect_diagnose import DetectDiagnoseSkill, DetectResult
from core_pipeline.skills.schemas import validate_input, validate_result
from core_pipeline.skills.servers.common import (Ctx, StepLog, alarm_id_of, code_of, engine_id_of, env, envelope,
                                                 is_test_dsn, jsonable, log, run_tool, sha256_file, window_bounds)
from core_pipeline.skills.state_store import DetectionStateStore, WindowRecord
from core_pipeline.skills.work_order import Draft, WorkOrderSkill
from core_pipeline.stream.state import EpisodeState, WindowState
from core_pipeline.stream.window_view import PROCESS_COLS, WindowView

SKILL = "skills_detect"
mcp = FastMCP(SKILL)


def _val(x):
    """property-or-method tolerant read (the interface dump hid decorators; reviewer 2026-10-21 point 1)."""
    return x() if callable(x) else x


class DetectServer:
    def __init__(self) -> None:
        self.dsn = env("SGP_PG_DSN", required=True)
        det_dir = Path(env("SGP_DETECTOR_DIR", "models/detector_v1"))
        diag_dir = Path(env("SGP_DIAGNOSER_DIR", "models/diagnoser_v1"))
        calib_path = Path(env("SGP_GATE_CALIB", "models/gate_v2/gate_calibration.json"))
        ref_path = Path(env("SGP_GATE_REF", "data/processed/gate_reference_v1.parquet"))
        self.processed_dir = env("SGP_PROCESSED_DIR", "data/processed/TEP")
        meta = json.loads((Path("outputs/model_meta") / "detector_v1.json").read_text())
        self.contract_sha = meta["input_contract_sha256"]
        self.detector = WindowIF.load(det_dir)
        self.detector_version = meta["version"]                     # artefact name (detector_v1): the key of episode_windows / alarms
        self.detector_model = self.detector.meta.get("version", "if_window_v1")   # the model's own tag, recorded only
        self.conformal = ConformalCalibration.load(det_dir / "conformal")
        self.diagnoser = FaultDiagnoser.load(diag_dir)
        self.diagnoser_version = self.diagnoser.meta.get("version", DIAG_VERSION)
        self.skill = DetectDiagnoseSkill(self.detector, self.conformal, self.diagnoser)
        self.calib = GateCalibration.load(calib_path)
        self.calib_sha = sha256_file(calib_path)
        self.ref = GateReference.load(ref_path)
        self.store = DetectionStateStore(self.dsn, self.detector_version, self.contract_sha, alpha=P.ALPHA)
        self.cmms = WorkOrderSkill(self.dsn, self.detector_version, self.contract_sha,
                                   tau_path=env("SGP_TAU_PATH", "models/retrieval_v1/tau.json"))
        self.versions = {"detector": self.detector_version, "detector_model": self.detector_model, "diagnoser": self.diagnoser_version, "gate": GATE_VERSION,
                         "calib_sha256": self.calib_sha, "contract_sha256": self.contract_sha, "alpha": P.ALPHA,
                         "tau": self.cmms.tau}
        self.steplog = StepLog(self.dsn, SKILL, self.versions)
        self.block_inject: set[tuple[str, int]] = set()
        inj = os.environ.get("SGP_TEST_BLOCK_WINDOWS", "")
        if inj:
            if not is_test_dsn(self.dsn):
                log.warning("SGP_TEST_BLOCK_WINDOWS ignored: DSN is not a test database")
            else:
                self.block_inject = {(s.split(":")[0], int(s.split(":")[1])) for s in inj.split(",") if s.strip()}
                log.warning("TEST ONLY: injecting input-contract violations into windows %s", sorted(self.block_inject))
        log.info("skills_detect ready: %s", self.versions)

    # ------------------------------------------------------------------ data
    @lru_cache(maxsize=8)
    def _frame(self, episode_id: UUID) -> pd.DataFrame:
        eid = engine_id_of(self.dsn, episode_id)
        if eid is None:
            raise ValueError("DATA_MISSING: episode has no engine key")
        _, fault, _ = decode_engine_id(eid)
        df = load_processed(self.processed_dir, P.SPLIT, faulty=fault > 0, columns=["cycle"] + PROCESS_COLS, engine_ids=[eid])
        if len(df) < P.RUN_LENGTH:
            raise ValueError(f"DATA_MISSING: {len(df)} rows for episode (expected {P.RUN_LENGTH})")
        return df.reset_index(drop=True)

    def _view(self, episode_id: UUID, t: int) -> WindowView:
        df = self._frame(episode_id)
        if (str(episode_id), t) in self.block_inject:                 # test-only corruption → integrity BLOCK
            df = df.copy()
            lo, hi = P.STRIDE * t + 1, P.STRIDE * t + P.WINDOW_SIZE
            df.loc[(df["cycle"] >= lo) & (df["cycle"] <= hi), PROCESS_COLS[:3]] = np.nan
        return WindowView.build(episode_id, t, df)

    def _episode_state(self, episode_id: UUID) -> EpisodeState:
        """Policy-side EpisodeState rebuilt from the COMMITTED rows (the DB is the authority, ST4-02)."""
        st = EpisodeState(episode_id, self.store.alpha)
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            rows = self.store.tested_prefix(cur, episode_id)
        for idx, gate, p, _ in rows:
            st.observe(idx, gate, p)
        return st

    def _gate_key(self):
        return (self.contract_sha, GATE_VERSION, self.calib_sha)

    def _stored_gate(self, episode_id: UUID, t: int) -> Optional[dict]:
        cols = ("gate, drift, drift_assessed, drift_stat, drift_threshold, composite_min, composite_mean, block_reasons, "
                "flag_reasons, n_history, history_age_windows")
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(f"""SELECT {cols} FROM mart.gate_checks WHERE episode_id=%s AND window_index=%s
                            AND contract_sha256=%s AND gate_version=%s AND calib_sha256=%s""", (episode_id, t, *self._gate_key()))
            row = cur.fetchone()
        if not row:
            return None
        keys = [c.strip() for c in cols.split(",")]
        d = dict(zip(keys, row))
        d.update(gate_version=GATE_VERSION, calib_sha256=self.calib_sha, contract_sha256=self.contract_sha, stored=True)
        return d

    def _gate_payload(self, g: GateDecision) -> dict:
        return {"gate": g.gate, "drift": g.drift, "drift_assessed": g.drift_assessed, "drift_stat": g.drift_stat,
                "drift_threshold": g.drift_threshold, "composite_min": g.composite_min, "composite_mean": g.composite_mean,
                "block_reasons": list(g.block_reasons), "flag_reasons": list(g.flag_reasons), "n_history": g.n_history,
                "history_age_windows": g.history_age_windows, "gate_version": GATE_VERSION, "calib_sha256": self.calib_sha,
                "contract_sha256": self.contract_sha, "stored": False}

    def _state_dict(self, rec: WindowRecord) -> dict:
        d = jsonable(rec)
        d["detector_version"], d["contract_sha256"] = self.detector_version, self.contract_sha
        d["alarm_id"] = alarm_id_of(rec.episode_id, rec.window_index, self.detector_version) if rec.alarm else None
        return d

    # ------------------------------------------------------------------ gate_check
    def gate_check(self, ctx: Ctx, episode_id: UUID, t: int) -> dict:
        if window_bounds(self.dsn, episode_id, t) is None:
            return envelope("ERROR", reason=("WINDOW_OUT_OF_RANGE", f"window {t} of episode {episode_id} is not in the grid"))
        existing = self.store.get(episode_id, t)
        stored = self._stored_gate(episode_id, t)
        if existing is not None:                                    # already observed under this detector/contract
            payload = stored or {**self._gate_payload(GateDecision(existing.gate, existing.drift, float("nan"), float("nan"), [], [],
                                                                  existing.drift_assessed, None, None, 0)), "stored": True}
            return envelope("OK", payload=payload, state=self._state_dict(WindowRecord(**{**existing.__dict__, "reused": True})))
        expected = self.store.next_expected_window(episode_id)
        if t != expected:
            return envelope("REFUSED", reason=("WINDOW_ORDER", f"next expected window is {expected}, got {t}"))
        if stored is not None:                                      # gate ran, detection not yet: return the stored verdict
            if stored["gate"] == "BLOCK":                            # partial commit: gate row stored, window row not → complete it
                rec = self.store.observe(episode_id, t, "BLOCK", stored["drift"], stored["drift_assessed"], None, [], None)
                return envelope("OK", payload=stored, state=self._state_dict(rec))
            return envelope("OK", payload=stored, state=None)
        view = self._view(episode_id, t)
        g = gate_check_fn(view, self.ref, self.calib, self._episode_state(episode_id))
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("""INSERT INTO mart.gate_checks (episode_id, window_index, contract_sha256, gate_version, calib_sha256, gate, drift,
                             drift_assessed, drift_stat, drift_threshold, composite_min, composite_mean, block_reasons, flag_reasons,
                             n_history, history_age_windows, request_id)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                        (episode_id, t, *self._gate_key(), g.gate, g.drift, g.drift_assessed, g.drift_stat, g.drift_threshold,
                         None if g.composite_min != g.composite_min else g.composite_min,
                         None if g.composite_mean != g.composite_mean else g.composite_mean,
                         json.dumps(list(g.block_reasons)), json.dumps(list(g.flag_reasons)), g.n_history, g.history_age_windows, ctx.request_id))
            conn.commit()
        state = None
        if g.gate == "BLOCK":                                       # §1.1: the BLOCK row is committed HERE, m_t unchanged
            rec = self.store.observe(episode_id, t, "BLOCK", g.drift, g.drift_assessed, None, [], None)
            state = self._state_dict(rec)
        return envelope("OK", payload=self._gate_payload(g), state=state)

    # ------------------------------------------------------------------ detect_diagnose
    def _diag_payload(self, r: DetectResult, rec: WindowRecord, detection_run: bool) -> dict:
        s = len(r.pred_set)
        return {"detection_run": detection_run, "score": r.score, "p_value": rec.p_value, "p_adj": rec.p_adj, "alarm": rec.alarm,
                "m_t": rec.m_t, "alpha": P.ALPHA,
                "diagnosis": {"set": [int(f) for f in r.pred_set], "set_size": s, "valid": 1 <= s <= P.SET_SIZE_MAX,
                              "coverage_target": P.COVERAGE_TARGET, "calibration_window": r.calibration_window or 20,
                              "top1": None if r.top1 is None else int(r.top1), "top1_prob": r.top1_prob,
                              "indeterminate": s == 0 or s > P.SET_SIZE_MAX},
                "contributions": list(r.contributions)[:5],
                "calibration_ref": {"n_cal": r.n_cal, "detector_version": self.detector_version, "diagnoser_version": self.diagnoser_version},
                "alarm_id": alarm_id_of(rec.episode_id, rec.window_index, self.detector_version) if rec.alarm else None}

    def _stored_detect_payload(self, rec: WindowRecord) -> dict:
        aid = alarm_id_of(rec.episode_id, rec.window_index, self.detector_version)
        top1_prob, contrib = None, []
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT top1_prob, contributions FROM mart.diagnoses WHERE alarm_id=%s", (aid,))
            row = cur.fetchone()
        if row:
            top1_prob = row[0]
            contrib = row[1] if isinstance(row[1], list) else json.loads(row[1] or "[]")
        r = DetectResult("OK", None, True, None, rec.p_value, list(rec.pred_set), rec.set_size, 1 <= rec.set_size <= P.SET_SIZE_MAX,
                         rec.top1, top1_prob, 20, contrib, _val(self.conformal.n_cal))
        return self._diag_payload(r, rec, detection_run=True)

    def _ensure_alarm_rows(self, rec: WindowRecord, gate: GateDecision) -> None:
        """Partial-commit completion (ST4-13): the window row says alarm but alarms/diagnoses are missing (crash after
        observe()). Recompute the deterministic detection on the stored window and write the rows idempotently."""
        if not rec.alarm:
            return
        aid = alarm_id_of(rec.episode_id, rec.window_index, self.detector_version)
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM mart.alarms a JOIN mart.diagnoses d USING (alarm_id) WHERE alarm_id=%s", (aid,))
            if cur.fetchone():
                return
        r = self.skill(self._view(rec.episode_id, rec.window_index), gate, WindowState(rec.window_index), expand=1)
        if r.status == "OK":
            self._record_alarm(rec, r)

    def _record_alarm(self, rec: WindowRecord, r: DetectResult) -> None:
        aid = alarm_id_of(rec.episode_id, rec.window_index, self.detector_version)
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("""INSERT INTO mart.alarms (alarm_id, episode_id, window_index, gate, drift, p_value, p_adj, m_t, alarm, detector)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,true,%s) ON CONFLICT DO NOTHING""",
                        (aid, rec.episode_id, rec.window_index, rec.gate, rec.drift, rec.p_value, rec.p_adj, rec.m_t, self.detector_version))
            cur.execute("""INSERT INTO mart.diagnoses (alarm_id, calibration_window, pred_set, set_size, top1, top1_prob, contributions)
                           VALUES (%s,20,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                        (aid, [int(f) for f in r.pred_set], len(r.pred_set), None if r.top1 is None else int(r.top1), r.top1_prob,
                         json.dumps(jsonable(list(r.contributions)[:5]), ensure_ascii=False)))
            conn.commit()

    def detect_diagnose(self, ctx: Ctx, episode_id: UUID, t: int, expand: int, query_id: Optional[UUID]) -> dict:
        if window_bounds(self.dsn, episode_id, t) is None:
            return envelope("ERROR", reason=("WINDOW_OUT_OF_RANGE", f"window {t} of episode {episode_id} is not in the grid"))
        existing = self.store.get(episode_id, t)
        if existing is not None and existing.gate == "BLOCK":
            return envelope("REFUSED", reason=("GATE_BLOCK", "window is BLOCK: detection is not run (contract §1.1)"))
        stored = self._stored_gate(episode_id, t)
        if stored is None:
            return envelope("REFUSED", reason=("GATE_NOT_RUN", "no stored gate result for this window under the current contract/gate/calibration"))
        if stored["gate"] == "BLOCK":
            return envelope("REFUSED", reason=("GATE_BLOCK", "window is BLOCK: detection is not run (contract §1.1)"))
        nan = float("nan")
        gate = GateDecision(stored["gate"], stored["drift"], nan if stored["composite_min"] is None else stored["composite_min"],
                            nan if stored["composite_mean"] is None else stored["composite_mean"],
                            list(stored["block_reasons"] or []), list(stored["flag_reasons"] or []), stored["drift_assessed"],
                            stored["drift_stat"], stored["drift_threshold"], stored["n_history"], stored["history_age_windows"])
        if expand == 1:
            if existing is not None:                                 # chat mode reads the stored observation (§2.2)
                self._ensure_alarm_rows(existing, gate)
                return envelope("OK", payload=self._stored_detect_payload(existing),
                                state=self._state_dict(WindowRecord(**{**existing.__dict__, "reused": True})))
            expected = self.store.next_expected_window(episode_id)
            if t != expected:
                return envelope("REFUSED", reason=("WINDOW_ORDER", f"next expected window is {expected}, got {t}"))
            r = self.skill(self._view(episode_id, t), gate, WindowState(t), expand=1)
            if r.status != "OK":
                return envelope("REFUSED", reason=(r.reason if r.reason in ("GATE_BLOCK",) else "BAD_REQUEST", f"{r.reason}"))
            rec = self.store.observe(episode_id, t, gate.gate, gate.drift, gate.drift_assessed, r.p_value,
                                     [int(f) for f in r.pred_set], None if r.top1 is None else int(r.top1))
            if rec.alarm:
                self._record_alarm(rec, r)
            return envelope("OK", payload=self._diag_payload(r, rec, detection_run=not rec.reused), state=self._state_dict(rec))
        # ---- expand == 2: diagnosis only, under the window's single query (spend_query → query_id; ST4-02/ST4-13, A-3)
        if existing is None:
            return envelope("REFUSED", reason=("DETECT_NOT_RUN", "expansion requires the primary detection of this window first"))
        if t == 0:
            return envelope("REFUSED", reason=("NO_EXPANSION", "no backward expansion at window 0 (prereg §3.2)"))
        if query_id is None:
            return envelope("REFUSED", reason=("QUERY_REQUIRED", "expand=2 needs the query_id returned by spend_query"))
        if not existing.q_used or existing.query_request_id != query_id:
            return envelope("REFUSED", reason=("QUERY_USED", "query_id does not own this window's query (not spent, or spent by another query)"))
        if existing.expanded_set is not None:
            rec = existing
            r = DetectResult("OK", None, False, None, rec.p_value, list(rec.expanded_set), len(rec.expanded_set),
                             1 <= len(rec.expanded_set) <= P.SET_SIZE_MAX, None, None, 40, [], _val(self.conformal.n_cal))
            return envelope("OK", payload=self._diag_payload(r, rec, detection_run=False), state=self._state_dict(WindowRecord(**{**rec.__dict__, "reused": True})))
        r = self.skill(self._view(episode_id, t), gate, WindowState(t, q_used=True), expand=2, episode_state=self._episode_state(episode_id))
        if r.status != "OK":
            code = r.reason if r.reason in ("NO_EXPANSION", "EXPANSION_INVALID") else "BAD_REQUEST"
            return envelope("REFUSED", reason=(code, f"{r.reason}"))
        self.store.set_expanded(episode_id, t, query_id, [int(f) for f in r.pred_set])
        rec = self.store.get(episode_id, t)
        return envelope("OK", payload=self._diag_payload(r, rec, detection_run=False), state=self._state_dict(rec))

    # ------------------------------------------------------------------ spend_query (one logical query per window, ST4-13)
    def spend_query(self, ctx: Ctx, episode_id: UUID, t: int) -> dict:
        existing = self.store.get(episode_id, t)
        if existing is None:
            return envelope("REFUSED", reason=("DETECT_NOT_RUN", "the window has no primary detection yet"))
        if existing.gate == "BLOCK":
            return envelope("REFUSED", reason=("GATE_BLOCK", "no query on a BLOCK window"))
        if not self.store.spend_query(episode_id, t, ctx.request_id):
            return envelope("REFUSED", reason=("QUERY_USED", "this window's single query was already spent by another request"))
        rec = self.store.get(episode_id, t)
        return envelope("OK", payload={"query_id": str(rec.query_request_id), "q_used": True, "expand_allowed": t > 0},
                        state=self._state_dict(rec))

    # ------------------------------------------------------------------ work_order (mock CMMS, ST4-03)
    @staticmethod
    def _draft(d: dict) -> Draft:
        return Draft(UUID(str(d["proposal_id"])), int(d["draft_version"]), str(d["content_sha256"]), dict(d["form"]))

    def work_order_draft(self, ctx: Ctx, inp: dict) -> dict:
        try:
            d = self.cmms.draft(inp["alarm_id"], UUID(inp["episode_id"]), int(inp["window_index"]), inp["policy"], ctx.thread_id,
                                inp["action_items"], UUID(inp["decision_id"]),
                                UUID(inp["base_proposal_id"]) if inp.get("base_proposal_id") else None, inp.get("created_by", "agent"))
        except (PermissionError, ValueError) as e:
            return envelope("REFUSED", reason=code_of(e, "BAD_REQUEST"))
        return envelope("OK", payload=jsonable(d))

    def work_order_approve(self, ctx: Ctx, inp: dict) -> dict:
        # the approver must be the registered owner of this thread (mart.threads, written at session start, S4a-04):
        # an agent tool call cannot name an approver into existence (ST4-13)
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT owner FROM mart.threads WHERE thread_id=%s", (ctx.thread_id,))
            row = cur.fetchone()
        if not row or row[0] != inp["approver"]:
            return envelope("REFUSED", reason=("APPROVER_MISMATCH", "approver is not the registered owner of this thread"))
        try:
            token = self.cmms.approve(self._draft(inp["draft"]), inp["approver"], ctx.thread_id, inp.get("decision", "approve"), inp.get("modifications"))
        except (PermissionError, ValueError) as e:
            return envelope("REFUSED", reason=code_of(e, "BAD_REQUEST"))
        return envelope("OK", payload={"approval_token": None if token is None else str(token), "decision": inp.get("decision", "approve")})

    def work_order_issue(self, ctx: Ctx, inp: dict) -> dict:
        tok = inp.get("approval_token")
        try:
            res = self.cmms.issue(self._draft(inp["draft"]), UUID(str(tok)) if tok else None, ctx.thread_id)
        except (PermissionError, ValueError) as e:
            return envelope("REFUSED", reason=code_of(e, "BAD_REQUEST"))
        if res.get("status") == "OK":
            return envelope("OK", payload={"work_order_id": res["work_order_id"], "cmms_ref": res["cmms_ref"], "idempotent_replay": bool(res.get("idempotent_replay"))})
        code = res.get("reason", "INTERNAL")
        return envelope("ERROR" if res.get("status") == "ERROR" else "REFUSED", reason=(code if code in ("NO_APPROVAL", "TOKEN_USED", "SUPERSEDED", "CONTENT_MISMATCH", "THREAD_MISMATCH", "APPROVAL_MISMATCH", "UNKNOWN_PROPOSAL") else "INTERNAL", f"{code}: {res.get('detail', '')}".strip()))

    def work_order_status(self, ctx: Ctx, proposal_id: str) -> dict:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT alarm_id, policy, current_version, thread_id FROM mart.proposals WHERE proposal_id=%s", (UUID(proposal_id),))
            p = cur.fetchone()
            if not p:
                return envelope("REFUSED", reason=("UNKNOWN_PROPOSAL", "no such proposal"))
            if p[3] != ctx.thread_id:
                return envelope("REFUSED", reason=("THREAD_MISMATCH", "proposal belongs to another thread"))
            cur.execute("SELECT draft_version, decision, token_used FROM mart.approvals WHERE proposal_id=%s ORDER BY approval_id", (UUID(proposal_id),))
            approvals = [{"draft_version": r[0], "decision": r[1], "token_used": r[2]} for r in cur.fetchall()]
            cur.execute("SELECT work_order_id, cmms_ref, status, draft_version FROM mart.work_orders WHERE proposal_id=%s", (UUID(proposal_id),))
            wos = [{"work_order_id": str(r[0]), "cmms_ref": r[1], "status": r[2], "draft_version": r[3]} for r in cur.fetchall()]
        return envelope("OK", payload={"proposal_id": proposal_id, "alarm_id": p[0], "policy": p[1], "current_version": p[2],
                                       "approvals": approvals, "work_orders": wos})


_SERVER: Optional[DetectServer] = None


def server() -> DetectServer:
    global _SERVER
    if _SERVER is None:
        _SERVER = DetectServer()
    return _SERVER


def _checked(tool: str, sub: str, inp: dict, fn) -> dict:
    s = server()
    errs = validate_input("work_order" if tool.startswith("work_order") else tool, inp, sub)
    if errs:
        return envelope("ERROR", reason=("SCHEMA_INVALID", "; ".join(errs)[:800]), trace={"skill": SKILL, "tool": tool, "contract_version": "1.0"})
    psub = {"gate_check": "payload", "detect_diagnose": "payload", "spend_query": "payload", "work_order_draft": "draft_payload",
            "work_order_approve": "approve_payload", "work_order_issue": "issue_payload"}.get(tool)
    schema = "work_order" if tool.startswith("work_order") else tool
    return run_tool(s.steplog, tool, inp["ctx"], {k: v for k, v in inp.items() if k != "ctx"}, fn,
                    violation_codes=("GATE_BLOCK", "NO_APPROVAL", "TOKEN_USED", "ARG_NOT_IN_STATE", "APPROVER_MISMATCH"),
                    validate=(lambda out: validate_result(schema, out, psub)) if psub else None)


@mcp.tool()
def gate_check(ctx: dict, episode_id: str, window_index: int) -> dict:
    """Gate + drift for one window (contract §2.1). Commits BLOCK windows to the state store itself."""
    inp = {"ctx": ctx, "episode_id": episode_id, "window_index": window_index}
    return _checked("gate_check", "input", inp, lambda c: server().gate_check(c, UUID(episode_id), window_index))


@mcp.tool()
def detect_diagnose(ctx: dict, episode_id: str, window_index: int, expand: int = 1, query_id: Optional[str] = None) -> dict:
    """Detection (IF + conformal p, prefix-BH alarm via the state store) and diagnosis set (contract §2.2).
    expand=2 (diagnosis only, t-1 ∪ t) requires the query_id from spend_query."""
    inp = {"ctx": ctx, "episode_id": episode_id, "window_index": window_index, "expand": expand, "query_id": query_id}
    return _checked("detect_diagnose", "input", inp,
                    lambda c: server().detect_diagnose(c, UUID(episode_id), window_index, expand, UUID(query_id) if query_id else None))


@mcp.tool()
def spend_query(ctx: dict, episode_id: str, window_index: int) -> dict:
    """Spend the window's single query_more (contract §3.5). Returns query_id, to be presented by detect_diagnose(expand=2)
    and retrieve(k=10). Retrying the same request_id returns the same query_id; another request gets QUERY_USED."""
    inp = {"ctx": ctx, "episode_id": episode_id, "window_index": window_index}
    return _checked("spend_query", "input", inp, lambda c: server().spend_query(c, UUID(episode_id), window_index))


@mcp.tool()
def work_order_draft(ctx: dict, alarm_id: str, episode_id: str, window_index: int, policy: str, decision_id: str,
                     action_items: list[dict], base_proposal_id: Optional[str] = None, created_by: str = "agent") -> dict:
    """Immutable draft version bound to the latest 'propose' decision of (alarm, policy) (contract §2.4, ST4-03)."""
    inp = {"ctx": ctx, "alarm_id": alarm_id, "episode_id": episode_id, "window_index": window_index, "policy": policy,
           "decision_id": decision_id, "action_items": action_items, "base_proposal_id": base_proposal_id, "created_by": created_by}
    return _checked("work_order_draft", "draft", inp, lambda c: server().work_order_draft(c, inp))


@mcp.tool()
def work_order_approve(ctx: dict, draft: dict, approver: str, decision: str = "approve", modifications: Optional[dict] = None) -> dict:
    """Record the human/oracle decision on a presented draft; 'approve' returns a one-shot token (contract §1.3)."""
    inp = {"ctx": ctx, "draft": draft, "approver": approver, "decision": decision, "modifications": modifications}
    return _checked("work_order_approve", "approve", inp, lambda c: server().work_order_approve(c, inp))


@mcp.tool()
def work_order_issue(ctx: dict, draft: dict, approval_token: Optional[str] = None) -> dict:
    """Mock CMMS issue: REFUSED(NO_APPROVAL) without a token; token + issue in one transaction (contract §1.3)."""
    inp = {"ctx": ctx, "draft": draft, "approval_token": approval_token}
    return _checked("work_order_issue", "issue", inp, lambda c: server().work_order_issue(c, inp))


@mcp.tool()
def work_order_status(ctx: dict, proposal_id: str) -> dict:
    """Current version, approvals and issued orders of a proposal (own thread only)."""
    inp = {"ctx": ctx, "proposal_id": proposal_id}
    return _checked("work_order_status", "status", inp, lambda c: server().work_order_status(c, proposal_id))


if __name__ == "__main__":
    mcp.run()          # models load lazily on the first tool call (see skills_retrieve.py)
