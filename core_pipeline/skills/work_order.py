"""
core_pipeline/skills/work_order.py — S4a-01 rev 2: mock CMMS, DB-authoritative (reviewer points 3–6).

Immutable drafts: every version is a row in mart.proposal_drafts (never updated); mart.proposals only points at the
current version. Version numbers are allocated under a row lock (no duplicate versions under concurrency).
Grounding is read from the SERVER's stored results, not from caller dicts: detection from mart.episode_windows,
and the policy decision the draft is built on from mart.policy_decisions by explicit decision_id. The decision must
be for this alarm/policy, action='propose', and the LATEST decision of (alarm, policy) (a later hold invalidates it;
retries of the same latest decision are allowed). Citable clauses are the decision's own scoped A/B evidence
SNAPSHOT at or above tau (recorded by the graph when it decided) — never a later retrieval. Each action item must
cite one of those keys and use a whitelisted action type; the forbidden-term check is only a secondary guard.
Approval binds to (proposal_id, draft_version, content_sha256, thread_id). issue() verifies against the DB: the
caller's draft must be the CURRENT version, its form must hash to the stored sha, the approval must be for that
version+sha+thread and unused; token consumption and the work-order INSERT are one transaction; retries and
concurrent issuers get the existing order back (unique (proposal_id, draft_version)).
Proposal EVENTS for the pre-registered cost accounting are recorded by the policy (replay_window_log), not here —
B1/B2 never draft in this skill; this is the L1 conversation path only.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from uuid import UUID

import psycopg

ACTION_TYPES = {"inspect", "test", "calibrate", "isolate_verify", "monitor", "sample_analyze", "notify"}   # 점검·시험·교정·격리확인·감시·시료분석·통보
FORBIDDEN_TERMS = ("설정값", "setpoint", "인터록", "interlock", "bypass", "우회", "긴급정지", "비상정지", "emergency stop", "shutdown")


def record_decision(dsn: str, alarm_id: str, policy: str, signals: dict, action: str, rule: int,
                    evidence: Optional[list[dict]] = None) -> UUID:
    """Server-side record of a decide_action outcome (written by the graph's decide node, read by draft()).
    `evidence` is the SNAPSHOT of the scoped retrieve the decision used ([{clause_key, grade, score}]); a draft may
    only cite keys from that snapshot, never from later retrievals."""
    did = uuid.uuid4()
    sha = hashlib.sha256(json.dumps(signals, sort_keys=True, default=str).encode()).hexdigest()
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO mart.policy_decisions (decision_id, alarm_id, policy, signals_sha256, signals, action, rule, evidence)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (did, alarm_id, policy, sha, json.dumps(signals, default=str), action, rule, json.dumps(evidence or [])))
        conn.commit()
    return did


def content_hash(form: dict) -> str:
    return hashlib.sha256(json.dumps(form, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Draft:
    proposal_id: UUID
    draft_version: int
    content_sha256: str
    form: dict


class WorkOrderSkill:
    def __init__(self, dsn: str, detector_version: str, contract_sha256: str, tau: Optional[float] = None,
                 tau_path: str = "models/retrieval_v1/tau.json"):
        self.dsn, self.detector_version, self.contract = dsn, detector_version, contract_sha256
        self.tau = tau if tau is not None else json.loads(Path(tau_path).read_text())["tau"]

    # ---- server-side grounding reads: ONE event ----
    def _grounding(self, cur, episode_id: UUID, window_index: int, alarm_id: str, policy: str, decision_id: UUID) -> tuple[dict, set[str], str]:
        # the alarm row is the event identity: it must be THIS episode/window and this detector
        cur.execute("SELECT episode_id, window_index, detector, alarm FROM mart.alarms WHERE alarm_id=%s", (alarm_id,))
        a = cur.fetchone()
        if not a or not a[3]:
            raise PermissionError("NO_ALARM: no stored alarm with this id")
        if a[0] != episode_id or a[1] != window_index or a[2] != self.detector_version:
            raise PermissionError("EVENT_MISMATCH: alarm_id does not belong to this episode/window/detector")
        cur.execute("""SELECT p_adj, alarm, pred_set, expanded_set FROM mart.episode_windows
                       WHERE episode_id=%s AND window_index=%s AND detector_version=%s AND contract_sha256=%s""",
                    (episode_id, window_index, self.detector_version, self.contract))
        d = cur.fetchone()
        if not d or not d[1]:
            raise PermissionError("NO_ALARM: no stored alarm observation for this window")
        # the decision the draft is built on: must be THIS alarm/policy, action='propose', and the LATEST decision for
        # (alarm, policy) — a later hold/query_more invalidates it (retries of the same latest decision are allowed)
        cur.execute("SELECT alarm_id, policy, action, signals_sha256, evidence, seq FROM mart.policy_decisions WHERE decision_id=%s", (decision_id,))
        dec = cur.fetchone()
        if not dec or dec[0] != alarm_id or dec[1] != policy:
            raise PermissionError("DECISION_MISMATCH: decision_id is not a decision of this alarm/policy")
        if dec[2] != "propose":
            raise PermissionError(f"NO_PROPOSE_DECISION: the referenced decision is '{dec[2]}', not 'propose'")
        cur.execute("SELECT max(seq) FROM mart.policy_decisions WHERE alarm_id=%s AND policy=%s", (alarm_id, policy))
        if cur.fetchone()[0] != dec[5]:
            raise PermissionError("STALE_DECISION: a later decision exists for this alarm/policy; the proposal is no longer current")
        # citable evidence = the decision's own scoped A/B snapshot at or above tau (never later retrievals)
        snap = dec[4] if isinstance(dec[4], list) else json.loads(dec[4] or "[]")
        ab = {e["clause_key"] for e in snap if e.get("grade") in ("A", "B") and float(e.get("score", 0)) >= self.tau}
        if not ab:
            raise PermissionError("NO_EVIDENCE: the decision's evidence snapshot has no scoped A/B clause at or above tau")
        pred = list(d[3]) if d[3] is not None else list(d[2] or [])
        return {"p_adj": d[0], "pred_set": pred, "set_size": len(pred), "decision_id": str(decision_id), "signals_sha256": dec[3]}, ab, str(decision_id)

    @staticmethod
    def _check_items(items: list[dict], ab_keys: set[str]) -> None:
        if not items:
            raise PermissionError("a draft needs at least one action item")
        for it in items:
            if it.get("action_type") not in ACTION_TYPES:
                raise PermissionError(f"FORBIDDEN_ACTION: action_type {it.get('action_type')!r} not in the allowed set {sorted(ACTION_TYPES)}")
            if it.get("clause_key") not in ab_keys:
                raise PermissionError(f"UNGROUNDED_ITEM: {it.get('clause_key')!r} is not among the stored scoped A/B evidence")
            text = (it.get("item") or "").lower()
            if any(term in text for term in FORBIDDEN_TERMS):
                raise PermissionError("FORBIDDEN_ACTION: setpoint change / interlock bypass / emergency stop are outside the agent's authority")

    # ---- draft (new proposal or new version of an existing one) ----
    def draft(self, alarm_id: str, episode_id: UUID, window_index: int, policy: str, thread_id: str, action_items: list[dict],
              decision_id: UUID, base_proposal_id: Optional[UUID] = None, created_by: str = "agent") -> Draft:
        if policy not in ("E", "B3"):
            raise PermissionError("only grounded policies (E, B3) draft in the CMMS; B1/B2 proposal events are recorded by the policy, not here")
        with psycopg.connect(self.dsn) as conn:
            with conn.transaction(), conn.cursor() as cur:
                detect, ab, decision_id = self._grounding(cur, episode_id, window_index, alarm_id, policy, decision_id)
                self._check_items(action_items, ab)
                form = {"alarm_id": alarm_id, "episode_id": str(episode_id), "window_index": window_index, "policy": policy,
                        "decision_id": decision_id, "signals_sha256": detect["signals_sha256"],
                        "p_adj": detect["p_adj"], "pred_set": detect["pred_set"], "evidence_keys": sorted(ab),
                        "action_items": [{"action_type": i["action_type"], "item": i["item"], "clause_key": i["clause_key"]} for i in action_items]}
                sha = content_hash(form)
                if base_proposal_id is None:
                    pid, version = uuid.uuid4(), 1
                    cur.execute("""INSERT INTO mart.proposals (proposal_id, alarm_id, policy, action, ev_ab, request_form, current_version, episode_id, window_index, thread_id)
                                   VALUES (%s,%s,%s,'propose',true,%s,1,%s,%s,%s)""",
                                (pid, alarm_id, policy, json.dumps(form, ensure_ascii=False), episode_id, window_index, thread_id))
                else:
                    cur.execute("SELECT current_version, thread_id, alarm_id, episode_id, window_index, policy FROM mart.proposals WHERE proposal_id=%s FOR UPDATE", (base_proposal_id,))
                    row = cur.fetchone()
                    if not row:
                        raise ValueError("unknown proposal to revise")
                    if row[1] != thread_id:
                        raise PermissionError("THREAD_MISMATCH: proposal belongs to another thread")
                    if (row[2], row[3], row[4], row[5]) != (alarm_id, episode_id, window_index, policy):
                        raise PermissionError("EVENT_MISMATCH: a revision must keep the proposal's alarm/episode/window/policy; use a new proposal")
                    pid, version = base_proposal_id, row[0] + 1                       # allocated under the row lock
                    cur.execute("UPDATE mart.proposals SET current_version=%s, request_form=%s WHERE proposal_id=%s",
                                (version, json.dumps(form, ensure_ascii=False), pid))
                cur.execute("""INSERT INTO mart.proposal_drafts (proposal_id, draft_version, content_sha256, form, created_by)
                               VALUES (%s,%s,%s,%s,%s)""", (pid, version, sha, json.dumps(form, ensure_ascii=False), created_by))
        return Draft(pid, version, sha, form)

    # ---- approval: one-shot token bound to version + content + thread ----
    def approve(self, draft: Draft, approver: str, thread_id: str, decision: str = "approve", modifications: Optional[dict] = None) -> Optional[UUID]:
        token = uuid.uuid4() if decision == "approve" else None
        with psycopg.connect(self.dsn) as conn:
            with conn.transaction(), conn.cursor() as cur:
                cur.execute("SELECT content_sha256 FROM mart.proposal_drafts WHERE proposal_id=%s AND draft_version=%s", (draft.proposal_id, draft.draft_version))
                row = cur.fetchone()
                if not row or row[0] != draft.content_sha256 or content_hash(draft.form) != row[0]:
                    raise PermissionError("APPROVAL_MISMATCH: the draft presented for approval is not the stored version")
                cur.execute("""INSERT INTO mart.approvals (proposal_id, approver, decision, approval_token, token_used, modifications, draft_version, content_sha256, thread_id)
                               VALUES (%s,%s,%s,%s,false,%s,%s,%s,%s)""",
                            (draft.proposal_id, approver, decision, token, json.dumps(modifications or {}), draft.draft_version, draft.content_sha256, thread_id))
        return token

    # ---- issue: DB-authoritative, one transaction, idempotent on (proposal_id, draft_version) ----
    def issue(self, draft: Draft, approval_token: Optional[UUID], thread_id: str) -> dict:
        with psycopg.connect(self.dsn) as conn:
            try:
                with conn.transaction(), conn.cursor() as cur:
                    cur.execute("SELECT current_version, thread_id FROM mart.proposals WHERE proposal_id=%s FOR UPDATE", (draft.proposal_id,))
                    prop = cur.fetchone()
                    if not prop:
                        return {"status": "REFUSED", "reason": "UNKNOWN_PROPOSAL"}
                    current_version, owner_thread = prop
                    if owner_thread != thread_id:
                        return {"status": "REFUSED", "reason": "THREAD_MISMATCH"}
                    cur.execute("SELECT work_order_id, cmms_ref FROM mart.work_orders WHERE proposal_id=%s AND draft_version=%s", (draft.proposal_id, draft.draft_version))
                    existing = cur.fetchone()
                    if existing:
                        return {"status": "OK", "work_order_id": str(existing[0]), "cmms_ref": existing[1], "idempotent_replay": True}
                    if draft.draft_version != current_version:
                        return {"status": "REFUSED", "reason": "SUPERSEDED", "detail": f"current version is {current_version}"}
                    cur.execute("SELECT content_sha256 FROM mart.proposal_drafts WHERE proposal_id=%s AND draft_version=%s", (draft.proposal_id, draft.draft_version))
                    stored = cur.fetchone()
                    if not stored or stored[0] != draft.content_sha256 or content_hash(draft.form) != stored[0]:
                        return {"status": "REFUSED", "reason": "CONTENT_MISMATCH", "detail": "presented draft differs from the stored version"}
                    if approval_token is None:
                        return {"status": "REFUSED", "reason": "NO_APPROVAL"}
                    cur.execute("""UPDATE mart.approvals SET token_used=true
                                   WHERE approval_token=%s AND proposal_id=%s AND draft_version=%s AND content_sha256=%s AND thread_id=%s
                                     AND decision='approve' AND token_used=false RETURNING approval_id""",
                                (approval_token, draft.proposal_id, draft.draft_version, stored[0], thread_id))
                    row = cur.fetchone()
                    if not row:
                        cur.execute("SELECT token_used, draft_version, thread_id FROM mart.approvals WHERE approval_token=%s", (approval_token,))
                        a = cur.fetchone()
                        if a is None: return {"status": "REFUSED", "reason": "NO_APPROVAL"}
                        if a[0]: return {"status": "REFUSED", "reason": "TOKEN_USED"}
                        if a[2] != thread_id: return {"status": "REFUSED", "reason": "THREAD_MISMATCH"}
                        return {"status": "REFUSED", "reason": "APPROVAL_MISMATCH", "detail": "approval binds to another draft version/content"}
                    wo_id = uuid.uuid4(); cmms_ref = f"MOCK-WO-{str(wo_id)[:8].upper()}"
                    cur.execute("""INSERT INTO mart.work_orders (work_order_id, proposal_id, approval_id, cmms_ref, idempotency_key, status, draft_version)
                                   VALUES (%s,%s,%s,%s,%s,'issued',%s)""",
                                (wo_id, draft.proposal_id, row[0], cmms_ref, f"{draft.proposal_id}:{draft.draft_version}", draft.draft_version))
                    return {"status": "OK", "work_order_id": str(wo_id), "cmms_ref": cmms_ref, "idempotent_replay": False}
            except psycopg.errors.UniqueViolation:            # lost a race on (proposal_id, draft_version): return the winner's order
                pass
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT work_order_id, cmms_ref FROM mart.work_orders WHERE proposal_id=%s AND draft_version=%s", (draft.proposal_id, draft.draft_version))
            ex = cur.fetchone()
            return {"status": "OK", "work_order_id": str(ex[0]), "cmms_ref": ex[1], "idempotent_replay": True} if ex else {"status": "ERROR", "reason": "RACE_UNRESOLVED"}
