"""core_pipeline/policy/initial_snapshot.py — freeze / load the first-observation policy input (ST4-50, option C).

The ONLY write path into mart.policy_input_snapshots. Driver-side direct SQL (like agent/plans.py): no Skill server
module is imported, and the Skill contract is unchanged.

freeze_policy_input runs in one transaction under the SAME per-(episode, detector, contract) advisory lock the detect
server's DetectionStateStore uses for spend_query / set_expanded, so "freeze" and "spend" serialise:
  1. a snapshot with the same key already exists → verify it (content hash, immutable source columns, expected
     retrieval versions) and return it. The CURRENT q_used is not consulted: a window whose query E spent after the
     freeze still returns its snapshot on restart.
  2. no snapshot → the window must still be unexpanded and unspent (q_used=false, expanded_set IS NULL), otherwise
     SNAPSHOT_TOO_LATE. The latest state is never reconstructed into a first observation.
  3. alarm window → the FIRST OK scoped k=5 envelope of (alarm, role, contract) in mart.agent_steps (the same row
     skills_retrieve._stored_scoped reuses) must exist and must have been computed on pred_set.
  4. INSERT.
load_policy_input never falls back: a missing row is SNAPSHOT_MISSING.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional
from uuid import UUID

import psycopg

from core_pipeline.policy.policy_input import K5Grounding, PolicyInput, versions_mismatch

K_PRIMARY = 5
RETRIEVE_SKILL = "skills_retrieve"
CONTRACT_VERSION = "1.0"
_WIN_COLS = "gate, drift, drift_assessed, p_value, p_adj, m_t, alarm, pred_set, set_size, top1, q_used, expanded_set"
_SNAP_COLS = ("snapshot_id, gate, drift, drift_assessed, p_value, p_adj, m_t, alarm, alarm_id, pred_set, set_size, top1, "
              "k5_request_id, k5_fault_set, k5_mode, k5_tau, k5_g_ab, k5_g_c, k5_ev_ab, k5_ev_c, k5_snapshot, k5_versions, content_sha256")


class SnapshotError(RuntimeError):
    CODES = ("SNAPSHOT_TOO_LATE", "SNAPSHOT_K5_MISSING", "SNAPSHOT_K5_MISMATCH", "SNAPSHOT_MISMATCH", "SNAPSHOT_STALE",
             "SNAPSHOT_MISSING", "SNAPSHOT_SOURCE_MISSING")

    def __init__(self, code: str, message: str):
        assert code in self.CODES, code
        super().__init__(f"{code}: {message}")
        self.code = code


def lock_key(episode_id: UUID | str, detector_version: str, contract_sha256: str) -> int:
    """Identical to DetectionStateStore._lock_key (tests/unit/test_initial_snapshot.py pins the equality)."""
    h = hashlib.sha256(f"{UUID(str(episode_id))}|{detector_version}|{contract_sha256}".encode()).digest()
    return int.from_bytes(h[:8], "big", signed=True)


def alarm_id_of(episode_id: UUID | str, window_index: int, detector_version: str) -> str:
    return f"{UUID(str(episode_id))}:{int(window_index)}:{detector_version}"


def _jsonb(v: Any) -> Any:
    return v if isinstance(v, (dict, list)) or v is None else json.loads(v)


# ------------------------------------------------------------------------------------------------- pure validation
def validate_k5(request_id: UUID | str, bound_input: dict, output: dict, pred_set: list[int], alarm_id: str, contract_sha256: str) -> K5Grounding:
    """Turn the first stored scoped k=5 envelope into a K5Grounding, refusing anything not computed on the first
    observation of this alarm under this contract."""
    args = bound_input.get("args") or {}
    gr = output.get("grounding") or {}
    payload = output.get("payload") or {}
    versions = ((output.get("trace") or {}).get("versions")) or {}
    problems = []
    if output.get("status") != "OK":
        problems.append(f"status {output.get('status')}")
    if str(args.get("mode")) != "scoped" or str(gr.get("mode")) != "scoped":
        problems.append("mode is not scoped")
    if int(args.get("k", -1)) != K_PRIMARY:
        problems.append(f"k {args.get('k')} != {K_PRIMARY}")
    if sorted(int(f) for f in (args.get("fault_set") or [])) != sorted(int(f) for f in pred_set):
        problems.append(f"envelope fault_set {sorted(args.get('fault_set') or [])} != first pred_set {sorted(pred_set)}")
    if str(args.get("alarm_id")) != alarm_id or str(payload.get("alarm_id")) != alarm_id:
        problems.append("alarm_id differs")
    if str(bound_input.get("contract_version")) != CONTRACT_VERSION or str(versions.get("contract_sha256")) != contract_sha256:
        problems.append("contract differs")
    if not isinstance(gr.get("snapshot"), list):
        problems.append("grounding.snapshot missing")
    if problems:
        raise SnapshotError("SNAPSHOT_K5_MISMATCH", "; ".join(problems))
    return K5Grounding(request_id=str(request_id), fault_set=[int(f) for f in args["fault_set"]], mode="scoped",
                       tau=gr.get("tau"), g_ab=gr.get("g_AB"), g_c=gr.get("g_C"), ev_ab=gr.get("ev_AB"), ev_c=gr.get("ev_C"),
                       snapshot=[{"clause_key": s["clause_key"], "grade": s["grade"], "score": float(s["score"])} for s in gr["snapshot"]],
                       versions=dict(versions))


def verify_against_source(pi: PolicyInput, win: Optional[tuple], expected_k5_versions: Optional[dict]) -> None:
    """Stored snapshot ↔ (its own hash, the immutable episode_windows columns, the run's expected retrieval versions)."""
    if pi.content_hash() != pi.content_sha256:
        raise SnapshotError("SNAPSHOT_MISMATCH", "content_sha256 does not match the stored columns")
    if win is None:
        raise SnapshotError("SNAPSHOT_SOURCE_MISSING", "episode_windows row of the snapshot is gone")
    gate, drift, da, p, padj, m_t, alarm, ps, ss, top1, _q, _x = win
    src = {"gate": gate, "drift": drift, "drift_assessed": bool(da), "p_value": p, "p_adj": padj, "m_t": int(m_t), "alarm": alarm,
           "pred_set": [int(x) for x in (ps or [])], "set_size": int(ss or 0), "top1": top1}
    mine = {k: v for k, v in pi.detection_block().items() if k != "alarm_id"}
    if mine != src:
        diff = [k for k in src if src[k] != mine.get(k)]
        raise SnapshotError("SNAPSHOT_MISMATCH", f"detection columns differ from episode_windows: {diff}")
    if expected_k5_versions is not None and pi.k5 is not None:
        bad = versions_mismatch(pi.k5.versions, expected_k5_versions)
        if bad:
            raise SnapshotError("SNAPSHOT_STALE", f"k=5 retrieval versions differ from the run's expectation: {bad}")


# ------------------------------------------------------------------------------------------------- DB access
def _read_window(cur, episode_id: UUID, t: int, detector: str, contract: str) -> Optional[tuple]:
    cur.execute(f"SELECT {_WIN_COLS} FROM mart.episode_windows WHERE episode_id=%s AND window_index=%s AND detector_version=%s AND contract_sha256=%s",
                (episode_id, t, detector, contract))
    return cur.fetchone()


def _read_snapshot(cur, episode_id: UUID, t: int, detector: str, contract: str, role: str) -> Optional[PolicyInput]:
    cur.execute(f"""SELECT {_SNAP_COLS} FROM mart.policy_input_snapshots
                    WHERE episode_id=%s AND window_index=%s AND detector_version=%s AND contract_sha256=%s AND caller_role=%s""",
                (episode_id, t, detector, contract, role))
    row = cur.fetchone()
    if not row:
        return None
    (sid, gate, drift, da, p, padj, m_t, alarm, aid, ps, ss, top1, k5_rid, k5_fs, k5_mode, k5_tau, k5_gab, k5_gc, k5_evab, k5_evc, k5_snap, k5_ver, sha) = row
    k5 = None
    if k5_rid is not None:
        k5 = K5Grounding(str(k5_rid), [int(x) for x in (k5_fs or [])], k5_mode, k5_tau, k5_gab, k5_gc, k5_evab, k5_evc, _jsonb(k5_snap) or [], _jsonb(k5_ver) or {})
    return PolicyInput(str(episode_id), int(t), detector, contract, role, gate, drift, bool(da), p, padj, int(m_t), alarm, aid,
                       [int(x) for x in (ps or [])], int(ss or 0), top1, k5, str(sid), sha)


def _first_k5(cur, alarm_id: str, role: str, contract: str) -> Optional[tuple]:
    """The same row skills_retrieve._stored_scoped reuses: earliest OK scoped k=5 of (alarm, role) under this contract."""
    cur.execute("""SELECT request_id, input, output FROM mart.agent_steps WHERE skill=%s AND status='OK'
                   AND input->'args'->>'alarm_id'=%s AND input->'args'->>'mode'='scoped' AND (input->'args'->>'k')::int=%s
                   AND input->'ctx'->>'caller_role'=%s
                   AND input->>'contract_version'=%s AND output->'trace'->'versions'->>'contract_sha256'=%s
                   ORDER BY step_id LIMIT 1""", (RETRIEVE_SKILL, alarm_id, K_PRIMARY, role, CONTRACT_VERSION, contract))
    return cur.fetchone()


def freeze_policy_input(dsn: str, episode_id: UUID | str, t: int, detector_version: str, contract_sha256: str, caller_role: str,
                        eval_run_id: Optional[UUID | str] = None, expected_k5_versions: Optional[dict] = None) -> PolicyInput:
    ep = UUID(str(episode_id))
    with psycopg.connect(dsn) as conn:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key(ep, detector_version, contract_sha256),))
            win = _read_window(cur, ep, t, detector_version, contract_sha256)
            existing = _read_snapshot(cur, ep, t, detector_version, contract_sha256, caller_role)
            if existing is not None:
                verify_against_source(existing, win, expected_k5_versions)
                return existing
            if win is None:
                raise SnapshotError("SNAPSHOT_SOURCE_MISSING", f"window {t} of episode {ep} has no committed detection row under this contract")
            gate, drift, da, p, padj, m_t, alarm, ps, ss, top1, q_used, expanded = win
            if q_used or expanded is not None:
                raise SnapshotError("SNAPSHOT_TOO_LATE", f"window {t} already spent its query (q_used={q_used}, expanded={expanded is not None}); "
                                                         "the first observation cannot be reconstructed from the latest state")
            pred_set = [int(x) for x in (ps or [])]
            k5, aid = None, None
            if alarm is True:
                aid = alarm_id_of(ep, t, detector_version)
                row = _first_k5(cur, aid, caller_role, contract_sha256)
                if row is None:
                    raise SnapshotError("SNAPSHOT_K5_MISSING", f"no OK scoped k={K_PRIMARY} envelope for alarm {aid} (role {caller_role}) under this contract")
                k5 = validate_k5(row[0], _jsonb(row[1]), _jsonb(row[2]), pred_set, aid, contract_sha256)
            pi = PolicyInput(str(ep), int(t), detector_version, contract_sha256, caller_role, gate, drift, bool(da), p, padj, int(m_t), alarm, aid,
                             pred_set, int(ss or 0), top1, k5)
            pi = PolicyInput(**{**pi.__dict__, "content_sha256": pi.content_hash()})
            if expected_k5_versions is not None and k5 is not None:
                bad = versions_mismatch(k5.versions, expected_k5_versions)
                if bad:
                    raise SnapshotError("SNAPSHOT_STALE", f"k=5 retrieval versions differ from the run's expectation: {bad}")
            kb = pi.k5_block()
            cur.execute("""INSERT INTO mart.policy_input_snapshots
                             (episode_id, window_index, detector_version, contract_sha256, caller_role, gate, drift, drift_assessed, p_value, p_adj,
                              m_t, alarm, alarm_id, pred_set, set_size, top1, k5_request_id, k5_fault_set, k5_mode, k5_tau, k5_g_ab, k5_g_c,
                              k5_ev_ab, k5_ev_c, k5_snapshot, k5_versions, content_sha256, frozen_by_eval_run)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                           ON CONFLICT (episode_id, window_index, detector_version, contract_sha256, caller_role) DO NOTHING
                           RETURNING snapshot_id""",
                        (ep, t, detector_version, contract_sha256, caller_role, gate, drift, bool(da), p, padj, int(m_t), alarm, aid, pred_set,
                         int(ss or 0), top1, UUID(kb["request_id"]) if kb else None, kb["fault_set"] if kb else None, kb["mode"] if kb else None,
                         kb["tau"] if kb else None, kb["g_ab"] if kb else None, kb["g_c"] if kb else None, kb["ev_ab"] if kb else None,
                         kb["ev_c"] if kb else None, json.dumps(kb["snapshot"], ensure_ascii=False) if kb else None,
                         json.dumps(kb["versions"], ensure_ascii=False, default=str) if kb else None, pi.content_sha256,
                         UUID(str(eval_run_id)) if eval_run_id else None))
            got = cur.fetchone()
            if got is None:                                      # lost a race despite the lock (should not happen): read back and verify
                stored = _read_snapshot(cur, ep, t, detector_version, contract_sha256, caller_role)
                verify_against_source(stored, win, expected_k5_versions)
                return stored
            return PolicyInput(**{**pi.__dict__, "snapshot_id": str(got[0])})


def load_policy_input(dsn: str, episode_id: UUID | str, t: int, detector_version: str, contract_sha256: str, caller_role: str,
                      expected_k5_versions: Optional[dict] = None) -> PolicyInput:
    ep = UUID(str(episode_id))
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        snap = _read_snapshot(cur, ep, t, detector_version, contract_sha256, caller_role)
        if snap is None:
            raise SnapshotError("SNAPSHOT_MISSING", f"no first-observation snapshot for window {t} of episode {ep} (role {caller_role})")
        win = _read_window(cur, ep, t, detector_version, contract_sha256)
    verify_against_source(snap, win, expected_k5_versions)
    return snap
