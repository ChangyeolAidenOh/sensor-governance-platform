"""core_pipeline/skills/public_skills.py: doubles around the real Skill client for the PUBLIC demo (demo-public branch).

The public demo runs on a hosted Postgres that was built locally (scripts/public_demo_build.py): the 10 demo episodes
were pre-computed through the two real Skill servers and every question the public UI offers was asked once through
the real graph ("priming"), so the retrieval results a visitor can reach are already stored in mart.agent_steps.
This module lets the public app answer those requests WITHOUT the bge embedder / reranker / index (4 GB), and lets one
labelled preset exercise the approval gate the way the test-DB synthetic case does (ST4-29, widened to the public demo).

  * StoredRetrieve(inner, dsn): `retrieve` is answered from rows the real skills_retrieve server stored earlier, with the
    same lookups RetrieveServer uses for its own reuse path (_stored_window / _stored_scoped) plus the same lookup keyed
    on the question text for open mode. A request that was never computed is REFUSED with DEMO_LIMIT; the graph handles
    a refused retrieve exactly as it does for the real server. Every call still goes through run_tool / StepLog, so
    request ownership, idempotent replay and trace.versions are recorded as for a server call. All other tools are
    delegated to `inner` (the real skills_detect server, which serves stored windows and runs the work-order tools).
  * SyntheticByPreset(inner, dsn, presets): demo.synthetic.GroundingOverride, but ONLY for threads whose
    mart.threads.preset is in `presets`. Other threads see the real envelope untouched.
  * policy_by_preset(dsn, presets): the real decide_action(sig, "E") for every thread except the synthetic presets, which
    get demo.synthetic.synthetic_policy. The thread is read from the session ContextVar (session.current_auth), so the
    graph code is unchanged.

Nothing here is imported by the evaluation code; the servers and the retrieval modules are not modified.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import psycopg

from core_pipeline.skills.schemas import validate_input, validate_result
from core_pipeline.skills.servers.common import Ctx, StepLog, envelope, run_tool, sha256_file

RETRIEVE_SKILL = "skills_retrieve"           # the stored rows were written by the real server under this skill name
K_PRIMARY, K_QUERY_MORE = 5, 10
PUBLIC_LIMIT_MESSAGE = ("공개 데모에서는 사전계산된 문서 검색만 재생합니다. 이 질문의 검색 결과는 저장되어 있지 않습니다. "
                        "추천 질문 버튼의 문장을 그대로 사용해 주세요.")


def _preset_of(dsn: str, thread_id: str, cache: dict) -> Optional[int]:
    if thread_id not in cache:
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT preset FROM mart.threads WHERE thread_id=%s", (thread_id,))
            r = cur.fetchone()
        cache[thread_id] = (int(r[0]) if r and r[0] is not None else None)
    return cache[thread_id]


class StoredRetrieve:
    """Answers `retrieve` from stored rows; delegates every other attribute to the inner client."""

    def __init__(self, inner: Any, dsn: str, meta_path: str = "outputs/model_meta/detector_v1.json",
                 tau_path: str = "models/retrieval_v1/tau.json", fault_map_path: str = "outputs/fault_to_guidance_map.json"):
        self._inner, self.dsn = inner, dsn
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        tau_doc = json.loads(Path(tau_path).read_text(encoding="utf-8"))
        self.contract_sha, self.detector_version, self.tau = meta["input_contract_sha256"], meta["version"], float(tau_doc["tau"])
        if Path(fault_map_path).is_file() and sha256_file(Path(fault_map_path)) != tau_doc.get("fault_map_sha256"):
            raise RuntimeError("INDEX_MISMATCH: fault map sha differs from the one tau was fixed on (tau.json)")
        self.versions = {"public_stored": True, "tau": self.tau, "fault_map_sha256": tau_doc.get("fault_map_sha256"),
                         "k_primary": K_PRIMARY, "k_query_more": K_QUERY_MORE, "contract_sha256": self.contract_sha, "detector": self.detector_version}
        self.steplog = StepLog(dsn, RETRIEVE_SKILL, self.versions)
        self.misses: list[dict] = []                      # (thread_id, mode, k, alarm_id, query) of every DEMO_LIMIT refusal, for the sidebar

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    # ---------------------------------------------------------------- the same lookups as RetrieveServer (ST4-18)
    def _stored_window(self, alarm_id: str) -> Optional[dict]:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT episode_id, window_index, detector FROM mart.alarms WHERE alarm_id=%s", (alarm_id,))
            a = cur.fetchone()
            if not a or a[2] != self.detector_version:
                return None
            cur.execute("""SELECT pred_set, expanded_set, q_used, query_request_id FROM mart.episode_windows
                           WHERE episode_id=%s AND window_index=%s AND detector_version=%s AND contract_sha256=%s""",
                        (a[0], a[1], a[2], self.contract_sha))
            w = cur.fetchone()
        if not w:
            return None
        return {"set": [int(x) for x in (w[1] if w[1] is not None else (w[0] or []))], "q_used": w[2], "query_request_id": w[3]}

    def _stored_scoped(self, alarm_id: str, k: int, caller_role: str) -> Optional[dict]:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("""SELECT output FROM mart.agent_steps WHERE skill=%s AND status='OK'
                           AND input->'args'->>'alarm_id'=%s AND input->'args'->>'mode'='scoped' AND (input->'args'->>'k')::int=%s
                           AND input->'ctx'->>'caller_role'=%s
                           AND input->>'contract_version'=%s AND output->'trace'->'versions'->>'contract_sha256'=%s
                           ORDER BY step_id LIMIT 1""", (RETRIEVE_SKILL, alarm_id, k, caller_role, "1.0", self.contract_sha))
            row = cur.fetchone()
        if not row:
            return None
        return row[0] if isinstance(row[0], dict) else json.loads(row[0])

    def _stored_open(self, query: str, k: int, caller_role: str, alarm_id: Optional[str]) -> Optional[dict]:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("""SELECT output FROM mart.agent_steps WHERE skill=%s AND status='OK'
                           AND input->'args'->>'mode'='open' AND input->'args'->>'query'=%s AND (input->'args'->>'k')::int=%s
                           AND input->'ctx'->>'caller_role'=%s AND coalesce(input->'args'->>'alarm_id','')=%s
                           AND input->>'contract_version'=%s AND output->'trace'->'versions'->>'contract_sha256'=%s
                           ORDER BY step_id LIMIT 1""", (RETRIEVE_SKILL, query, k, caller_role, alarm_id or "", "1.0", self.contract_sha))
            row = cur.fetchone()
        if not row:
            return None
        return row[0] if isinstance(row[0], dict) else json.loads(row[0])

    def _answer(self, ctx: Ctx, query: str, mode: str, fault_set: list[int], k: int, alarm_id: Optional[str]) -> dict:
        prior = None
        if alarm_id:
            w = self._stored_window(alarm_id)
            if w is None:
                return envelope("REFUSED", reason=("NO_ALARM", "alarm_id has no stored alarm / detection row under this detector and contract"))
            if mode == "scoped":
                if sorted(int(f) for f in fault_set) != sorted(w["set"]):
                    return envelope("REFUSED", reason=("ARG_NOT_IN_STATE", f"fault_set {sorted(fault_set)} != stored set {sorted(w['set'])} for this alarm"))
                prior = self._stored_scoped(alarm_id, k, ctx.caller_role)
        if mode == "open":
            prior = self._stored_open(query, k, ctx.caller_role, alarm_id)
        if prior is None:
            self.misses.append({"thread_id": ctx.thread_id, "mode": mode, "k": k, "alarm_id": alarm_id, "query": query})
            return envelope("REFUSED", reason=("DEMO_LIMIT", PUBLIC_LIMIT_MESSAGE))
        payload = dict(prior["payload"])
        payload["reused"] = True
        return envelope("OK", payload=payload, grounding=prior["grounding"])

    def retrieve(self, ctx: dict, query: str, mode: str, fault_set: list[int], k: int = 5, alarm_id: Optional[str] = None,
                 query_id: Optional[str] = None) -> dict:
        inp = {"ctx": ctx, "query": query, "mode": mode, "fault_set": [int(f) for f in fault_set], "k": int(k), "alarm_id": alarm_id, "query_id": query_id}
        errs = validate_input("retrieve", inp)
        if errs:
            return envelope("ERROR", reason=("SCHEMA_INVALID", "; ".join(errs)[:800]), trace={"skill": RETRIEVE_SKILL, "tool": "retrieve", "contract_version": "1.0"})
        return run_tool(self.steplog, "retrieve", ctx, {k_: v for k_, v in inp.items() if k_ != "ctx"},
                        lambda c: self._answer(c, query, mode, inp["fault_set"], inp["k"], alarm_id),
                        violation_codes=("ARG_NOT_IN_STATE", "ACL_DENIED"), validate=lambda out: validate_result("retrieve", out))


class SyntheticByPreset:
    """demo.synthetic.GroundingOverride restricted to the threads of the given presets (public synthetic approval case)."""

    def __init__(self, inner: Any, dsn: str, presets: tuple[int, ...]):
        self._inner, self.dsn, self.presets, self._cache = inner, dsn, tuple(int(p) for p in presets), {}

    def is_synthetic(self, thread_id: str) -> bool:
        return _preset_of(self.dsn, thread_id, self._cache) in self.presets

    def __getattr__(self, name: str):
        fn = getattr(self._inner, name)
        if name != "retrieve":
            return fn

        def retrieve(ctx, query, mode, fault_set, k=5, alarm_id=None, query_id=None):
            env = fn(ctx, query, mode, fault_set, k, alarm_id=alarm_id, query_id=query_id)
            if not self.is_synthetic(str(ctx.get("thread_id", ""))):
                return env
            if env.get("status") == "OK" and mode == "scoped" and env.get("grounding"):
                g = dict(env["grounding"])
                snap = [dict(s, score=(0.9 if s["grade"] in ("A", "B") else s["score"])) for s in g["snapshot"]]
                if not any(s["grade"] in ("A", "B") for s in snap):
                    return env
                g.update(snapshot=snap, g_AB=0.9, ev_AB=True)
                env = {**env, "grounding": g, "trace": {**(env.get("trace") or {}), "synthetic": True}}
            return env
        return retrieve


def policy_by_preset(dsn: str, presets: tuple[int, ...]):
    from core_pipeline.agent.graph import POLICY
    from core_pipeline.agent.session import current_auth
    from core_pipeline.policy.decide_action import decide_action
    from demo.synthetic import synthetic_policy
    cache: dict = {}
    syn = tuple(int(p) for p in presets)

    def policy(sig):
        if _preset_of(dsn, current_auth().thread_id, cache) in syn:
            return synthetic_policy(sig)
        return decide_action(sig, POLICY)
    return policy
