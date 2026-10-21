"""core_pipeline/skills/servers/skills_retrieve.py — FastMCP server 2 of 2 (ST4-01): retrieve. Imports torch via
sentence-transformers (bge-m3 + bge-reranker-v2-m3); NEVER lightgbm.

Contract v1.0 points enforced here:
  §1.2/§1.8 (A-4)  a successful search returns status OK; grounding is a separate block {mode, tau, g_AB, g_C, ev_AB, ev_C,
                   snapshot}. The action path is grounded only by scoped ∧ ev_AB === true. NO_EVIDENCE is no longer a
                   retrieve status — it is work_order_draft's refusal code.
  §1.4             scoped + alarm_id: fault_set MUST equal the set the detect server stored for that alarm's window under
                   the SAME input contract (expanded_set if the expansion ran, else pred_set) → REFUSED(ARG_NOT_IN_STATE).
  §3.5 (ST4-13)    scoped k=10 is the query_more re-search: it must present the query_id that spend_query stored for the
                   alarm's window (QUERY_REQUIRED / QUERY_USED). scoped k=5 for an alarm is computed ONCE; later calls
                   return the stored envelope (payload.reused=true) — no repeated re-ranking outside the query budget.
  §1.6             ACL pre-filter then mapping hard filter (scoped) — both inside Retriever, before ranking.
  §2.3             open-mode results are stored as evidence rows (mode='open') but are never a basis for ev_AB.
Evidence rows (mart.evidence) are written only when an alarm_id is given; the decide node's snapshot is grounding.snapshot.

Run:   python -m core_pipeline.skills.servers.skills_retrieve     (stdio; the client spawns it)
Env:   SGP_PG_DSN, SGP_RETRIEVAL_DIR, SGP_FAULT_MAP, SGP_TAU_PATH, SGP_DEVICE (cpu|mps)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import psycopg
from fastmcp import FastMCP

from core_pipeline.retrieval.acl import load_roles
from core_pipeline.retrieval.index import ChunkIndex
from core_pipeline.retrieval.models import STEmbedder, STReranker, resolve_pins
from core_pipeline.retrieval.retrieve import K_PRIMARY, K_QUERY_MORE, RetrieveResult, Retriever
from core_pipeline.retrieval.scope import FaultScope
from core_pipeline.skills.schemas import validate_input, validate_result
from core_pipeline.skills.servers.common import Ctx, StepLog, env, envelope, jsonable, log, run_tool, sha256_file

SKILL = "skills_retrieve"
mcp = FastMCP(SKILL)


def _guide_key(chunks: list[dict]) -> str:
    for k in ("guide_no", "guide", "guide_id"):
        if chunks and k in chunks[0]:
            return k
    raise RuntimeError(f"cannot find the guide field in chunk records: keys={list(chunks[0].keys()) if chunks else []}")


class RetrieveServer:
    def __init__(self) -> None:
        self.dsn = env("SGP_PG_DSN", required=True)
        rdir = Path(env("SGP_RETRIEVAL_DIR", "models/retrieval_v1"))
        fmap = Path(env("SGP_FAULT_MAP", "outputs/fault_to_guidance_map.json"))
        tau_path = Path(env("SGP_TAU_PATH", "models/retrieval_v1/tau.json"))
        device = env("SGP_DEVICE", None)
        meta = json.loads(Path("outputs/model_meta/detector_v1.json").read_text())
        self.contract_sha, self.detector_version = meta["input_contract_sha256"], meta["version"]
        tau_doc = json.loads(tau_path.read_text())
        self.tau = float(tau_doc["tau"])
        if sha256_file(fmap) != tau_doc.get("fault_map_sha256"):
            raise RuntimeError("INDEX_MISMATCH: fault map sha differs from the one tau was fixed on (tau.json)")
        index = ChunkIndex.load(rdir)
        guides = {c[_guide_key(index.chunks)] for c in index.chunks}
        pins = resolve_pins()
        self.retriever = Retriever(index=index, embedder=STEmbedder(device=device, pins=pins), reranker=STReranker(device=device, pins=pins),
                                   scope=FaultScope(fmap), roles=load_roles(fmap, guides), tau=self.tau)
        self.versions = {"embedder": index.embedder_name, "reranker": getattr(self.retriever.reranker, "name", "bge-reranker-v2-m3"), "tau": self.tau,
                         "fault_map_sha256": tau_doc.get("fault_map_sha256"), "index_dir": str(rdir), "k_primary": K_PRIMARY, "k_query_more": K_QUERY_MORE,
                         "contract_sha256": self.contract_sha, "detector": self.detector_version}
        self.steplog = StepLog(self.dsn, SKILL, self.versions)
        log.info("skills_retrieve ready: %s", self.versions)

    def _stored_window(self, alarm_id: str) -> Optional[dict]:
        """The window row the detect server stored for the alarm under THIS contract (expanded_set wins once it ran)."""
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
        """An earlier OK scoped retrieval for (alarm, k) under this contract, computed for the SAME caller role (ST4-18:
        results are reusable across sessions, never across ACL roles) — the stored envelope is the authority."""
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("""SELECT output FROM mart.agent_steps WHERE skill=%s AND status='OK'
                           AND input->'args'->>'alarm_id'=%s AND input->'args'->>'mode'='scoped' AND (input->'args'->>'k')::int=%s
                           AND input->'ctx'->>'caller_role'=%s
                           AND input->>'contract_version'=%s AND output->'trace'->'versions'->>'contract_sha256'=%s
                           ORDER BY step_id LIMIT 1""", (SKILL, alarm_id, k, caller_role, "1.0", self.contract_sha))
            row = cur.fetchone()
        if not row:
            return None
        return row[0] if isinstance(row[0], dict) else json.loads(row[0])

    def retrieve(self, ctx: Ctx, query: str, mode: str, fault_set: list[int], k: int, alarm_id: Optional[str], query_id: Optional[str]) -> dict:
        if ctx.caller_role not in self.retriever.roles:
            return envelope("REFUSED", reason=("ACL_DENIED", f"role {ctx.caller_role!r} has no ACL entry"))
        if alarm_id:
            w = self._stored_window(alarm_id)
            if w is None:
                return envelope("REFUSED", reason=("NO_ALARM", "alarm_id has no stored alarm / detection row under this detector and contract"))
            if mode == "scoped":
                if sorted(int(f) for f in fault_set) != sorted(w["set"]):
                    return envelope("REFUSED", reason=("ARG_NOT_IN_STATE", f"fault_set {sorted(fault_set)} != stored set {sorted(w['set'])} for this alarm"))
                prior = self._stored_scoped(alarm_id, k, ctx.caller_role)
                if prior is not None:                                            # computed once per (alarm, k, role); reuse needs no query ownership
                    payload = dict(prior["payload"]); payload["reused"] = True
                    return envelope("OK", payload=payload, grounding=prior["grounding"])
                if k == K_QUERY_MORE:
                    if query_id is None:
                        return envelope("REFUSED", reason=("QUERY_REQUIRED", "k=10 is the query_more re-search: present the query_id from spend_query"))
                    if not w["q_used"] or str(w["query_request_id"]) != str(query_id):
                        return envelope("REFUSED", reason=("QUERY_USED", "query_id does not own this window's query"))
        elif mode == "scoped" and k == K_QUERY_MORE:
            return envelope("REFUSED", reason=("QUERY_REQUIRED", "k=10 needs an alarm_id and its query_id"))
        r: RetrieveResult = self.retriever(query, [int(f) for f in fault_set], ctx.caller_role, mode=mode, k=k)
        if str(r.status).upper().startswith("ERROR"):
            return envelope("ERROR", reason=("INDEX_MISMATCH", f"retriever status {r.status}"))
        clauses = [jsonable(c) for c in r.chunks]
        snapshot = [{"clause_key": c["clause_key"], "grade": c["grade"], "score": float(c["score"])} for c in clauses]
        if alarm_id:
            with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
                for c in clauses:
                    cur.execute("INSERT INTO mart.evidence (alarm_id, mode, k, clause_key, grade, rank, score, tau) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                                (alarm_id, mode, k, c["clause_key"], c["grade"], int(c["rank"]), float(c["score"]), self.tau))
                conn.commit()
        ev_ab = None if mode != "scoped" or r.ev_AB is None else bool(r.ev_AB)       # open mode never grounds (ST4-07)
        if mode == "open":
            ev_ab = False
        grounding = {"mode": mode, "tau": r.tau if r.tau is not None else self.tau, "g_AB": r.g_AB, "g_C": r.g_C,
                     "ev_AB": ev_ab, "ev_C": None if r.ev_C is None else bool(r.ev_C), "snapshot": snapshot}
        payload = {"clauses": clauses, "k": k, "mode": mode, "scope_size": int(r.scope_size), "n_candidates": int(r.n_candidates), "alarm_id": alarm_id, "reused": False}
        return envelope("OK", payload=payload, grounding=grounding)


_SERVER: Optional[RetrieveServer] = None


def server() -> RetrieveServer:
    global _SERVER
    if _SERVER is None:
        _SERVER = RetrieveServer()
    return _SERVER


@mcp.tool()
def retrieve(ctx: dict, query: str, mode: str, fault_set: list[int], k: int = 5, alarm_id: Optional[str] = None,
             query_id: Optional[str] = None) -> dict:
    """Hybrid KOSHA clause retrieval (ACL → scope → BM25+dense → RRF → reranker). Grounding is returned, never decided.
    k=10 (query_more re-search) requires the query_id from spend_query."""
    inp = {"ctx": ctx, "query": query, "mode": mode, "fault_set": fault_set, "k": k, "alarm_id": alarm_id, "query_id": query_id}
    errs = validate_input("retrieve", inp)
    if errs:
        return envelope("ERROR", reason=("SCHEMA_INVALID", "; ".join(errs)[:800]), trace={"skill": SKILL, "tool": "retrieve", "contract_version": "1.0"})
    return run_tool(server().steplog, "retrieve", ctx, {k_: v for k_, v in inp.items() if k_ != "ctx"},
                    lambda c: server().retrieve(c, query, mode, fault_set, k, alarm_id, query_id),
                    violation_codes=("ARG_NOT_IN_STATE", "ACL_DENIED"), validate=lambda out: validate_result("retrieve", out))


if __name__ == "__main__":
    # models load lazily on the first tool call (the MCP initialize handshake must not wait for torch); the client
    # allows SGP_MCP_TIMEOUT seconds per call for that first load
    mcp.run()
