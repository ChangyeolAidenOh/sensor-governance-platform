"""
core_pipeline/retrieval/retrieve.py — S3-05: the retrieve function contract (skill contract v0.1 §2.3).
  order: ACL pre-filter → (scoped) mapping hard filter → BM25 + dense candidates → RRF → reranker → top-k
  mode='scoped' (action path, replay): scope = predicted set's A∪B∪C (empty set → C only); grades A/B/C
  mode='open'   (conversation evidence queries only): ACL only; grade may be OUT; NEVER a basis for ev_AB
  returns g_AB (best reranker score among A/B chunks), g_C, and ev_AB / ev_C when tau is given (tau is fixed at the
  end of Stage 3 on the dev set; before that ev_* are None and the policy cannot propose).
  Two things are kept apart: `status` says whether relevant chunks were RETURNED (OK / NO_RESULTS); `ev_AB` says
  whether they QUALIFY as grounding for a proposal (scoped ∧ A/B ∧ score >= tau). open mode can return useful
  results with status OK while ev_AB stays False. `grounding` summarises: 'AB' | 'C_only' | 'none' | 'unknown_tau'.
  caller_role comes from the authenticated execution context (graph state), never from the LLM.
Every returned chunk is a real indexed chunk (citation traceability, H3); the LLM never writes clause text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from core_pipeline.retrieval.acl import allowed
from core_pipeline.retrieval.index import ChunkIndex, Embedder, Reranker
from core_pipeline.retrieval.scope import FaultScope, GRADE_RANK

K_PRIMARY, K_QUERY_MORE = 5, 10


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_key: str
    clause_key: str
    guide_no: str
    title: str
    text: str
    page: Optional[int]
    grade: str
    rank: int
    score: float            # reranker score
    rrf: float


@dataclass(frozen=True)
class RetrieveResult:
    status: str             # OK (chunks returned) | NO_RESULTS (nothing in ACL ∩ scope)
    grounding: str          # AB | C_only | none | unknown_tau   (proposal eligibility, separate from status)
    mode: str
    k: int
    chunks: list[RetrievedChunk] = field(default_factory=list)
    g_AB: Optional[float] = None
    g_C: Optional[float] = None
    tau: Optional[float] = None
    ev_AB: Optional[bool] = None
    ev_C: Optional[bool] = None
    scope_size: int = 0
    n_candidates: int = 0


@dataclass
class Retriever:
    index: ChunkIndex
    embedder: Embedder
    reranker: Reranker
    scope: FaultScope
    roles: dict[str, set[str]]
    tau: Optional[float] = None          # fixed at Stage 3 end (prereg §4.3); None = not yet fixed
    last_rerank_texts: list[str] = field(default_factory=list)   # every candidate text sent to the reranker in the last call (diagnostics)

    def __post_init__(self):
        # index ↔ embedder consistency (ST3-01 condition): the stored vectors must come from this embedder
        if self.index.embedder_name != self.embedder.name:
            raise ValueError(f"index embedded with {self.index.embedder_name!r} but query embedder is {self.embedder.name!r}")

    def __call__(self, query: str, pred_set: list[int], caller_role: str, mode: str = "scoped", k: int = K_PRIMARY) -> RetrieveResult:
        assert mode in ("scoped", "open") and k in (K_PRIMARY, K_QUERY_MORE)
        chunks = self.index.chunks
        acl_ok = np.array([allowed(caller_role, self.roles, c["guide_no"]) for c in chunks])
        if mode == "scoped":
            keys = self.scope.scope_keys(pred_set)
            in_scope = np.array([c["clause_key"] in keys for c in chunks])
            mask = acl_ok & in_scope
        else:
            mask = acl_ok
        allowed_idx = np.flatnonzero(mask)
        scope_size = int(mask.sum())
        qvec = self.embedder.encode([query])[0]
        cands = self.index.candidates(query, qvec, allowed_idx)
        if not cands:
            return RetrieveResult("NO_RESULTS", "none" if self.tau is not None else "unknown_tau", mode, k, [], None, None,
                                  self.tau, False if self.tau is not None else None, False if self.tau is not None else None,
                                  scope_size, 0)
        idxs = [i for i, _ in cands]
        rr = {i: r for i, r in cands}
        rerank_texts = [f"{chunks[i]['title']}\n{chunks[i]['text']}" for i in idxs]
        self.last_rerank_texts = rerank_texts
        scores = self.reranker.score(query, rerank_texts)
        order = np.argsort(-scores, kind="stable")[:k]
        out = []
        for rank, o in enumerate(order, 1):
            i = idxs[o]; c = chunks[i]
            out.append(RetrievedChunk(c["chunk_key"], c["clause_key"], c["guide_no"], c["title"], c["text"], c.get("page"),
                                      self.scope.grade(c["clause_key"], pred_set), rank, float(scores[o]), float(rr[i])))
        ab = [x.score for x in out if x.grade in ("A", "B")]
        cc = [x.score for x in out if x.grade == "C"]
        g_AB = max(ab) if ab else None
        g_C = max(cc) if cc else None
        if self.tau is None:
            ev_AB = ev_C = None; grounding = "unknown_tau"
        else:
            ev_AB = (mode == "scoped") and (g_AB is not None) and (g_AB >= self.tau)   # open-mode results never ground a proposal
            ev_C = (g_C is not None) and (g_C >= self.tau)
            grounding = "AB" if ev_AB else ("C_only" if ev_C else "none")
        return RetrieveResult("OK", grounding, mode, k, out, g_AB, g_C, self.tau, ev_AB, ev_C, scope_size, len(cands))
