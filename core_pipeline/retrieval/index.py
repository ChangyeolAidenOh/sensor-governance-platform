"""
core_pipeline/retrieval/index.py — S3-03: hybrid retrieval over clause chunks.
  BM25 (rank-bm25, character bigram tokens — Korean particles make whitespace tokens weak; no morphological analyser to
  keep dependencies and reproducibility simple) + dense (bge-m3 via an injectable Embedder; vectors kept in a numpy
  array — ~800 chunks × 1024 dims, deterministic, no vector-DB service; chromadb left unused) → RRF (k=60, prereg §4.3,
  20 candidates from each) → cross-encoder reranker (bge-reranker-v2-m3 via an injectable Reranker) → top-k.
Models are injected so the pipeline is unit-testable without torch (HashEmbedder / LexicalReranker stubs in tests).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

RRF_K = 60
CANDIDATES_PER_SOURCE = 20


class Embedder(Protocol):
    name: str
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...        # (n, d), L2-normalised


class Reranker(Protocol):
    name: str
    def score(self, query: str, texts: Sequence[str]) -> np.ndarray: ...   # (n,), higher = more relevant


def bigram_tokens(text: str) -> list[str]:
    s = re.sub(r"\s+", " ", text.strip().lower())
    toks = []
    for w in s.split(" "):
        if len(w) == 1:
            toks.append(w)
        toks.extend(w[i:i + 2] for i in range(len(w) - 1))
    return toks


@dataclass
class ChunkIndex:
    chunks: list[dict]                 # from outputs/kosha_chunks.jsonl
    bm25: BM25Okapi
    dense: np.ndarray                  # (n, d) normalised
    embedder_name: str

    @classmethod
    def build(cls, chunks: list[dict], embedder: Embedder) -> "ChunkIndex":
        texts = [f"{c['title']}\n{c['text']}" for c in chunks]
        bm25 = BM25Okapi([bigram_tokens(t) for t in texts])
        dense = embedder.encode(texts)
        dense = dense / np.clip(np.linalg.norm(dense, axis=1, keepdims=True), 1e-12, None)
        return cls(chunks, bm25, dense.astype(np.float32), embedder.name)

    def save(self, dir_: Path, extra_meta: dict | None = None) -> None:
        dir_ = Path(dir_); dir_.mkdir(parents=True, exist_ok=True)
        np.save(dir_ / "dense.npy", self.dense)
        chunks_txt = "\n".join(json.dumps(c, ensure_ascii=False) for c in self.chunks) + "\n"
        (dir_ / "chunks.jsonl").write_text(chunks_txt, encoding="utf-8")
        import hashlib
        meta = {"embedder": self.embedder_name, "n_chunks": len(self.chunks), "dim": int(self.dense.shape[1]),
                "chunks_sha256": hashlib.sha256(chunks_txt.encode("utf-8")).hexdigest(),
                "rrf_k": RRF_K, "candidates_per_source": CANDIDATES_PER_SOURCE, "bm25_tokens": "char-bigram (fixed choice)",
                **(extra_meta or {})}
        (dir_ / "index_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, dir_: Path) -> "ChunkIndex":
        """Refuses an inconsistent artefact set: chunk file hash, vector row count and dim must match index_meta."""
        import hashlib
        dir_ = Path(dir_)
        chunks_txt = (dir_ / "chunks.jsonl").read_text(encoding="utf-8")
        chunks = [json.loads(l) for l in chunks_txt.splitlines() if l.strip()]
        meta = json.loads((dir_ / "index_meta.json").read_text())
        dense = np.load(dir_ / "dense.npy")
        sha = hashlib.sha256(chunks_txt.encode("utf-8")).hexdigest()
        if sha != meta.get("chunks_sha256"):
            raise ValueError("chunks.jsonl differs from the file the vectors were built from")
        if dense.shape != (meta["n_chunks"], meta["dim"]) or len(chunks) != meta["n_chunks"]:
            raise ValueError(f"dense {dense.shape} / chunks {len(chunks)} do not match index_meta {meta['n_chunks']}×{meta['dim']}")
        keys = [c["chunk_key"] for c in chunks]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate chunk_key in index")
        texts = [f"{c['title']}\n{c['text']}" for c in chunks]
        return cls(chunks, BM25Okapi([bigram_tokens(t) for t in texts]), dense, meta["embedder"])

    # ---- retrieval ----
    def candidates(self, query: str, qvec: np.ndarray, allowed_idx: np.ndarray) -> list[tuple[int, float]]:
        """RRF-fused candidates restricted to allowed_idx (ACL ∩ scope applied BEFORE ranking). Returns [(idx, rrf)]."""
        if len(allowed_idx) == 0:
            return []
        bm = self.bm25.get_scores(bigram_tokens(query))[allowed_idx]
        de = self.dense[allowed_idx] @ (qvec / max(np.linalg.norm(qvec), 1e-12))
        top_bm = allowed_idx[np.argsort(-bm, kind="stable")[:CANDIDATES_PER_SOURCE]]
        top_de = allowed_idx[np.argsort(-de, kind="stable")[:CANDIDATES_PER_SOURCE]]
        rrf: dict[int, float] = {}
        for ranked in (top_bm, top_de):
            for r, i in enumerate(ranked):
                rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (RRF_K + r + 1)
        return sorted(rrf.items(), key=lambda x: (-x[1], x[0]))
