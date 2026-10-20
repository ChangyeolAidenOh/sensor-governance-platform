"""
core_pipeline/retrieval/models.py — real embedder / reranker (sentence-transformers) and test stubs.
Real models (prereg §4.3): BAAI/bge-m3 (dense, 1024-d) and BAAI/bge-reranker-v2-m3 (cross-encoder).
  * REVISIONS ARE PINNED: core_pipeline/retrieval/model_pins.json holds the Hub commit sha of each model. On the first
    build (no pins file) the current Hub sha is resolved and WRITTEN; every later load passes revision=<sha>. The
    embedder's name includes the sha, and ChunkIndex stores it, so a query embedder that differs from the one that
    built the vectors is refused (Retriever.__post_init__).
  * RERANKER SCALE CONTRACT: the reranker returns a relevance score in [0, 1] = sigmoid(logit) applied EXACTLY ONCE.
    CrossEncoder is built with default_activation_function=Identity so predict() yields raw logits; sigmoid is applied
    here. The stub reranker follows the same scale. tau (prereg §4.3 grid {0.2..0.8}) lives on this scale. This is a
    relevance score, NOT a calibrated probability of correctness.
  * max_length=2048 tokens: a 1000-character Korean chunk + query stays below it (bge-reranker-v2-m3 supports 8192).
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

import json
from pathlib import Path

BGE_M3 = "BAAI/bge-m3"
BGE_RERANKER = "BAAI/bge-reranker-v2-m3"
PINS_PATH = Path(__file__).with_name("model_pins.json")
RERANKER_MAX_LENGTH = 2048


def resolve_pins(pins_path: Path = PINS_PATH) -> dict[str, str]:
    """{model_id: hub_commit_sha}. Resolved from the Hub once and written; afterwards read-only (commit the file)."""
    if pins_path.exists():
        return json.loads(pins_path.read_text())
    from huggingface_hub import model_info
    pins = {m: model_info(m).sha for m in (BGE_M3, BGE_RERANKER)}
    pins_path.write_text(json.dumps(pins, indent=2))
    return pins


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class STEmbedder:
    def __init__(self, device: str | None = None, pins: dict[str, str] | None = None):
        from sentence_transformers import SentenceTransformer
        pins = pins or resolve_pins()
        self.revision = pins[BGE_M3]
        self.name = f"{BGE_M3}@{self.revision[:12]}"
        self.model = SentenceTransformer(BGE_M3, device=device, revision=self.revision)
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True, batch_size=16, show_progress_bar=False), dtype=np.float32)


class STReranker:
    def __init__(self, device: str | None = None, pins: dict[str, str] | None = None):
        import torch
        from sentence_transformers import CrossEncoder
        pins = pins or resolve_pins()
        self.revision = pins[BGE_RERANKER]
        self.name = f"{BGE_RERANKER}@{self.revision[:12]}"
        try:                                                                          # sentence-transformers >= 5: activation_fn
            self.model = CrossEncoder(BGE_RERANKER, device=device, max_length=RERANKER_MAX_LENGTH, revision=self.revision,
                                      activation_fn=torch.nn.Identity())              # raw logits
        except TypeError:                                                             # older name
            self.model = CrossEncoder(BGE_RERANKER, device=device, max_length=RERANKER_MAX_LENGTH, revision=self.revision,
                                      default_activation_function=torch.nn.Identity())
    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        logits = np.asarray(self.model.predict([(query, t) for t in texts], batch_size=16, show_progress_bar=False), dtype=np.float64)
        return _sigmoid(logits)                                                       # sigmoid exactly once → [0, 1]


# ---------- stubs for tests (no torch) ----------
class HashEmbedder:
    """Deterministic bag-of-bigrams hashed into 256 dims — enough to make lexical overlap rank higher."""
    name = "hash-bigram-256"
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        from core_pipeline.retrieval.index import bigram_tokens
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in bigram_tokens(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % 256] += 1.0
        n = np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)
        return out / n


class LexicalReranker:
    """Jaccard overlap of bigrams → pseudo-logit 10·J − 5 → sigmoid, so the stub lives on the same [0, 1] scale."""
    name = "lexical-jaccard"
    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        from core_pipeline.retrieval.index import bigram_tokens
        q = set(bigram_tokens(query))
        out = []
        for t in texts:
            s = set(bigram_tokens(t))
            j = len(q & s) / max(1, len(q | s))
            out.append(10 * j - 5)
        return _sigmoid(np.asarray(out, dtype=np.float64))
