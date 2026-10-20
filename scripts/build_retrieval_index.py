"""
scripts/build_retrieval_index.py — S3-03: embed outputs/kosha_chunks.jsonl with bge-m3 and write models/retrieval_v1/
(dense.npy, chunks.jsonl, index_meta.json). Deterministic for a fixed model revision. ~800 chunks: a minute on M2.
Usage: python scripts/build_retrieval_index.py [--chunks outputs/kosha_chunks.jsonl] [--out models/retrieval_v1]
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_pipeline.retrieval.index import ChunkIndex
from core_pipeline.retrieval.models import STEmbedder, STReranker, resolve_pins


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="outputs/kosha_chunks.jsonl")
    ap.add_argument("--out", default="models/retrieval_v1")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    chunks = [json.loads(l) for l in Path(a.chunks).read_text(encoding="utf-8").splitlines() if l.strip()]
    pins = resolve_pins()                                  # written on first run → commit core_pipeline/retrieval/model_pins.json
    emb = STEmbedder(device=a.device, pins=pins)
    rr = STReranker(device=a.device, pins=pins)             # loaded here only to pin + record its revision
    idx = ChunkIndex.build(chunks, emb)
    idx.save(Path(a.out), extra_meta={"source_chunks_file_sha256": hashlib.sha256(Path(a.chunks).read_bytes()).hexdigest(),
                                      "embedder_revision": emb.revision, "reranker": rr.name, "reranker_revision": rr.revision,
                                      "reranker_scale": "sigmoid(logit) once, [0,1]", "reranker_max_length": 2048})
    # reranker input truncation check (token budget is RERANKER_MAX_LENGTH = 2048): longest title+text chunk plus a
    # generous 128-token query must fit, otherwise the strict 1000-char bound is not enough for this tokenizer
    tok = rr.model.tokenizer
    n_tok = [len(tok(f"{c['title']}\n{c['text']}", add_special_tokens=True)["input_ids"]) for c in chunks]
    meta_path = Path(a.out) / "index_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["reranker_tokens_max_chunk"] = int(max(n_tok)); meta["reranker_tokens_p99_chunk"] = int(sorted(n_tok)[int(0.99 * len(n_tok))])
    meta["reranker_truncation_risk"] = bool(max(n_tok) + 128 > 2048)
    meta_path.write_text(json.dumps(meta, indent=2))
    ChunkIndex.load(Path(a.out))                            # round-trip consistency check
    print(json.dumps(json.loads((Path(a.out) / "index_meta.json").read_text()), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
