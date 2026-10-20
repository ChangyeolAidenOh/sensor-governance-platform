"""
scripts/fix_tau.py — S3-04: fix tau on the DEVELOPMENT set only (prereg §4.3 / §5.3; evals/rag_dev_set.json, 10 items).
Runs the real scoped retriever (engineer role, k=5) with each item's pred_set, records g_AB, then scans the
pre-registered grid tau ∈ {0.2, 0.3, …, 0.8}:
  balanced accuracy = (TPR over 'answer' items with g_AB >= tau  +  TNR over 'trap' items with NOT(g_AB >= tau)) / 2
  ties → higher trap-NONE recall (TNR) → still tied → the LARGEST tau (more conservative grounding). The last rule was
  NOT in prereg-v1; it was specified in S3-04 (ST3-09) before the first run and is reported as such.
Writes models/retrieval_v1/tau.json (grid, per-item g_AB, chosen tau, dev-set sha256, index meta ref) and the
reranker pair-length check (query + title + text + special tokens) the reviewer asked for. Never touches the 50-item
evaluation set. hit@3 on answer items is reported as a dev diagnostic only.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from core_pipeline.retrieval.acl import load_roles
from core_pipeline.retrieval.index import ChunkIndex
from core_pipeline.retrieval.models import STEmbedder, STReranker, resolve_pins, RERANKER_MAX_LENGTH
from core_pipeline.retrieval.retrieve import Retriever
from core_pipeline.retrieval.scope import FaultScope

GRID = [round(0.2 + 0.1 * i, 1) for i in range(7)]          # 0.2 … 0.8 (prereg §4.3)


def choose_tau(g: dict[str, float | None], kinds: dict[str, str]) -> tuple[float, list[dict]]:
    rows = []
    for tau in GRID:
        found = {i: (g[i] is not None and g[i] >= tau) for i in g}
        ans = [i for i in g if kinds[i] == "answer"]; trap = [i for i in g if kinds[i] == "trap"]
        tpr = sum(found[i] for i in ans) / len(ans); tnr = sum(not found[i] for i in trap) / len(trap)
        rows.append({"tau": tau, "tpr": tpr, "tnr": tnr, "balanced_acc": (tpr + tnr) / 2})
    best = max(rows, key=lambda r: (r["balanced_acc"], r["tnr"], r["tau"]))   # ties: TNR, then largest tau
    return best["tau"], rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="evals/rag_dev_set.json")
    ap.add_argument("--index", default="models/retrieval_v1")
    ap.add_argument("--fault-map", default="outputs/fault_to_guidance_map.json")
    ap.add_argument("--out", default="models/retrieval_v1/tau.json")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = json.loads(Path(a.dev).read_text(encoding="utf-8"))
    items = dev["items"]; assert len(items) == 10 and sum(i["kind"] == "answer" for i in items) == 5
    pins = resolve_pins(); emb = STEmbedder(device=a.device, pins=pins); rr = STReranker(device=a.device, pins=pins)
    idx = ChunkIndex.load(Path(a.index))
    scope = FaultScope(Path(a.fault_map)); roles = load_roles(Path(a.fault_map), {c["guide_no"] for c in idx.chunks})
    retr = Retriever(idx, emb, rr, scope, roles, tau=None)
    tok = rr.model.tokenizer
    g, kinds, per_item, pair_tokens = {}, {}, [], []
    for it in items:
        r = retr(it["question"], it["pred_set"], "engineer", mode="scoped", k=5)
        g[it["id"]] = r.g_AB; kinds[it["id"]] = it["kind"]
        top3 = [c.clause_key for c in r.chunks[:3]]
        hit3 = (any(k in top3 for k in it["answer_keys"]) if it["kind"] == "answer" else None)
        for t in retr.last_rerank_texts:                       # EVERY candidate pair the reranker scored, not only the returned top-k
            pair_tokens.append(len(tok(it["question"], t, add_special_tokens=True)["input_ids"]))
        per_item.append({"id": it["id"], "kind": it["kind"], "pred_set": it["pred_set"], "status": r.status, "grounding": r.grounding,
                         "g_AB": r.g_AB, "g_C": r.g_C, "scope_size": r.scope_size, "hit@3": hit3,
                         "top": [{"key": c.clause_key, "grade": c.grade, "score": round(c.score, 4)} for c in r.chunks]})
    tau, grid = choose_tau(g, kinds)
    im = json.loads((Path(a.index) / "index_meta.json").read_text())
    out = {"tau": tau, "grid": grid,
           "rule": "balanced accuracy; ties → TNR → largest tau (largest-tau tie rule pre-specified in S3-04/ST3-09, not in prereg-v1)",
           "contract": {"mode": "scoped", "role": "engineer", "k": 5, "score_scale": "sigmoid(logit) once, [0,1]"},
           "dev_set": a.dev, "dev_set_sha256": hashlib.sha256(Path(a.dev).read_bytes()).hexdigest(),
           "fault_map": a.fault_map, "fault_map_sha256": hashlib.sha256(Path(a.fault_map).read_bytes()).hexdigest(),
           "reranker": rr.name, "reranker_revision": rr.revision, "embedder": emb.name, "embedder_revision": emb.revision,
           "index_meta": im,
           "reranker_pair_tokens_max": int(max(pair_tokens)), "reranker_pair_tokens_n": len(pair_tokens),
           "reranker_pair_tokens_over_limit": int(sum(t > RERANKER_MAX_LENGTH for t in pair_tokens)),
           "pair_tokens_scope": "all candidate pairs scored by the reranker (RRF-fused BM25∪dense), not only the returned top-k",
           "hit@3_answer_items": sum(1 for p in per_item if p["hit@3"]) , "per_item": per_item,
           "note": "development diagnostic; H3 is evaluated on the 50-item set in Stage 5"}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_item", "index_meta")}, ensure_ascii=False, indent=1))
    print("per item:", json.dumps([{k: p[k] for k in ("id", "kind", "grounding", "g_AB", "hit@3")} for p in per_item], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
