"""
scripts/build_clause_corpus.py — S3-01: parse every docling JSON in data/processed/kosha/docling/ into clauses,
VERIFY against the Stage 0-C index (all 699 rows: keys, level, page, n_items, n_chars must match → otherwise abort),
then write outputs/kosha_clauses.jsonl (full text) and outputs/kosha_chunks.jsonl (<=1000-char chunks, '#k' keys).
Usage: python scripts/build_clause_corpus.py [--docling data/processed/kosha/docling] [--index outputs/kosha_clause_index.json]
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_pipeline.docs.clause_parser import parse_file, index_rows, compat_rows, structural_diff
from core_pipeline.docs.chunker import chunk_clause, MAX_CHARS


def dump_diagnostics(files, idx_rows, clauses, guides, out_dir: Path) -> None:
    """For each differing guide: 0-C rows vs parser rows side by side, then every docling text that looks like a
    heading candidate (section_header of any text, or any text starting with a number pattern) with its label."""
    import re
    out_dir.mkdir(parents=True, exist_ok=True)
    num = re.compile(r"^\s*\d")
    for g in guides:
        lines = [f"# {g}", "", "## 0-C rows", *[f"{r['clause_key']:<28} L{r['level']} p{r['page']} items={r['n_items']} chars={r['n_chars']} | {r['title']}" for r in idx_rows if r["guide_no"] == g],
                 "", "## parser rows", *[f"{c.source_key:<28} L{c.level} p{c.page} items={len(c.items)} chars={c.n_chars} | {c.title}" for c in clauses if c.guide_no == g],
                 "", "## docling heading candidates (idx label page | text)"]
        f = next((f for f in files if f.stem == g), None)
        if f:
            doc = json.loads(f.read_text(encoding="utf-8"))
            for i, t in enumerate(doc.get("texts", [])):
                txt = (t.get("text") or "").strip(); orig = (t.get("orig") or txt).strip(); lab = t.get("label")
                if lab in ("section_header", "page_header", "page_footer", "title") or num.match(orig):
                    pg = t["prov"][0]["page_no"] if t.get("prov") else "?"
                    mk = t.get("marker", ""); en = t.get("enumerated", "")
                    lines.append(f"{i:4d} {lab:<15} p{pg} mk={mk!r:<8} en={en!s:<5} len_text={len(txt):3d} len_orig={len(orig):3d} | text={txt[:60]!r} | orig={orig[:60]!r}")
        (out_dir / f"{g}.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docling", default="data/processed/kosha/docling")
    ap.add_argument("--index", default="outputs/kosha_clause_index.json")
    ap.add_argument("--out-clauses", default="outputs/kosha_clauses.jsonl")
    ap.add_argument("--out-chunks", default="outputs/kosha_chunks.jsonl")
    ap.add_argument("--diag-dir", default="outputs/corpus_diag", help="where to dump side-by-side diagnostics for guides that differ")
    ap.add_argument("--known-diffs", default="outputs/corpus_known_diffs.json",
                    help="reviewed exceptions: {\"extra_clauses\": {\"guide:part:id\": \"reason\"}} — clauses the 0-C index missed")
    a = ap.parse_args()
    files = sorted(Path(a.docling).glob("*.json"))
    idx_rows = json.loads(Path(a.index).read_text(encoding="utf-8"))
    clauses = [c for f in files for c in parse_file(f)]
    known = json.loads(Path(a.known_diffs).read_text(encoding="utf-8")) if Path(a.known_diffs).exists() else {"extra_clauses": {}}
    # known diffs apply narrowly: the source file hash recorded for each exception must match the file on disk
    file_sha = {f.stem: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    known_extra = set()
    for k, v in known.get("extra_clauses", {}).items():
        g = k.split(":")[0]
        if v.get("source_file_sha256") not in (None, file_sha.get(g)):
            print(f"abort: known diff {k} was recorded for a different {g}.json (sha mismatch)"); return 2
        known_extra.add(k)
    known_deltas = {}
    for k, v in known.get("item_deltas", {}).items():
        g = k.split(":")[0]
        if v.get("source_file_sha256") not in (None, file_sha.get(g)):
            print(f"abort: known item delta {k} was recorded for a different {g}.json (sha mismatch)"); return 2
        if v.get("mapping_referenced", False):
            print(f"abort: known item delta {k} is on a mapping-referenced clause — not allowed"); return 2
        known_deltas[k] = (int(v["d_items"]), int(v["d_chars"]))
    crows, removed_for_compat = compat_rows(clauses)
    problems = structural_diff(idx_rows, crows, known_extra, known_deltas)   # row lists, duplicates preserved, titles compared
    json_versions = sorted({json.loads(f.read_text(encoding="utf-8")).get("version", "?") for f in files})
    print(f"version field recorded in the input docling JSON: {json_versions}  (not the package version that produced them, and 0-C input identity is unverified)")
    print(f"compat-only removals (kept in corpus): {len(removed_for_compat)} section_header items repeated >=3x")
    print(f"files={len(files)} clauses={len(clauses)} index_rows={len(idx_rows)} structural_problems={len(problems)}")
    for pr in problems[:20]: print("  ", pr)
    if problems or len(clauses) - len(known_extra) != len(idx_rows):
        bad_guides = sorted({pr.split(":")[0].split(" row")[0] for pr in problems})
        dump_diagnostics(files, idx_rows, clauses, bad_guides, Path(a.diag_dir))
        print(f"abort: not structurally equivalent to the 0-C index; diagnostics for {bad_guides} in {a.diag_dir}/"); return 2
    keys = [c.clause_key for c in clauses]
    dups = sorted({k for k in keys if keys.count(k) > 1})
    if dups:
        print("abort: duplicate clause keys after disambiguation:", dups); return 2
    chunks = [ch.to_dict() for c in clauses for ch in chunk_clause(c)]
    ckeys = [c["chunk_key"] for c in chunks]
    if len(ckeys) != len(set(ckeys)):
        print("abort: duplicate chunk keys"); return 2
    assert all(len(c["text"]) <= MAX_CHARS for c in chunks)
    disamb = [(c.clause_key, c.title) for c in clauses if c.n_occurrences > 1]
    print("disambiguated keys:", disamb)
    Path(a.out_clauses).write_text("\n".join(json.dumps({"clause_key": c.clause_key, "guide_no": c.guide_no, "part": c.part,
                                                           "clause_id": c.clause_id, "title": c.title, "level": c.level,
                                                           "page": c.page, "n_items": len(c.items), "n_chars": c.n_chars,
                                                           "text": c.text()}, ensure_ascii=False) for c in clauses) + "\n", encoding="utf-8")
    Path(a.out_chunks).write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n", encoding="utf-8")
    n_split = sum(1 for c in clauses if len(chunk_clause(c)) > 1)
    sha = hashlib.sha256(Path(a.out_chunks).read_bytes()).hexdigest()
    print(json.dumps({"clauses": len(clauses), "empty_clauses": sum(1 for c in clauses if not c.items), "chunks": len(chunks),
                      "clauses_split": n_split, "max_chars": MAX_CHARS, "chunks_sha256": sha, "max_chunk_chars": max(len(c["text"]) for c in chunks),
                      "disambiguated": disamb, "equivalence": "index STRUCTURE (keys/level/page/n_items/n_chars/title) equals 0-C except the reviewed extra clauses; not full-text equality",
                      "known_extra_clauses": known.get("extra_clauses", {}), "known_item_deltas": known.get("item_deltas", {}),
                      "input_json_version_field": json_versions,
                      "compat_only_removed_items": removed_for_compat, "source_file_sha256": file_sha,
                      "guides": sorted({c.guide_no for c in clauses})}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
