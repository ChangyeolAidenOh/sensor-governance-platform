"""0-C spike: pymupdf vs docling on the KOSHA GUIDE P-series corpus.

Run docling first (isolated venv, CLI), then this script in the main venv:

    .venv-docling/bin/docling convert data/raw/kosha/files --to md --to json \\
        --no-ocr --tables --table-mode accurate --output data/processed/kosha_spike/docling
    python scripts/spike_pdf_parsers.py --files data/raw/kosha/files \\
        --docling-dir data/processed/kosha_spike/docling --report docs/pdf_parser_spike.md

KOSHA GUIDE PDFs carry no table of contents (verified 2026-10-18), so clause
ground truth is (a) a manual list per document in data/raw/kosha/clause_gt/<guide>.txt
(one clause id per line, e.g. "4.1.2"; '#' comments allowed) where present, and
(b) the union of both parsers' detected clause headings otherwise.

Metrics per document and parser:
  clause_recall_gt     share of manual ground-truth clause ids found as headings (if gt file)
  clause_cov_union     share of the union of both parsers' clause ids found by this parser
  clause_ids           detected clause ids (numeric heading lines; list markers excluded)
  nospace_line_rate    share of long lines (>=30 chars) with no inner spaces — Korean text
                       exported without word spacing is unusable for BM25 / rerankers
  tables               detected tables; broken = ragged rows, >50% empty cells, or <2 rows
  scanned_pages        pages (cover excluded) with <30 chars (would need OCR)
docling structure is read from its JSON texts[] (using `orig`, or marker + text, since a
numbered heading that docling labels list_item keeps its number only in `marker`), not from
the markdown, which renumbers list items as "N." and repeats running headers as headings.
Repeated headings (>=3 identical) are dropped as page headers.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pymupdf

# ---- heading / toc patterns -------------------------------------------------------
# "4.1 반응기의 설계"  "4 . 1 반응기"  "## 4.1 반응기" (docling md)  "1. 목 적"
HEADING_RE = re.compile(r"^\s*(?:#+\s*)?(\d{1,2}(?:\s*\.\s*\d{1,2}){0,3})\s*\.?\s+(\S.*)$")
# TOC line: id, title, dotted leader / spaces, page number at end
TOC_LINE_RE = re.compile(r"^\s*(\d{1,2}(?:\s*\.\s*\d{1,2}){0,3})\s*\.?\s+(.+?)\s*[.·…‧ ]{2,}\s*(\d{1,3})\s*$")
TOC_TITLE_WORDS = ("목차", "목 차", "차례", "차 례")
MD_STRIP_RE = re.compile(r"[*_`]+")


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def norm_id(s: str) -> str:
    return re.sub(r"\s*\.\s*", ".", s.strip()).strip(".")


# ---- ground truth: toc ------------------------------------------------------------

def toc_from_bookmarks(doc) -> list[dict]:
    out = []
    for level, title, page in doc.get_toc(simple=True):
        m = HEADING_RE.match(title)
        if m:
            out.append({"id": norm_id(m.group(1)), "title": m.group(2).strip(), "level": level, "page": page})
    return out


def toc_from_text(doc, max_pages: int = 6) -> list[dict]:
    """TOC entries from the first pages: a page is a TOC page if it carries a TOC title word
    or at least 3 lines of the form 'id title ... page'."""
    entries, started = [], False
    for pno in range(min(max_pages, len(doc))):
        text = doc[pno].get_text("text", sort=True)
        n_toc_lines = sum(1 for l in text.splitlines() if TOC_LINE_RE.match(l))
        if not started and (any(w in norm(text) for w in [norm(w) for w in TOC_TITLE_WORDS]) or n_toc_lines >= 3):
            started = True
        if not started:
            continue
        for line in text.splitlines():
            m = TOC_LINE_RE.match(line)
            if m:
                cid = norm_id(m.group(1))
                entries.append({"id": cid, "title": m.group(2).strip(), "level": cid.count(".") + 1, "page": int(m.group(3))})
        if started and entries and pno >= 1 and not any(TOC_LINE_RE.match(l) for l in text.splitlines()):
            break
    # de-duplicate by id keeping first
    seen, uniq = set(), []
    for e in entries:
        if e["id"] not in seen:
            seen.add(e["id"]); uniq.append(e)
    return uniq


# ---- parser outputs ---------------------------------------------------------------

def pymupdf_extract(pdf: Path) -> dict:
    t0 = time.time()
    doc = pymupdf.open(pdf)
    pages_text, chars = [], []
    tables = []
    for page in doc:
        txt = page.get_text("text", sort=True)
        pages_text.append(txt)
        chars.append(len(txt.strip()))
        try:
            tf = page.find_tables()
            for t in tf.tables:
                rows = t.extract()
                tables.append(table_stats(rows))
        except Exception as exc:  # find_tables can raise on odd pages
            tables.append({"error": str(exc)[:80]})
    bm = toc_from_bookmarks(doc)
    toc = bm if bm else toc_from_text(doc)
    text = "\n".join(pages_text)
    doc.close()
    return {"text": text, "chars_per_page": chars, "n_pages": len(chars), "tables": tables,
            "toc": toc, "toc_source": "bookmarks" if bm else "toc_page", "seconds": round(time.time() - t0, 2)}


def table_stats(rows) -> dict:
    """rows: list of row lists (cells may be None/str)."""
    if not rows:
        return {"rows": 0, "cols": 0, "empty_ratio": 1.0, "ragged": False, "broken": True}
    n_rows = len(rows)
    lens = [len(r) for r in rows]
    n_cols = max(lens)
    cells = [c for r in rows for c in r]
    empty = sum(1 for c in cells if c is None or str(c).strip() == "")
    ragged = len(set(lens)) > 1
    empty_ratio = empty / max(1, len(cells))
    broken = ragged or empty_ratio > 0.5 or n_rows < 2
    return {"rows": n_rows, "cols": n_cols, "empty_ratio": round(empty_ratio, 2), "ragged": ragged, "broken": broken}


def docling_load(docling_dir: Path, stem: str) -> dict | None:
    md_path = docling_dir / f"{stem}.md"
    js_path = docling_dir / f"{stem}.json"
    if not md_path.exists():
        return None
    md = md_path.read_text(encoding="utf-8")
    tables, table_note = [], ""
    labels: dict[str, int] = {}
    header_lines: list[str] = []
    item_lines: list[str] = []
    struct_note = ""
    if js_path.exists():
        js = json.loads(js_path.read_text(encoding="utf-8"))
        texts = js.get("texts") if isinstance(js, dict) else None
        if isinstance(texts, list):
            for t in texts:
                if not isinstance(t, dict):
                    continue
                lab = str(t.get("label", "?"))
                labels[lab] = labels.get(lab, 0) + 1
                # list items keep the number in `marker` (text has it stripped); `orig` holds the raw line
                txt = str(t.get("orig") or ((str(t.get("marker", "")) + " " if t.get("marker") else "") + str(t.get("text", ""))))
                item_lines.append(txt)
                if lab == "section_header":
                    header_lines.append(txt)
        else:
            struct_note = f"no texts[] in json; top-level keys {list(js)[:12] if isinstance(js, dict) else type(js).__name__}"
        tlist = js.get("tables") if isinstance(js, dict) else None
        if isinstance(tlist, list):
            for t in tlist:
                data = t.get("data", {}) if isinstance(t, dict) else {}
                grid = data.get("grid")
                if isinstance(grid, list) and grid:
                    rows = [[(c.get("text") if isinstance(c, dict) else c) for c in row] for row in grid]
                    tables.append(table_stats(rows))
                else:
                    tables.append({"rows": data.get("num_rows", 0), "cols": data.get("num_cols", 0),
                                   "empty_ratio": None, "ragged": None, "broken": None,
                                   "note": f"no grid; keys={list(data)[:6]}"})
        else:
            table_note = f"json top-level keys: {list(js)[:12] if isinstance(js, dict) else type(js).__name__}"
    else:
        table_note = "no json"
    # items_text = every text item one per line (raw text, no markdown renumbering) -> clause detection
    # headers_text = section_header items only -> structure-label quality
    items_text = "\n".join(item_lines) if item_lines else MD_STRIP_RE.sub("", md)
    headers_text = "\n".join(header_lines)
    return {"text": MD_STRIP_RE.sub("", md), "items_text": items_text, "headers_text": headers_text,
            "labels": labels, "struct_note": struct_note, "tables": tables, "table_note": table_note}


# ---- metrics ----------------------------------------------------------------------

LIST_MARKER_RE = re.compile(r"^[(\[]?[가-힣a-zA-Z0-9①-⑳㉠-㉿]{1,3}[)\]]|^[①-⑳㉠-㉿·•\-]")


def is_clause_heading(line: str):
    """Return (id, title) if the line looks like a numbered clause heading, else None."""
    m = HEADING_RE.match(line)
    if not m:
        return None
    title = m.group(2).strip()
    if LIST_MARKER_RE.match(title):          # "2. (가) ..." / "3. ① ..." = renumbered list items
        return None
    if len(title) > 45 or title.endswith(("다.", "다", "한다.", "것")):   # sentences, not headings
        return None
    if re.search(r"[.·…‧ ]{3,}\d{1,3}\s*$", line):   # toc-like dotted leader
        return None
    return norm_id(m.group(1)), title


def body_headings(text: str, drop_repeated: int = 3) -> dict[str, str]:
    found: dict[str, str] = {}
    counts: dict[str, int] = {}
    for line in text.splitlines():
        h = is_clause_heading(line)
        if not h:
            continue
        cid, title = h
        key = cid + "|" + norm(title)
        counts[key] = counts.get(key, 0) + 1
        found.setdefault(cid, title)
    # running headers repeat on every page; a clause id seen >= drop_repeated times with the same title is one
    for cid, title in list(found.items()):
        if counts.get(cid + "|" + norm(title), 0) >= drop_repeated:
            found.pop(cid)
    return found


def space_per_hangul(text: str) -> float | None:
    """Spaces per Hangul character. Normal Korean prose is roughly 0.25-0.4; text exported
    without word spacing falls to a few hundredths."""
    hangul = len(re.findall(r"[가-힣]", text))
    if hangul < 200:
        return None
    spaces = sum(1 for line in text.splitlines() for ch in line.strip() if ch == " ")
    return round(spaces / hangul, 3)


def load_gt(gt_dir: Path, stem: str) -> list[str] | None:
    p = gt_dir / f"{stem}.txt"
    if not p.exists():
        return None
    ids = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.split("#")[0].strip()
        if ln:
            ids.append(norm_id(ln))
    return ids or None


def clause_recall(toc: list[dict], text: str) -> tuple[float | None, list[str]]:
    if not toc:
        return None, []
    heads = body_headings(text)
    missing = []
    hit = 0
    for e in toc:
        title_key = norm(e["title"])[:3]
        h = heads.get(e["id"])
        if h is not None and (not title_key or norm(h).startswith(title_key) or title_key in norm(h)):
            hit += 1
        else:
            missing.append(f"{e['id']} {e['title']}")
    return hit / len(toc), missing


def clause_tree_stats(text: str) -> dict:
    ids = list(body_headings(text))
    depth = {}
    for cid in ids:
        d = cid.count(".") + 1
        depth[d] = depth.get(d, 0) + 1
    # leaf clause lengths: split text at heading lines, measure chars between consecutive headings
    positions = [m.start() for m in re.finditer(r"^\s*(?:#+\s*)?\d{1,2}(?:\s*\.\s*\d{1,2}){0,3}\s*\.?\s+\S", text, re.M)]
    lengths = [b - a for a, b in zip(positions, positions[1:] + [len(text)])]
    if lengths:
        lengths.sort()
        q = lambda p: lengths[min(len(lengths) - 1, int(p * len(lengths)))]
        stats = {"n_segments": len(lengths), "chars_p50": q(0.5), "chars_p90": q(0.9), "chars_max": lengths[-1]}
    else:
        stats = {"n_segments": 0}
    return {"headings_by_depth": dict(sorted(depth.items())), **stats}


# ---- main -------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default="data/raw/kosha/files")
    ap.add_argument("--docling-dir", default="data/processed/kosha_spike/docling")
    ap.add_argument("--report", default="docs/pdf_parser_spike.md")
    ap.add_argument("--json", default="outputs/pdf_parser_spike.json")
    ap.add_argument("--excerpt-dir", default="data/processed/kosha_spike/compare")
    ap.add_argument("--excerpts", nargs="*", default=["P-116-2012", "P-133-2013", "P-65-2012"],
                    help="guide numbers for side-by-side excerpts (manual check)")
    ap.add_argument("--excerpt-chars", type=int, default=6000)
    ap.add_argument("--gt-dir", default="data/raw/kosha/clause_gt")
    args = ap.parse_args()
    gt_dir = Path(args.gt_dir)

    files = sorted(Path(args.files).glob("*.pdf"))
    docling_dir = Path(args.docling_dir)
    rows = []
    for pdf in files:
        stem = pdf.stem
        pm = pymupdf_extract(pdf)
        dl = docling_load(docling_dir, stem)
        rec = {"guide_no": stem, "n_pages": pm["n_pages"],
               "min_chars_page": min(pm["chars_per_page"]) if pm["chars_per_page"] else 0,
               "scanned_pages": sum(1 for c in pm["chars_per_page"][1:] if c < 30)}
        gt = load_gt(gt_dir, stem)
        pm_ids = body_headings(pm["text"])
        dl_ids = body_headings(dl["items_text"]) if dl else {}
        dl_hdr_ids = body_headings(dl["headers_text"]) if dl else {}
        union = set(pm_ids) | set(dl_ids)

        def score(ids: dict[str, str]) -> dict:
            out = {"clause_ids": sorted(ids, key=lambda x: [int(p) for p in x.split(".")]),
                   "n_clauses": len(ids),
                   "clause_cov_union": round(len(set(ids) & union) / len(union), 3) if union else None,
                   "clause_recall_gt": None, "gt_missing": [], "gt_extra": []}
            if gt:
                out["clause_recall_gt"] = round(sum(1 for g in gt if g in ids) / len(gt), 3)
                out["gt_missing"] = [g for g in gt if g not in ids]
                out["gt_extra"] = sorted(set(ids) - set(gt), key=lambda x: [int(p) for p in x.split(".")])
            return out

        rec["gt_n"] = len(gt) if gt else 0
        rec["pymupdf"] = {**score(pm_ids), "space_per_hangul": space_per_hangul(pm["text"]),
                          "tables": len(pm["tables"]), "tables_broken": sum(1 for t in pm["tables"] if t.get("broken")),
                          "seconds": pm["seconds"], "tree": clause_tree_stats(pm["text"])}
        if dl is None:
            rec["docling"] = None
        else:
            rec["docling"] = {**score(dl_ids), "space_per_hangul": space_per_hangul(dl["text"]),
                              "label_recall": round(len(set(dl_hdr_ids) & set(dl_ids)) / len(dl_ids), 3) if dl_ids else None,
                              "labels": dl["labels"], "struct_note": dl["struct_note"],
                              "tables": len(dl["tables"]), "tables_broken": sum(1 for t in dl["tables"] if t.get("broken")),
                              "tables_unscored": sum(1 for t in dl["tables"] if t.get("broken") is None),
                              "table_note": dl["table_note"], "tree": clause_tree_stats(dl["items_text"])}
        rows.append(rec)
        pmr, dlr = rec["pymupdf"], rec["docling"]
        print(f"{stem:<12} p{rec['n_pages']:>3} gt {rec['gt_n']:>2} | pymupdf clauses {pmr['n_clauses']:>3} cov {fmt(pmr['clause_cov_union'])} gt {fmt(pmr['clause_recall_gt'])} "
              f"sp/ha {fmt(pmr['space_per_hangul'])} tables {pmr['tables']}/{pmr['tables_broken']} | "
              + (f"docling clauses {dlr['n_clauses']:>3} cov {fmt(dlr['clause_cov_union'])} gt {fmt(dlr['clause_recall_gt'])} "
                 f"sp/ha {fmt(dlr['space_per_hangul'])} label {fmt(dlr['label_recall'])} tables {dlr['tables']}/{dlr['tables_broken']}" if dlr else "docling: missing"))

        if stem in args.excerpts:
            out = Path(args.excerpt_dir); out.mkdir(parents=True, exist_ok=True)
            parts = [f"# {stem} side-by-side (first {args.excerpt_chars} chars)\n",
                     "## clause ids detected — pymupdf\n", ", ".join(f"{k} {v}" for k, v in pm_ids.items()) or "(none)",
                     "\n## clause ids detected — docling (all text items)\n", ", ".join(f"{k} {v}" for k, v in dl_ids.items()) or "(none)",
                     "\n## of which labeled section_header\n", ", ".join(dl_hdr_ids) or "(none)",
                     "\n## pymupdf text\n```\n" + pm["text"][:args.excerpt_chars] + "\n```\n"]
            if dl:
                parts.append("## docling section_header items\n```\n" + dl["headers_text"][:args.excerpt_chars] + "\n```\n")
                parts.append("## docling markdown\n```\n" + dl["text"][:args.excerpt_chars] + "\n```\n")
            (out / f"{stem}.md").write_text("\n".join(parts), encoding="utf-8")

    # ---- aggregate --------------------------------------------------------------
    def agg(parser: str):
        rs = [r[parser]["clause_cov_union"] for r in rows if r.get(parser) and r[parser]["clause_cov_union"] is not None]
        gts = [r[parser]["clause_recall_gt"] for r in rows if r.get(parser) and r[parser]["clause_recall_gt"] is not None]
        ns = [r[parser]["space_per_hangul"] for r in rows if r.get(parser) and r[parser]["space_per_hangul"] is not None]
        lr = [r[parser].get("label_recall") for r in rows if r.get(parser) and r[parser].get("label_recall") is not None]
        tb = sum(r[parser]["tables"] for r in rows if r.get(parser))
        bk = sum(r[parser]["tables_broken"] for r in rows if r.get(parser))
        sec = sum(r[parser].get("seconds", 0) or 0 for r in rows if r.get(parser))
        return {"docs": sum(1 for r in rows if r.get(parser)),
                "clause_cov_union_mean": round(sum(rs) / len(rs), 3) if rs else None,
                "docs_cov_ge_0.8": sum(1 for x in rs if x >= 0.8),
                "gt_docs": len(gts), "clause_recall_gt_mean": round(sum(gts) / len(gts), 3) if gts else None,
                "gt_docs_ge_0.8": sum(1 for x in gts if x >= 0.8),
                "space_per_hangul_mean": round(sum(ns) / len(ns), 3) if ns else None,
                "label_recall_mean": round(sum(lr) / len(lr), 3) if lr else None,
                "tables": tb, "tables_broken": bk, "table_broken_rate": round(bk / tb, 3) if tb else None,
                "seconds_total": round(sec, 1)}

    summary = {"n_docs": len(rows), "docs_with_gt": sum(1 for r in rows if r["gt_n"]),
               "docs_with_scanned_pages": sum(1 for r in rows if r["scanned_pages"]),
               "pymupdf": agg("pymupdf"), "docling": agg("docling") if any(r.get("docling") for r in rows) else None}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({"summary": summary, "docs": rows}, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- markdown report --------------------------------------------------------
    lines = ["# 0-C PDF parser spike: pymupdf vs docling (KOSHA GUIDE P-series)", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Docs: {summary['n_docs']}, with manual clause ground truth: {summary['docs_with_gt']}, "
             f"with scanned pages: {summary['docs_with_scanned_pages']}. KOSHA PDFs have no TOC page; clause coverage is measured against "
             "the union of both parsers' detected clause headings, and against the manual lists where present.", "",
             "## Summary", "",
             "| parser | docs | clause cov (union) | docs ≥ 0.80 | gt docs | clause recall (gt) | gt docs ≥ 0.80 | spaces per hangul | label recall | tables | broken | broken rate | seconds |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in ("pymupdf", "docling"):
        sm = summary[p]
        if sm:
            lines.append(f"| {p} | {sm['docs']} | {sm['clause_cov_union_mean']} | {sm['docs_cov_ge_0.8']} | {sm['gt_docs']} | {sm['clause_recall_gt_mean']} | {sm['gt_docs_ge_0.8']} | "
                         f"{sm['space_per_hangul_mean']} | {sm['label_recall_mean']} | {sm['tables']} | {sm['tables_broken']} | {sm['table_broken_rate']} | {sm['seconds_total']} |")
    lines += ["", "Go/No-Go (plan §4.1): clause preservation ≥ 80% on the manual ground truth. 'spaces per hangul' = word spacing preserved "
              "(normal Korean prose ~0.25-0.4; near 0 = spacing lost). docling clauses are detected from all JSON text items; 'label recall' = "
              "share of those also labeled section_header (structure quality). Table 'broken' is a proxy (ragged / >50% empty / <2 rows).", "",
              "## Per document", "",
              "| guide | pages | gt | pm clauses | pm cov | pm gt | pm sp/ha | pm tables (broken) | dl clauses | dl cov | dl gt | dl sp/ha | dl label | dl tables (broken) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        pm, dl = r["pymupdf"], r["docling"]
        lines.append(f"| {r['guide_no']} | {r['n_pages']} | {r['gt_n']} | {pm['n_clauses']} | {fmt(pm['clause_cov_union'])} | {fmt(pm['clause_recall_gt'])} | {fmt(pm['space_per_hangul'])} | "
                     f"{pm['tables']} ({pm['tables_broken']}) | " + (f"{dl['n_clauses']} | {fmt(dl['clause_cov_union'])} | {fmt(dl['clause_recall_gt'])} | {fmt(dl['space_per_hangul'])} | {fmt(dl['label_recall'])} | {dl['tables']} ({dl['tables_broken']}) |" if dl else "- | - | - | - | - | - |"))
    lines += ["", "## Ground-truth misses (docs with a manual list)", ""]
    for r in rows:
        if r["gt_n"]:
            lines.append(f"- **{r['guide_no']}** pymupdf missing {r['pymupdf']['gt_missing']} extra {r['pymupdf']['gt_extra']}"
                         + (f" | docling missing {r['docling']['gt_missing']} extra {r['docling']['gt_extra']}" if r["docling"] else ""))
    lines += ["", "## Clause ids per document (pymupdf | docling)", ""]
    for r in rows:
        lines.append(f"- {r['guide_no']}: {' '.join(r['pymupdf']['clause_ids'])} | {' '.join(r['docling']['clause_ids']) if r['docling'] else '-'}")
    lines += ["", "## docling item labels per document", ""]
    for r in rows:
        if r["docling"]:
            lines.append(f"- {r['guide_no']}: {r['docling']['labels']}" + (f" ({r['docling']['struct_note']})" if r["docling"]["struct_note"] else ""))
    lines += ["", "## Clause segment length (pymupdf text between headings)", ""]
    for r in rows:
        t = r["pymupdf"]["tree"]
        lines.append(f"- {r['guide_no']}: depth {t['headings_by_depth']}, segments {t.get('n_segments')}, p50 {t.get('chars_p50', '-')}, p90 {t.get('chars_p90', '-')}, max {t.get('chars_max', '-')} chars")
    notes = [r["docling"]["table_note"] for r in rows if r["docling"] and r["docling"]["table_note"]]
    if notes:
        lines += ["", "## docling table parsing notes", "", *[f"- {n}" for n in sorted(set(notes))]]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n[summary]", json.dumps(summary, ensure_ascii=False))
    print(f"[done] report {args.report}, json {args.json}, excerpts {args.excerpt_dir}")


def fmt(x):
    return "-" if x is None else f"{x:.2f}"


if __name__ == "__main__":
    main()
