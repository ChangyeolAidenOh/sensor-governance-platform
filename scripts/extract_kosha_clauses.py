"""Clause index of the KOSHA GUIDE corpus from docling JSON (0-G input, Stage 3 precursor).

Rule (decision ST0-15): a clause heading is a text item whose raw line (`orig`, or
marker + text) starts with a numeric id like "4.1.2" followed by a short title; list
markers ((가), ①, ...) are not clauses; headings repeated >= 3 times are page headers.
Everything between one heading and the next belongs to that clause.

Output:
    outputs/kosha_clause_index.json   [{guide_no, name, clause_id, title, level, n_items,
                                       n_chars, preview, page}]
    docs/kosha_clause_index.md        human-readable listing per guide (local doc)

Usage:  python scripts/extract_kosha_clauses.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

DOCLING_DIR = Path("data/processed/kosha/docling")
INDEX_CSV = Path("data/raw/kosha/index_P.csv")
OUT_JSON = Path("outputs/kosha_clause_index.json")
OUT_MD = Path("docs/kosha_clause_index.md")

HEADING_RE = re.compile(r"^\s*(\d{1,2}(?:\s*\.\s*\d{1,2}){0,3})\s*\.?\s+(\S.*)$")
LIST_MARKER_RE = re.compile(r"^[(\[]?[가-힣a-zA-Z0-9①-⑳㉠-㉿]{1,3}[)\]]|^[①-⑳㉠-㉿·•\-]")


def norm_id(s: str) -> str:
    return re.sub(r"\s*\.\s*", ".", s.strip()).strip(".")


def clause_heading(line: str):
    m = HEADING_RE.match(line)
    if not m:
        return None
    title = m.group(2).strip()
    if LIST_MARKER_RE.match(title) or len(title) > 45 or title.endswith(("다.", "다", "한다.", "것")):
        return None
    return norm_id(m.group(1)), title


def item_line(t: dict) -> str:
    return str(t.get("orig") or ((str(t.get("marker", "")) + " ") if t.get("marker") else "") + str(t.get("text", "")))


def item_page(t: dict):
    prov = t.get("prov") or []
    if prov and isinstance(prov, list) and isinstance(prov[0], dict):
        return prov[0].get("page_no")
    return None


def build(guide_no: str, name: str, js: dict) -> list[dict]:
    texts = [t for t in js.get("texts", []) if isinstance(t, dict)]
    # drop running headers: identical heading lines repeated >= 3 times
    counts: dict[str, int] = {}
    header_counts: dict[str, int] = {}
    for t in texts:
        h = clause_heading(item_line(t))
        if h:
            key = h[0] + "|" + re.sub(r"\s+", "", h[1])
            counts[key] = counts.get(key, 0) + 1
        if t.get("label") == "section_header":
            k2 = re.sub(r"\s+", "", str(t.get("text", "")))
            header_counts[k2] = header_counts.get(k2, 0) + 1
    clauses: list[dict] = []
    current = None
    part, max_top = 0, 0          # numbering restarts (appendices, case studies) open a new part
    for t in texts:
        line = item_line(t)
        # running page headers (same section_header text on >= 3 pages) belong to no clause
        if t.get("label") == "section_header" and header_counts.get(re.sub(r"\s+", "", str(t.get("text", ""))), 0) >= 3:
            continue
        h = clause_heading(line)
        if h and counts.get(h[0] + "|" + re.sub(r"\s+", "", h[1]), 0) < 3:
            top = int(h[0].split(".")[0])
            if "." not in h[0]:
                if top < max_top:           # top-level id went backwards -> appendix / case-study block
                    part += 1
                    max_top = 0
                max_top = max(max_top, top)
            current = {"guide_no": guide_no, "name": name, "part": part, "clause_id": h[0], "title": h[1],
                       "clause_key": f"{guide_no}:{part}:{h[0]}",
                       "level": h[0].count(".") + 1, "page": item_page(t), "n_items": 0, "n_chars": 0, "preview": ""}
            clauses.append(current)
            continue
        if current is not None and t.get("label") not in ("page_header", "page_footer"):
            body = str(t.get("text", "")).strip()
            if body:
                current["n_items"] += 1
                current["n_chars"] += len(body)
                if len(current["preview"]) < 160:
                    current["preview"] = (current["preview"] + " " + body).strip()[:160]
    return clauses


def main() -> None:
    names = {}
    if INDEX_CSV.exists():
        names = {r["guide_no"]: r["name"] for r in csv.DictReader(INDEX_CSV.open(encoding="utf-8"))}
    all_clauses: list[dict] = []
    md = ["# KOSHA GUIDE clause index (from docling JSON, rule ST0-15)", ""]
    for js_path in sorted(DOCLING_DIR.glob("*.json")):
        guide_no = js_path.stem
        js = json.loads(js_path.read_text(encoding="utf-8"))
        clauses = build(guide_no, names.get(guide_no, ""), js)
        all_clauses.extend(clauses)
        md.append(f"## {guide_no} {names.get(guide_no, '')}  ({len(clauses)} clauses)")
        md.append("")
        for c in clauses:
            indent = "  " * (c["level"] - 1)
            tag = f"[part {c['part']}] " if c["part"] else ""
            md.append(f"{indent}- {tag}**{c['clause_id']}** {c['title']} (p.{c['page']}, {c['n_chars']} chars) — {c['preview'][:90]}")
        md.append("")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(all_clauses, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    by_guide = {}
    for c in all_clauses:
        by_guide[c["guide_no"]] = by_guide.get(c["guide_no"], 0) + 1
    n_parts = sum(1 for c in all_clauses if c["part"] and c["level"] == 1 and c["clause_id"] == "1")
    print(f"[done] {len(all_clauses)} clauses over {len(by_guide)} guides -> {OUT_JSON}, {OUT_MD}; appendix/restart blocks: {n_parts}")
    print("per guide:", by_guide)
    chars = sorted(c["n_chars"] for c in all_clauses if c["n_chars"] > 0)
    if chars:
        q = lambda p: chars[min(len(chars) - 1, int(p * len(chars)))]
        print(f"clause body chars: p50 {q(0.5)}, p90 {q(0.9)}, max {chars[-1]}, empty clauses {sum(1 for c in all_clauses if c['n_chars'] == 0)}")


if __name__ == "__main__":
    main()
