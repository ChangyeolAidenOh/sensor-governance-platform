r"""
core_pipeline/docs/clause_parser.py — S3-01: docling JSON → KOSHA GUIDE clauses, reproducing the Stage 0-C index
(outputs/kosha_clause_index.json) exactly: same clause keys `guide:part:id`, same n_items / n_chars.

Conventions (reverse-engineered from the 0-C index and verified on P-53-2012, 18/18 rows identical):
  * body order = `texts` order; `page_footer` and `page_header` labelled items are dropped. Running headers that
    docling labelled as plain `text` ("KOSHA GUIDE", "P - 53 - 2012" in P-53) are NOT dropped — 0-C kept them — and are
    stripped only at chunking time (RUNNING_HEADER_RE)
  * a clause header is recognised on the ORIGINAL text (`orig`, which keeps the list marker docling strips from
    `text`: "6.3 습도 조절…" → text "습도 조절…", marker "6.3") of a `section_header`, `list_item` OR plain `text` item
    (0-C accepted all three: P-159 "2.2 저온 고압 산소 벤트" and P-178 "5.3.6 …" are `text`). The grammar does the
    filtering: a dotted multi-level number ("6.3", "5.2.1", depth ≤ 4; "5.5." with an attached dot allowed), a single number
    followed by a dot ("1.", "10.") or a single number followed by a space ("10 습식 테스트기 교정", P-164's checklist
    rows, which 0-C indexed; "200 pF의 … 얼마인가?" is dropped by the length cap); the title must contain a letter, not
    start with "-",
    docling's `text` of the item (number included for section_headers, marker-stripped for list_items) must be
    ≤ MAX_HEADER_CHARS = 45 (the cut-off 0-C's own index implies: kept "5. CGA(…미국)" text 43, "3.1 모든 압축이음매…"
    44; dropped "2.3 모든 라인을 추적하고…" 46, "2010년 BP Deepwater…" 50) and not end with a Korean sentence ending followed by a period ("…다." / "…니다." / "…음." …): "0.1 MPa에서…
    상승한다." and "3.1 이 지침에서…같다." are body text; "5.2 일시적 작동 시스템：1차 결함" and "3. 교정일자 :
    2000. 00. 00." are headers.
  * running-header lines ("KOSHA GUIDE P-77-2011", "KOSHA GUIDE", "P - 53 - 2012") are items in the index view
    exactly as in 0-C (P-77 7.2 keeps its single "KOSHA GUIDE P-77-2011" line; the 22-char lines in P-120 vanish
    only because they repeat ≥ 3×, see compat view) and are stripped from CHUNK TEXT by RUNNING_HEADER_RE
  * COMPAT VIEW vs CORPUS: 0-C also dropped section_headers whose text (whitespace ignored) repeats >= 3 times in a
    document ("(3) 문제점" ×3, "위험성평가 검토결과 기록지 (예시)" ×7 + "…기록지(예시)" ×1). Those are real sub-headings, so the corpus KEEPS them as items; `compat_rows()` removes them only for the
    structural comparison with the 0-C index, and reports which ones it removed.
  * any other header-like item ("(가) 온도", "(1) 사고 개요") is an item of the current clause
  * the part increments when top-level numbering RESTARTS (a level-1 clause id <= the previous top-level id, e.g.
    "1. 폭발사고사례 A" after "6. 방호대책"); the "<부록>" header itself stays an item of the preceding clause (0-C did so)
  * text before the first numbered header (cover, 개요) belongs to no clause and is skipped
  * n_chars = len("".join(item texts)); page = page of the clause header
  * DUPLICATE clause ids inside one (guide, part) exist in the source (P-132-2013 has two "4.8": 교반기 정지 p.15 and
    미반응 물질의 축적 p.16; the 0-C index carries both rows under one key, 699 rows / 698 unique). Here EVERY occurrence
    of a repeated id gets an occurrence suffix `~n` (`P-132-2013:0:4.8~1`, `~2`) so no bare key can silently mean
    either one; the frozen mapping is re-pointed explicitly (fault_to_guidance_map.json amendment). `source_key`
    keeps the bare 0-C key for the structural-equivalence check, which compares ROW LISTS (duplicates preserved) and
    titles — it certifies index structure (keys, level, page, n_items, n_chars), not full-text equality.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# "6.3 x" / "5.2.1 x" / "1. x" / "5.5 . x" (stray dot absorbed); NOT "200 x". Depth capped at 4 components (0-C).
# multi-level "6.3 x" / "5.5. x" (depth ≤ 4), single "1. x" / "10. x", or single "10 x" with a space (P-164 checklist rows;
# "200 pF의 …?" is removed by the 45-char `text` cap, not by the number form)
CLAUSE_RE = re.compile(r"^\s*(?:(\d+(?:\.\d+){1,3})(?!\.?\d)\s*\.?(?!\d)|(\d+)\s*\.(?![.\d])|(\d+)(?=\s))\s*(\S.*)$")
MAX_HEADER_CHARS = 45                              # on docling `text` (marker-stripped): 0-C kept ≤45 ("5. CGA(…)" text 43), dropped 46+
SENTENCE_END_RE = re.compile(r"(다|니다|음|함|됨)\.\s*$")                  # Korean sentence ending WITH a period: "…상승한다." is body; "…1차 결함" / "교정일자 : 2000. 00. 00." are headers
REPEATED_HEADER_MIN = 3                           # 0-C quirk reproduced in compat_rows() ONLY: "(3) 문제점" ×3 dropped, "(1) 사고 개요" ×2 kept


DROP_LABELS = {"page_footer", "page_header"}      # docling furniture: 0-C dropped both (P-120 headers are page_header)
RUNNING_HEADER_RE = re.compile(r"^\s*(KOSHA\s*GUIDE(\s*P\s*-\s*\d+\s*-\s*\d{4})?|P\s*-\s*\d+\s*-\s*\d{4})\s*$")


@dataclass
class Clause:
    guide_no: str
    part: int
    clause_id: str
    title: str
    level: int
    page: int | None
    items: list[str] = field(default_factory=list)
    item_meta: list[dict] = field(default_factory=list)   # per item: docling text index, label, page (for diagnostics)

    occurrence: int = 0          # 0 = unique id; n>=1 = n-th occurrence of a repeated id within (guide, part)
    n_occurrences: int = 1

    @property
    def source_key(self) -> str:
        """Bare 0-C key (may be shared by duplicates)."""
        return f"{self.guide_no}:{self.part}:{self.clause_id}"

    @property
    def clause_key(self) -> str:
        return self.source_key if self.n_occurrences == 1 else f"{self.source_key}~{self.occurrence}"

    @property
    def n_chars(self) -> int:
        return len("".join(self.items))

    def text(self, strip_running: bool = True) -> str:
        items = [t for t in self.items if not (strip_running and RUNNING_HEADER_RE.match(t))]
        return "\n".join(items)


def parse_docling(doc: dict, guide_no: str) -> list[Clause]:
    clauses: list[Clause] = []
    part, cur, last_top = 0, None, 0
    for ti, t in enumerate(doc.get("texts", [])):
        label, text = t.get("label"), (t.get("text") or "").strip()
        orig = (t.get("orig") or text).strip()
        if label in DROP_LABELS or not text:
            continue                                   # running-header LINES are kept as items (0-C kept "KOSHA GUIDE P-77-2011" ×1)
                                                       # and stripped from chunk text only (RUNNING_HEADER_RE in text()/chunker)
        page = t["prov"][0]["page_no"] if t.get("prov") else None
        m = CLAUSE_RE.match(orig)
        if m and label in ("section_header", "list_item", "text"):
            cid, title = (m.group(1) or m.group(2) or m.group(3)), m.group(4).strip()
            if (re.search(r"[A-Za-z가-힣]", title)                       # a title needs a letter: "2012. 7." is a date
                    and not title.startswith(("-", "–", "—"))              # "3.4-1 공급원…": hyphenated checklist sub-items are not clauses
                    and len(text) <= MAX_HEADER_CHARS and not SENTENCE_END_RE.search(title)):   # body sentences are items
                level = cid.count(".") + 1
                if level == 1:
                    top = int(cid.split(".")[0])
                    if top <= last_top:                 # numbering restarted → appendix / next part
                        part += 1
                    last_top = top
                cur = Clause(guide_no, part, cid, title, level, page)
                clauses.append(cur)
                continue
        if cur is not None:
            cur.items.append(text)
            cur.item_meta.append({"idx": ti, "label": label, "page": page})
    # disambiguate repeated ids within (guide, part)
    from collections import Counter
    counts = Counter(c.source_key for c in clauses)
    seen: dict[str, int] = {}
    for c in clauses:
        if counts[c.source_key] > 1:
            seen[c.source_key] = seen.get(c.source_key, 0) + 1
            c.occurrence, c.n_occurrences = seen[c.source_key], counts[c.source_key]
    keys = [c.clause_key for c in clauses]
    assert len(keys) == len(set(keys)), "clause keys still not unique after disambiguation"
    return clauses


def guide_no_from_filename(path: Path) -> str:
    return path.stem                       # e.g. P-53-2012.json → P-53-2012


def parse_file(path: Path) -> list[Clause]:
    return parse_docling(json.loads(Path(path).read_text(encoding="utf-8")), guide_no_from_filename(Path(path)))


def index_rows(clauses: list[Clause]) -> list[dict]:
    """Same shape as outputs/kosha_clause_index.json rows, in document order, with the bare 0-C key (duplicates
    preserved) — the CORPUS view (all items kept)."""
    return [{"guide_no": c.guide_no, "part": str(c.part), "clause_id": c.clause_id, "title": c.title,
             "clause_key": c.source_key, "level": str(c.level), "page": str(c.page), "n_items": str(len(c.items)),
             "n_chars": str(c.n_chars)} for c in clauses]


def compat_rows(clauses: list[Clause]) -> tuple[list[dict], list[dict]]:
    """0-C-COMPATIBLE view for the structural comparison only: drops, per document, section_header items whose text
    repeats >= REPEATED_HEADER_MIN times (a 0-C quirk; the corpus keeps them). Returns (rows, removed_items)."""
    from collections import Counter, defaultdict
    by_doc = defaultdict(list)
    for c in clauses: by_doc[c.guide_no].append(c)
    rows, removed = [], []
    for g, cs in by_doc.items():
        norm = lambda x: re.sub(r"\s+", "", x)         # 0-C counted repeats ignoring whitespace ("기록지(예시)" ≡ "기록지 (예시)")
        counts = Counter(norm(it) for c in cs for it, mt in zip(c.items, c.item_meta) if mt["label"] == "section_header")
        for c in cs:
            keep = [(it, mt) for it, mt in zip(c.items, c.item_meta) if not (mt["label"] == "section_header" and counts[norm(it)] >= REPEATED_HEADER_MIN)]
            for it, mt in zip(c.items, c.item_meta):
                if mt["label"] == "section_header" and counts[norm(it)] >= REPEATED_HEADER_MIN:
                    removed.append({"guide": g, "clause": c.clause_key, "idx": mt["idx"], "page": mt["page"], "text": it, "repeats": counts[norm(it)]})
            rows.append({"guide_no": c.guide_no, "part": str(c.part), "clause_id": c.clause_id, "title": c.title,
                         "clause_key": c.source_key, "level": str(c.level), "page": str(c.page),
                         "n_items": str(len(keep)), "n_chars": str(len("".join(it for it, _ in keep)))})
    return rows, removed


STRUCT_FIELDS = ("clause_key", "title", "level", "page", "n_items", "n_chars")


def structural_diff(index_rows_0c: list[dict], mine: list[dict], known_extra: set[str] = frozenset(),
                    known_item_deltas: dict[str, tuple[int, int]] | None = None) -> list[str]:
    """Row-list comparison per guide (order preserved, duplicates preserved, titles compared).
    known_extra: parser clause keys (guide:part:id form) that are REVIEWED, recorded exceptions — clauses the 0-C
    index missed. For the comparison each such clause is FOLDED BACK into the preceding parser row the way 0-C saw
    it (its header line as one item + its items), so the fold itself verifies that 0-C merely failed to see the
    header. known_item_deltas: {clause_key: (d_items, d_chars)} reviewed per-clause differences (parser − 0-C) in
    form/table regions of guides the frozen mapping does not reference; applied only to that clause and only for
    exactly that delta. Anything else must match exactly."""
    known_item_deltas = known_item_deltas or {}
    from collections import defaultdict
    by_g0, by_g1 = defaultdict(list), defaultdict(list)
    for r in index_rows_0c: by_g0[r["guide_no"]].append(r)
    for r in mine:
        if r["clause_key"] in known_extra:
            rows = by_g1[r["guide_no"]]
            if not rows:
                return [f"{r['clause_key']}: known extra clause has no preceding row to fold into"]
            prev = dict(rows[-1])
            prev["n_items"] = str(int(prev["n_items"]) + 1 + int(r["n_items"]))
            prev["n_chars"] = str(int(prev["n_chars"]) + len(r["title"]) + int(r["n_chars"]))
            rows[-1] = prev
        else:
            by_g1[r["guide_no"]].append(dict(r))
    problems = []
    for g in sorted(set(by_g0) | set(by_g1)):
        a, b = by_g0.get(g, []), by_g1.get(g, [])
        if len(a) != len(b):
            ka, kb = [f"{r['clause_key']}|{r['title']}" for r in a], [f"{r['clause_key']}|{r['title']}" for r in b]
            problems.append(f"{g}: row count {len(a)} (0-C) vs {len(b)} (parser); extra={[x for x in kb if x not in ka]} missing={[x for x in ka if x not in kb]}")
            continue
        for i, (ra, rb) in enumerate(zip(a, b)):
            rb = dict(rb)
            if ra["clause_key"] in known_item_deltas and rb["clause_key"] == ra["clause_key"]:
                di, dc = known_item_deltas[ra["clause_key"]]
                rb["n_items"] = str(int(rb["n_items"]) - di); rb["n_chars"] = str(int(rb["n_chars"]) - dc)
            for f in STRUCT_FIELDS:
                if str(ra[f]) != str(rb[f]):
                    problems.append(f"{g} row {i} {ra['clause_key']}: {f} {ra[f]!r} vs {rb[f]!r}")
    return problems
