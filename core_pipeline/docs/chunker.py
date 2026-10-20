"""
core_pipeline/docs/chunker.py — S3-01: clause → chunks (prereg §4.3, ST0-20). MAX_CHARS = 1000 is a STRICT upper bound
on len(chunk.text), separators included: a clause whose stripped text fits is one chunk keyed by its clause_key; a longer
clause is split at item (paragraph) boundaries into greedy packs, keyed `clause_key#k`. An item longer than MAX_CHARS is
itself split at sentence boundaries (다./니다./. ) and, failing that, at MAX_CHARS, so no chunk ever exceeds the bound
(the reranker's max_length is set above the token count of a 1000-char Korean chunk; see models.py). Running page headers
("KOSHA GUIDE", "P - 53 - 2012") are stripped from chunk text (they were kept in the 0-C count only).
Every chunk carries guide_no, part, clause_id, clause_key, title, level, page so citations resolve to a real clause.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from core_pipeline.docs.clause_parser import Clause, RUNNING_HEADER_RE

MAX_CHARS = 1000


@dataclass(frozen=True)
class Chunk:
    chunk_key: str
    clause_key: str
    guide_no: str
    part: int
    clause_id: str
    title: str
    level: int
    page: int | None
    k: int
    n_chunks: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


_SENT_RE = re.compile(r"(?<=다\.)\s*|(?<=\. )")      # split after Korean sentence end 다./니다. or ". "


def split_long_item(item: str, max_chars: int) -> list[str]:
    if len(item) <= max_chars:
        return [item]
    pieces, cur = [], ""
    for sent in _SENT_RE.split(item):
        if cur and len(cur) + len(sent) > max_chars:
            pieces.append(cur); cur = ""
        cur += sent
    if cur:
        pieces.append(cur)
    out = []
    for pc in pieces:                                   # a single sentence longer than max_chars: hard cut
        out.extend(pc[i:i + max_chars] for i in range(0, len(pc), max_chars))
    return [x for x in out if x]


def chunk_clause(c: Clause, max_chars: int = MAX_CHARS) -> list[Chunk]:
    items = [p for t in c.items if not RUNNING_HEADER_RE.match(t) for p in split_long_item(t, max_chars)]
    if not items:
        return []
    packs: list[list[str]] = []
    cur, cur_len = [], 0
    for it in items:
        add = len(it) + (1 if cur else 0)                # "\n" separator counts
        if cur and cur_len + add > max_chars:
            packs.append(cur); cur, cur_len = [], 0; add = len(it)
        cur.append(it); cur_len += add
    if cur:
        packs.append(cur)
    assert all(len("\n".join(p)) <= max_chars for p in packs)
    base = dict(clause_key=c.clause_key, guide_no=c.guide_no, part=c.part, clause_id=c.clause_id, title=c.title,
                level=c.level, page=c.page, n_chunks=len(packs))
    if len(packs) == 1:
        return [Chunk(chunk_key=c.clause_key, k=0, text="\n".join(packs[0]), **base)]
    return [Chunk(chunk_key=f"{c.clause_key}#{k}", k=k, text="\n".join(p), **base) for k, p in enumerate(packs)]
