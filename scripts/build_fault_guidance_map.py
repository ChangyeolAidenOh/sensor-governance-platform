"""Fault -> guidance clause candidates (0-G draft, rule-based; see docs/fault_to_guidance_map.md).

The mapping is an assumption of this project (plan §1.3): KOSHA GUIDE P-series do not
describe the TEP process, so each TEP fault is linked to guidance *topics*, and clause
candidates are selected from the clause index by keyword match on clause title + preview,
restricted to the guides named for that fault. Nothing here is a claim that the clause is
the engineering-correct response; it is the retrieval prior the agent starts from.

Inputs : outputs/kosha_clause_index.json (scripts/extract_kosha_clauses.py)
Outputs: outputs/fault_guidance_candidates.json, docs/fault_guidance_candidates.md

Usage:  python scripts/build_fault_guidance_map.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

INDEX = Path("outputs/kosha_clause_index.json")
OUT_JSON = Path("outputs/fault_guidance_candidates.json")
OUT_MD = Path("docs/fault_guidance_candidates.md")

# TEP faults (Downs & Vogel 1993; Rieth et al. 2017 keep 1..20)
FAULTS = {
    1: ("A/C 피드 비율 변화, B 조성 일정 (스트림 4)", "step", "feed_composition"),
    2: ("B 조성 변화, A/C 비율 일정 (스트림 4)", "step", "feed_composition"),
    3: ("D 피드 온도 변화 (스트림 2)", "step", "feed_temperature"),
    4: ("반응기 냉각수 입구 온도 변화", "step", "reactor_cooling"),
    5: ("응축기 냉각수 입구 온도 변화", "step", "condenser_cooling"),
    6: ("A 피드 손실 (스트림 1)", "step", "feed_loss"),
    7: ("C 헤더 압력 손실 — 공급 가용성 저하 (스트림 4)", "step", "header_pressure"),
    8: ("A, B, C 피드 조성 변동 (스트림 4)", "random", "feed_composition"),
    9: ("D 피드 온도 변동 (스트림 2)", "random", "feed_temperature"),
    10: ("C 피드 온도 변동 (스트림 4)", "random", "feed_temperature"),
    11: ("반응기 냉각수 입구 온도 변동", "random", "reactor_cooling"),
    12: ("응축기 냉각수 입구 온도 변동", "random", "condenser_cooling"),
    13: ("반응 속도론 완만한 드리프트", "drift", "reaction_kinetics"),
    14: ("반응기 냉각수 밸브 고착", "sticking", "valve_sticking"),
    15: ("응축기 냉각수 밸브 고착", "sticking", "valve_sticking"),
    16: ("미공개 결함", "unknown", "unknown"),
    17: ("미공개 결함", "unknown", "unknown"),
    18: ("미공개 결함", "unknown", "unknown"),
    19: ("미공개 결함", "unknown", "unknown"),
    20: ("미공개 결함", "unknown", "unknown"),
}

# topic -> (primary guides, secondary guides, keywords matched against title+preview)
TOPICS = {
    "feed_composition": (["P-132-2013", "P-167-2020"], ["P-179-2022", "P-129-2013"],
                         ["조성", "혼합", "원료", "시료", "분석", "비율", "농도", "폭발하한"]),
    "feed_temperature": (["P-38-2012", "P-53-2012"], ["P-129-2013", "P-116-2012"],
                         ["온도", "발열", "열량", "안전운전 한계", "계측", "설정값"]),
    "reactor_cooling": (["P-53-2012", "P-38-2012", "P-65-2012"], ["P-133-2013", "P-116-2012"],
                        ["냉각", "폭주", "온도", "압력", "파열판", "방호", "인터록", "긴급"]),
    "condenser_cooling": (["P-60-2012", "P-53-2012"], ["P-129-2013"],
                          ["냉각", "냉매", "응축", "온도", "압력", "점검"]),
    "feed_loss": (["P-18-2012", "P-52-2012", "P-77-2011"], ["P-58-2012"],
                  ["누출", "차단", "격리", "비상", "공급", "밸브", "조치"]),
    "header_pressure": (["P-18-2012", "P-65-2012", "P-159-2017"], ["P-121-2012"],
                        ["압력", "벤트", "방출", "누출", "공급", "운전"]),
    "reaction_kinetics": (["P-38-2012", "P-53-2012"], ["P-120-2012"],
                          ["반응", "열량", "위험확인", "설계", "변경", "검토", "한계"]),
    "valve_sticking": (["P-77-2011", "P-129-2013"], ["P-133-2013"],
                       ["밸브", "작동", "페일", "정비", "검사", "테스트", "계측기", "점검", "불량", "우회"]),
    "unknown": ([], ["P-116-2012", "P-129-2013", "P-58-2012"],
                ["경보", "우선순위", "점검", "사고대응", "비상"]),
}

# clauses every alarm may cite regardless of fault (alarm handling, instrument check, interlock rules)
CROSS_CUTTING = {
    "P-116-2012": ["5.1", "5.2.2", "5.2.4", "5.2.5"],
    "P-129-2013": ["4.1", "4.3", "5.2.2"],
    "P-133-2013": ["5.1", "5.2.2", "5.2.3"],
}

SKIP_TITLES = ("목적", "목 적", "적용범위", "정의", "용어의 정의")


def main() -> None:
    clauses = json.loads(INDEX.read_text(encoding="utf-8"))
    by_guide: dict[str, list[dict]] = {}
    for c in clauses:
        by_guide.setdefault(c["guide_no"], []).append(c)

    def candidates(guides: list[str], keywords: list[str], limit: int = 8) -> list[dict]:
        out = []
        for g in guides:
            for c in by_guide.get(g, []):
                if c["part"] != 0 or c["title"].strip() in SKIP_TITLES or c["n_chars"] == 0:
                    continue
                text = c["title"] + " " + c["preview"]
                hits = [k for k in keywords if k in text]
                if hits:
                    out.append({"clause_key": c["clause_key"], "guide_no": g, "clause_id": c["clause_id"],
                                "title": c["title"], "page": c["page"], "n_chars": c["n_chars"], "hits": hits})
        out.sort(key=lambda x: (-len(x["hits"]), x["guide_no"], [int(p) for p in x["clause_id"].split(".")]))
        return out[:limit]

    result = []
    md = ["# Fault -> guidance clause candidates (rule-based draft, 0-G)", "",
          "Generated by scripts/build_fault_guidance_map.py from outputs/kosha_clause_index.json. "
          "Candidates are keyword matches within the guides assigned to each fault topic; review and prune "
          "before freezing docs/fault_to_guidance_map.md. Faults 16-20 have no primary guides by design.", ""]
    for fid, (desc, kind, topic) in FAULTS.items():
        primary, secondary, kws = TOPICS[topic]
        prim = candidates(primary, kws)
        sec = candidates(secondary, kws, limit=5)
        result.append({"fault": fid, "description": desc, "kind": kind, "topic": topic,
                       "primary_guides": primary, "secondary_guides": secondary, "keywords": kws,
                       "primary_candidates": prim, "secondary_candidates": sec})
        md.append(f"## Fault {fid} — {desc} ({kind}, topic `{topic}`)")
        md.append("")
        md.append(f"primary guides: {', '.join(primary) or '(none)'} · secondary: {', '.join(secondary)}")
        for label, lst in (("primary", prim), ("secondary", sec)):
            md.append(f"- {label} ({len(lst)}):")
            for c in lst:
                md.append(f"  - {c['clause_key']} **{c['title']}** p.{c['page']} ({c['n_chars']} chars) hits={c['hits']}")
        md.append("")
    cross = {g: [c for c in by_guide.get(g, []) if c["part"] == 0 and c["clause_id"] in ids]
             for g, ids in CROSS_CUTTING.items()}
    md.append("## Cross-cutting clauses (any alarm)")
    md.append("")
    for g, lst in cross.items():
        md.append(f"- {g}: " + ", ".join(f"{c['clause_id']} {c['title']}" for c in lst) if lst else f"- {g}: (to fill)")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"faults": result, "cross_cutting": {g: [c["clause_key"] for c in l] for g, l in cross.items()}},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    n_prim = {r["fault"]: len(r["primary_candidates"]) for r in result}
    print(f"[done] {OUT_JSON}, {OUT_MD}")
    print("primary candidates per fault:", n_prim)
    print("faults with 0 primary candidates:", [f for f, n in n_prim.items() if n == 0])


if __name__ == "__main__":
    main()
