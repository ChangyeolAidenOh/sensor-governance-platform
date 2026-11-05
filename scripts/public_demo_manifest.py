"""scripts/public_demo_manifest.py: turn the precompute output of the public build into the public manifest (demo-public).

    python scripts/public_demo_manifest.py --raw build/public/demo_presets_public_raw.json --out outputs/demo_presets_public.json --synthetic-presets 11

The precompute entries are kept as they are (schema demo_presets/1, no truth labels). The synthetic preset(s) get
kind = "synthetic", an explicit note and `synthetic: true`; the document gets `public: true` and the list of synthetic
presets, so the app, the landing page and anyone reading the file see the labelling (decision 1 (a)/(b) of the public
demo plan). The synthetic preset must be an alarm episode in the precompute (otherwise there is nothing to propose).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SYNTHETIC_NOTE = ("합성 승인 사례: 실제 탐지 결과가 아닙니다. 이 프리셋의 세션에서는 근거 등급을 인위적으로 A/B 이상으로 바꾸고 정책 결정을 '제안'으로 고정해 "
                  "승인 게이트(approve / modify / reject)를 보여 줍니다. 탐지 창, 알람, 진단 집합은 저장된 실제 값이지만 조치안과 승인 흐름은 시연용입니다.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic-presets", default="11")
    a = ap.parse_args()
    syn = {int(x) for x in a.synthetic_presets.split(",") if x.strip()}
    doc = json.loads(Path(a.raw).read_text(encoding="utf-8"))
    if doc.get("schema") != "demo_presets/1":
        print(f"unexpected schema {doc.get('schema')!r}", file=sys.stderr)
        return 2
    seen = set()
    for e in doc["presets"]:
        p = int(e["preset"])
        seen.add(p)
        if p in syn:
            if e.get("kind") != "alarm":
                print(f"synthetic preset {p} is kind={e.get('kind')!r}; it must be an alarm episode (re-run prune with another --synthetic-run-id)", file=sys.stderr)
                return 2
            e["kind"], e["synthetic"], e["note"] = "synthetic", True, SYNTHETIC_NOTE
        else:
            e["synthetic"] = False
    missing = syn - seen
    if missing:
        print(f"synthetic presets {sorted(missing)} are not in the precompute output", file=sys.stderr)
        return 2
    for forbidden in ("fault_scenario", "run_id", "reveal"):
        if any(forbidden in e for e in doc["presets"]):
            print(f"refusing: field {forbidden!r} in a preset entry (truth label)", file=sys.stderr)
            return 2
    doc["public"], doc["synthetic_presets"] = True, sorted(syn)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    kinds = {int(e["preset"]): e["kind"] for e in doc["presets"]}
    print(f"wrote {a.out}: {kinds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
