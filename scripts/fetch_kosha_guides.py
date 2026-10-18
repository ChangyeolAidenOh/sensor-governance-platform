"""KOSHA GUIDE corpus acquisition for the 0-C parsing spike.

Source: data.go.kr open API 15144147 (한국산업안전보건공단 기술지원규정 조회),
endpoint https://apis.data.go.kr/B552468/koshaguide/getKoshaGuide.
Request (verified 2026-10-18 against the portal spec and a live call):
    serviceKey (Decoding key), callApiId=1050 (fixed), pageNo, numOfRows,
    optional techGdlnNo (series letter or full number), techGdlnNm, ofancYmd.
Response: body.totalCount, body.items.item[] with techGdlnNm, techGdlnNo,
    techGdlnOfancYmd, fileDownloadUrl; header.resultCode ("00" = ok).

Steps:
    python scripts/fetch_kosha_guides.py index [--series P]
        -> data/raw/kosha/index_raw.json (items verbatim)
        -> data/raw/kosha/index_P.csv     (guide_no, name, date, url)
        prints every title so the corpus selection can be made from the list
    python scripts/fetch_kosha_guides.py download --select data/raw/kosha/selection.txt
        -> data/raw/kosha/files/<guide_no>.<ext>, data/raw/kosha/files_manifest.json
        selection.txt: one guide number per line (e.g. P-123-2026); '#' comments allowed

Requires DATA_GO_KR_KEY in the environment or .env (일반 인증키 Decoding).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

BASE = "https://apis.data.go.kr/B552468/koshaguide/getKoshaGuide"
GUIDE_NO_RE = re.compile(r"\b([A-Z]{1,2})\s*-\s*(\d{1,4})\s*-\s*(\d{4})\b")
DATE_RE = re.compile(r"^\d{4}[-.]?\d{2}[-.]?\d{2}")
RAW_DIR = Path("data/raw/kosha")
DOWNLOAD_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
                    "Accept": "application/pdf,*/*;q=0.8"}


def load_key() -> str:
    key = os.environ.get("DATA_GO_KR_KEY")
    if not key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            key = os.environ.get("DATA_GO_KR_KEY")
        except ImportError:
            pass
    if not key:
        sys.exit("DATA_GO_KR_KEY not set (export it or put it in .env)")
    return key


def gateway_error(node, depth=0):
    """data.go.kr returns errors as HTTP 200 with resultCode/returnReasonCode."""
    if not isinstance(node, dict) or depth > 5:
        return None
    code = msg = None
    for k, v in node.items():
        if isinstance(v, (str, int)):
            if re.fullmatch(r"(returnReasonCode|resultCode|errorCode)", k, re.I):
                code = str(v).strip()
            if re.fullmatch(r"(returnAuthMsg|resultMsg|errMsg|errorMessage)", k, re.I):
                msg = str(v).strip()
    if code is not None and not re.match(r"^(00|0|NORMAL)", code, re.I):
        return f"gateway code {code}: {msg}"
    for v in node.values():
        found = gateway_error(v, depth + 1)
        if found:
            return found
    return None


def pick_list(node, depth=0):
    if isinstance(node, list):
        return [x for x in node if isinstance(x, dict)]
    if isinstance(node, dict) and depth < 4:
        for v in node.values():
            found = pick_list(v, depth + 1)
            if found:
                return found
        if depth >= 2:
            return [node]
    return []


def fetch_page(key: str, page: int, rows: int = 100, series: str | None = None) -> tuple[list[dict], int]:
    params = {"serviceKey": key, "callApiId": "1050", "pageNo": page, "numOfRows": rows}
    if series:
        params["techGdlnNo"] = series
    r = requests.get(BASE, params=params, timeout=60)
    if r.status_code != 200:
        sys.exit(f"HTTP {r.status_code}: {r.text[:500]}")
    try:
        parsed = json.loads(r.text)
    except json.JSONDecodeError:
        sys.exit(f"non-JSON response (first 500 chars):\n{r.text[:500]}")
    err = gateway_error(parsed)
    if err:
        sys.exit(f"data.go.kr error: {err}\nraw: {r.text[:500]}")
    body = parsed.get("body") or {}
    items = (body.get("items") or {}).get("item") or []
    if isinstance(items, dict):
        items = [items]
    return items, int(body.get("totalCount") or 0)


def normalize(item: dict) -> dict:
    return {"guide_no": str(item.get("techGdlnNo", "")).replace(" ", "").upper(),
            "name": str(item.get("techGdlnNm", "")).strip(),
            "date": str(item.get("techGdlnOfancYmd", "")).strip(),
            "url": str(item.get("fileDownloadUrl", "")).strip()}


def cmd_index(args) -> None:
    key = load_key()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    total = None
    page = 1
    while True:
        page_items, total = fetch_page(key, page, rows=100, series=args.series)
        if not page_items:
            break
        items.extend(page_items)
        print(f"[page {page}] {len(page_items)} items (totalCount {total})")
        if len(items) >= total or page >= args.max_pages:
            break
        page += 1
        time.sleep(0.3)
    (RAW_DIR / "index_raw.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[saved] {RAW_DIR / 'index_raw.json'}: {len(items)} items")

    norm = [normalize(it) for it in items]
    series_counts: dict[str, int] = {}
    for n in norm:
        s = n["guide_no"].split("-")[0] if n["guide_no"] else "?"
        series_counts[s] = series_counts.get(s, 0) + 1
    print("[series counts]", dict(sorted(series_counts.items())))
    years: dict[str, int] = {}
    for n in norm:
        years[n["date"][:4]] = years.get(n["date"][:4], 0) + 1
    print("[publication years]", dict(sorted(years.items())))

    p_items = sorted([n for n in norm if n["guide_no"].startswith(f"{args.series}-")],
                     key=lambda x: int(x["guide_no"].split("-")[1]))
    out_csv = RAW_DIR / f"index_{args.series}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["guide_no", "name", "date", "url"])
        w.writeheader()
        w.writerows(p_items)
    print(f"[saved] {out_csv}: {len(p_items)} items")
    for n in p_items:
        print(f"    {n['guide_no']:<12} {n['date']}  {n['name']}")

def detect_kind(head: bytes) -> str:
    if head.startswith(b"%PDF-"):
        return "pdf"
    if head[:2] == b"PK":
        return "hwpx"
    if head[:4] == b"\xd0\xcf\x11\xe0":
        return "hwp"
    return "unknown"


def cmd_download(args) -> None:
    index = {r["guide_no"]: r for r in csv.DictReader((RAW_DIR / "index_P.csv").open(encoding="utf-8"))}
    wanted = [ln.split("#")[0].strip().upper() for ln in Path(args.select).read_text(encoding="utf-8").splitlines()]
    wanted = [w for w in wanted if w]
    out_dir = RAW_DIR / "files"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for g in wanted:
        row = index.get(g)
        if not row or not row["url"]:
            print(f"[skip] {g}: not in index_P.csv or no url")
            manifest.append({"guide_no": g, "status": "missing"})
            continue
        # portal.kosha.or.kr answers 400 (HTML) to non-browser User-Agents; a browser UA gets the PDF
        r = requests.get(row["url"], timeout=120, headers=DOWNLOAD_HEADERS)
        kind = detect_kind(r.content[:8])
        if kind == "unknown":
            print(f"[warn] {g}: not a known document type (HTTP {r.status_code}, {len(r.content)} bytes, "
                  f"content-type {r.headers.get('content-type')})")
        cd = r.headers.get("content-disposition", "")
        ext = {"pdf": "pdf", "hwpx": "hwpx", "hwp": "hwp"}.get(kind, "bin")
        path = out_dir / f"{g}.{ext}"
        path.write_bytes(r.content)
        manifest.append({"guide_no": g, "name": row["name"], "date": row["date"], "url": row["url"],
                         "file": path.name, "kind": kind, "bytes": len(r.content), "http": r.status_code,
                         "content_type": r.headers.get("content-type", ""), "content_disposition": cd})
        print(f"[ok] {g} -> {path.name} ({kind}, {len(r.content)/1e6:.1f} MB)")
        time.sleep(0.5)
    (RAW_DIR / "files_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    kinds = {}
    for m in manifest:
        kinds[m.get("kind", "missing")] = kinds.get(m.get("kind", "missing"), 0) + 1
    print("[done] kinds:", kinds)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("index"); s1.add_argument("--max-pages", type=int, default=20)
    s1.add_argument("--series", default="P", help="series letter sent as techGdlnNo (P = 공정안전)")
    s2 = sub.add_parser("download"); s2.add_argument("--select", default=str(RAW_DIR / "selection.txt"))
    args = ap.parse_args()
    {"index": cmd_index, "download": cmd_download}[args.cmd](args)


if __name__ == "__main__":
    main()
