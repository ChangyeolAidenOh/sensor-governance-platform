"""
scripts/check_mart_inputs.py — ST1-01: before a replay (or any evaluation) confirm the mart was built from the
parquet set and split file that are on disk NOW. Exit 1 on mismatch → do not replay against a stale mart.
Compares: last ok load_log (manifest sha256, split sha256, grid sha256, prereg commit) vs. the current files and a
fresh derivation; and raw.tep_files sidecar identity vs. the current <file>.done.json sidecars.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import psycopg
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_pipeline import prereg_constants as P  # noqa: E402
from scripts.load_mart import derive, sha256_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--split", default="outputs/stage0_episode_split.json")
    ap.add_argument("--manifest", default="data/processed/TEP/_manifest.json")
    a = ap.parse_args()
    bad = []
    with psycopg.connect(a.dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT load_id, manifest_sha256, split_json_sha256, grid_sha256, prereg_commit, prereg_n FROM mart.load_log WHERE ok ORDER BY load_id DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            print("FAIL no successful load"); return 1
        load_id, msha, ssha, gsha, pc, n = row
        if pc != P.PREREG_COMMIT: bad.append(f"prereg commit {pc} != {P.PREREG_COMMIT}")
        if ssha != sha256_file(Path(a.split)): bad.append("split JSON on disk differs from the loaded one")
        if derive(json.loads(Path(a.split).read_text()), n)[3] != gsha: bad.append("derived grid differs from loaded grid")
        mp = Path(a.manifest)
        if msha is None: bad.append("load recorded no manifest identity (dev load) — not valid for replay")
        elif not mp.exists(): bad.append(f"manifest missing at {mp}")
        elif sha256_file(mp) != msha: bad.append("manifest on disk differs from the loaded one (parquet set replaced?)")
        cur.execute("SELECT file_name, size_bytes, written_at, raw_size, raw_mtime FROM raw.tep_files")
        for fname, size, written, rsize, rmtime in cur.fetchall():
            sc = mp.parent / f"{fname}.done.json"
            if not sc.exists(): bad.append(f"sidecar missing {sc}"); continue
            d = json.loads(sc.read_text())
            if (d.get("size_bytes"), d.get("written_at"), d.get("raw_size"), d.get("raw_mtime")) != (size, written, rsize, rmtime):
                bad.append(f"{fname}: sidecar identity differs from registered")
            pq = mp.parent / fname
            if pq.exists() and pq.stat().st_size != size: bad.append(f"{fname}: parquet size differs")
    for b in bad: print("FAIL", b)
    print("RESULT:", "PASS" if not bad else "FAIL", f"(load_id={load_id}, n={n})")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
