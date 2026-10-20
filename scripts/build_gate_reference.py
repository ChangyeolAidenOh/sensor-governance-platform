"""
scripts/build_gate_reference.py — S1-04: build the external gate reference from the calibration group's fault-free
episodes (prereg §3.0) and save it as data/processed/gate_reference_v1.parquet (+ .json provenance).
Reads the calibration engine ids from staging (loader role). Deterministic (seed = prereg SEED).
Usage: python scripts/build_gate_reference.py --dsn $SGP_PG_DSN --processed data/processed/TEP
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psycopg
from core_pipeline import prereg_constants as P
from core_pipeline.data.tep_loader import load_processed
from core_pipeline.governance.gate_adapter import GateReference
from core_pipeline.stream.window_view import PROCESS_COLS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--processed", default="data/processed/TEP")
    ap.add_argument("--out", default="data/processed/gate_reference_v1.parquet")
    a = ap.parse_args()
    with psycopg.connect(a.dsn) as conn, conn.cursor() as cur:
        cur.execute("""SELECT episode_id, engine_id FROM staging.episodes
                       WHERE "group"='calibration' AND fault_scenario=0 ORDER BY engine_id""")
        rows = cur.fetchall()
    assert len(rows) == 100, f"expected 100 calibration fault-free episodes, got {len(rows)}"
    eids = [e for _, e in rows]
    df = load_processed(a.processed, P.SPLIT, faulty=False, columns=PROCESS_COLS, engine_ids=eids)
    frames = [g for _, g in df.groupby("engine_id", sort=True)]
    ref = GateReference.from_calibration(frames, [str(e) for e, _ in rows])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ref.save(Path(a.out))
    print(json.dumps({k: v for k, v in ref.meta.items() if k != "sensors"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
