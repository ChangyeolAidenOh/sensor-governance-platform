"""
scripts/replay_smoke.py — S1-03 smoke on DEV run ids only (never evaluation): replays dev episodes with a dummy
policy (no model, no LLM), checks the order sha, reports wall time per run id. Writes nothing to the DB.
Usage: python scripts/replay_smoke.py --dsn $SGP_PG_DSN --processed data/processed/TEP [--max-run-ids 2]
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from core_pipeline.stream.replayer import DbParquetSource, Replayer, ReplayRecord
from core_pipeline.anomaly.prefix_bh import bh_adjusted
from core_pipeline import prereg_constants as P


def dummy_policy(view, ep, w):
    h = hashlib.sha256(pd.util.hash_pandas_object(view.current(), index=False).values.tobytes()).digest()
    p = int.from_bytes(h[:4], "big") / 2**32
    adj = bh_adjusted([q for _, q in ep._tested] + [p])[-1]
    return ("no_alarm" if adj > P.ALPHA else "hold"), "PASS", p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--processed", default="data/processed/TEP")
    ap.add_argument("--max-run-ids", type=int, default=2)
    a = ap.parse_args()
    src = DbParquetSource(a.dsn, a.processed, group="dev")
    expected = src.expected_sha()

    class Limited:                                   # first k run ids of the dev order
        def __iter__(self):
            seen = []
            for ep in src:
                if ep.run_id not in seen:
                    seen.append(ep.run_id)
                    if len(seen) > a.max_run_ids:
                        return
                yield ep
    recs: list[ReplayRecord] = []
    # expected sha for the limited prefix is derived by replaying the order query the same way
    import psycopg
    from core_pipeline.stream.replayer import order_sha
    with psycopg.connect(a.dsn) as conn, conn.cursor() as cur:
        cur.execute(src._sql()); rows = cur.fetchall()
    runs = sorted({r for _, _, r, _ in rows})[: a.max_run_ids]
    exp_limited = order_sha((r, f) for _, _, r, f in rows if r in runs)
    t0 = time.perf_counter()
    out = Replayer(Limited(), n=len(runs), expected_sha=exp_limited, on_record=recs.append).run(dummy_policy)
    out["wall_per_run_id"] = (time.perf_counter() - t0) / max(1, len(runs))
    out["windows"] = len(recs)
    out["alarms"] = sum(1 for r in recs if r.alarm)
    out["projected_first20_seconds_dummy"] = out["wall_per_run_id"] * 20
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
