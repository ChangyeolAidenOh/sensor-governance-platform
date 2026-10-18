"""Determine the exact fault-onset sample in TEP by paired comparison.

Hypothesis (from the raw inventory): faulty run r and fault-free run r share the
same RNG seed, so their trajectories are identical up to the sample at which the
fault is introduced. The first sample where any sensor column differs is the
onset. This is exact, unlike a variance-based detector.

Usage (from repo root, .venv active):
    python scripts/check_tep_onset.py --split training --runs 1 2 3 4 5
    python scripts/check_tep_onset.py --split testing  --runs 1 2 3 4 5

Appends a section to docs/tep_raw_inventory.md by default.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

FILES = {
    "training": ("TEP_FaultFree_Training.RData", "TEP_Faulty_Training.RData"),
    "testing": ("TEP_FaultFree_Testing.RData", "TEP_Faulty_Testing.RData"),
}
ID_COLS = ["faultNumber", "simulationRun", "sample"]


def load_single(path: Path) -> pd.DataFrame:
    res = pyreadr.read_r(str(path))
    (name, df), = res.items()
    for c in ID_COLS:
        df[c] = df[c].astype(np.int64)
    return df


def first_diff_sample(a: pd.DataFrame, b: pd.DataFrame, sensor_cols: list[str]) -> tuple[int | None, int]:
    """Return (first differing sample id or None, number of identical leading samples)."""
    a = a.sort_values("sample")
    b = b.sort_values("sample")
    n = min(len(a), len(b))
    xa = a[sensor_cols].to_numpy()[:n]
    xb = b[sensor_cols].to_numpy()[:n]
    diff = np.any(xa != xb, axis=1)
    idx = np.where(diff)[0]
    if len(idx) == 0:
        return None, n
    return int(a["sample"].to_numpy()[idx[0]]), int(idx[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/raw/TEP")
    ap.add_argument("--split", choices=list(FILES), required=True)
    ap.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--faults", nargs="*", type=int, default=None, help="default = all faults present")
    ap.add_argument("--out", default="docs/tep_raw_inventory.md")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    ff_name, fy_name = FILES[args.split]

    t0 = time.time()
    ff = load_single(data_dir / ff_name)
    fy = load_single(data_dir / fy_name)
    print(f"[load] both files in {time.time() - t0:.1f} s")

    sensor_cols = [c for c in fy.columns if c not in ID_COLS]
    faults = args.faults or sorted(fy["faultNumber"].unique().tolist())

    rows = []
    for f in faults:
        for r in args.runs:
            a = fy[(fy["faultNumber"] == f) & (fy["simulationRun"] == r)]
            b = ff[(ff["faultNumber"] == 0) & (ff["simulationRun"] == r)]
            if a.empty or b.empty:
                rows.append((f, r, None, None, "missing pair"))
                continue
            onset, n_same = first_diff_sample(a, b, sensor_cols)
            rows.append((f, r, onset, n_same, ""))

    res = pd.DataFrame(rows, columns=["fault", "run", "first_diff_sample", "identical_leading_samples", "note"])
    print(res.to_string(index=False))

    onset_vals = res["first_diff_sample"].dropna().astype(int)
    summary = (
        f"- split: {args.split} · faults: {len(faults)} · runs checked: {args.runs}\n"
        f"- first_diff_sample distribution: {onset_vals.value_counts().sort_index().to_dict()}\n"
        f"- runs with no difference at all (fault never diverges from fault-free): "
        f"{int(res['first_diff_sample'].isna().sum())}"
    )
    print(summary)

    out = Path(args.out)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## Fault onset by paired comparison ({args.split})\n\n")
        fh.write("Faulty run r vs fault-free run r (same simulationRun id); first sample where any sensor differs.\n\n")
        fh.write(summary + "\n\n```\n" + res.to_string(index=False) + "\n```\n")
    print(f"[done] appended to {out}")

    del ff, fy
    gc.collect()


if __name__ == "__main__":
    main()
