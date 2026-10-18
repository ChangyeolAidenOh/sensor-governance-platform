"""Inspect the raw TEP (Rieth et al. 2017) RData files and write an inventory.

Usage (from repo root, .venv active):
    python scripts/inspect_tep_raw.py --data-dir data/raw/TEP --out docs/tep_raw_inventory.md
    python scripts/inspect_tep_raw.py --files TEP_FaultFree_Training.RData   # one file at a time

Each file is loaded fully by pyreadr (no partial read), so the faulty files take
several GB of RAM. Files are processed one at a time and released before the next.
Nothing under data/ is written.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyreadr
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyreadr is required: pip install pyreadr") from exc


EXPECTED_FILES = [
    "TEP_FaultFree_Training.RData",
    "TEP_FaultFree_Testing.RData",
    "TEP_Faulty_Training.RData",
    "TEP_Faulty_Testing.RData",
]

ID_COLS_CANDIDATES = ["faultNumber", "simulationRun", "sample"]


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def inspect_file(path: Path) -> tuple[str, dict]:
    t0 = time.time()
    result = pyreadr.read_r(str(path))
    load_s = time.time() - t0

    # pyreadr returns an OrderedDict {object_name: DataFrame}
    names = list(result.keys())
    md: list[str] = []
    md.append(f"## {path.name}")
    md.append("")
    md.append(f"- file size: {_fmt_bytes(path.stat().st_size)}")
    md.append(f"- load time: {load_s:.1f} s")
    md.append(f"- R objects: {names}")

    summary: dict = {"file": path.name, "objects": {}}

    for name in names:
        df = result[name]
        md.append("")
        md.append(f"### object `{name}`")
        md.append("")
        md.append(f"- shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
        md.append(f"- memory (pandas): {_fmt_bytes(df.memory_usage(deep=True).sum())}")
        dtypes = df.dtypes.astype(str).value_counts().to_dict()
        md.append(f"- dtypes: {dtypes}")
        md.append(f"- columns: {list(df.columns)}")

        id_cols = [c for c in ID_COLS_CANDIDATES if c in df.columns]
        md.append(f"- id columns found: {id_cols}")

        nan_total = int(df.isna().sum().sum())
        md.append(f"- NaN cells: {nan_total:,}")

        obj: dict = {"shape": df.shape, "id_cols": id_cols, "nan": nan_total}

        if "faultNumber" in df.columns:
            faults = sorted(df["faultNumber"].unique().tolist())
            md.append(f"- faultNumber values ({len(faults)}): {faults}")
            obj["faults"] = faults
        if "simulationRun" in df.columns:
            runs = df["simulationRun"]
            md.append(
                f"- simulationRun: min {int(runs.min())}, max {int(runs.max())}, "
                f"n unique {runs.nunique()}"
            )
            obj["runs"] = (int(runs.min()), int(runs.max()), int(runs.nunique()))
        if "sample" in df.columns:
            smp = df["sample"]
            md.append(f"- sample: min {int(smp.min())}, max {int(smp.max())}, n unique {smp.nunique()}")
            obj["samples"] = (int(smp.min()), int(smp.max()), int(smp.nunique()))

        if {"faultNumber", "simulationRun"}.issubset(df.columns):
            per = df.groupby(["faultNumber", "simulationRun"], sort=False).size()
            md.append(
                f"- rows per (fault, run): min {int(per.min())}, max {int(per.max())}, "
                f"groups {len(per):,}"
            )
            runs_per_fault = df.groupby("faultNumber")["simulationRun"].nunique()
            md.append(f"- runs per fault: {runs_per_fault.to_dict()}")
            obj["rows_per_group"] = (int(per.min()), int(per.max()), int(len(per)))

            # Fault onset check: within one faulty run, the first sample index at
            # which the measurement vector departs from its pre-onset mean by more
            # than 6 sigma on any sensor. Reported only as a hint; the authoritative
            # onset is documented by Rieth et al. (train: sample 20, test: sample 160).
            sensor_cols = [c for c in df.columns if c not in ID_COLS_CANDIDATES]
            onset_rows = []
            for fault in [f for f in obj.get("faults", []) if f != 0][:3]:
                first_run = int(df.loc[df["faultNumber"] == fault, "simulationRun"].iloc[0])
                g = df[(df["faultNumber"] == fault) & (df["simulationRun"] == first_run)]
                g = g.sort_values("sample")
                x = g[sensor_cols].to_numpy(dtype=np.float64)
                base_n = min(15, len(x) - 1)
                mu = x[:base_n].mean(axis=0)
                sd = x[:base_n].std(axis=0) + 1e-9
                z = np.abs((x - mu) / sd)
                hit = np.where((z > 6).any(axis=1))[0]
                first = int(g["sample"].iloc[hit[0]]) if len(hit) else None
                onset_rows.append((fault, first_run, first))
            if onset_rows:
                md.append(
                    "- onset hint (fault, run, first sample with any |z|>6 vs first "
                    f"{base_n} samples): {onset_rows}"
                )
                obj["onset_hint"] = onset_rows

        head = df.head(3).to_string(max_cols=12)
        md.append("")
        md.append("```")
        md.append(head)
        md.append("```")
        summary["objects"][name] = obj

    del result
    gc.collect()
    return "\n".join(md), summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/raw/TEP")
    ap.add_argument("--files", nargs="*", default=None, help="subset of file names; default = all four")
    ap.add_argument("--out", default="docs/tep_raw_inventory.md")
    ap.add_argument("--append", action="store_true", help="append to --out instead of overwriting")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = args.files or EXPECTED_FILES
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []
    if not args.append:
        sections.append("# TEP raw inventory (Rieth et al. 2017, Harvard Dataverse doi:10.7910/DVN/6C3JR1)")
        sections.append("")
        sections.append(f"Generated by scripts/inspect_tep_raw.py on {time.strftime('%Y-%m-%d %H:%M')}")
        sections.append("")

    for fname in files:
        path = data_dir / fname
        if not path.exists():
            print(f"[skip] missing: {path}")
            sections.append(f"## {fname}\n\n- MISSING\n")
            continue
        print(f"[load] {path} ({_fmt_bytes(path.stat().st_size)})")
        md, summary = inspect_file(path)
        print(md)
        print()
        sections.append(md)
        sections.append("")

    mode = "a" if args.append else "w"
    with out.open(mode, encoding="utf-8") as f:
        f.write("\n".join(sections))
        f.write("\n")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
