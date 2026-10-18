"""
TEP (Tennessee Eastman Process) Data Loading and Preprocessing

Rieth et al. 2017, "Additional Tennessee Eastman Process Simulation Data for
Anomaly Detection Evaluation", Harvard Dataverse doi:10.7910/DVN/6C3JR1.

Four RData files: fault-free / faulty x training / testing. Each file holds
500 simulation runs per fault scenario (fault 0 = fault-free, 1..20 = faults),
sampled every 3 minutes. Training runs are 500 samples long, testing runs 960.
52 process variables: xmeas_1..41 (measurements) and xmv_1..11 (manipulated).

Output follows the v1 preprocess contract so that Layer 1 (quality gate) and
Layer 2A (anomaly) code consume it unchanged:
    engine_id   int   one simulated run (episode), globally unique:
                      split_code * 100000 + fault_scenario * 1000 + run_id
                      (training 1..20500, testing 100001..120500)
    cycle       int   sample index within the run (1-based)
plus process columns passed explicitly as `TEP_PROCESS_COLS`. Two identifiers
are deliberately distinct: `engine_id` is the episode unit for gate/window
processing; (`split`, `run_id`) is the shared-seed group used for data splits
(decision ST0-09), since fault-free run r and faulty runs r of the same split
share their pre-onset trajectory.

Labels (decision ST0-08):
    fault_scenario    run-level scenario id, 0..20 (== raw faultNumber)
    is_fault_active   sample-level: fault_scenario > 0 and cycle >= onset,
                      onset = 21 (training, 1 h) / 161 (testing, 8 h) per the
                      dataset specification. NOT derived from the data.
The observed divergence cycle (first cycle where a faulty run departs from the
fault-free run with the same run id) is written separately as metadata and is
never used as a label.

Memory: the faulty testing file is ~3.9 GB in pandas. Columns are renamed and
added in place, column order is applied only at write time via pyarrow, and
validation uses reshape on fixed-length episode blocks instead of groupby. Row
order is checked; the fault-free files arrive in (engine_id, cycle) order, the
faulty files do not (observed 2026-10-18), so those are sorted once, which
copies the frame. Measured peak RSS for the full conversion on an M2 16 GB:
8.6 GB (faulty testing step). Rows are written fault-major so that parquet
row-group statistics prune `engine_id` filters.

Completion and resume: a parquet is written to a temp file, renamed into place,
read back and re-verified, and only then gets a `<name>.done.json` sidecar that
records the raw file identity (size, mtime), loader version, onset, run length
and compression. A rerun skips a file only when its sidecar matches the current
raw file and options and the parquet size still matches. `_manifest.json` is
written atomically last; `--force` removes any existing manifest first so a
failed rerun cannot leave a stale completion marker. Readers refuse a directory
without a manifest.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TEP_LOADER_VERSION = "0.3"

# Column definitions
TEP_XMEAS_COLS = [f"xmeas_{i}" for i in range(1, 42)]
TEP_XMV_COLS = [f"xmv_{i}" for i in range(1, 12)]
TEP_PROCESS_COLS = TEP_XMEAS_COLS + TEP_XMV_COLS
TEP_INDEX_COLS = ["engine_id", "cycle"]
TEP_LABEL_COLS = ["fault_scenario", "is_fault_active", "run_id", "split"]
TEP_CONTRACT_COLS = TEP_INDEX_COLS + TEP_LABEL_COLS + TEP_PROCESS_COLS
TEP_RAW_ID_COLS = ["faultNumber", "simulationRun", "sample"]
RAW_TO_CONTRACT = {"faultNumber": "fault_scenario", "simulationRun": "run_id", "sample": "cycle"}

# Fault-onset sample per split, from the dataset specification (3-min sampling;
# fault introduced after 1 h in training, after 8 h in testing).
TEP_ONSET = {"training": 21, "testing": 161}
TEP_RUN_LENGTH = {"training": 500, "testing": 960}
TEP_N_RUNS = 500
TEP_N_FAULTS = 20

SPLIT_CODE = {"training": 0, "testing": 1}
ENGINE_ID_BASE = 1000        # fault_scenario * 1000 + run_id   (run_id <= 500)
ENGINE_ID_SPLIT_BASE = 100_000

TEP_RAW_FILES = {
    ("training", False): "TEP_FaultFree_Training.RData",
    ("testing", False): "TEP_FaultFree_Testing.RData",
    ("training", True): "TEP_Faulty_Training.RData",
    ("testing", True): "TEP_Faulty_Testing.RData",
}
TEP_PROCESSED_FILES = {
    ("training", False): "fault_free_training.parquet",
    ("testing", False): "fault_free_testing.parquet",
    ("training", True): "faulty_training.parquet",
    ("testing", True): "faulty_testing.parquet",
}
DIVERGENCE_META_FILE = "divergence_meta.parquet"
MANIFEST_FILE = "_manifest.json"
ROW_GROUP_EPISODES = 100   # rows per parquet row group = 100 episodes (50k training / 96k testing)


def make_engine_id(split: str, fault_scenario, run_id):
    """Globally unique episode id; accepts scalars or numpy arrays."""
    return SPLIT_CODE[split] * ENGINE_ID_SPLIT_BASE + np.asarray(fault_scenario) * ENGINE_ID_BASE + np.asarray(run_id)


def decode_engine_id(engine_id: int) -> tuple[str, int, int]:
    """Inverse of make_engine_id -> (split, fault_scenario, run_id)."""
    split = {v: k for k, v in SPLIT_CODE.items()}[engine_id // ENGINE_ID_SPLIT_BASE]
    rest = engine_id % ENGINE_ID_SPLIT_BASE
    return split, rest // ENGINE_ID_BASE, rest % ENGINE_ID_BASE


def _peak_rss_gb() -> float:
    """Peak resident set size of this process in GB (macOS reports bytes, Linux KB)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9


def _log(msg: str) -> None:
    print(f"{msg}  [peak rss {_peak_rss_gb():.2f} GB]", flush=True)


# Raw loading

def raw_identity(data_dir: str, split: str, faulty: bool) -> dict:
    """Identity of the raw file a sidecar was derived from (size + mtime, cheap)."""
    path = Path(data_dir) / TEP_RAW_FILES[(split, faulty)]
    if not path.exists():
        raise FileNotFoundError(path)
    st = path.stat()
    return {"raw_file": path.name, "raw_size": st.st_size, "raw_mtime": int(st.st_mtime)}


def load_raw(data_dir: str, split: str, faulty: bool) -> pd.DataFrame:
    """Load one RData file as-is (raw column names), id columns cast to int64."""
    import pyreadr  # local import: only the conversion step needs it

    path = Path(data_dir) / TEP_RAW_FILES[(split, faulty)]
    if not path.exists():
        raise FileNotFoundError(path)
    result = pyreadr.read_r(str(path))
    (_, df), = result.items()
    del result
    for c in TEP_RAW_ID_COLS:
        df[c] = df[c].to_numpy().astype(np.int64)
    missing = [c for c in TEP_PROCESS_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing process columns {missing}")
    return df


# Contract conversion (in place: no whole-frame copy)

def _is_sorted_by_episode(engine_id: np.ndarray, cycle: np.ndarray) -> bool:
    key = engine_id.astype(np.int64) * (max(TEP_RUN_LENGTH.values()) + 1) + cycle.astype(np.int64)
    return bool(np.all(np.diff(key) > 0))


def to_contract(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Rename/derive contract columns on the raw frame itself and return it.

    The returned object is the same DataFrame (mutated). Column order is not
    changed here; `write_contract` applies the contract order at write time.
    """
    if split not in TEP_ONSET:
        raise ValueError(f"split must be one of {list(TEP_ONSET)}, got {split!r}")
    df.rename(columns=RAW_TO_CONTRACT, inplace=True)
    fault = df["fault_scenario"].to_numpy()
    run = df["run_id"].to_numpy()
    cycle = df["cycle"].to_numpy()
    df["engine_id"] = make_engine_id(split, fault, run).astype(np.int64)
    df["is_fault_active"] = (fault > 0) & (cycle >= TEP_ONSET[split])
    df["split"] = pd.Categorical.from_codes(np.full(len(df), SPLIT_CODE[split], dtype=np.int8),
                                            categories=list(SPLIT_CODE))

    if not _is_sorted_by_episode(df["engine_id"].to_numpy(), cycle):
        _log("[warn] raw rows not in (engine_id, cycle) order; sorting (this copies the frame)")
        order = np.lexsort((cycle, df["engine_id"].to_numpy()))
        df = df.take(order)
        df.reset_index(drop=True, inplace=True)
    return df


def validate_contract(df: pd.DataFrame, split: str, faulty: bool) -> dict:
    """Structural checks on the (verified-sorted) contract frame. Raises on violation.

    Uses the block structure: with rows sorted by (engine_id, cycle) and every
    run exactly n_len long, reshaping to (n_runs, n_len) must give a constant
    engine_id per row and cycles 1..n_len in every row. That also rules out
    duplicate (engine_id, cycle) pairs and missing cycles without a groupby.
    """
    n_len = TEP_RUN_LENGTH[split]
    n = len(df)
    if n % n_len:
        raise ValueError(f"row count {n} is not a multiple of run length {n_len}")
    n_runs = n // n_len

    cyc = df["cycle"].to_numpy().reshape(n_runs, n_len)
    expected_cycles = np.arange(1, n_len + 1)
    bad_rows = np.where((cyc != expected_cycles).any(axis=1))[0]
    if len(bad_rows):
        raise ValueError(f"cycles are not exactly 1..{n_len} in every run (first bad run block {int(bad_rows[0])})")

    eid = df["engine_id"].to_numpy().reshape(n_runs, n_len)
    if not (eid == eid[:, :1]).all():
        raise ValueError("a run block contains more than one engine_id")
    ids = eid[:, 0]
    if len(np.unique(ids)) != len(ids):
        raise ValueError("duplicate engine_id blocks")

    fault_per_run = df["fault_scenario"].to_numpy().reshape(n_runs, n_len)[:, 0]
    faults = sorted(set(int(f) for f in np.unique(fault_per_run)))
    expected_faults = list(range(1, TEP_N_FAULTS + 1)) if faulty else [0]
    if faults != expected_faults:
        raise ValueError(f"fault_scenario values {faults} != expected {expected_faults}")
    run_per_run = df["run_id"].to_numpy().reshape(n_runs, n_len)[:, 0]
    for f in faults:
        n_r = len(np.unique(run_per_run[fault_per_run == f]))
        if n_r != TEP_N_RUNS:
            raise ValueError(f"fault {f}: {n_r} runs != {TEP_N_RUNS}")

    nan = 0
    for c in TEP_PROCESS_COLS:           # column by column: no 52-column copy
        nan += int(np.isnan(df[c].to_numpy()).sum())
    if nan:
        raise ValueError(f"{nan} NaN cells in process columns")

    active = int(df["is_fault_active"].to_numpy().sum())
    expected_active = 0 if not faulty else n_runs * (n_len - TEP_ONSET[split] + 1)
    if active != expected_active:
        raise ValueError(f"is_fault_active rows {active} != expected {expected_active}")

    return {"rows": int(n), "engines": int(n_runs), "faults": faults,
            "run_length": n_len, "active_rows": active}


def write_contract(df: pd.DataFrame, path: Path, split: str, compression: str = "zstd") -> None:
    """Write atomically (temp file + rename) with the contract column order applied by pyarrow."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pandas(df, columns=TEP_CONTRACT_COLS, preserve_index=False)
    pq.write_table(table, tmp, compression=compression,
                   row_group_size=ROW_GROUP_EPISODES * TEP_RUN_LENGTH[split])
    del table
    os.replace(tmp, path)


def verify_written(path: Path, split: str, faulty: bool, expected_rows: int) -> dict:
    """Read the label columns back from the written parquet and re-check the contract row by row.

    Independent of the in-memory frame: block structure (fixed-length episodes,
    one engine_id per block, cycles 1..n), engine_id <-> (split, fault, run)
    correspondence, and is_fault_active == (fault > 0) & (cycle >= onset) on
    every row.
    """
    lab = pd.read_parquet(path, columns=TEP_INDEX_COLS + ["fault_scenario", "run_id", "is_fault_active"])
    n_len = TEP_RUN_LENGTH[split]
    if len(lab) != expected_rows or len(lab) % n_len:
        raise ValueError(f"{path.name}: {len(lab)} rows read back, expected {expected_rows}")
    n_runs = len(lab) // n_len
    cyc = lab["cycle"].to_numpy().reshape(n_runs, n_len)
    if (cyc != np.arange(1, n_len + 1)).any():
        raise ValueError(f"{path.name}: cycles not 1..{n_len} in every block after write")
    eid = lab["engine_id"].to_numpy().reshape(n_runs, n_len)
    fault = lab["fault_scenario"].to_numpy().reshape(n_runs, n_len)
    run = lab["run_id"].to_numpy().reshape(n_runs, n_len)
    if not ((eid == eid[:, :1]).all() and (fault == fault[:, :1]).all() and (run == run[:, :1]).all()):
        raise ValueError(f"{path.name}: a block mixes engine_id / fault / run")
    if not np.array_equal(eid[:, 0], make_engine_id(split, fault[:, 0], run[:, 0])):
        raise ValueError(f"{path.name}: engine_id does not match (split, fault, run)")
    if len(np.unique(eid[:, 0])) != n_runs:
        raise ValueError(f"{path.name}: duplicate episodes after write")
    expected_active = (lab["fault_scenario"].to_numpy() > 0) & (lab["cycle"].to_numpy() >= TEP_ONSET[split])
    if not np.array_equal(lab["is_fault_active"].to_numpy(), expected_active):
        raise ValueError(f"{path.name}: is_fault_active disagrees with (fault > 0) & (cycle >= onset) row-wise")
    expected_faults = list(range(1, TEP_N_FAULTS + 1)) if faulty else [0]
    if sorted(set(int(f) for f in np.unique(fault[:, 0]))) != expected_faults:
        raise ValueError(f"{path.name}: fault set mismatch after write")
    return {"rows_read_back": int(len(lab)), "episodes_read_back": int(n_runs)}


# Divergence metadata (ST0-08: metadata only, never a label)

def compute_divergence_meta(df_faulty: pd.DataFrame, df_fault_free: pd.DataFrame, split: str) -> pd.DataFrame:
    """First cycle at which each faulty run differs from the fault-free run with the same run_id.

    Both frames must be validated contract frames of the same split (rows in
    (engine_id, cycle) order). Processes one fault block at a time so the only
    large temporary is one fault's (500, n_len, 52) array.
    """
    n_len = TEP_RUN_LENGTH[split]
    n_cols = len(TEP_PROCESS_COLS)

    ff_runs = df_fault_free["run_id"].to_numpy()[::n_len]
    ff_arr = df_fault_free[TEP_PROCESS_COLS].to_numpy().reshape(len(ff_runs), n_len, n_cols)
    run_pos = {int(r): i for i, r in enumerate(ff_runs)}

    fault_col = df_faulty["fault_scenario"].to_numpy()
    run_col = df_faulty["run_id"].to_numpy()
    rows = []
    for fault in sorted(set(int(f) for f in np.unique(fault_col))):
        lo = int(np.searchsorted(fault_col, fault, side="left"))
        hi = int(np.searchsorted(fault_col, fault, side="right"))
        runs = run_col[lo:hi][::n_len]
        arr = df_faulty.iloc[lo:hi][TEP_PROCESS_COLS].to_numpy().reshape(len(runs), n_len, n_cols)
        ref = ff_arr[[run_pos[int(r)] for r in runs]]
        diff_any = (arr != ref).any(axis=2)                    # (n_runs, n_len)
        has_diff = diff_any.any(axis=1)
        first_cycle = np.where(has_diff, diff_any.argmax(axis=1) + 1, -1)
        rows.append(pd.DataFrame({
            "split": split,
            "fault_scenario": fault,
            "run_id": runs.astype(np.int64),
            "engine_id": make_engine_id(split, fault, runs.astype(np.int64)),
            "spec_onset_cycle": TEP_ONSET[split],
            "observed_divergence_cycle": first_cycle.astype(np.int64),
        }))
        del arr, ref, diff_any
    del ff_arr
    return pd.concat(rows, ignore_index=True)


# Completion bookkeeping

def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".done.json")


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _options(split: str, compression: str) -> dict:
    return {"loader_version": TEP_LOADER_VERSION, "onset": TEP_ONSET[split],
            "run_length": TEP_RUN_LENGTH[split], "compression": compression,
            "row_group_episodes": ROW_GROUP_EPISODES, "columns": TEP_CONTRACT_COLS}


def _write_sidecar(path: Path, stats: dict, identity: dict, options: dict) -> None:
    payload = {"file": path.name, "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "size_bytes": path.stat().st_size, **identity, **options, **stats}
    _atomic_write_json(_sidecar(path), payload)


def _valid_sidecar(path: Path, identity: dict, options: dict) -> Optional[dict]:
    """Sidecar content if it exists AND matches the current raw file, options and parquet size."""
    sc = _sidecar(path)
    if not (sc.exists() and path.exists()):
        return None
    payload = json.loads(sc.read_text(encoding="utf-8"))
    for k, v in {**identity, **options}.items():
        if payload.get(k) != v:
            return None
    if payload.get("size_bytes") != path.stat().st_size:
        return None
    return payload


def is_complete(output_dir: str, splits: Iterable[str] = ("training", "testing")) -> bool:
    out = Path(output_dir)
    if not (out / MANIFEST_FILE).exists():
        return False
    manifest = json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest.get("loader_version") != TEP_LOADER_VERSION:
        return False
    needed = [TEP_PROCESSED_FILES[(s, f)] for s in splits for f in (False, True)] + [DIVERGENCE_META_FILE]
    for name in needed:
        path = out / name
        if name not in manifest["files"] or not path.exists() or not _sidecar(path).exists():
            return False
        if json.loads(_sidecar(path).read_text(encoding="utf-8")).get("size_bytes") != path.stat().st_size:
            return False
    return True


def _require_complete(output_dir: str, split: str) -> None:
    if not is_complete(output_dir, splits=(split,)):
        raise RuntimeError(f"{output_dir}: processed TEP data incomplete for split {split!r} "
                           f"(no {MANIFEST_FILE} or files missing). Run: python -m core_pipeline.data.tep_loader")


# Processed-data access

def load_processed(output_dir: str, split: str, faulty: bool,
                   columns: Optional[Iterable[str]] = None,
                   engine_ids: Optional[Iterable[int]] = None) -> pd.DataFrame:
    """Read one processed parquet, optionally a column subset and/or a set of engine_ids."""
    _require_complete(output_dir, split)
    path = Path(output_dir) / TEP_PROCESSED_FILES[(split, faulty)]
    cols = None if columns is None else list(dict.fromkeys(list(TEP_INDEX_COLS) + list(columns)))
    filters = None if engine_ids is None else [("engine_id", "in", [int(e) for e in engine_ids])]
    return pd.read_parquet(path, columns=cols, filters=filters)


def load_split(output_dir: str, split: str, columns: Optional[Iterable[str]] = None,
               engine_ids: Optional[Iterable[int]] = None) -> pd.DataFrame:
    """Fault-free + faulty of one split, concatenated. Large for testing (~4 GB) if unfiltered."""
    parts = [load_processed(output_dir, split, faulty, columns, engine_ids) for faulty in (False, True)]
    return pd.concat(parts, ignore_index=True)


def load_divergence_meta(output_dir: str) -> pd.DataFrame:
    if not (Path(output_dir) / MANIFEST_FILE).exists():
        raise RuntimeError(f"{output_dir}: no {MANIFEST_FILE}; run tep_loader first")
    return pd.read_parquet(Path(output_dir) / DIVERGENCE_META_FILE)


# Pipeline entry point

def preprocess_tep(data_dir: str, output_dir: str,
                   splits: Iterable[str] = ("training", "testing"),
                   compression: str = "zstd", force: bool = False) -> dict:
    """Convert raw RData to contract parquet files plus divergence metadata.

    One raw file in memory at a time. Files with a valid `.done.json` sidecar are
    skipped unless `force`. The manifest is written only after every requested
    file and the divergence metadata exist.
    """
    splits = list(splits)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST_FILE
    if manifest_path.exists():
        # Any (re)run invalidates the completion marker until everything is verified again.
        manifest_path.unlink()
        _log(f"[init] removed existing {MANIFEST_FILE}; will be rewritten only on full success")
    stats: dict = {}
    meta_parts: list[pd.DataFrame] = []
    meta_path = out_dir / DIVERGENCE_META_FILE
    meta_identity = {"raw_file": "paired", "raw_size": 0, "raw_mtime": 0}
    meta_options = {"loader_version": TEP_LOADER_VERSION, "onset": TEP_ONSET, "compression": compression}
    existing_meta = None
    if not force and _valid_sidecar(meta_path, meta_identity, meta_options) is not None:
        existing_meta = pd.read_parquet(meta_path)

    for split in splits:
        frames: dict[bool, Optional[pd.DataFrame]] = {False: None, True: None}
        for faulty in (False, True):
            path = out_dir / TEP_PROCESSED_FILES[(split, faulty)]
            name = f"{split}/{'faulty' if faulty else 'fault_free'}"
            identity = raw_identity(data_dir, split, faulty)
            options = _options(split, compression)
            sc = None if force else _valid_sidecar(path, identity, options)
            if sc is not None:
                stats[name] = {k: sc[k] for k in ("rows", "engines", "faults", "run_length", "active_rows")}
                _log(f"[{name}] skip (sidecar matches raw file, options and parquet size) {stats[name]}")
                continue
            t0 = time.time()
            df = to_contract(load_raw(data_dir, split, faulty), split)
            _log(f"[{name}] loaded+converted in {time.time() - t0:.0f} s")
            stats[name] = validate_contract(df, split, faulty)
            _log(f"[{name}] validated {stats[name]}")
            write_contract(df, path, split, compression)
            _log(f"[{name}] written {path.name} ({path.stat().st_size / 1e6:.0f} MB)")
            stats[name].update(verify_written(path, split, faulty, stats[name]["rows"]))
            _write_sidecar(path, stats[name], identity, options)
            _log(f"[{name}] read-back verified, sidecar written ({time.time() - t0:.0f} s total)")
            frames[faulty] = df

        need_meta = force or existing_meta is None or not (existing_meta["split"] == split).any()
        if need_meta:
            t0 = time.time()
            for faulty in (False, True):
                if frames[faulty] is None:   # file was skipped: reload from parquet for the comparison
                    frames[faulty] = pd.read_parquet(out_dir / TEP_PROCESSED_FILES[(split, faulty)])
                    _log(f"[{split}/divergence] reloaded {'faulty' if faulty else 'fault_free'} from parquet")
            meta = compute_divergence_meta(frames[True], frames[False], split)
            meta_parts.append(meta)
            _log(f"[{split}/divergence] {len(meta)} runs, observed cycle range "
                 f"{int(meta['observed_divergence_cycle'].min())}..{int(meta['observed_divergence_cycle'].max())} "
                 f"({time.time() - t0:.0f} s)")
            del meta
        else:
            meta_parts.append(existing_meta[existing_meta["split"] == split])
            _log(f"[{split}/divergence] reused existing metadata")

        frames.clear()
        gc.collect()
        _log(f"[{split}] done, frames released")

    meta_all = pd.concat(meta_parts, ignore_index=True)
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    meta_all.to_parquet(tmp, index=False, compression=compression)
    os.replace(tmp, meta_path)
    _write_sidecar(meta_path, {"rows": int(len(meta_all)), "splits": sorted(meta_all["split"].unique().tolist())},
                   meta_identity, meta_options)
    stats["divergence_meta_rows"] = int(len(meta_all))

    manifest = {
        "loader_version": TEP_LOADER_VERSION,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_dir": str(data_dir),
        "splits": splits,
        "onset": TEP_ONSET,
        "files": {TEP_PROCESSED_FILES[(s, f)]: stats[f"{s}/{'faulty' if f else 'fault_free'}"]
                  for s in splits for f in (False, True)},
    }
    manifest["files"][DIVERGENCE_META_FILE] = {"rows": stats["divergence_meta_rows"]}
    manifest["peak_rss_gb"] = round(_peak_rss_gb(), 2)
    _atomic_write_json(manifest_path, manifest)
    _log(f"[done] manifest written: {manifest_path}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert raw TEP RData files to contract parquet.")
    ap.add_argument("--data-dir", default="data/raw/TEP")
    ap.add_argument("--output-dir", default="data/processed/TEP")
    ap.add_argument("--splits", nargs="+", default=["training", "testing"], choices=list(TEP_ONSET))
    ap.add_argument("--force", action="store_true", help="ignore existing .done.json sidecars and redo every file")
    args = ap.parse_args()
    preprocess_tep(args.data_dir, args.output_dir, splits=args.splits, force=args.force)


if __name__ == "__main__":
    main()
