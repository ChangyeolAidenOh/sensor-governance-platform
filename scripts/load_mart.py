"""
scripts/load_mart.py — Stage 1 (S1-02 rev 2): derive staging.episodes + mart.windows + mart.window_labels from the
frozen split (outputs/stage0_episode_split.json) and register the parquet identity actually in use.

Rules (ST1-01..03):
  * constants come from core_pipeline.prereg_constants (git-frozen), never from the DB or this file;
  * n = 60 by default; n = 40 only with --n 40 --n-reduction-decision <ID> (prereg §3.1 time contract);
  * episode_id is an opaque uuid minted once per engine_id in staging.episode_keys and NEVER re-minted;
  * only the derived tables are rebuilt (episodes, windows, labels). If the derived grid would change and any
    execution table already references it, the load is REFUSED (logged, exit 2). If the grid is unchanged the
    load is a no-op that only appends a load_log row (action='unchanged'). Checkpointer tables are never touched;
  * the parquet identity AND content (sha256 of every parquet's bytes) are recorded in raw.tep_files; mart.load_log
    stores input_contract_sha256 = sha256(grid, split, manifest, content shas, loader/schema/prereg versions,
    window/stride, n). "unchanged" means the whole contract is equal to the last ok load, not just the grid;
    a contract change with existing execution rows is refused (old alarms must not attach to new data);
  * --n-reduction-decision must name a real entry in docs/decision_log.md that records the 60→40 time contract.

Usage: python scripts/load_mart.py --dsn $SGP_PG_DSN [--n 40 --n-reduction-decision ST5-01]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_pipeline import prereg_constants as P  # noqa: E402

SCHEMA_VERSION = 3
EXECUTION_TABLES = ["mart.alarms", "mart.proposals", "mart.approvals", "mart.work_orders",
                    "mart.replay_runs", "mart.replay_window_log", "mart.agent_steps"]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def derive(split_doc: dict, n: int):
    assert split_doc["seed"] == P.SEED, f"split seed {split_doc['seed']} != prereg {P.SEED}"
    assert split_doc["split"] == P.SPLIT and split_doc["run_length"] == P.RUN_LENGTH
    assert split_doc["spec_onset_cycle"] == P.ONSET_CYCLE
    groups = split_doc["groups"]
    seen: dict[int, str] = {}
    for g, spec in groups.items():
        for r in spec["run_ids"]:
            assert r not in seen, f"run_id {r} in both {seen[r]} and {g}"
            seen[r] = g
    assert groups["evaluation_pool"]["ordered"] is True
    eval_rank = {r: i for i, r in enumerate(groups["evaluation_pool"]["run_ids"])}
    demo = {(f, r): k for k, f, r in P.DEMO_PRESETS}
    for k, f, r in P.DEMO_PRESETS:
        assert r in groups["demo_pool"]["run_ids"], f"demo preset {k} run {r} not in demo_pool"

    episodes, windows, labels = [], [], []
    for g, spec in groups.items():
        for r in spec["run_ids"]:
            for f in range(0, P.N_FAULTS + 1):
                eid = P.engine_id(P.SPLIT, f, r)
                rank = eval_rank.get(r) if g == "evaluation_pool" else None
                episodes.append((eid, P.SPLIT, r, f, P.RUN_LENGTH, g, rank,
                                 rank is not None and rank < n, demo.get((f, r)) if g == "demo_pool" else None))
                for k in range(P.N_WINDOWS):
                    s = k * P.STRIDE + 1
                    windows.append((eid, k, s, s + P.WINDOW_SIZE - 1))
                    labels.append((eid, k, f, f > 0 and k >= P.ONSET_WINDOW))
    h = hashlib.sha256()
    for row in episodes + windows + labels:
        h.update(repr(row).encode())
    return episodes, windows, labels, h.hexdigest()


def copy_rows(cur, table, cols, rows):
    buf = io.StringIO()
    for row in rows:
        buf.write("\t".join("\\N" if v is None else "t" if v is True else "f" if v is False else str(v) for v in row) + "\n")
    buf.seek(0)
    with cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN") as cp:
        cp.write(buf.read())


def dvc_lock_md5(out_dir: Path):
    """md5 DVC recorded for the processed out directory in dvc.lock (provenance only; content check is sha256)."""
    lock = Path("dvc.lock")
    if not lock.exists():
        return None
    try:
        import yaml
        doc = yaml.safe_load(lock.read_text())
        for st in doc.get("stages", {}).values():
            for o in st.get("outs", []):
                if Path(o.get("path", "")).resolve() == out_dir.resolve():
                    return o.get("md5")
    except Exception:
        return None
    return None


def input_contract_sha(grid_sha, split_sha, ident, n):
    h = hashlib.sha256()
    for part in [grid_sha, split_sha, ident["manifest_sha256"] if ident else "", ident["loader_version"] if ident else "",
                 *(f["file_name"] + ":" + f["content_sha256"] for f in (ident["files"] if ident else [])),
                 str(SCHEMA_VERSION), P.PREREG_COMMIT, str(P.WINDOW_SIZE), str(P.STRIDE), str(n)]:
        h.update(str(part).encode() + b"\x00")
    return h.hexdigest()


def decision_authorises_reduction(decision_id: str, log_path: Path) -> bool:
    """The ID must head a real decision-log entry whose body records the 60→40 time contract."""
    if not log_path.exists():
        return False
    text = log_path.read_text()
    m = re.search(rf"^### {re.escape(decision_id)} \|.*?(?=^### |\Z)", text, re.S | re.M)
    return bool(m) and "40" in m.group(0) and "시간 계약" in m.group(0)


def parquet_identity(manifest_path: Path, dvc_file: Path):
    """Identity of the processed parquet set: manifest sha, per-file sidecars, DVC outs md5."""
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    out_dir = manifest_path.parent
    files = []
    for fname, st in manifest["files"].items():
        if not fname.endswith(".parquet") or fname.startswith("divergence"):
            continue
        sc_path = out_dir / f"{fname}.done.json"
        sc = json.loads(sc_path.read_text()) if sc_path.exists() else {}
        pq = out_dir / fname
        if not pq.exists():
            raise FileNotFoundError(f"manifest lists {fname} but it is missing in {out_dir}")
        files.append({"file_name": fname, "split": "training" if "training" in fname else "testing",
                      "faulty": fname.startswith("faulty"), "rows": st.get("rows"), "episodes": st.get("engines"),
                      "size_bytes": sc.get("size_bytes"), "written_at": sc.get("written_at"),
                      "raw_size": sc.get("raw_size"), "raw_mtime": sc.get("raw_mtime"),
                      "content_sha256": sha256_file(pq)})            # full-file hash: once per load, not per window
    dvc_md5 = dvc_lock_md5(out_dir)
    return {"manifest_sha256": sha256_file(manifest_path), "loader_version": manifest.get("loader_version"),
            "dvc_md5": dvc_md5, "files": files}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--split", default="outputs/stage0_episode_split.json")
    ap.add_argument("--manifest", default="data/processed/TEP/_manifest.json")
    ap.add_argument("--dvc-file", default="data/processed/TEP.dvc")
    ap.add_argument("--schema", default="sql/001_schema.sql")
    ap.add_argument("--n", type=int, default=P.N_EVAL_DEFAULT)
    ap.add_argument("--n-reduction-decision", default=None, help="decision ID authorising n=40 (§3.1 time contract)")
    ap.add_argument("--receipt", default=None, help="write the load_log row as JSON here (DVC out)")
    ap.add_argument("--decision-log", default="docs/decision_log.md")
    a = ap.parse_args()

    if a.n not in P.N_EVAL_ALLOWED:
        print(f"refuse: n={a.n} not in pre-registered set {sorted(P.N_EVAL_ALLOWED)}"); return 2
    if a.n != P.N_EVAL_DEFAULT and not a.n_reduction_decision:
        print("refuse: n=40 requires --n-reduction-decision <decision ID> (prereg §3.1)"); return 2
    if a.n == P.N_EVAL_DEFAULT and a.n_reduction_decision:
        print("refuse: --n-reduction-decision given but n is the default"); return 2
    if a.n_reduction_decision and not decision_authorises_reduction(a.n_reduction_decision, Path(a.decision_log)):
        print(f"refuse: {a.n_reduction_decision} is not a decision-log entry recording the 60→40 시간 계약 ({a.decision_log})"); return 2

    split_path = Path(a.split)
    split_doc = json.loads(split_path.read_text())
    ident = parquet_identity(Path(a.manifest), Path(a.dvc_file))
    if ident is None:
        print(f"[warn] manifest not found at {a.manifest}; parquet identity NOT recorded (dev only)")
    episodes, windows, labels, grid_sha = derive(split_doc, a.n)
    split_sha = sha256_file(split_path)
    contract_sha = input_contract_sha(grid_sha, split_sha, ident, a.n)

    with psycopg.connect(a.dsn) as conn, conn.cursor() as cur:
        cur.execute(Path(a.schema).read_text())

        # mint opaque keys once; existing keys are kept
        cur.execute("CREATE TEMP TABLE tmp_eids (engine_id integer PRIMARY KEY) ON COMMIT DROP")
        copy_rows(cur, "tmp_eids", ["engine_id"], [(e[0],) for e in episodes])
        cur.execute("INSERT INTO staging.episode_keys (engine_id) SELECT engine_id FROM tmp_eids ON CONFLICT DO NOTHING")
        cur.execute("SELECT count(*) FROM staging.episode_keys")
        n_keys = cur.fetchone()[0]

        cur.execute("SELECT input_contract_sha256 FROM mart.load_log WHERE ok ORDER BY load_id DESC LIMIT 1")
        last = cur.fetchone()
        cur.execute(" UNION ALL ".join(f"SELECT '{t}', count(*) FROM {t}" for t in EXECUTION_TABLES))
        exec_counts = dict(cur.fetchall())
        exec_rows = sum(exec_counts.values())

        if last and last[0] == contract_sha:
            action = "unchanged"                      # whole input contract equal: no DML on derived tables
        elif exec_rows > 0:
            action = "refused"                        # contract changed (data, split, grid, versions or n) while execution rows reference it
        else:
            action = "rebuilt"
            # derived tables only (DELETE, not TRUNCATE: execution tables hold FKs to windows and must be untouched)
            cur.execute("DELETE FROM mart.window_labels"); cur.execute("DELETE FROM mart.windows"); cur.execute("DELETE FROM staging.episodes")
            cur.execute("CREATE TEMP TABLE tmp_ep (engine_id integer, split text, run_id integer, fault_scenario smallint, "
                        "run_length integer, grp text, eval_rank integer, eval_selected boolean, demo_preset smallint) ON COMMIT DROP")
            copy_rows(cur, "tmp_ep", ["engine_id", "split", "run_id", "fault_scenario", "run_length", "grp",
                                      "eval_rank", "eval_selected", "demo_preset"], episodes)
            cur.execute("""INSERT INTO staging.episodes (episode_id, engine_id, split, run_id, fault_scenario, run_length,
                             "group", eval_rank, eval_selected, demo_preset)
                           SELECT k.episode_id, t.engine_id, t.split, t.run_id, t.fault_scenario, t.run_length, t.grp,
                                  t.eval_rank, t.eval_selected, t.demo_preset
                           FROM tmp_ep t JOIN staging.episode_keys k USING (engine_id)""")
            cur.execute("CREATE TEMP TABLE tmp_w (engine_id integer, window_index smallint, start_cycle integer, end_cycle integer) ON COMMIT DROP")
            copy_rows(cur, "tmp_w", ["engine_id", "window_index", "start_cycle", "end_cycle"], windows)
            cur.execute("""INSERT INTO mart.windows SELECT k.episode_id, t.window_index, t.start_cycle, t.end_cycle
                           FROM tmp_w t JOIN staging.episode_keys k USING (engine_id)""")
            cur.execute("CREATE TEMP TABLE tmp_l (engine_id integer, window_index smallint, fault_scenario smallint, is_fault_active boolean) ON COMMIT DROP")
            copy_rows(cur, "tmp_l", ["engine_id", "window_index", "fault_scenario", "is_fault_active"], labels)
            cur.execute("""INSERT INTO mart.window_labels SELECT k.episode_id, t.window_index, t.fault_scenario, t.is_fault_active
                           FROM tmp_l t JOIN staging.episode_keys k USING (engine_id)""")
            if ident:
                cur.execute("DELETE FROM raw.tep_files")
                for f in ident["files"]:
                    cur.execute("""INSERT INTO raw.tep_files (file_name, split, faulty, rows, episodes, size_bytes, written_at,
                                     raw_size, raw_mtime, loader_version, manifest_sha256, content_sha256, dvc_md5)
                                   VALUES (%(file_name)s,%(split)s,%(faulty)s,%(rows)s,%(episodes)s,%(size_bytes)s,%(written_at)s,
                                           %(raw_size)s,%(raw_mtime)s,%(lv)s,%(ms)s,%(content_sha256)s,%(dm)s)""",
                                {**f, "lv": ident["loader_version"], "ms": ident["manifest_sha256"], "dm": ident["dvc_md5"]})

        cur.execute("""SELECT e."group", e.eval_selected, count(DISTINCT e.episode_id), count(w.*)
                       FROM staging.episodes e JOIN mart.windows w USING (episode_id) GROUP BY 1,2 ORDER BY 1,2""")
        counts = {f"{g}{'/selected' if s else ''}": {"episodes": ne, "windows": nw} for g, s, ne, nw in cur.fetchall()}
        counts["episode_keys"] = n_keys
        counts["execution_rows"] = exec_counts
        ok = action != "refused"
        if ok:
            sel = counts.get("evaluation_pool/selected", {})
            assert sel.get("episodes") == a.n * 21 and sel.get("windows") == a.n * 21 * P.N_WINDOWS, sel
            cur.execute("SELECT count(*) FROM mart.windows w LEFT JOIN mart.window_labels l USING (episode_id, window_index) WHERE l.episode_id IS NULL")
            assert cur.fetchone()[0] == 0, "windows without labels"

        cur.execute("""INSERT INTO mart.load_log (schema_version, prereg_commit, split_json_sha256, split_seed, manifest_sha256,
                         loader_version, dvc_md5, grid_sha256, input_contract_sha256, prereg_n, n_reduction_decision, window_size, stride, action, counts, ok)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING load_id, loaded_at""",
                    (SCHEMA_VERSION, P.PREREG_COMMIT, split_sha, split_doc["seed"],
                     ident["manifest_sha256"] if ident else None, ident["loader_version"] if ident else None,
                     ident["dvc_md5"] if ident else None, grid_sha, contract_sha, a.n, a.n_reduction_decision,
                     P.WINDOW_SIZE, P.STRIDE, action, json.dumps(counts), ok))
        load_id, loaded_at = cur.fetchone()
        conn.commit()

    stable_counts = {k: v for k, v in counts.items() if k != "execution_rows"}
    # receipt = DVC out: deterministic for an unchanged input contract (no load_id / timestamp / action), so
    # `dvc status` stays clean on idempotent re-runs. The volatile fields go to stdout only.
    receipt = {"ok": ok, "grid_sha256": grid_sha, "input_contract_sha256": contract_sha, "schema_version": SCHEMA_VERSION,
               "prereg_commit": P.PREREG_COMMIT, "prereg_n": a.n, "n_reduction_decision": a.n_reduction_decision,
               "manifest_sha256": ident["manifest_sha256"] if ident else None, "counts": stable_counts}
    if a.receipt and ok:
        Path(a.receipt).parent.mkdir(parents=True, exist_ok=True)
        Path(a.receipt).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**receipt, "load_id": load_id, "loaded_at": str(loaded_at), "action": action,
                      "execution_rows": counts["execution_rows"]}))
    if action == "refused":
        print("REFUSED: input contract changed (data/split/grid/version/n) but execution tables hold rows", exec_counts,
              "\n  → archive/clear execution tables deliberately (decision log), or load as a new data version.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
