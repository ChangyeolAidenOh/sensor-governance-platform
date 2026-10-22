"""S4a-06 first task — read-only check of the sgp_agent privileges the Streamlit demo process needs (ST4-24).

    python scripts/check_checkpoint_grants.py                       # admin DSN from SGP_PG_DSN (.env), role sgp_agent
    python scripts/check_checkpoint_grants.py --dsn "$SGP_TEST_PG_DSN"
    python scripts/check_checkpoint_grants.py --agent-dsn "$SGP_AGENT_PG_DSN"   # also probe as the role (SELECT ... LIMIT 0)

Reports, without changing anything:
  A. which SQL verbs the installed langgraph-checkpoint-postgres package runs on which public table (DELETE expected
     only inside delete_thread, which the demo never calls) — the ground truth for sql/009;
  B. existence + SELECT/INSERT/UPDATE/DELETE of the role on the four checkpoint tables and USAGE on schema public;
  C. every `mart.<table>` referenced by the modules that run INSIDE the demo process (core_pipeline/agent/*,
     skills/work_order.py for record_decision, policy/*): table privileges, and the column-level UPDATE grants where
     the table-level UPDATE is false (006/008 grant UPDATE per column);
  D. with --agent-dsn: a `SELECT ... LIMIT 0` on each of those tables as the role (read probe only).
  E. with --write-probe (needs --agent-dsn on a DB whose name contains "test"): as the role, PostgresSaver(conn) WITHOUT
     setup(): put a checkpoint on a fresh probe thread, read it back, put_writes, read again — the smallest real exercise of
     the saver's own SQL (upserts, any DELETE in put/put_writes). Probe rows stay (the role has no DELETE).
This is a DIAGNOSTIC listing (reviewer 2026-10-22): module-level SQL constants show as func=None, the regular saver
(__init__.py/base.py) and the shallow saver (shallow.py) are listed separately, and the GRANT candidates are a draft for
sql/009 — the completion criterion is the demo running as the role. Exit: 0 clean, 1 with --strict when any candidate or
probe failure remains, 2 usage error.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

CHECKPOINT_TABLES = ("checkpoint_migrations", "checkpoints", "checkpoint_blobs", "checkpoint_writes")
VERBS = ("SELECT", "INSERT", "UPDATE", "DELETE")
AGENT_PROCESS_GLOBS = ("core_pipeline/agent/*.py", "core_pipeline/skills/work_order.py", "core_pipeline/policy/*.py")
SQL_VERB_RE = re.compile(r"\b(INSERT\s+INTO|UPDATE|DELETE\s+FROM|FROM|JOIN)\s+((?:public\.)?checkpoint\w*|(?:public\.)?checkpoints)\b", re.I)
UPSERT_RE = re.compile(r"ON\s+CONFLICT\b.*?\bDO\s+UPDATE\b", re.I | re.S)


def scan_package_sql() -> list[dict]:
    """Verb/table pairs per statement. A statement that INSERTs into a table and contains ON CONFLICT ... DO UPDATE is
    also recorded as UPDATE on that table (the upsert needs the privilege). saver = regular | shallow | aio."""
    pkg = importlib.import_module("langgraph.checkpoint.postgres")
    root = Path(pkg.__file__).parent
    found = []
    for py in sorted(root.rglob("*.py")):
        text = py.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        saver = "shallow" if "shallow" in py.name else ("aio" if py.name == "aio.py" else "regular")
        func = None
        for i, line in enumerate(lines, 1):
            m = re.match(r"\s*(?:async\s+)?def\s+(\w+)", line)
            if m:
                func = m.group(1)
            for vm in SQL_VERB_RE.finditer(line):
                verb = vm.group(1).upper().split()[0]
                verb = {"FROM": "SELECT", "JOIN": "SELECT"}.get(verb, verb)
                table = vm.group(2).replace("public.", "").lower()
                found.append({"file": py.name, "line": i, "func": func, "saver": saver, "verb": verb, "table": table})
                if verb == "INSERT":
                    tail = "\n".join(lines[i - 1:i + 40])
                    tail = tail.split('"""', 2)[0] if tail.count('"""') >= 1 else tail
                    if UPSERT_RE.search(tail):
                        found.append({"file": py.name, "line": i, "func": func, "saver": saver, "verb": "UPDATE", "table": table, "via": "ON CONFLICT DO UPDATE"})
    return found


def scan_agent_process_tables(repo: Path) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = {}
    for pattern in AGENT_PROCESS_GLOBS:
        for py in sorted(repo.glob(pattern)):
            text = py.read_text(encoding="utf-8", errors="replace")
            for t in set(re.findall(r"\bmart\.(\w+)", text)):
                refs.setdefault(t, set()).add(str(py.relative_to(repo)))
    return refs


def table_privs(cur, role: str, schema: str, table: str) -> dict:
    cur.execute("SELECT to_regclass(%s) IS NOT NULL", (f"{schema}.{table}",))
    exists = cur.fetchone()[0]
    if not exists:
        return {"exists": False}
    out = {"exists": True}
    for v in VERBS:
        cur.execute("SELECT has_table_privilege(%s, %s, %s)", (role, f"{schema}.{table}", v))
        out[v] = cur.fetchone()[0]
    if not out["UPDATE"]:
        cur.execute("""SELECT column_name FROM information_schema.column_privileges
                       WHERE grantee=%s AND table_schema=%s AND table_name=%s AND privilege_type='UPDATE' ORDER BY column_name""",
                    (role, schema, table))
        out["UPDATE_columns"] = [r[0] for r in cur.fetchall()]
    return out


def write_probe(agent_dsn: str) -> dict:
    """As the role: PostgresSaver on one autocommit connection, NO setup(); put → get_tuple → put_writes → get_tuple."""
    import uuid
    from langgraph.checkpoint.base import empty_checkpoint
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    out: dict = {"thread_id": f"probe-{uuid.uuid4()}", "steps": []}
    try:
        with psycopg.connect(agent_dsn, autocommit=True, prepare_threshold=0, row_factory=dict_row) as conn:
            saver = PostgresSaver(conn)
            config = {"configurable": {"thread_id": out["thread_id"], "checkpoint_ns": ""}}
            cp = empty_checkpoint()
            cfg2 = saver.put(config, cp, {"source": "input", "step": -1, "parents": {}}, {})
            out["steps"].append("put ok")
            t = saver.get_tuple(config)
            out["steps"].append("get_tuple ok" if t is not None else "get_tuple returned None")
            saver.put_writes(cfg2, [("probe_channel", {"v": 1})], task_id="probe")
            out["steps"].append("put_writes ok")
            t2 = saver.get_tuple(config)
            out["steps"].append(f"get_tuple ok, pending_writes={len(t2.pending_writes or [])}" if t2 is not None else "get_tuple returned None")
            out["ok"] = t is not None and t2 is not None
    except Exception as e:
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
    return out


def main() -> int:
    load_dotenv(".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.environ.get("SGP_PG_DSN"), help="admin DSN (privilege lookup)")
    ap.add_argument("--role", default="sgp_agent")
    ap.add_argument("--agent-dsn", default=None, help="optional: probe SELECT ... LIMIT 0 as the role")
    ap.add_argument("--write-probe", action="store_true", help="with --agent-dsn on a *test* DB: put/get/put_writes a probe checkpoint as the role")
    ap.add_argument("--strict", action="store_true", help="exit 1 when any grant candidate or probe failure remains")
    ap.add_argument("--repo", default=".")
    a = ap.parse_args()
    a.dsn = a.dsn or os.environ.get("SGP_PG_DSN")               # an empty "--dsn $VAR" falls back to .env
    a.agent_dsn = a.agent_dsn or None
    if not a.dsn:
        print("SGP_PG_DSN is required", file=sys.stderr)
        return 2
    repo = Path(a.repo).resolve()
    report: dict = {"role": a.role}

    # A. installed package SQL
    pkg_sql = scan_package_sql()
    by_saver: dict[str, dict[str, dict[str, set[str]]]] = {}
    for f in pkg_sql:
        tag = f"{f['func']}@{f['file']}:{f['line']}" + (" (upsert)" if f.get("via") else "")
        by_saver.setdefault(f["saver"], {}).setdefault(f["table"], {}).setdefault(f["verb"], set()).add(tag)
    report["package_sql_by_saver"] = {sv: {t: {v: sorted(x) for v, x in vs.items()} for t, vs in sorted(ts.items())} for sv, ts in sorted(by_saver.items())}
    report["package_delete_sites_regular"] = sorted({f"{f['func']}@{f['file']}:{f['line']}" for f in pkg_sql if f["verb"] == "DELETE" and f["saver"] == "regular"})
    report["note"] = "func=None means a module-level SQL constant; which method runs it must be read from the source (dump)"

    with psycopg.connect(a.dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT current_database(), current_user, EXISTS(SELECT 1 FROM pg_roles WHERE rolname=%s)", (a.role,))
        db, me, role_exists = cur.fetchone()
        report.update({"database": db, "connected_as": me, "role_exists": role_exists})
        if not role_exists:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        # B. checkpoint tables
        cur.execute("SELECT has_schema_privilege(%s, 'public', 'USAGE'), has_schema_privilege(%s, 'mart', 'USAGE')", (a.role, a.role))
        report["schema_usage"] = dict(zip(("public", "mart"), cur.fetchone()))
        report["checkpoint_tables"] = {t: table_privs(cur, a.role, "public", t) for t in CHECKPOINT_TABLES}
        # C. mart tables referenced by the demo-process modules
        refs = scan_agent_process_tables(repo)
        report["mart_tables_referenced"] = {t: {"referenced_by": sorted(fs), **table_privs(cur, a.role, "mart", t)} for t, fs in sorted(refs.items())}
        cur.execute("""SELECT sequence_name, has_sequence_privilege(%s, 'mart.'||sequence_name, 'USAGE')
                       FROM information_schema.sequences WHERE sequence_schema='mart' ORDER BY 1""", (a.role,))
        report["mart_sequences_usage"] = {r[0]: r[1] for r in cur.fetchall()}

    # D. probe as the role — the probe connection must be the checked role on the checked database
    probe_failures = 0
    if a.agent_dsn:
        probe = {}
        with psycopg.connect(a.agent_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_user, current_database()")
                pu, pdb = cur.fetchone()
            report["probe_identity"] = {"user": pu, "database": pdb, "matches_role": pu == a.role, "matches_database": pdb == db}
            if pu != a.role or pdb != db:
                print(f"--agent-dsn connects as {pu}@{pdb}, expected {a.role}@{db}; probe skipped", file=sys.stderr)
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return 2
            for schema, tables in (("public", CHECKPOINT_TABLES), ("mart", tuple(report["mart_tables_referenced"]))):
                for t in tables:
                    try:
                        with conn.cursor() as cur:
                            cur.execute(f"SELECT 1 FROM {schema}.{t} LIMIT 0")
                        probe[f"{schema}.{t}"] = "ok"
                    except psycopg.Error as e:
                        conn.rollback()
                        probe[f"{schema}.{t}"] = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
        report["probe_as_role"] = probe
        probe_failures += sum(1 for v in probe.values() if v != "ok")
        if a.write_probe:
            if "test" not in db:
                print("--write-probe is allowed only on a database whose name contains 'test'", file=sys.stderr)
                return 2
            report["write_probe"] = write_probe(a.agent_dsn)
            probe_failures += 0 if report["write_probe"].get("ok") else 1

    # draft GRANT candidates (missing only; DELETE never proposed — the demo never calls delete_thread)
    grants = []
    for t in CHECKPOINT_TABLES:
        p = report["checkpoint_tables"][t]
        if not p.get("exists"):
            grants.append(f"-- public.{t} does not exist in {db}: run PostgresSaver.setup() ONCE as admin before granting")
            continue
        need = ["SELECT"] if t == "checkpoint_migrations" else ["SELECT", "INSERT", "UPDATE"]
        missing = [v for v in need if not p.get(v)]
        if missing:
            grants.append(f"GRANT {', '.join(missing)} ON public.{t} TO {a.role};")
    for t, p in report["mart_tables_referenced"].items():
        if not p.get("exists"):
            continue
        missing = [v for v in ("SELECT", "INSERT") if not p.get(v)]
        if missing:
            grants.append(f"-- mart.{t}: {', '.join(missing)} missing (check whether the demo path writes it before granting)")
        if not p.get("UPDATE"):
            grants.append(f"-- mart.{t}: table-level UPDATE false; column grants = {p.get('UPDATE_columns')} (compare with the SET lists in the referencing modules)")
    report["grant_candidates"] = grants
    print(json.dumps(report, ensure_ascii=False, indent=2))
    hard = [g for g in grants if g.startswith("GRANT")]
    return 1 if a.strict and (hard or probe_failures) else 0


if __name__ == "__main__":
    sys.exit(main())
