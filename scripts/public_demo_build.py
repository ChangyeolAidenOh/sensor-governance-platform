"""scripts/public_demo_build.py: build the public demo database locally, then move it to the hosted Postgres (demo-public).

    python scripts/public_demo_build.py local            # every local step in order (resume with --from <step>)
    python scripts/public_demo_build.py neon             # restore the dump on Neon, create roles, grants, check
    python scripts/public_demo_build.py local --from prune
    python scripts/public_demo_build.py local --only verify

Local steps (database sgp_public on the docker Postgres, derived from SGP_PG_DSN / SGP_AGENT_PG_DSN by swapping the dbname):
  createdb  -> load (scripts/load_mart.py: 001 + full grid; the loader has no subset option and is not modified)
  -> schema (003 004 005 007 010) -> roles (002 via psql, 006 008) -> saver (PostgresSaver.setup(), then 009, 011)
  -> verify (scripts/verify_mart.py on the FULL load: 3b compares group sizes with the frozen split)
  -> prune (scripts/public_demo_prune.py: keep presets 1..10 + one synthetic episode as preset 11)
  -> precompute (scripts/precompute_demo.py on the pruned DB, raw manifest to build/public/)
  -> manifest (scripts/public_demo_manifest.py -> outputs/demo_presets_public.json)
  -> prime (scripts/public_demo_prime.py through the real graph; bge models load locally)
  -> check (scripts/public_demo_check.py) -> dump (pg_dump -Fc --no-owner --no-privileges -> build/public/sgp_public.dump)
Neon steps (SGP_PUBLIC_NEON_ADMIN_DSN in .env, direct connection, database sgp_v2):
  restore (pg_restore --no-owner --no-acl) -> roles (002 + 011 with NEW passwords, written to build/public/neon_roles.env,
  plus 006 007 008 009 010 grants) -> check (public_demo_check with the agent and skill DSNs)
Rules: psql / pg_dump / pg_restore run inside the docker Postgres container (same major as the local server); every
local DSN must name sgp_public; any DSN whose dbname contains "eval" is refused; SGP_LLM_* is cleared for the priming.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.conninfo import conninfo_to_dict, make_conninfo

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "public"
LOCAL_DB = "sgp_public"
SYN_PRESET = 11
SCHEMA_FILES = ("sql/003_skill_state.sql", "sql/004_skill_state_rev2.sql", "sql/005_gate_checks.sql", "sql/007_graph_state.sql", "sql/010_policy_isolation.sql")
GRANT_FILES = ("sql/006_roles_stage4a.sql", "sql/007_graph_state.sql", "sql/008_roles_llm_ledger.sql", "sql/009_roles_checkpointer.sql", "sql/010_policy_isolation.sql")
LOCAL_STEPS = ("createdb", "load", "schema", "roles", "saver", "verify", "prune", "precompute", "manifest", "prime", "check", "dump")
NEON_STEPS = ("restore", "roles", "check")


def with_dbname(dsn: str, dbname: str, user: str | None = None, password: str | None = None) -> str:
    d = conninfo_to_dict(dsn)
    d["dbname"] = dbname
    if user is not None:
        d["user"] = user
    if password is not None:
        d["password"] = password
    return make_conninfo(**{k: v for k, v in d.items() if v is not None})


def guard(dsn: str, expect_db: str | None) -> str:
    name = conninfo_to_dict(dsn).get("dbname") or ""
    if "eval" in name:
        sys.exit(f"refusing to touch an evaluation database ({name})")
    if expect_db and name != expect_db:
        sys.exit(f"DSN names database {name!r}, expected {expect_db!r}")
    return dsn


def sh(cmd: list[str], env: dict | None = None, stdin_path: Path | None = None, stdout_path: Path | None = None) -> None:
    shown = [c for c in cmd if "password" not in c.lower() and "://" not in c]
    print("$", " ".join(shown[:8]), "..." if len(shown) > 8 else "")
    fin = open(stdin_path, "rb") if stdin_path else None
    fout = open(stdout_path, "wb") if stdout_path else None
    try:
        r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT), **(env or {})}, stdin=fin if fin else subprocess.DEVNULL, stdout=fout)
    finally:
        for f in (fin, fout):
            if f:
                f.close()
    if r.returncode != 0:
        sys.exit(f"step failed (rc={r.returncode}): {cmd[0]} {cmd[1] if len(cmd) > 1 else ''}")


def psql_docker(dsn: str, sql_file: Path, variables: dict | None = None) -> None:
    cmd = ["docker", "exec", "-i", "sgp_postgres", "psql", "-v", "ON_ERROR_STOP=1", dsn]
    for k, v in (variables or {}).items():
        cmd += ["-v", f"{k}={v}"]
    cmd += ["-f", "-"]
    sh(cmd, stdin_path=sql_file)


def run_sql_files(dsn: str, files: tuple[str, ...]) -> None:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        for f in files:
            cur.execute((ROOT / f).read_text(encoding="utf-8"))
            conn.commit()
            print(f"applied {f}")


def docker_dsn(dsn: str) -> str:
    """The local server as seen from inside the container (the compose service listens on 5432 there)."""
    d = conninfo_to_dict(dsn)
    if d.get("host") in ("localhost", "127.0.0.1"):
        d["host"], d["port"] = "localhost", "5432"
    return make_conninfo(**{k: v for k, v in d.items() if v is not None})


# ------------------------------------------------------------------------------------------------------ local
def local_steps(a) -> None:
    admin_src = os.environ.get("SGP_PG_DSN") or sys.exit("SGP_PG_DSN (local admin DSN) is required in .env")
    agent_src = os.environ.get("SGP_AGENT_PG_DSN") or sys.exit("SGP_AGENT_PG_DSN (local sgp_agent DSN) is required in .env")
    agent_pw = conninfo_to_dict(agent_src).get("password") or sys.exit("SGP_AGENT_PG_DSN has no password")
    admin = guard(with_dbname(admin_src, LOCAL_DB), LOCAL_DB)
    agent = guard(with_dbname(admin_src, LOCAL_DB, user="sgp_agent", password=agent_pw), LOCAL_DB)     # same host/port text as admin (check_dsns)
    BUILD.mkdir(parents=True, exist_ok=True)
    steps = select(LOCAL_STEPS, a)
    for step in steps:
        print(f"\n== local: {step}")
        if step == "createdb":
            with psycopg.connect(with_dbname(admin_src, "postgres"), autocommit=True) as conn, conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (LOCAL_DB,))
                if cur.fetchone():
                    if not a.reset:
                        sys.exit(f"{LOCAL_DB} already exists; pass --reset to drop it, or --from <step> to resume")
                    cur.execute(f"DROP DATABASE {LOCAL_DB} WITH (FORCE)")
                cur.execute(f"CREATE DATABASE {LOCAL_DB}")
            print(f"created {LOCAL_DB}")
        elif step == "load":
            sh([sys.executable, "scripts/load_mart.py", "--dsn", admin, "--receipt", str(BUILD / "mart_load_receipt.json")])
        elif step == "schema":
            run_sql_files(admin, SCHEMA_FILES)
        elif step == "roles":
            psql_docker(docker_dsn(admin), ROOT / "sql/002_roles.sql", {"agent_password": agent_pw})
            run_sql_files(admin, ("sql/006_roles_stage4a.sql", "sql/007_graph_state.sql", "sql/008_roles_llm_ledger.sql", "sql/010_policy_isolation.sql"))
        elif step == "saver":
            from langgraph.checkpoint.postgres import PostgresSaver
            with PostgresSaver.from_conn_string(admin) as saver:
                saver.setup()
            run_sql_files(admin, ("sql/009_roles_checkpointer.sql",))
            psql_docker(docker_dsn(admin), ROOT / "sql/011_roles_public_skill.sql", {"skill_password": agent_pw})
            print("checkpointer tables created; local sgp_skill shares the local agent password")
        elif step == "verify":
            sh([sys.executable, "scripts/verify_mart.py", "--dsn", admin, "--agent-dsn", agent])
        elif step == "prune":
            cmd = [sys.executable, "scripts/public_demo_prune.py", "--dsn", admin, "--synthetic-preset", str(SYN_PRESET), "--synthetic-fault", str(a.synthetic_fault)]
            if a.synthetic_run_id:
                cmd += ["--synthetic-run-id", str(a.synthetic_run_id)]
            sh(cmd)
        elif step == "precompute":
            presets = ",".join(str(p) for p in list(range(1, 11)) + [SYN_PRESET])
            sh([sys.executable, "-m", "scripts.precompute_demo", "--presets", presets, "--manifest", str(BUILD / "demo_presets_public_raw.json"),
                "--truth", str(BUILD / "truth_DO_NOT_COMMIT.json"), "--raw", str(BUILD / "demo_presets_public_windows.json"),
                "--goldens", "", "--thread", "demo-precompute-public"], env={"SGP_PG_DSN": admin})
        elif step == "manifest":
            sh([sys.executable, "scripts/public_demo_manifest.py", "--raw", str(BUILD / "demo_presets_public_raw.json"),
                "--out", "outputs/demo_presets_public.json", "--synthetic-presets", str(SYN_PRESET)])
        elif step == "prime":
            sh([sys.executable, "scripts/public_demo_prime.py", "--app-dsn", agent, "--skill-dsn", admin, "--manifest", "outputs/demo_presets_public.json",
                "--synthetic-presets", str(SYN_PRESET), "--report", str(BUILD / "prime_report.json")])
        elif step == "check":
            skill = with_dbname(admin_src, LOCAL_DB, user="sgp_skill", password=agent_pw)
            sh([sys.executable, "scripts/public_demo_check.py", "--dsn", admin, "--agent-dsn", agent, "--skill-dsn", skill, "--manifest", "outputs/demo_presets_public.json"])
        elif step == "dump":
            out = BUILD / "sgp_public.dump"
            sh(["docker", "exec", "-i", "sgp_postgres", "pg_dump", "-Fc", "--no-owner", "--no-privileges", "-d", docker_dsn(admin)], stdout_path=out)
            print(f"dump {out} ({out.stat().st_size / 1e6:.1f} MB)")


# ------------------------------------------------------------------------------------------------------ neon
def neon_steps(a) -> None:
    admin = os.environ.get("SGP_PUBLIC_NEON_ADMIN_DSN") or sys.exit("SGP_PUBLIC_NEON_ADMIN_DSN is required in .env (direct connection, database sgp_v2)")
    if "-pooler" in admin:
        sys.exit("use the direct Neon connection string (no -pooler host)")
    guard(admin, "sgp_v2")
    roles_env = BUILD / "neon_roles.env"
    steps = select(NEON_STEPS, a)
    for step in steps:
        print(f"\n== neon: {step}")
        if step == "restore":
            dump = BUILD / "sgp_public.dump"
            if not dump.is_file():
                sys.exit(f"{dump} not found; run the local build first")
            with psycopg.connect(admin, autocommit=True) as conn, conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM information_schema.schemata WHERE schema_name IN ('raw','staging','mart')")
                if cur.fetchone()[0]:
                    if not a.reset:
                        sys.exit("the Neon database already holds a build; pass --reset to drop raw/staging/mart and the checkpointer tables first")
                    cur.execute("DROP SCHEMA IF EXISTS raw, staging, mart CASCADE")
                    cur.execute("DROP TABLE IF EXISTS public.checkpoint_writes, public.checkpoint_blobs, public.checkpoints, public.checkpoint_migrations")
                    print("wiped the previous build (roles and their passwords are kept)")
            sh(["docker", "exec", "-i", "sgp_postgres", "pg_restore", "--no-owner", "--no-acl", "--no-comments", "--exit-on-error", "-d", admin], stdin_path=dump)
            with psycopg.connect(admin) as conn, conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM mart.episode_windows")
                print(f"restored: {cur.fetchone()[0]} episode_windows rows")
        elif step == "roles":
            if roles_env.is_file() and not a.reset:
                sys.exit(f"{roles_env} exists; the roles were created before (pass --reset to re-apply grants with the same passwords)")
            if roles_env.is_file():
                pw = dict(line.split("=", 1) for line in roles_env.read_text().splitlines() if "=" in line)
                agent_pw, skill_pw = pw["SGP_PUBLIC_NEON_AGENT_PASSWORD"], pw["SGP_PUBLIC_NEON_SKILL_PASSWORD"]
            else:
                agent_pw, skill_pw = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
            psql_docker(admin, ROOT / "sql/002_roles.sql", {"agent_password": agent_pw})
            run_sql_files(admin, GRANT_FILES)
            psql_docker(admin, ROOT / "sql/011_roles_public_skill.sql", {"skill_password": skill_pw})
            from urllib.parse import quote
            host = conninfo_to_dict(admin)["host"]
            agent_dsn = f"postgresql://sgp_agent:{quote(agent_pw, safe='')}@{host}/sgp_v2?sslmode=require&channel_binding=require"
            skill_dsn = f"postgresql://sgp_skill:{quote(skill_pw, safe='')}@{host}/sgp_v2?sslmode=require&channel_binding=require"
            roles_env.write_text("\n".join([f"SGP_PUBLIC_NEON_AGENT_PASSWORD={agent_pw}", f"SGP_PUBLIC_NEON_SKILL_PASSWORD={skill_pw}",
                                            f"SGP_PUBLIC_NEON_AGENT_DSN={agent_dsn}", f"SGP_PUBLIC_NEON_SKILL_DSN={skill_dsn}", ""]), encoding="utf-8")
            os.chmod(roles_env, 0o600)
            print(f"roles ready; DSNs written to {roles_env} (append the two DSN lines to .env; never commit)")
        elif step == "check":
            if not roles_env.is_file():
                sys.exit(f"{roles_env} not found; run the roles step first")
            pw = dict(line.split("=", 1) for line in roles_env.read_text().splitlines() if "=" in line)
            sh([sys.executable, "scripts/public_demo_check.py", "--dsn", admin, "--agent-dsn", pw["SGP_PUBLIC_NEON_AGENT_DSN"],
                "--skill-dsn", pw["SGP_PUBLIC_NEON_SKILL_DSN"], "--manifest", "outputs/demo_presets_public.json"])


def select(all_steps: tuple[str, ...], a) -> list[str]:
    if a.only:
        if a.only not in all_steps:
            sys.exit(f"unknown step {a.only}; choose from {all_steps}")
        return [a.only]
    start = all_steps.index(a.from_step) if a.from_step else 0
    return list(all_steps[start:])


def main() -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=("local", "neon"))
    ap.add_argument("--from", dest="from_step", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--reset", action="store_true", help="local: drop sgp_public first; neon roles: re-apply with the stored passwords")
    ap.add_argument("--synthetic-fault", type=int, default=1)
    ap.add_argument("--synthetic-run-id", type=int, default=None)
    a = ap.parse_args()
    if a.from_step and a.from_step not in (LOCAL_STEPS + NEON_STEPS):
        sys.exit(f"unknown step {a.from_step}")
    for k in list(os.environ):
        if k.startswith("SGP_LLM_"):
            os.environ.pop(k)
    (local_steps if a.target == "local" else neon_steps)(a)
    print("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
