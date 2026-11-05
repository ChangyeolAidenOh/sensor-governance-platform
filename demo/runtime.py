"""demo/runtime.py — configuration guards and the process-lifetime runtime of the Streamlit demo (ST4-24, ST4-30, ST4-31).

The demo process connects as `sgp_agent` (SGP_AGENT_PG_DSN) for everything it touches itself — session, turns,
checkpointer, ledger, plans, decisions. The two Skill servers are spawned by SyncSkillClient with SGP_PG_DSN in THEIR
environment only (the detect server needs the admin connection in development, 006 comment); the app never opens a
connection with it. Guards (load_config):
  * both DSNs required; same host/port/dbname; different users; app user == sgp_agent
  * SGP_DEMO_SYNTHETIC=1 (ST4-29) only when both dbnames contain "test"
  * manifest = SGP_DEMO_MANIFEST (default outputs/demo_presets.json), validated by core_pipeline.agent.manifest
The runtime (build_runtime) holds ONE SyncSkillClient (two server subprocesses), ONE psycopg ConnectionPool feeding a
PostgresSaver (no setup(): that is an admin step), the TurnStore / GraphDeps / AgentSession, and a lock that serialises
turns (single-operator demo). It is created once per process by app.py (st.cache_resource keyed on fingerprint()).

Public mode (demo-public branch, SGP_PUBLIC_DEMO=1): the same runtime with three differences and no server change.
  * only the skills_detect server is spawned (it serves the pre-computed windows and runs the work-order tools);
    `retrieve` is answered from stored rows by core_pipeline.skills.public_skills.StoredRetrieve (no bge models);
  * the connection pool keeps no idle connection (min_size=0) and checks a connection before handing it out, so the
    hosted Postgres can scale to zero between visitors and a suspended compute never surfaces a dead connection;
  * SGP_DEMO_SYNTHETIC_PRESETS=11 marks the labelled synthetic approval preset(s): their threads get the test-DB
    grounding override and the forced-propose policy (ST4-29 widened to the public demo); every other preset is real.
    This is the ONLY way to enable synthetic behaviour outside a test database and it requires public mode.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from psycopg.conninfo import conninfo_to_dict

APP_ROLE = "sgp_agent"
DEFAULT_MANIFEST = "outputs/demo_presets.json"
DEFAULT_OWNER = "engineer"


class DemoConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class DemoConfig:
    app_dsn: str
    skill_dsn: str
    manifest_path: str
    synthetic: bool
    dsn_facts: dict = field(default_factory=dict)      # host/port/dbname/users for the sidebar (no passwords)
    public: bool = False                                # SGP_PUBLIC_DEMO=1 (demo-public branch)
    synthetic_presets: tuple = ()                       # SGP_DEMO_SYNTHETIC_PRESETS, public mode only


def _facts(dsn: str, label: str) -> dict:
    try:
        d = conninfo_to_dict(dsn)
    except Exception as e:                              # the message never echoes the DSN
        raise DemoConfigError(f"{label}: cannot parse the DSN ({type(e).__name__})") from e
    return {"host": d.get("host"), "port": str(d.get("port") or ""), "dbname": d.get("dbname"), "user": d.get("user")}


def check_dsns(app_dsn: str, skill_dsn: str, synthetic: bool) -> dict:
    """ST4-30 guard: same target database, different users, app user is the agent role; synthetic → both test DBs."""
    a, s = _facts(app_dsn, "SGP_AGENT_PG_DSN"), _facts(skill_dsn, "SGP_PG_DSN")
    if (a["host"], a["port"], a["dbname"]) != (s["host"], s["port"], s["dbname"]):
        raise DemoConfigError("SGP_AGENT_PG_DSN and SGP_PG_DSN must point at the same host/port/dbname")
    if not a["user"] or a["user"] == s["user"]:
        raise DemoConfigError("SGP_AGENT_PG_DSN must use a different user from SGP_PG_DSN")
    if a["user"] != APP_ROLE:
        raise DemoConfigError(f"SGP_AGENT_PG_DSN must connect as {APP_ROLE} (got user {a['user']!r})")
    if synthetic and not ("test" in (a["dbname"] or "") and "test" in (s["dbname"] or "")):
        raise DemoConfigError("SGP_DEMO_SYNTHETIC=1 is allowed only when both DSNs name a database containing 'test'")
    return {"host": a["host"], "port": a["port"], "dbname": a["dbname"], "app_user": a["user"], "skill_user": s["user"]}


def load_config(environ: Optional[dict] = None) -> DemoConfig:
    env = os.environ if environ is None else environ
    app_dsn, skill_dsn = (env.get("SGP_AGENT_PG_DSN") or "").strip(), (env.get("SGP_PG_DSN") or "").strip()
    if not app_dsn:
        raise DemoConfigError("SGP_AGENT_PG_DSN is required (app role; see .env.example)")
    if not skill_dsn:
        raise DemoConfigError("SGP_PG_DSN is required (passed to the Skill servers only)")
    synthetic = (env.get("SGP_DEMO_SYNTHETIC") or "").strip() in ("1", "true", "yes")
    public = (env.get("SGP_PUBLIC_DEMO") or "").strip() in ("1", "true", "yes")
    raw_presets = (env.get("SGP_DEMO_SYNTHETIC_PRESETS") or "").strip()
    try:
        synthetic_presets = tuple(sorted({int(x) for x in raw_presets.split(",") if x.strip()}))
    except ValueError as e:
        raise DemoConfigError(f"SGP_DEMO_SYNTHETIC_PRESETS must be comma-separated preset numbers ({e})")
    if synthetic_presets and not public:
        raise DemoConfigError("SGP_DEMO_SYNTHETIC_PRESETS is allowed only with SGP_PUBLIC_DEMO=1 (ST4-29: the local demo uses SGP_DEMO_SYNTHETIC on a test DB)")
    if synthetic_presets and synthetic:
        raise DemoConfigError("SGP_DEMO_SYNTHETIC and SGP_DEMO_SYNTHETIC_PRESETS cannot be combined")
    facts = check_dsns(app_dsn, skill_dsn, synthetic)
    manifest_path = (env.get("SGP_DEMO_MANIFEST") or DEFAULT_MANIFEST).strip()
    if not Path(manifest_path).exists():
        raise DemoConfigError(f"demo manifest not found: {manifest_path}")
    return DemoConfig(app_dsn=app_dsn, skill_dsn=skill_dsn, manifest_path=manifest_path, synthetic=synthetic, dsn_facts=facts,
                      public=public, synthetic_presets=synthetic_presets)


def fingerprint(cfg: DemoConfig, environ: Optional[dict] = None) -> str:
    """Cache key for the process runtime: both DSNs, the manifest CONTENT, the LLM mode variables, the synthetic flag."""
    env = os.environ if environ is None else environ
    h = hashlib.sha256()
    h.update(cfg.app_dsn.encode()); h.update(b"|"); h.update(cfg.skill_dsn.encode()); h.update(b"|")
    h.update(Path(cfg.manifest_path).read_bytes()); h.update(b"|")
    for k in ("SGP_LLM_ROUTER", "SGP_LLM_NARRATOR", "SGP_LLM_MODEL", "SGP_LLM_BUDGET_USD", "SGP_TURN_LEASE_SECONDS", "SGP_MCP_TIMEOUT"):
        h.update(f"{k}={env.get(k, '')}".encode()); h.update(b"|")
    h.update(b"synthetic=1" if cfg.synthetic else b"synthetic=0"); h.update(b"|")
    h.update(b"public=1" if cfg.public else b"public=0"); h.update(b"|")
    h.update(",".join(str(p) for p in cfg.synthetic_presets).encode())
    return h.hexdigest()[:16]


class DemoRuntime:
    """Process-lifetime resources. Public calls mirror AgentSession and run under one lock."""

    def __init__(self, cfg: DemoConfig, manifest: dict, skills, pool, saver, deps, session, llm_cfg: dict):
        self.cfg, self.manifest, self.skills, self.pool, self.saver, self.deps, self.session, self.llm_cfg = cfg, manifest, skills, pool, saver, deps, session, llm_cfg
        self.lock = threading.Lock()
        self.closed = False

    def invoke(self, thread_id: str, secret: str, message: str, client_request_id: str) -> dict:
        with self.lock:
            return self.session.invoke(thread_id, secret, message, client_request_id)

    def resume(self, thread_id: str, secret: str, parent_turn_id: str, decision: str, client_request_id: str, modifications: Optional[dict] = None) -> dict:
        with self.lock:
            return self.session.resume(thread_id, secret, parent_turn_id, decision, client_request_id, modifications=modifications)

    def pending(self, thread_id: str, secret: str) -> Optional[dict]:
        return self.session.pending_turn(thread_id, secret)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.skills.__exit__(None, None, None)
        finally:
            try:
                self.pool.close()
            except Exception:
                pass


def build_runtime(cfg: DemoConfig) -> DemoRuntime:
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
    from langgraph.checkpoint.postgres import PostgresSaver

    from core_pipeline.agent.graph import GraphDeps
    from core_pipeline.agent.llm_factory import install_llm
    from core_pipeline.agent.manifest import load_manifest
    from core_pipeline.agent.runner import AgentSession
    from core_pipeline.agent.turns import TurnStore
    from core_pipeline.skills.client import SyncSkillClient

    manifest = load_manifest(cfg.manifest_path)
    if cfg.public:
        from core_pipeline.skills.client import SERVERS
        skills = SyncSkillClient(servers={"skills_detect": SERVERS["skills_detect"]}, env={"SGP_PG_DSN": cfg.skill_dsn})
    else:
        skills = SyncSkillClient(env={"SGP_PG_DSN": cfg.skill_dsn})
    skills.__enter__()
    try:
        if cfg.public:                                                        # idle connections would keep the hosted compute awake
            pool = ConnectionPool(cfg.app_dsn, min_size=0, max_size=4, max_idle=60.0, open=True, check=ConnectionPool.check_connection,
                                  kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
        else:
            pool = ConnectionPool(cfg.app_dsn, min_size=1, max_size=4, open=True,
                                  kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
        saver = PostgresSaver(pool)                                           # no setup(): admin step (ST4-24)
        deps_kw: dict[str, Any] = {}
        graph_skills: Any = skills
        if cfg.public:
            from core_pipeline.skills.public_skills import StoredRetrieve
            graph_skills = StoredRetrieve(skills, cfg.app_dsn)
        if cfg.synthetic_presets:
            from core_pipeline.skills.public_skills import SyntheticByPreset, policy_by_preset
            graph_skills, deps_kw["policy"] = SyntheticByPreset(graph_skills, cfg.app_dsn, cfg.synthetic_presets), policy_by_preset(cfg.app_dsn, cfg.synthetic_presets)
        elif cfg.synthetic:
            from demo.synthetic import GroundingOverride, synthetic_policy
            graph_skills, deps_kw["policy"] = GroundingOverride(skills), synthetic_policy
        deps = GraphDeps(dsn=cfg.app_dsn, skills=graph_skills, turns=TurnStore(cfg.app_dsn), manifest=manifest, **deps_kw)
        llm_cfg = install_llm(deps)
        session = AgentSession(deps, saver)
    except Exception:
        skills.__exit__(None, None, None)
        raise
    return DemoRuntime(cfg, manifest, skills, pool, saver, deps, session, llm_cfg)


def mode_summary(cfg: DemoConfig, llm_cfg: dict, environ: Optional[dict] = None) -> dict:
    env = os.environ if environ is None else environ
    budget = (env.get("SGP_LLM_BUDGET_USD") or "").strip()
    return {"router": llm_cfg.get("router"), "narrator": llm_cfg.get("narrator"), "model": llm_cfg.get("model"), "paid": bool(llm_cfg.get("paid")),
            "budget_usd": (float(budget) if budget else None), "budget_set": bool(budget), "synthetic": cfg.synthetic,
            "database": cfg.dsn_facts.get("dbname"), "app_user": cfg.dsn_facts.get("app_user"), "manifest": cfg.manifest_path,
            "public": cfg.public, "synthetic_presets": list(cfg.synthetic_presets)}


def summary_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
