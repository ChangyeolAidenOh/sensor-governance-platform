"""core_pipeline/agent/llm_factory.py — choose the Router / Narrator implementation from the environment (ST4-20/23).

  SGP_LLM_ROUTER   = rule (default) | llm
  SGP_LLM_NARRATOR = template (default) | llm | llm_all
  SGP_LLM_MODEL    = claude-haiku-5-5 (default; must be in ALLOWED_MODELS)

Defaults make no LLM call at all. `install_llm(deps)` replaces deps.router / deps.narrator in place (GraphDeps fields;
the graph itself is unchanged) and returns the effective configuration for the trace / smoke report. Tests pass a
FakeLLMClient / MemoryLedger through `build(..., client=, ledger=)`.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from core_pipeline.agent.llm_client import ALLOWED_MODELS, DEFAULT_MODEL, LLMClient
from core_pipeline.agent.llm_ledger import LlmLedger
from core_pipeline.agent.narrator import TemplateNarrator
from core_pipeline.agent.router import RuleRouter

ROUTER_MODES = ("rule", "llm")
NARRATOR_MODES = ("template", "llm", "llm_all")


def llm_config() -> dict:
    r = (os.environ.get("SGP_LLM_ROUTER") or "rule").strip().lower()
    n = (os.environ.get("SGP_LLM_NARRATOR") or "template").strip().lower()
    m = (os.environ.get("SGP_LLM_MODEL") or DEFAULT_MODEL).strip()
    if r not in ROUTER_MODES:
        raise ValueError(f"SGP_LLM_ROUTER must be one of {ROUTER_MODES}, got {r!r}")
    if n not in NARRATOR_MODES:
        raise ValueError(f"SGP_LLM_NARRATOR must be one of {NARRATOR_MODES}, got {n!r}")
    if (r == "llm" or n != "template") and m not in ALLOWED_MODELS:
        raise ValueError(f"SGP_LLM_MODEL {m!r} is not allowed ({ALLOWED_MODELS})")
    return {"router": r, "narrator": n, "model": m, "paid": (r == "llm" or n != "template")}


def build(cfg: dict, dsn: Optional[str] = None, client: Optional[LLMClient] = None, ledger: Optional[LlmLedger] = None) -> tuple[Any, Any]:
    """→ (router, narrator). A real client / ledger is created lazily only when a paid mode is selected."""
    rule, template = RuleRouter(), TemplateNarrator()
    if not cfg["paid"]:
        return rule, template
    if client is None:
        from core_pipeline.agent.llm_client import AnthropicClient
        client = AnthropicClient()
    if ledger is None:
        if not dsn:
            raise ValueError("dsn is required for the LLM ledger")
        from core_pipeline.agent.llm_ledger import PostgresLedger
        ledger = PostgresLedger(dsn)
    router: Any = rule
    narrator: Any = template
    if cfg["router"] == "llm":
        from core_pipeline.agent.llm_router import LLMRouter
        router = LLMRouter(client, ledger, rule=rule, model=cfg["model"])
    if cfg["narrator"] != "template":
        from core_pipeline.agent.llm_narrator import LLMNarrator
        narrator = LLMNarrator(client, ledger, template=template, mode=cfg["narrator"], model=cfg["model"])
    return router, narrator


def install_llm(deps: Any, client: Optional[LLMClient] = None, ledger: Optional[LlmLedger] = None) -> dict:
    cfg = llm_config()
    deps.router, deps.narrator = build(cfg, dsn=getattr(deps, "dsn", None), client=client, ledger=ledger)
    return cfg
