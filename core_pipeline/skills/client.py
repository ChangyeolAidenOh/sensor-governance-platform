"""core_pipeline/skills/client.py — the graph-side MCP client for the two Skill servers (S4a-00, ST4-01).

The graph process imports NEITHER server module (no torch, no lightgbm here). Each server is spawned once as a stdio
subprocess (fastmcp Client) and kept open for the life of the SkillClient; model loading happens once per server.
Tool inputs are validated against the JSON Schemas before they are sent; every call needs a Ctx dict (common.new_ctx).

    async with SkillClient() as sk:
        g = await sk.gate_check(ctx, episode_id, t)
        d = await sk.detect_diagnose(ctx2, episode_id, t)
        r = await sk.retrieve(ctx3, query, "scoped", d["state"]["pred_set"], 5, alarm_id=d["payload"]["alarm_id"])

SyncSkillClient wraps the same calls for synchronous code (Streamlit, scripts) on a background event loop.

Transport note (ST4-01 / decision e): stdio by default; SGP_MCP_TRANSPORT=http is reserved for Stage 7 and not wired yet.
The fastmcp result object differs between fastmcp majors; _unwrap() reads structured content first and falls back to
the text block — the installed version (3.4.0) is verified by tests/integration/test_h0_servers.py on Aiden's machine.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from core_pipeline.skills.schemas import validate_envelope, validate_input

ROOT = Path(__file__).resolve().parents[2]
SERVERS = {
    "skills_detect": ROOT / "core_pipeline" / "skills" / "servers" / "skills_detect.py",
    "skills_retrieve": ROOT / "core_pipeline" / "skills" / "servers" / "skills_retrieve.py",
}
TOOL_SERVER = {"gate_check": "skills_detect", "detect_diagnose": "skills_detect", "spend_query": "skills_detect", "work_order_draft": "skills_detect",
               "work_order_approve": "skills_detect", "work_order_issue": "skills_detect", "work_order_status": "skills_detect",
               "retrieve": "skills_retrieve"}
TOOL_SCHEMA = {"gate_check": ("gate_check", "input"), "detect_diagnose": ("detect_diagnose", "input"), "spend_query": ("spend_query", "input"),
               "retrieve": ("retrieve", "input"),
               "work_order_draft": ("work_order", "draft"), "work_order_approve": ("work_order", "approve"),
               "work_order_issue": ("work_order", "issue"), "work_order_status": ("work_order", "status")}


class SkillCallError(RuntimeError):
    pass


def _unwrap(res: Any) -> dict:
    """fastmcp CallToolResult → the envelope dict returned by the tool."""
    for attr in ("structured_content", "data"):
        v = getattr(res, attr, None)
        if isinstance(v, dict):
            return v.get("result", v) if set(v.keys()) == {"result"} else v
    content = getattr(res, "content", res)
    if isinstance(content, list) and content:
        text = getattr(content[0], "text", None)
        if text is not None:
            return json.loads(text)
    raise SkillCallError(f"cannot unwrap tool result of type {type(res).__name__}")


class SkillClient:
    def __init__(self, servers: Optional[dict] = None, env: Optional[dict] = None, python: Optional[str] = None):
        from fastmcp import Client
        from fastmcp.client.transports import PythonStdioTransport
        self._servers = servers or SERVERS
        merged = {**os.environ, **(env or {})}
        self._clients = {name: Client(PythonStdioTransport(str(path), python_cmd=python or sys.executable, env=merged, cwd=str(ROOT)))
                         for name, path in self._servers.items()}

    async def __aenter__(self) -> "SkillClient":
        for c in self._clients.values():
            await c.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        for c in self._clients.values():
            await c.__aexit__(*exc)

    async def call(self, tool: str, inp: dict) -> dict:
        schema, sub = TOOL_SCHEMA[tool]
        errs = validate_input(schema, inp, sub)
        if errs:
            raise SkillCallError(f"{tool}: input schema violation: {errs}")
        timeout = float(os.environ.get("SGP_MCP_TIMEOUT", "600"))        # first call loads the models
        res = await self._clients[TOOL_SERVER[tool]].call_tool(tool, inp, timeout=timeout, raise_on_error=False)
        if getattr(res, "is_error", False):
            raise SkillCallError(f"{tool}: MCP error: {getattr(res, 'content', res)}")
        out = _unwrap(res)
        errs = validate_envelope(out)
        if errs:
            raise SkillCallError(f"{tool}: envelope schema violation: {errs}")
        return out

    async def gate_check(self, ctx: dict, episode_id: str, window_index: int) -> dict:
        return await self.call("gate_check", {"ctx": ctx, "episode_id": str(episode_id), "window_index": int(window_index)})

    async def detect_diagnose(self, ctx: dict, episode_id: str, window_index: int, expand: int = 1, query_id: Optional[str] = None) -> dict:
        return await self.call("detect_diagnose", {"ctx": ctx, "episode_id": str(episode_id), "window_index": int(window_index), "expand": int(expand), "query_id": query_id})

    async def spend_query(self, ctx: dict, episode_id: str, window_index: int) -> dict:
        return await self.call("spend_query", {"ctx": ctx, "episode_id": str(episode_id), "window_index": int(window_index)})

    async def retrieve(self, ctx: dict, query: str, mode: str, fault_set: list[int], k: int = 5, alarm_id: Optional[str] = None,
                       query_id: Optional[str] = None) -> dict:
        return await self.call("retrieve", {"ctx": ctx, "query": query, "mode": mode, "fault_set": [int(f) for f in fault_set], "k": int(k),
                                            "alarm_id": alarm_id, "query_id": query_id})

    async def work_order_draft(self, ctx: dict, **kw) -> dict:
        return await self.call("work_order_draft", {"ctx": ctx, **kw})

    async def work_order_approve(self, ctx: dict, draft: dict, approver: str, decision: str = "approve", modifications: Optional[dict] = None) -> dict:
        return await self.call("work_order_approve", {"ctx": ctx, "draft": draft, "approver": approver, "decision": decision, "modifications": modifications})

    async def work_order_issue(self, ctx: dict, draft: dict, approval_token: Optional[str]) -> dict:
        return await self.call("work_order_issue", {"ctx": ctx, "draft": draft, "approval_token": approval_token})

    async def work_order_status(self, ctx: dict, proposal_id: str) -> dict:
        return await self.call("work_order_status", {"ctx": ctx, "proposal_id": proposal_id})


class SyncSkillClient:
    """Same API, synchronous: runs one event loop on a daemon thread so the server subprocesses stay alive between calls."""

    def __init__(self, **kw):
        self._kw = kw
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="sgp-skill-client")
        self._thread.start()
        self._client: Optional[SkillClient] = None

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def __enter__(self) -> "SyncSkillClient":
        async def _open():
            c = SkillClient(**self._kw)
            await c.__aenter__()
            return c
        self._client = self._run(_open())
        return self

    def __exit__(self, *exc) -> None:
        if self._client is not None:
            self._run(self._client.__aexit__(None, None, None))
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    def __getattr__(self, name: str):
        if self._client is None or not hasattr(self._client, name):
            raise AttributeError(name)
        fn = getattr(self._client, name)
        return lambda *a, **kw: self._run(fn(*a, **kw))
