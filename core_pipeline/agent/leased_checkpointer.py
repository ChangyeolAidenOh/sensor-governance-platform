"""core_pipeline/agent/leased_checkpointer.py — checkpoint writes bound to the execution lease (ST4-16, reviewer
2026-10-21 point 3b). guard() at node START is not enough: a lagging executor whose lease was reclaimed could still
RETURN from its running node and LangGraph would then write that checkpoint. This wrapper verifies the current
session's lease (contextvar executor → mart.graph_turns, conditioned UPDATE) on EVERY put / put_writes and refuses
the write with LostLease when the lease is gone. Reads are passed through unchanged.
Residual (recorded): the lease check and the checkpoint write run on different connections, so they are not one
transaction; the window is the check-to-write interval of a single put, no longer a whole node.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from langgraph.checkpoint.base import BaseCheckpointSaver

from core_pipeline.agent.session import _AUTH
from core_pipeline.agent.turns import LostLease, TurnStore


class LeasedCheckpointer(BaseCheckpointSaver):
    def __init__(self, inner: BaseCheckpointSaver, turns: TurnStore):
        super().__init__(serde=getattr(inner, "serde", None))
        self.inner, self.turns = inner, turns

    # ---- lease -------------------------------------------------------------------------------------------------
    def _assert_lease(self, config: dict) -> None:
        auth = _AUTH.get()
        if auth is None:
            raise LostLease("checkpoint write outside a verified session scope")
        thread_id = (config.get("configurable") or {}).get("thread_id")
        if thread_id != auth.thread_id:
            raise LostLease("checkpoint write for a thread other than the session's")
        self.turns.touch(auth.thread_id, auth.lease_turn, auth.executor)          # raises LostLease when reclaimed

    # ---- delegated reads ---------------------------------------------------------------------------------------
    @property
    def config_specs(self):
        return self.inner.config_specs

    def get_tuple(self, config):
        return self.inner.get_tuple(config)

    def list(self, config, *, filter=None, before=None, limit=None):
        return self.inner.list(config, filter=filter, before=before, limit=limit)

    def get_next_version(self, current, channel=None):
        return self.inner.get_next_version(current, channel)

    def delete_thread(self, thread_id):
        return self.inner.delete_thread(thread_id)

    def setup(self):
        return self.inner.setup()

    # ---- guarded writes ----------------------------------------------------------------------------------------
    def put(self, config, checkpoint, metadata, new_versions):
        self._assert_lease(config)
        return self.inner.put(config, checkpoint, metadata, new_versions)

    def put_writes(self, config, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = ""):
        self._assert_lease(config)
        return self.inner.put_writes(config, writes, task_id, task_path)

    async def aget_tuple(self, config):
        return await self.inner.aget_tuple(config)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        async for item in self.inner.alist(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(self, config, checkpoint, metadata, new_versions):
        self._assert_lease(config)
        return await self.inner.aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        self._assert_lease(config)
        return await self.inner.aput_writes(config, writes, task_id, task_path)
