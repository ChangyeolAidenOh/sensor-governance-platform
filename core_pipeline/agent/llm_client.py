"""core_pipeline/agent/llm_client.py — the single LLM transport for the two call sites (ST4-20).

Rules enforced here:
  * no tool binding, ever (ST4-15): the request carries no `tools`; scripts/llm_payload_check.py dumps the real payload
  * no SDK-level retries (max_retries=0): every request that leaves the process has a ledger reservation (ST4-21)
  * the model must be in ALLOWED_MODELS — only models with a price row can be reserved and settled
  * the reservation estimate is a CONSERVATIVE ESTIMATE of the input cost plus the max_tokens output bound; it controls the
    sum of concurrent reservations, it is NOT a proven upper bound of the vendor's bill (ST4-21)
  * any failure whose billing is not documented as zero raises LLMOutcomeUnknown (timeouts, connection errors and
    every HTTP error status outside REJECTED_NO_CHARGE_STATUSES): the caller keeps the reservation PENDING and never
    re-calls automatically. LLMRequestRejected is raised only for HTTP statuses listed in REJECTED_NO_CHARGE_STATUSES —
    empty until the vendor documentation has been checked (ST4-20 paid-call precondition).
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Union

# Prices verified against the official model/pricing docs on 2026-10-22 (ST4-20): standard API, USD per MTok,
# tiered by prompt length — tiers[0] applies when input_tokens <= boundary_input_tokens, tiers[1] above it (input AND output).
PRICE_TABLE_VERSION = "docs/2026-10-22"
PRICE_TABLES: dict[str, dict] = {
    "claude-haiku-5-5": {"version": PRICE_TABLE_VERSION, "boundary_input_tokens": 100_000, "tiers": [[0.10, 0.50], [0.50, 2.50]]},
}
INPUT_GUARD_TOKENS = 100_000                      # development guard on large inputs: estimated_input_tokens above this is refused.
                                                  # It does NOT guarantee the actual prompt is <= 100K: settlement is tier-aware.
ALLOWED_MODELS: tuple[str, ...] = ("claude-haiku-5-5",)   # 0-F selection (ST0-23)
DEFAULT_MODEL = "claude-haiku-5-5"
EFFORT = "medium"                                 # same adapter argument as 0-F (reasoning_effort)
RESERVE_FORMULA_VERSION = "v1:utf8_bytes+64"      # estimated_input_tokens = utf8 bytes of system+user + 64 (margin without evidence)
RESERVE_FRAMING_TOKENS = 64
ROUTER_MAX_TOKENS, NARRATE_MAX_TOKENS = 1500, 3000
ROUTER_TIMEOUT_S, NARRATE_TIMEOUT_S = 30.0, 60.0


def sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


REJECTED_NO_CHARGE_STATUSES: frozenset[int] = frozenset()   # HTTP statuses documented as not billed; filled after the docs check


class LLMOutcomeUnknown(RuntimeError):
    """No usable response and no documented proof of zero billing: keep the reservation PENDING, never auto re-call."""


LLMNoResponse = LLMOutcomeUnknown                           # former name


class LLMRequestRejected(RuntimeError):
    """HTTP status in REJECTED_NO_CHARGE_STATUSES: documented as not billed, nothing generated (settle at cost 0)."""


def tier_for(table: dict, input_tokens: int) -> int:
    return 0 if int(input_tokens) <= int(table["boundary_input_tokens"]) else 1


def cost_from_table(table: dict, input_tokens: int, output_tokens: int) -> float:
    """Tier-aware cost from a price table in the PRICE_TABLES shape (the one stored in the reservation row)."""
    pi, po = table["tiers"][tier_for(table, input_tokens)]
    return round(int(input_tokens) / 1e6 * float(pi) + int(output_tokens) / 1e6 * float(po), 6)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    return cost_from_table(PRICE_TABLES[model], input_tokens, output_tokens)


@dataclass(frozen=True)
class LLMRequest:
    model: str
    system: str
    user: str
    max_tokens: int
    timeout_s: float
    effort: str = EFFORT

    def __post_init__(self) -> None:
        if self.model not in ALLOWED_MODELS:
            raise ValueError(f"model {self.model!r} is not allowed (ALLOWED_MODELS={ALLOWED_MODELS})")
        if self.model not in PRICE_TABLES:
            raise ValueError(f"model {self.model!r} has no price table")
        if int(self.max_tokens) <= 0:
            raise ValueError("max_tokens must be positive")
        if self.estimated_input_tokens() > INPUT_GUARD_TOKENS:
            raise ValueError(f"estimated input {self.estimated_input_tokens()} tokens exceeds the {INPUT_GUARD_TOKENS} guard")

    def bound_input(self) -> dict:
        """What the ledger compares for replay/mismatch: same stage key + same bound input = same logical request."""
        return {"model": self.model, "effort": self.effort, "max_tokens": int(self.max_tokens),
                "system_sha256": sha256_text(self.system), "user_sha256": sha256_text(self.user),
                "price_table_version": PRICE_TABLE_VERSION}

    def estimated_input_tokens(self) -> int:
        return len((self.system + self.user).encode("utf-8")) + RESERVE_FRAMING_TOKENS

    def price_table(self) -> dict:
        return copy.deepcopy(PRICE_TABLES[self.model])

    def reserved_usd(self) -> float:
        """Estimate: estimated input tokens at the tier they fall in + max_tokens output at that tier."""
        return cost_from_table(self.price_table(), self.estimated_input_tokens(), int(self.max_tokens))

    def reserve_record(self) -> dict:
        """Everything a later settlement needs, preserved in the row: estimate, formula and the FULL price table."""
        t = self.price_table()
        return {"reserved_usd": self.reserved_usd(), "estimated_input_tokens": self.estimated_input_tokens(),
                "max_tokens": int(self.max_tokens), "formula_version": RESERVE_FORMULA_VERSION,
                "price_table_version": t["version"], "price_table": t,
                "prices_per_mtok": list(t["tiers"][tier_for(t, self.estimated_input_tokens())])}   # tier used by the estimate


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    model: str
    usage_known: bool = True            # False → actual cost unknown: settle at the reserved amount and block (ST4-21)
    stop_reason: Optional[str] = None
    usage: dict = field(default_factory=dict)


class LLMClient(Protocol):
    def complete(self, req: LLMRequest) -> LLMResponse: ...


# ----------------------------------------------------------------------------------------------------- anthropic
def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, dict):
            if block.get("type", "text") == "text" and block.get("text"):
                parts.append(str(block["text"]))
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts)


class AnthropicClient:
    """langchain_anthropic.ChatAnthropic, built exactly like the 0-F spike except max_retries=0 and a per-call timeout.
    Nothing is bound to the model: no bind_tools, no with_structured_output (both can add `tools` to the request)."""

    def __init__(self) -> None:
        self._models: dict[tuple, Any] = {}

    def _model(self, req: LLMRequest):
        key = (req.model, int(req.max_tokens), float(req.timeout_s), req.effort)
        if key not in self._models:
            from langchain_anthropic import ChatAnthropic
            self._models[key] = ChatAnthropic(model=req.model, max_tokens=int(req.max_tokens), reasoning_effort=req.effort,
                                              max_retries=0, timeout=float(req.timeout_s))
        return self._models[key]

    @staticmethod
    def messages(req: LLMRequest) -> list:
        from langchain_core.messages import HumanMessage, SystemMessage
        return [SystemMessage(content=req.system), HumanMessage(content=req.user)]

    @staticmethod
    def _classify(exc: BaseException) -> type:
        """Only an HTTP status documented as not billed is a rejection; everything else is an unknown outcome."""
        code = getattr(exc, "status_code", None)
        if isinstance(code, int) and code in REJECTED_NO_CHARGE_STATUSES:
            return LLMRequestRejected
        return LLMOutcomeUnknown

    def complete(self, req: LLMRequest) -> LLMResponse:
        model = self._model(req)
        t0 = time.perf_counter()
        try:
            ai = model.invoke(self.messages(req))
        except Exception as e:
            raise self._classify(e)(f"{type(e).__name__}: {e}") from e
        latency = int((time.perf_counter() - t0) * 1000)
        um = getattr(ai, "usage_metadata", None) or {}
        rm = getattr(ai, "response_metadata", None) or {}
        inp, out = um.get("input_tokens"), um.get("output_tokens")
        known = inp is not None and out is not None
        return LLMResponse(text=_text_of(getattr(ai, "content", "")), input_tokens=int(inp or 0), output_tokens=int(out or 0),
                           latency_ms=latency, model=str(rm.get("model") or req.model), usage_known=known,
                           stop_reason=rm.get("stop_reason"), usage=dict(um))

    def request_payload(self, req: LLMRequest) -> dict:
        """The request body the installed adapter would send. Runs entirely under a network guard (socket connect and
        httpx send raise), so no request can leave the process whichever SDK path the adapter takes."""
        model = self._model(req)
        msgs = self.messages(req)
        with network_blocked():
            fn = getattr(model, "_get_request_payload", None)
            if callable(fn):
                payload = fn(msgs)
                return dict(payload) if isinstance(payload, dict) else {"_repr": repr(payload)}
            captured: dict = {}

            def _capture(**kwargs):
                captured.update(kwargs)
                raise _Captured()

            client = getattr(model, "_client", None)
            messages_api = getattr(client, "messages", None)
            if messages_api is None or not hasattr(messages_api, "create"):
                raise RuntimeError("cannot capture the request payload with this langchain_anthropic version; inspect manually")
            original = messages_api.create
            try:
                messages_api.create = _capture
                try:
                    model.invoke(msgs)
                except _Captured:
                    pass
            finally:
                messages_api.create = original
            if not captured:
                raise RuntimeError("request body was not captured (adapter took another path; the network guard blocked it)")
            return captured


class _Captured(Exception):
    pass


class NetworkBlocked(RuntimeError):
    pass


@contextlib.contextmanager
def network_blocked():
    """Belt and braces for $0 checks: any socket connect or httpx send raises NetworkBlocked."""
    import socket

    def _deny(*a, **k):
        raise NetworkBlocked("network access is blocked in this check")

    patches: list[tuple[Any, str, Any]] = [(socket.socket, "connect", socket.socket.connect), (socket, "create_connection", socket.create_connection)]
    try:
        import httpx
        patches.append((httpx.Client, "send", httpx.Client.send))
    except Exception:
        pass
    for obj, name, _ in patches:
        setattr(obj, name, _deny)
    try:
        yield
    finally:
        for obj, name, orig in patches:
            setattr(obj, name, orig)


# ----------------------------------------------------------------------------------------------------- fake
FakeScript = Union[str, BaseException, Callable[[LLMRequest], "LLMResponse"]]


class FakeLLMClient:
    """Scripted responses for tests and dry runs ($0). Each entry: text → response with the default token counts,
    an exception instance → raised, a callable → called with the request."""

    def __init__(self, responses: Optional[list[FakeScript]] = None, input_tokens: int = 100, output_tokens: int = 50,
                 latency_ms: int = 5, usage_known: bool = True) -> None:
        self.responses = list(responses or [])
        self.input_tokens, self.output_tokens, self.latency_ms, self.usage_known = input_tokens, output_tokens, latency_ms, usage_known
        self.calls: list[LLMRequest] = []

    def complete(self, req: LLMRequest) -> LLMResponse:
        self.calls.append(req)
        if not self.responses:
            raise AssertionError("FakeLLMClient: no scripted response left")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(req)
        return LLMResponse(text=str(item), input_tokens=self.input_tokens, output_tokens=self.output_tokens,
                           latency_ms=self.latency_ms, model=req.model, usage_known=self.usage_known)
