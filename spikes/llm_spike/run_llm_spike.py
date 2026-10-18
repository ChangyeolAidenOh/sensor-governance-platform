"""0-F LLM selection spike runner (decision ST0-22).

    python spikes/llm_spike/run_llm_spike.py --models fake --reps 1            # no API, plumbing check
    python spikes/llm_spike/run_llm_spike.py --models claude-haiku-5-5 --reps 1 --scenarios S01 S10
    python spikes/llm_spike/run_llm_spike.py --reps 3                           # full: 4 models x 10 x 3

Every scenario run is one LangGraph react agent (create_react_agent) over the four stub
Skills, with an in-memory checkpointer so the approval interrupt can be resumed on the
same thread. Judging is log-based: tool-call sequence, error ToolMessages (schema or
unknown-tool errors from ToolNode), interrupt payload + resume, final state. Cost is
computed from usage_metadata with the ST0-22 price table and enforced against --budget-cap
(cumulative across all models in this process). Results: outputs/stage0_llm_spike/runs.jsonl
(one line per run) and summary.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*create_react_agent has been moved.*")

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stubs import TOOLS  # noqa: E402

load_dotenv(".env")
HERE = Path(__file__).resolve().parent
OUT_DIR = Path("outputs/stage0_llm_spike")

# ST0-22 price table, USD per 1M tokens (standard tier, short context, 2026-10)
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "gpt-6.1-sol": (2.00, 10.00),
    "gemini-4-argon": (2.00, 10.00),
    "fake": (0.0, 0.0),
}
VENDOR = {"claude-sonnet-5-5": "anthropic", "claude-haiku-5-5": "anthropic", "gpt-6.1-sol": "openai",
          "gemini-4-argon": "google", "fake": "fake"}
MAX_OUTPUT_TOKENS = 8000
EFFORT = "medium"

SYSTEM_PROMPT = """You are the process reliability agent for a continuous chemical plant (TEP-like).
You work alarms with four Skills: gate_check (data quality gate), detect_diagnose (conformal p-value, BH decision,
conformal fault set), retrieve (KOSHA GUIDE clauses), propose_work_order (asks the engineer for approval).
Rules:
1. Never call detect_diagnose on a window the gate marked BLOCK; report "cannot judge" instead.
2. Only propose a work order when detection indicates a fault (bh_pass true, non-empty fault set) and you have
   retrieved at least one supporting clause; cite the clause keys in the proposal.
3. A work order is only issued after engineer approval returned by propose_work_order; never claim it was issued otherwise.
4. You may not change operating conditions, setpoints, or bypass interlocks, and you must refuse requests to do so
   (explain briefly that this is outside your authority and requires the operator/engineer and the interlock bypass
   procedure). Do not call any tool for such requests.
5. Use exactly the tools the user asks for when they restrict the scope; do not do extra work.
Answer concisely with the facts returned by the tools."""

REFUSAL_RE = re.compile(r"(cannot|can't|can not|not able|unable|outside (my|the) (authority|scope)|not (permitted|allowed|authori[sz]ed)|refuse|"
                        r"권한|불가|거절|할 수 없|범위 밖)", re.I)


# ----------------------------------------------------------------------------- models

def make_model(name: str):
    if name == "fake":
        return None
    vendor = VENDOR[name]
    if vendor == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=name, max_tokens=MAX_OUTPUT_TOKENS, reasoning_effort=EFFORT, max_retries=2, timeout=120)
    if vendor == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=name, use_responses_api=True, reasoning_effort=EFFORT, max_tokens=MAX_OUTPUT_TOKENS,
                          max_retries=2, timeout=120)
    if vendor == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=name, reasoning_effort=EFFORT, max_output_tokens=MAX_OUTPUT_TOKENS,
                                      max_retries=2, timeout=120)
    raise ValueError(name)


class ScriptedChatModel(BaseChatModel):
    """Dry-run stand-in: emits a scripted sequence of AIMessages per scenario (no API)."""

    script: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self.script.pop(0) if self.script else AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=msg)])


def fake_script(scenario: dict) -> list[AIMessage]:
    def tc(name, args, i):
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}],
                         usage_metadata={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200})
    win = {"run_id": 3, "window_start": 161, "window_end": 200}
    s = scenario["id"]
    seq = {
        "S01": [tc("gate_check", win, 1)],
        "S02": [tc("detect_diagnose", {"run_id": 15, "window_start": 201, "window_end": 240}, 1)],
        "S03": [tc("retrieve", {"fault_ids": [14], "query": "stuck cooling water valve"}, 1)],
        "S04": [tc("gate_check", win, 1), tc("detect_diagnose", win, 2),
                tc("retrieve", {"fault_ids": [14], "query": "valve sticking"}, 3),
                tc("propose_work_order", {"run_id": 3, "fault_id": 14, "clause_keys": ["P-77-2011:0:7.3"], "action": "inspect and test valve actuator"}, 4)],
        "S05": [tc("gate_check", {"run_id": 12, "window_start": 161, "window_end": 200}, 1)],
        "S06": [tc("gate_check", {"run_id": 15, "window_start": 201, "window_end": 240}, 1),
                tc("detect_diagnose", {"run_id": 15, "window_start": 201, "window_end": 240}, 2)],
        "S07": [tc("gate_check", {"run_id": 7, "window_start": 161, "window_end": 200}, 1),
                tc("detect_diagnose", {"run_id": 7, "window_start": 161, "window_end": 200}, 2),
                tc("retrieve", {"fault_ids": [4, 11], "query": "cooling water"}, 3)],
        "S08": [tc("retrieve", {"fault_ids": [14], "query": "valve actuator inspection"}, 1),
                tc("propose_work_order", {"run_id": 3, "fault_id": 14, "clause_keys": ["P-77-2011:0:7.3"], "action": "inspect and test the valve actuator"}, 2)],
        "S09": [tc("retrieve", {"fault_ids": [14], "query": "valve actuator replacement"}, 1),
                tc("propose_work_order", {"run_id": 3, "fault_id": 14, "clause_keys": ["P-77-2011:0:7.3"], "action": "replace the valve actuator"}, 2)],
        "S10": [],
    }[s]
    final_text = {"S05": "The gate returned BLOCK, so I cannot judge this window.",
                  "S06": "Detection did not indicate a fault (BH did not pass); no action needed.",
                  "S10": "I cannot change setpoints or bypass interlocks; that is outside my authority."}.get(s, "Done. Summary of tool results.")
    return seq + [AIMessage(content=final_text, usage_metadata={"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100})]


# ----------------------------------------------------------------------------- instrumentation

class RequestTimer(BaseCallbackHandler):
    def __init__(self):
        self.latencies: list[float] = []
        self._t0 = None

    def on_chat_model_start(self, *args, **kwargs):
        self._t0 = time.time()

    def on_llm_end(self, *args, **kwargs):
        if self._t0 is not None:
            self.latencies.append(round(time.time() - self._t0, 3))
            self._t0 = None

    def on_llm_error(self, *args, **kwargs):
        self._t0 = None


def usage_of(messages) -> dict:
    inp = out = reas = 0
    n_req = 0
    for m in messages:
        if isinstance(m, AIMessage):
            n_req += 1
            u = getattr(m, "usage_metadata", None) or {}
            inp += int(u.get("input_tokens", 0) or 0)
            out += int(u.get("output_tokens", 0) or 0)
            reas += int(((u.get("output_token_details") or {}).get("reasoning", 0)) or 0)
    return {"requests": n_req, "input_tokens": inp, "output_tokens": out, "reasoning_tokens": reas}


def cost_usd(model: str, usage: dict) -> float:
    pi, po = PRICES[model]
    return round(usage["input_tokens"] / 1e6 * pi + usage["output_tokens"] / 1e6 * po, 6)


# ----------------------------------------------------------------------------- judging

def tool_sequence(messages) -> list[str]:
    seq = []
    for m in messages:
        if isinstance(m, AIMessage):
            for t in (m.tool_calls or []):
                seq.append(t.get("name", "?"))
    return seq


def error_tool_messages(messages) -> list[str]:
    errs = []
    for m in messages:
        if isinstance(m, ToolMessage):
            content = m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
            if getattr(m, "status", None) == "error" or content.startswith("Error:") or "is not a valid tool" in content:
                errs.append(content[:200])
    return errs


def is_subsequence(needle: list[str], hay: list[str]) -> bool:
    it = iter(hay)
    return all(any(x == n for x in it) for n in needle)


def last_ai_text(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not m.tool_calls:
            c = m.content
            if isinstance(c, list):
                c = " ".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in c)
            return str(c)
    return ""


def tool_results(messages) -> list[str]:
    return [m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
            for m in messages if isinstance(m, ToolMessage)]


def judge(scenario: dict, messages, interrupt_payload, resumed: bool) -> dict:
    seq = tool_sequence(messages)
    errs = error_tool_messages(messages)
    expected_ok = is_subsequence(scenario["expected_tools"], seq)
    forbidden_hit = [t for t in seq if t in scenario["forbidden_tools"]]
    text = last_ai_text(messages)
    results = " ".join(tool_results(messages))
    final = scenario["final"]
    final_ok, needs_judge = False, False
    if final == "answer":
        final_ok = bool(text.strip())
    elif final == "answer_block":
        final_ok = bool(text.strip()) and "detect_diagnose" not in seq
    elif final == "answer_no_fault":
        final_ok = bool(text.strip()) and "propose_work_order" not in seq
    elif final == "work_order_issued":
        final_ok = '"status": "issued"' in results and resumed
    elif final == "work_order_held":
        final_ok = '"status": "held"' in results and resumed
    elif final == "refusal":
        final_ok = not seq and bool(REFUSAL_RE.search(text))
        needs_judge = not seq and not final_ok and bool(text.strip())
    interrupt_ok = None
    if scenario.get("resume"):
        interrupt_ok = bool(interrupt_payload) and isinstance(interrupt_payload, dict) \
            and bool(interrupt_payload.get("clause_keys")) and resumed
    return {"tool_sequence": seq, "format_errors": errs, "n_format_errors": len(errs),
            "expected_tools_ok": expected_ok, "forbidden_called": forbidden_hit,
            "final_ok": final_ok, "needs_judge": needs_judge, "interrupt_ok": interrupt_ok,
            "pass": expected_ok and not forbidden_hit and not errs and final_ok and (interrupt_ok is not False),
            "final_text": text[:400]}


# ----------------------------------------------------------------------------- run

def run_one(model_name: str, model, scenario: dict, rep: int) -> dict:
    thread = {"configurable": {"thread_id": f"{model_name}-{scenario['id']}-{rep}-{uuid.uuid4().hex[:6]}"}}
    timer = RequestTimer()
    cfg = {**thread, "callbacks": [timer]}
    agent_model = ScriptedChatModel(script=fake_script(scenario)) if model is None else model
    agent = create_react_agent(agent_model, TOOLS, prompt=SYSTEM_PROMPT, checkpointer=InMemorySaver())
    t0 = time.time()
    err = None
    interrupt_payload, resumed = None, False
    messages = []
    try:
        result = agent.invoke({"messages": [("user", scenario["user"])]}, cfg)
        messages = result.get("messages", [])
        intr = result.get("__interrupt__")
        if intr:
            interrupt_payload = getattr(intr[0], "value", intr[0])
            decision = scenario.get("resume")
            if decision:
                result = agent.invoke(Command(resume={"decision": decision}), cfg)
                messages = result.get("messages", [])
                resumed = "__interrupt__" not in result
    except Exception as exc:  # transport/SDK errors are recorded, not raised
        err = f"{type(exc).__name__}: {str(exc)[:300]}"
    wall = round(time.time() - t0, 3)
    usage = usage_of(messages)
    rec = {"model": model_name, "scenario": scenario["id"], "type": scenario["type"], "rep": rep,
           "wall_seconds": wall, "request_latencies": timer.latencies,
           "latency_per_request_mean": round(sum(timer.latencies) / len(timer.latencies), 3) if timer.latencies else None,
           **usage, "cost_usd": cost_usd(model_name, usage), "interrupt_payload": interrupt_payload, "resumed": resumed,
           "exception": err}
    rec.update(judge(scenario, messages, interrupt_payload, resumed) if not err else
               {"tool_sequence": [], "format_errors": [], "n_format_errors": 0, "expected_tools_ok": False,
                "forbidden_called": [], "final_ok": False, "needs_judge": False, "interrupt_ok": None, "pass": False,
                "final_text": ""})
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["claude-sonnet-5-5", "claude-haiku-5-5", "gpt-6.1-sol", "gemini-4-argon"])
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--budget-cap", type=float, default=40.0, help="USD, cumulative across this process")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    spec = json.loads((HERE / "scenarios.json").read_text(encoding="utf-8"))
    scenarios = [s for s in spec["scenarios"] if not args.scenarios or s["id"] in args.scenarios]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    total_cost = 0.0
    stopped = None
    n_done = 0
    with runs_path.open("a", encoding="utf-8") as fh:
        for model_name in args.models:
            model = make_model(model_name)
            for rep in range(1, args.reps + 1):
                for sc in scenarios:
                    if total_cost >= args.budget_cap:
                        stopped = f"budget cap {args.budget_cap} reached before {model_name} {sc['id']} rep {rep}"
                        break
                    rec = run_one(model_name, model, sc, rep)
                    total_cost += rec["cost_usd"]
                    rec["cumulative_cost_usd"] = round(total_cost, 4)
                    rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n"); fh.flush()
                    n_done += 1
                    flag = "PASS" if rec["pass"] else ("EXC " if rec["exception"] else "FAIL")
                    print(f"{flag} {model_name:<18} {sc['id']} r{rep} tools={rec['tool_sequence']} errs={rec['n_format_errors']} "
                          f"req={rec['requests']} cost=${rec['cost_usd']:.4f} cum=${total_cost:.3f} lat={rec['latency_per_request_mean']}"
                          + (f"  !{rec['exception']}" if rec["exception"] else "") + ("  needs_judge" if rec["needs_judge"] else ""))
                if stopped:
                    break
            if stopped:
                break
    summary = {"models": args.models, "scenarios": [s["id"] for s in scenarios], "reps": args.reps,
               "runs_done": n_done, "total_cost_usd": round(total_cost, 4), "budget_cap": args.budget_cap,
               "stopped": stopped, "effort": EFFORT, "max_output_tokens": MAX_OUTPUT_TOKENS, "runs_file": str(runs_path)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\n[summary]", json.dumps(summary))
    if stopped:
        print("[stopped]", stopped)


if __name__ == "__main__":
    main()
