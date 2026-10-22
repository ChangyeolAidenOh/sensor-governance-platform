"""0-D spike: LangGraph + Postgres checkpointer + interrupt() approval gate.

No LLM is called. Nodes are deterministic so that the only thing under test is
the runtime: does a thread paused at interrupt() resume correctly from Postgres
in a NEW process/connection, with state preserved, over two turns, and does the
same protocol produce identical state on two threads?

Graph:  START -> detect -> approve(interrupt) -> act -> END
    detect   : builds a proposal for the alarm in the input (no side effects)
    approve  : interrupt({"proposal", "question"}) -> waits for Command(resume={"decision": ...})
    act      : approve -> audit "work_order_issued"; otherwise "held"

Protocol per thread (run twice on two thread ids, compared at the end):
    turn 1: invoke(alarm A)         -> expect pause with __interrupt__ payload
            [new checkpointer + graph instance = simulated restart]
            invoke(Command(resume={"decision": "approve"}))  -> work order issued
    turn 2: invoke(alarm B)         -> pause again (history preserved: 1 prior audit entry)
            invoke(Command(resume={"decision": "reject"}))   -> held

Usage:
    PG_DSN=postgresql://<user>:<password>@localhost:5436/sgp_v2 python spikes/langgraph_interrupt_spike.py   # PG_DSN is required
Report: outputs/stage0_interrupt_spike.json (and stdout).
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

DSN = os.environ.get("PG_DSN")
if not DSN:
    raise SystemExit("PG_DSN is required (no default credentials; see .env)")
REPORT = Path(os.environ.get("REPORT", "outputs/stage0_interrupt_spike.json"))


def _append(a: list, b: list) -> list:
    return list(a or []) + list(b or [])


class State(TypedDict, total=False):
    alarm_id: str
    proposal: str
    decision: str
    audit: Annotated[list[str], _append]      # reducer: appended across turns
    turns: Annotated[list[str], _append]


def detect(state: State) -> dict:
    return {"proposal": f"issue work order for alarm {state['alarm_id']}", "decision": ""}


def approve(state: State) -> dict:
    answer = interrupt({"proposal": state["proposal"], "question": "approve work order?",
                        "alarm_id": state["alarm_id"]})
    decision = str(answer.get("decision", "")) if isinstance(answer, dict) else str(answer)
    return {"decision": decision}


def act(state: State) -> dict:
    if state["decision"] == "approve":
        entry = f"work_order_issued:{state['alarm_id']}"
    else:
        entry = f"held:{state['alarm_id']}:{state['decision'] or 'no_decision'}"
    return {"audit": [entry], "turns": [state["alarm_id"]]}


def build_graph(checkpointer):
    g = StateGraph(State)
    g.add_node("detect", detect)
    g.add_node("approve", approve)
    g.add_node("act", act)
    g.add_edge(START, "detect")
    g.add_edge("detect", "approve")
    g.add_edge("approve", "act")
    g.add_edge("act", END)
    return g.compile(checkpointer=checkpointer)


def snapshot(graph, config) -> dict:
    st = graph.get_state(config)
    pending = []
    for task in getattr(st, "tasks", ()) or ():
        for intr in getattr(task, "interrupts", ()) or ():
            pending.append(getattr(intr, "value", intr))
    return {"values": dict(st.values), "next": list(st.next), "pending_interrupts": pending}


def run_protocol(thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    log: dict = {"thread_id": thread_id, "steps": []}

    def step(name: str, fn):
        t0 = time.time()
        out = fn()
        rec = {"step": name, "seconds": round(time.time() - t0, 3), **out}
        log["steps"].append(rec)
        print(f"[{thread_id[:8]}] {name}: next={rec.get('next')} pending={len(rec.get('pending_interrupts', []))} "
              f"decision={rec.get('values', {}).get('decision')!r} audit={rec.get('values', {}).get('audit')}")
        return rec

    # ---- turn 1: start, expect pause ---------------------------------------------------
    with PostgresSaver.from_conn_string(DSN) as cp:
        cp.setup()
        graph = build_graph(cp)
        result = graph.invoke({"alarm_id": "A-001"}, config)
        step("t1_invoke", lambda: {"interrupted": "__interrupt__" in result,
                                   "interrupt_payload": [getattr(i, "value", i) for i in result.get("__interrupt__", [])],
                                   **snapshot(graph, config)})
    # ---- simulated restart: new connection, new graph object, resume --------------------
    with PostgresSaver.from_conn_string(DSN) as cp2:
        graph2 = build_graph(cp2)
        before = step("t1_after_restart_state", lambda: snapshot(graph2, config))
        events = []
        for ev in graph2.stream(Command(resume={"decision": "approve"}), config, stream_mode="updates"):
            events.append({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()}) for k, v in ev.items()})
        step("t1_resume_approve", lambda: {"events": events, **snapshot(graph2, config)})
        # ---- turn 2 on the same thread -------------------------------------------------
        result2 = graph2.invoke({"alarm_id": "B-002"}, config)
        step("t2_invoke", lambda: {"interrupted": "__interrupt__" in result2,
                                   "interrupt_payload": [getattr(i, "value", i) for i in result2.get("__interrupt__", [])],
                                   **snapshot(graph2, config)})
    with PostgresSaver.from_conn_string(DSN) as cp3:
        graph3 = build_graph(cp3)
        final = step("t2_resume_reject", lambda: {**(graph3.invoke(Command(resume={"decision": "reject"}), config) and {}),
                                                   **snapshot(graph3, config)})
        n_checkpoints = sum(1 for _ in cp3.list(config))
        log["n_checkpoints"] = n_checkpoints
        log["final_values"] = final["values"]
        log["before_restart_next"] = before["next"]
    return log


def main() -> None:
    print(f"[dsn] {DSN.split('@')[-1]}")
    runs = [run_protocol(str(uuid.uuid4())) for _ in range(2)]

    def comparable(r: dict) -> dict:
        v = r["final_values"]
        return {"audit": v.get("audit"), "turns": v.get("turns"), "decision": v.get("decision"),
                "proposal": v.get("proposal"), "n_checkpoints": r["n_checkpoints"]}

    checks = {
        "t1_paused_with_payload": all(r["steps"][0]["interrupted"] and r["steps"][0]["interrupt_payload"] for r in runs),
        "paused_state_survives_restart": all(r["before_restart_next"] == ["approve"] and r["steps"][1]["pending_interrupts"] for r in runs),
        "t1_resume_issued_work_order": all(any(a.startswith("work_order_issued:A-001") for a in r["final_values"]["audit"]) for r in runs),
        "t2_paused_and_history_kept": all(r["steps"][3]["interrupted"] and r["steps"][3]["values"]["audit"] == [f"work_order_issued:A-001"] for r in runs),
        "t2_reject_held": all(r["final_values"]["audit"][-1].startswith("held:B-002:reject") for r in runs),
        "two_turns_recorded": all(r["final_values"]["turns"] == ["A-001", "B-002"] for r in runs),
        "deterministic_across_threads": comparable(runs[0]) == comparable(runs[1]),
        "no_work_order_without_approval": all(
            not any(a.startswith("work_order_issued") for a in r["steps"][0]["values"].get("audit", [])) for r in runs),
    }
    report = {"dsn_host": DSN.split("@")[-1], "runs": runs, "checks": checks, "go": all(checks.values())}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("\n[checks]")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"[go] {report['go']}  (checkpoints per thread: {[r['n_checkpoints'] for r in runs]})")
    print(f"[report] {REPORT}")
    sys.exit(0 if report["go"] else 1)


if __name__ == "__main__":
    main()
