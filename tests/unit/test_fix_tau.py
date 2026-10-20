"""S3-04: tau selection rule on the pre-registered grid — balanced accuracy, ties → TNR → largest tau."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.fix_tau import choose_tau, GRID


def test_grid_and_rule():
    assert GRID == [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    g = {"a1": 0.9, "a2": 0.75, "a3": 0.62, "a4": 0.55, "a5": 0.45, "t1": None, "t2": None, "t3": 0.38, "t4": 0.57, "t5": 0.42}
    kinds = {k: ("answer" if k.startswith("a") else "trap") for k in g}
    tau, rows = choose_tau(g, kinds)
    ba = {r["tau"]: r["balanced_acc"] for r in rows}
    assert ba[0.4] == 0.8 and ba[0.5] == 0.8 and ba[0.6] == 0.8 and ba[0.3] < 0.8 and ba[0.7] < 0.8   # three-way tie
    assert tau == 0.6                                                                                   # tie → TNR (1.0 at 0.6) wins


def test_none_g_counts_as_not_found():
    g = {"a1": 0.9, "t1": None}; kinds = {"a1": "answer", "t1": "trap"}
    tau, rows = choose_tau(g, kinds)
    assert all(r["tnr"] == 1.0 for r in rows) and tau == 0.8      # all tie at 1.0 → largest tau
