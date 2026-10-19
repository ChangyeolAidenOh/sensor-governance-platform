"""
core_pipeline/prereg_constants.py — single source of pre-registered constants (docs/preregistration.md, prereg-v1 rev 3,
tag cdd9a3b). Git-frozen: any change here is a "사전등록 수정(동결 후·수정 등록)" and must cite a decision ID.
Loaders, the replayer and the evaluator import from here; the DB never redefines these values.
"""
from __future__ import annotations

PREREG_TAG = "prereg-v1"
PREREG_COMMIT = "cdd9a3b"

SEED = 20261018                      # split + bootstrap seed (ST0-13, §5.1)
SPLIT = "testing"
RUN_LENGTH = 960
ONSET_CYCLE = 161                    # ST0-08 spec onset
N_FAULTS = 20

WINDOW_SIZE = 20                     # §3.2
STRIDE = 20
N_WINDOWS = (RUN_LENGTH - WINDOW_SIZE) // STRIDE + 1      # 48
ONSET_WINDOW = (ONSET_CYCLE - 1) // STRIDE                 # 8  (window containing cycle 161)

N_EVAL_DEFAULT = 60                  # §3.1
N_EVAL_ALLOWED = {60, 40}            # 40 only under the §3.1 time contract, with a logged decision ID
ALPHA = 0.10                         # §4.1
COVERAGE_TARGET = 0.90               # §4.2
SET_SIZE_MAX = 3                     # §4.2 valid set: 1 ≤ s ≤ 3

COST = {"c_fp": 1.0, "c_fn": 0.3, "c_hold": 0.1, "c_query": 0.05}   # §4.4
COST_SENSITIVITY_C_FP = (0.5, 1.0, 2.0)

# §3.3 demo presets: (preset, fault_scenario, run_id)
DEMO_PRESETS = (
    (1, 1, 22), (2, 4, 26), (3, 5, 39), (4, 6, 43), (5, 7, 133),
    (6, 13, 174), (7, 14, 176), (8, 18, 178), (9, 0, 250), (10, 0, 251),
)

SPLIT_CODE = {"training": 0, "testing": 1}


def engine_id(split: str, fault: int, run: int) -> int:
    """Label-bearing identifier (loader contract). Evaluator/loader side only — never exposed to the agent."""
    return SPLIT_CODE[split] * 100_000 + fault * 1000 + run
