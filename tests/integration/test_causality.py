"""
tests/integration/test_causality.py — H0-(e) **replayer part** (S1-05, dummy p-values, no DB/parquet).
Re-verified with real detect/diagnose/retrieve in Stage 5.
  (a) changing or truncating samples after window t leaves alarms/p_adj/actions for windows 0..t identical
  (b) BLOCK: no detect call, no p-value appended, m_t unchanged; fewer tests than a same-length non-BLOCK run
  (c) m_t non-decreasing within a run
  (d) a confirmed alarm never changes with later windows
  (e) WindowView refuses access beyond end_cycle and carries no label/identity columns
  (f) replay order equals the frozen list; any reordering is detected by the order sha
  (g) q_used resets per window
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from core_pipeline import prereg_constants as P
from core_pipeline.anomaly.prefix_bh import bh_adjusted
from core_pipeline.stream.replayer import Episode, ListSource, Replayer, ReplayRecord, frozen_order, order_sha
from core_pipeline.stream.state import EpisodeState, WindowState
from core_pipeline.stream.window_view import FutureAccessError, WindowView, PROCESS_COLS

SPLIT = Path(__file__).resolve().parents[2] / "outputs" / "stage0_episode_split.json"


def synth_frame(seed: int, n_cycles: int = P.RUN_LENGTH, with_labels: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n_cycles, len(PROCESS_COLS))), columns=PROCESS_COLS)
    df.insert(0, "cycle", np.arange(1, n_cycles + 1))
    if with_labels:
        df["fault_scenario"] = 3
        df["is_fault_active"] = df["cycle"] >= P.ONSET_CYCLE
    return df


def dummy_p(view: WindowView) -> float:
    """Deterministic p-value from the window's OWN samples only (so a future change cannot affect it)."""
    h = hashlib.sha256(pd.util.hash_pandas_object(view.current(), index=False).values.tobytes()).digest()
    return int.from_bytes(h[:4], "big") / 2**32


def make_policy(block_windows=(), alpha=P.ALPHA):
    def policy(view, ep: EpisodeState, w: WindowState):
        if view.window_index in block_windows:
            return "undecidable", "BLOCK", None          # no detect call ⇒ no p
        p = dummy_p(view)
        # the policy cannot know its own prefix-BH outcome before observe(); emulate with a local copy
        adj = bh_adjusted([q for _, q in ep._tested] + [p])[-1]
        if adj > alpha:
            return "no_alarm", "PASS", p
        w.q_used = True                                    # pretend one query_more happened
        return ("propose" if view.window_index % 2 else "hold"), "PASS", p
    return policy


def replay(frame, block=(), n_windows=P.N_WINDOWS):
    recs: list[ReplayRecord] = []
    ep = Episode(uuid4(), run_id=1, fault=0, frame=frame)
    r = Replayer(ListSource([ep]), n=1, expected_sha=order_sha([(1, 0)]), on_record=recs.append)
    # single-episode run: expected sha covers only (1,0); run() would also require fault 0..20 – bypass via _replay_episode
    r._replay_episode(ep, make_policy(block))
    return recs


def test_a_future_change_and_truncation_do_not_alter_past():
    base = synth_frame(1)
    recs0 = replay(base)
    t = 17
    changed = base.copy(); changed.loc[changed["cycle"] > (t + 1) * P.STRIDE, PROCESS_COLS] += 100.0
    recs1 = replay(changed)
    key = lambda r: (r.window_index, r.gate, r.p_value, r.p_adj, r.m_t, r.alarm, r.action)
    assert [key(r) for r in recs0[: t + 1]] == [key(r) for r in recs1[: t + 1]]
    # truncation: an episode cut at window t gives identical records for 0..t
    cut = base[base["cycle"] <= (t + 1) * P.STRIDE]
    st = EpisodeState(uuid4(), P.ALPHA)
    pol = make_policy()
    for k in range(t + 1):
        v = WindowView.build(st.episode_id, k, cut)
        a, g, p = pol(v, st, WindowState(k)); d = st.observe(k, g, p)
        assert (d.p_adj, d.alarm, st.m_t) == (recs0[k].p_adj, recs0[k].alarm, recs0[k].m_t)


def test_b_block_adds_no_pvalue_and_fewer_tests():
    base = synth_frame(2)
    plain = replay(base)
    blocked = replay(base, block=(3, 10, 11))
    assert plain[-1].m_t == P.N_WINDOWS
    assert blocked[-1].m_t == P.N_WINDOWS - 3                        # fewer tests than same-length non-BLOCK run
    for r in blocked:
        if r.gate == "BLOCK":
            assert r.p_value is None and r.alarm is None and r.action == "undecidable"
    for i in (3, 10, 11):                                            # m_t unchanged across a BLOCK window
        assert blocked[i].m_t == blocked[i - 1].m_t


def test_c_m_t_non_decreasing():
    for block in ((), (0, 1, 2), (5, 20, 47)):
        recs = replay(synth_frame(3), block=block)
        m = [r.m_t for r in recs]
        assert all(b >= a for a, b in zip(m, m[1:]))


def test_d_confirmed_alarm_immutable():
    st = EpisodeState(uuid4(), P.ALPHA)
    ps = [0.001, 0.5, 0.9, 0.02, 0.0001, 0.3]
    decisions = []
    for t, p in enumerate(ps):
        decisions.append(st.observe(t, "PASS", p))
        for earlier in decisions[:-1]:                               # earlier decisions untouched by later p's
            assert st.decision_at(earlier.window_index) == earlier
    with pytest.raises(ValueError):
        st.observe(3, "PASS", 0.5)                                   # out of order / re-observation refused
    with pytest.raises(ValueError):
        EpisodeState(uuid4(), P.ALPHA).observe(0, "BLOCK", 0.1)      # BLOCK must not carry a p


def test_e_view_refuses_future_and_strips_labels():
    frame = synth_frame(4, with_labels=True)
    v = WindowView.build(uuid4(), 5, frame)
    assert v.end_cycle == 120 and len(v) == 120
    assert not ({"fault_scenario", "is_fault_active"} & set(v.history().columns))
    with pytest.raises(FutureAccessError):
        v.cycles(1, 121)
    assert len(v.expanded()) == 40 and WindowView.build(uuid4(), 0, frame).expanded() is None


def test_f_order_equals_frozen_and_shuffle_detected():
    doc = json.loads(SPLIT.read_text())
    order = frozen_order(doc, 3)
    eps = [Episode(uuid4(), r, f, synth_frame(100 + 21 * r + f, n_cycles=P.RUN_LENGTH)) for r, f in order]
    rep = Replayer(ListSource(eps), n=3, expected_sha=order_sha(order))
    out = rep.run(make_policy())
    assert out["episodes"] == 63 and out["order_sha256"] == order_sha(order)
    shuffled = [eps[1], eps[0]] + eps[2:]
    with pytest.raises(RuntimeError):
        Replayer(ListSource(shuffled), n=3, expected_sha=order_sha(order)).run(make_policy())


def test_g_q_used_resets_each_window():
    recs = replay(synth_frame(5))
    assert any(r.q_used for r in recs) and any(not r.q_used for r in recs)
    assert all(r.q_used == (r.action in ("propose", "hold")) for r in recs)
