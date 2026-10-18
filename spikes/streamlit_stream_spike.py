"""0-D spike: does Streamlit Community Cloud stream tokens and keep multi-turn state?

No LLM, no database: a deterministic token generator stands in for the model so the
test isolates the hosting runtime (st.write_stream, session_state across reruns,
button-driven approval gate). Deploy this file as the app's main file on
Community Cloud (branch v2); it only needs `streamlit` from the root requirements.txt.

Local check:  streamlit run spikes/streamlit_stream_spike.py
"""

import time

import streamlit as st

st.set_page_config(page_title="0-D stream spike", layout="centered")
st.title("0-D: streaming + multi-turn + approval gate (no LLM)")

if "history" not in st.session_state:
    st.session_state.history = []          # list of {"role", "content"}
if "pending" not in st.session_state:
    st.session_state.pending = None        # proposal awaiting approval
if "audit" not in st.session_state:
    st.session_state.audit = []
if "timings" not in st.session_state:
    st.session_state.timings = []


def token_stream(prompt: str, n_tokens: int = 60, delay_s: float = 0.03):
    """Deterministic 'model': echoes a canned answer token by token."""
    words = (f"Received '{prompt}'. This reply is streamed token by token from a generator, "
             f"not from a model. Turn {len(st.session_state.history) // 2 + 1}. ").split()
    out = []
    while len(out) < n_tokens:
        out.extend(words)
    for w in out[:n_tokens]:
        yield w + " "
        time.sleep(delay_s)


for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if st.session_state.pending:
    with st.container(border=True):
        st.write(f"**Proposal awaiting approval:** {st.session_state.pending}")
        c1, c2 = st.columns(2)
        if c1.button("Approve", type="primary"):
            st.session_state.audit.append(f"approved: {st.session_state.pending}")
            st.session_state.pending = None
            st.rerun()
        if c2.button("Reject"):
            st.session_state.audit.append(f"rejected: {st.session_state.pending}")
            st.session_state.pending = None
            st.rerun()

prompt = st.chat_input("say something (type 'propose' to trigger the approval gate)")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        t0 = time.time()
        text = st.write_stream(token_stream(prompt))
        dt = round(time.time() - t0, 2)
    st.session_state.history.append({"role": "assistant", "content": text})
    st.session_state.timings.append(dt)
    if "propose" in prompt.lower():
        st.session_state.pending = f"work order for '{prompt}'"
    st.rerun()

with st.sidebar:
    st.subheader("Session evidence")
    st.write(f"turns: {len(st.session_state.history) // 2}")
    st.write(f"stream seconds per turn: {st.session_state.timings}")
    st.write("expected ≈ 60 tokens × 0.03 s = 1.8 s if streaming is incremental; "
             "a value near 0 s with text appearing all at once means buffering")
    st.write("audit:", st.session_state.audit)
    if st.button("Reset session"):
        for k in ("history", "pending", "audit", "timings"):
            st.session_state.pop(k, None)
        st.rerun()
