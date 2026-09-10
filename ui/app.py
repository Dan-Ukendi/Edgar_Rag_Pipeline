"""Minimal demo UI for the Edgar RAG Pipeline API.

Usage:
    uv run streamlit run ui/app.py
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

TICKERS = ["", "AAPL", "JPM", "XOM", "JNJ", "WMT", "TSLA"]


def escape_dollars(text: str) -> str:
    """Streamlit renders text between two $ signs as LaTeX math, which mangles
    dollar amounts (e.g. "$21.354 billion ... $22.247 billion" gets parsed as
    one math block). Escaping $ as \\$ shows a literal dollar sign instead."""
    return text.replace("$", "\\$")

st.set_page_config(page_title="Edgar RAG Pipeline", page_icon="📊")
st.title("Edgar RAG Pipeline")
st.caption("Ask questions about 6 companies' SEC 10-K filings, grounded with inline citations.")

with st.sidebar:
    st.subheader("Settings")
    ticker = st.selectbox("Ticker filter", TICKERS, index=0, format_func=lambda t: t or "All companies")
    backend = st.selectbox("LLM backend", ["groq", "local"], index=0)
    top_k = st.slider("Chunks retrieved (top_k)", 1, 10, 5)

    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.subheader("API health")
        st.json(health)
    except requests.RequestException:
        st.error(f"Cannot reach API at {API_URL}")

question = st.text_input("Question", placeholder="What are Apple's main risk factors?")

if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving and generating..."):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={
                    "question": question,
                    "ticker": ticker or None,
                    "top_k": top_k,
                    "backend": backend,
                },
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
            result = None

    if result:
        st.subheader("Answer")
        st.write(escape_dollars(result["answer"]))
        st.caption(f"Backend: {result['backend']} | Cited: {result['cited_indices'] or 'none'}")

        st.subheader("Retrieved excerpts")
        for ex in result["excerpts"]:
            badge = "✅ cited" if ex["cited"] else "not cited"
            with st.expander(f"[{ex['index']}] {ex['ticker']} | {ex['filing_date']} | {ex['item_section']} | {badge}"):
                st.write(escape_dollars(ex["text"]))
