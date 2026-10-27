"""Streamlit Chat UI for Multi-Source Agentic RAG System.

Features:
- Multi-turn conversation memory using LangGraph MemorySaver via st.session_state.thread_id
- Real-time tool invocation indicators (document_search, web_search, wikipedia, stock, calculator)
- Hybrid RAG retrieval (BM25 + Chroma Vector with RRF fusion)
- Clean, modern chat interface with sidebar metrics and controls
"""
import uuid
from pathlib import Path

import streamlit as st

from graph import ask, get_last_tools_called

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Source Agentic RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 Multi-Source Agentic RAG System")
st.caption("Powered by LangGraph, Groq (openai/gpt-oss-120b), Hybrid Retrieval (Chroma + BM25 with RRF), and LangSmith")

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session-{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Sidebar Controls & System Status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Session Controls")
    st.write(f"**Current Thread ID:** `{st.session_state.thread_id}`")
    
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.thread_id = f"session-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.header("📚 Multi-Source Tools")
    st.markdown("""
    - 📄 **Document Search:** Hybrid RRF search (Chroma + BM25) over local PDFs.
    - 🌐 **Web Search:** Tavily Search for real-time web news and events.
    - 📖 **Wikipedia:** General conceptual, historical, and biographical queries.
    - 📈 **Stock Price:** Real-time Yahoo Finance ticker pricing.
    - 🧮 **Calculator:** High-precision NumExpr mathematical engine.
    """)

    st.divider()
    st.header("📊 Knowledge Base Status")
    chroma_exists = Path("chroma_db").exists()
    chunks_exists = Path("chunks.pkl").exists()

    if chroma_exists and chunks_exists:
        st.success("✅ Hybrid Index Ready (ChromaDB + chunks.pkl)")
    else:
        st.warning("⚠️ Index not found. Run ingest.py first.")

    st.divider()
    st.header("💡 Example Queries")
    st.markdown("""
    - *What is the course code and title for Module 1 in the uploaded document?*
    - *What is 15% of 2480 plus 350?*
    - *What is the latest closing stock price for NVDA?*
    - *Who was Alan Turing and what was his major contribution?*
    - *Search the web for the latest updates on James Webb Telescope.*
    """)

# ---------------------------------------------------------------------------
# Chat Display
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tools"):
            tools_str = ", ".join(f"`{t}`" for t in msg["tools"])
            st.caption(f"🛠️ **Tools used:** {tools_str}")

# ---------------------------------------------------------------------------
# User Input & Agent Processing
# ---------------------------------------------------------------------------
prompt = st.chat_input("Ask a question about documents, math, stocks, history, or the web...")

if prompt:
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt, "tools": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Invoke Agent via graph.ask()
    with st.chat_message("assistant"):
        with st.spinner("Agent is reasoning and selecting tools..."):
            try:
                response = ask(prompt, thread_id=st.session_state.thread_id)
                tools_called = get_last_tools_called(st.session_state.thread_id)

                st.markdown(response)
                if tools_called:
                    tools_str = ", ".join(f"`{t}`" for t in tools_called)
                    st.caption(f"🛠️ **Tools used:** {tools_str}")
                else:
                    st.caption("ℹ️ *Direct response (no external tools required)*")

                # Save assistant response to state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "tools": tools_called
                })
            except Exception as e:
                error_msg = f"An error occurred while running the agent: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "tools": []
                })
