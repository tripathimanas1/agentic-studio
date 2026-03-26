from __future__ import annotations

import textwrap

import streamlit as st

from agentic_studio.agents.orchestrator import AgenticOrchestrator
from agentic_studio.core.memory import WorkingMemory
from agentic_studio.rag.index import HybridRAGIndex
from agentic_studio.rag.ingest import Ingestor


def _seed_documents(ingestor: Ingestor) -> None:
    docs = [
        (
            "playbook-1",
            "Agentic Product Architecture",
            "Agentic systems should separate planning, execution, verification, and memory. "
            "A robust production stack includes task decomposition, tool contracts, and rollback-safe execution. "
            "Human-in-the-loop checkpoints are critical for high-impact actions.",
        ),
        (
            "playbook-2",
            "RAG 2.0 Patterns",
            "Modern RAG combines dense retrieval, lexical retrieval, and reranking. "
            "Retrieval quality improves with query rewriting, metadata filtering, and reciprocal rank fusion. "
            "Citations must map to trusted sources and evidence freshness windows.",
        ),
        (
            "playbook-3",
            "GenAI Reliability",
            "Reliable GenAI requires layered guardrails: policy filters, prompt hardening, output validation, and audit logs. "
            "Self-critique and adversarial agents can lower hallucination rates and improve decision quality.",
        ),
        (
            "playbook-4",
            "Agent Evaluation",
            "Evaluation should track task success, citation coverage, groundedness, latency, and intervention rate. "
            "Continuous evaluation with fixed test sets and shadow traffic catches regressions early.",
        ),
    ]
    for source, title, text in docs:
        ingestor.add_text(source=source, title=title, text=text)


def _ensure_state() -> None:
    if "ingestor" not in st.session_state:
        st.session_state.ingestor = Ingestor()
    if "memory" not in st.session_state:
        st.session_state.memory = WorkingMemory(max_items=60)
    if "index" not in st.session_state:
        st.session_state.index = None
    if "result" not in st.session_state:
        st.session_state.result = None


def _build_index() -> None:
    chunks = st.session_state.ingestor.chunks
    st.session_state.index = HybridRAGIndex(chunks=chunks)


def _render_header() -> None:
    st.set_page_config(page_title="Agentic Strategy Studio", page_icon="AI", layout="wide")
    st.markdown(
        """
        <style>
        :root {
            --ink: #18212f;
            --muted: #334155;
            --panel: rgba(255, 255, 255, 0.72);
            --panel-border: #d6dbe8;
        }
        .stApp {
            color: var(--ink);
            background: radial-gradient(circle at 20% 10%, #f9f4e8, #eef5ff 50%, #f5f2ff 100%);
        }
        .stApp, .stMarkdown, .stText, p, li, label, span, div {
            color: var(--ink);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255, 250, 241, 0.96), rgba(238, 247, 255, 0.96));
            border-right: 1px solid var(--panel-border);
        }
        section[data-testid="stSidebar"] * {
            color: var(--ink) !important;
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            color: var(--ink);
            font-weight: 600;
        }
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
            background: var(--panel);
            border: 1px solid var(--panel-border);
        }
        input, textarea {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }
        input::placeholder, textarea::placeholder {
            color: var(--muted) !important;
            opacity: 0.9;
        }
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
            color: var(--ink) !important;
        }
        .hero {
            border: 1px solid #d9d9ef;
            border-radius: 20px;
            padding: 1.2rem 1.4rem;
            background: linear-gradient(120deg, rgba(255,248,235,0.92), rgba(236,247,255,0.92));
            box-shadow: 0 8px 24px rgba(30, 40, 70, 0.08);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            letter-spacing: 0.3px;
            color: var(--ink);
        }
        .hero p {
            color: #27364b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero">
            <h1>Agentic Strategy Studio</h1>
            <p>Multi-agent intelligence system for high-stakes strategy questions with hybrid RAG, critique loops, and evals.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run() -> None:
    _ensure_state()
    _render_header()

    with st.sidebar:
        st.subheader("Mission")
        objective = st.text_area(
            "Objective",
            value="Design a defensible AI product strategy grounded in evidence and clear risk controls.",
            height=120,
        )
        st.caption("Objective steers planner behavior across all agents.")

        if st.button("Load Curated Agentic Docs"):
            _seed_documents(st.session_state.ingestor)
            _build_index()
            st.success("Loaded expert seed docs and rebuilt index.")

        if st.button("Rebuild Retrieval Index"):
            _build_index()
            st.success("Index rebuilt.")

        st.divider()
        st.write(f"Chunks in knowledge base: **{len(st.session_state.ingestor.chunks)}**")

    col_left, col_right = st.columns([1.05, 1.35], gap="large")

    with col_left:
        st.subheader("Knowledge Ingestion")
        title = st.text_input("Document title", value="Internal Strategy Memo")
        source = st.text_input("Source id", value="user-note")
        text = st.text_area(
            "Document text",
            height=220,
            placeholder="Paste research notes, incident reports, customer interviews, design docs, or policy briefs...",
        )
        if st.button("Add Document"):
            if not text.strip():
                st.warning("Please add non-empty document text.")
            else:
                n = st.session_state.ingestor.add_text(source=source, title=title, text=text)
                _build_index()
                st.success(f"Added {n} chunk(s) and rebuilt index.")

        st.subheader("Ask the Agentic System")
        question = st.text_area(
            "Question",
            value=(
                "Given our evidence, what is the best 90-day roadmap for launching an agentic AI copilot in "
                "a regulated B2B setting, and what failure modes should we proactively mitigate?"
            ),
            height=150,
        )
        run_clicked = st.button("Run Multi-Agent Analysis", type="primary")

        if run_clicked:
            if st.session_state.index is None:
                _build_index()
            orchestrator = AgenticOrchestrator(
                index=st.session_state.index,
                memory=st.session_state.memory,
            )
            with st.spinner("Agents are planning, retrieving, critiquing, and synthesizing..."):
                st.session_state.result = orchestrator.run(question=question, objective=objective)

    with col_right:
        st.subheader("Results")
        result = st.session_state.result
        if result is None:
            st.info("Run an analysis to view outputs, citations, critique, and metrics.")
        else:
            tabs = st.tabs(["Final", "Metrics", "Trace", "Citations", "Critique", "Plan"])
            with tabs[0]:
                st.markdown(result.final_answer)
            with tabs[1]:
                m = result.metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Citation Coverage", m.get("citation_coverage", 0.0))
                c2.metric("Context Precision", m.get("context_precision", 0.0))
                c3.metric("Risk Score", m.get("risk_score", 0.0))
                st.caption("Lower risk score is better. Metrics are lightweight and can be replaced by formal evals.")
            with tabs[2]:
                for step in result.trace:
                    with st.expander(step.name.capitalize(), expanded=False):
                        st.write(step.output)
            with tabs[3]:
                for i, hit in enumerate(result.citations, start=1):
                    st.markdown(
                        textwrap.dedent(
                            f"""
                            **[{i}] {hit.chunk.title}** (`{hit.chunk.source}` | score={hit.score:.4f})

                            {hit.chunk.text[:450]}...
                            """
                        )
                    )
            with tabs[4]:
                st.write(result.critique)
            with tabs[5]:
                st.write(result.plan)


if __name__ == "__main__":
    run()