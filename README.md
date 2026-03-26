# Agentic Strategy Studio

An opinionated, production-minded starter app that demonstrates advanced Agentic AI patterns in one cohesive workflow:

- Multi-agent orchestration (`Planner -> Retriever -> Analyst -> Skeptic -> Synthesizer`)
- Hybrid RAG (semantic + lexical retrieval with reciprocal rank fusion)
- Working memory and run history
- Guardrails (PII sanitization + prompt-injection hardening)
- Self-reflection and critique loop
- Evidence-backed answers with source citations
- Lightweight evaluation panel (citation coverage, context precision, response risk)
- Human-in-the-loop controls from the UI

## Use Case

The app is focused on **strategy and due-diligence intelligence**.  
You can load internal notes/documents and ask high-stakes business questions.  
Agents collaborate to produce an evidence-grounded answer plus concerns and next actions.

## Quick Start

```bash
pip install -r requirements.txt
set PYTHONPATH=src
streamlit run streamlit_app.py
```

Optional:

```bash
set GEMINI_API_KEY=your_key
set GEMINI_MODEL=gemini-2.0-flash
```

Without an API key, the app runs in deterministic mock mode for local testing.

## Architecture

- `src/agentic_studio/rag`: ingestion, chunking, and hybrid retrieval
- `src/agentic_studio/agents`: specialized agents and orchestrator
- `src/agentic_studio/core`: LLM adapter, guardrails, memory, schemas
- `src/agentic_studio/evals`: quality and risk metrics
- `src/agentic_studio/ui`: Streamlit interface

## Topics Demonstrated

1. Agent planning and role specialization  
2. Tool-using retrieval agent  
3. Multi-stage reasoning pipelines  
4. Retrieval-grounded generation  
5. Defense-in-depth prompt safety  
6. Iterative critique and revision  
7. Memory and traceability  
8. Evaluation and quality gates  
9. Human approval touchpoints  
10. Extensible model provider abstraction

## Next Extensions

- Add SQL and web tools with explicit tool calling
- Add graph-RAG over entities/relationships
- Add automated eval harness with benchmark datasets
- Add async task queue for long-running autonomous jobs

