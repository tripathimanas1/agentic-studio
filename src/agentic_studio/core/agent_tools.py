"""Agent tools for the agentic orchestrator.

These tools wrap the individual agents so they can be called by the orchestrator
based on reasoning rather than a fixed workflow.
"""

from __future__ import annotations

from typing import Any

from agentic_studio.core.tools import ToolParameter, ToolRegistry, register_tool
from agentic_studio.core.models import RetrievalHit


# Global references to agents - set by orchestrator
_planner_agent = None
_retriever_agent = None
_analyst_agent = None
_skeptic_agent = None
_synthesizer_agent = None

# Global storage for retrieved hits (for later use)
_last_retrieved_hits: list[RetrievalHit] = []


def set_agents(
    planner: Any,
    retriever: Any,
    analyst: Any,
    skeptic: Any,
    synthesizer: Any,
) -> None:
    """Set agent references for tool calls."""
    global _planner_agent, _retriever_agent, _analyst_agent, _skeptic_agent, _synthesizer_agent
    _planner_agent = planner
    _retriever_agent = retriever
    _analyst_agent = analyst
    _skeptic_agent = skeptic
    _synthesizer_agent = synthesizer


def get_last_retrieved_hits() -> list[RetrievalHit]:
    """Get the last retrieved hits."""
    return _last_retrieved_hits


@register_tool(
    name="create_analysis_plan",
    description="Create a structured analysis plan with goals, evidence requirements, and success criteria",
    parameters=[
        ToolParameter(
            name="question",
            description="The question to analyze",
            type="string",
        ),
        ToolParameter(
            name="objective",
            description="The overall objective or context",
            type="string",
        ),
    ],
)
def plan_analysis(question: str, objective: str) -> str:
    """Plan tool: Create analysis plan."""
    if _planner_agent is None:
        return "[ERROR] Planner agent not initialized"
    return _planner_agent.run(question, objective)


@register_tool(
    name="retrieve_evidence",
    description="Retrieve relevant documents and evidence from the knowledge base",
    parameters=[
        ToolParameter(
            name="query",
            description="Search query to find relevant documents",
            type="string",
        ),
        ToolParameter(
            name="top_k",
            description="Number of documents to retrieve (default: 6)",
            type="number",
        ),
    ],
)
def retrieve_documents(query: str, top_k: int = 6) -> str:
    """Retriever tool: Fetch relevant documents."""
    global _last_retrieved_hits
    if _retriever_agent is None:
        return "[ERROR] Retriever agent not initialized"

    hits: list[RetrievalHit] = _retriever_agent.run(query, top_k=top_k)
    _last_retrieved_hits = hits

    evidence_summary = "\n".join(
        f"- [{h.chunk.title}] {h.chunk.source} (score: {h.score:.3f})\n  {h.chunk.text[:200]}..."
        for h in hits[:top_k]
    )
    return evidence_summary if evidence_summary else "No relevant documents found."


@register_tool(
    name="analyze_with_evidence",
    description="Draft an analysis using the plan and retrieved evidence",
    parameters=[
        ToolParameter(
            name="question",
            description="The original question",
            type="string",
        ),
        ToolParameter(
            name="plan",
            description="The analysis plan from the planner",
            type="string",
        ),
        ToolParameter(
            name="memory_context",
            description="Previous analysis context from memory",
            type="string",
            required=False,
        ),
    ],
)
def analyze_evidence(
    question: str,
    plan: str,
    memory_context: str = "",
) -> str:
    """Analyst tool: Analyze evidence against a plan."""
    if _analyst_agent is None:
        return "[ERROR] Analyst agent not initialized"

    return _analyst_agent.run(
        question=question,
        plan=plan,
        hits=_last_retrieved_hits,
        memory_text=memory_context,
    )


@register_tool(
    name="critique_draft",
    description="Critically review a draft for hallucinations, weak points, attacks, and logical flaws",
    parameters=[
        ToolParameter(
            name="draft",
            description="The draft answer to critique",
            type="string",
        ),
        ToolParameter(
            name="question",
            description="The original question",
            type="string",
        ),
    ],
)
def critique_answer(draft: str, question: str) -> str:
    """Skeptic tool: Critique draft answer."""
    if _skeptic_agent is None:
        return "[ERROR] Skeptic agent not initialized"

    structured_critique = _skeptic_agent.run(draft, question)
    return structured_critique.to_text()


@register_tool(
    name="synthesize_final_answer",
    description="Integrate critique into a final executive-grade answer",
    parameters=[
        ToolParameter(
            name="draft",
            description="The draft answer",
            type="string",
        ),
        ToolParameter(
            name="critique",
            description="The critique/feedback on the draft",
            type="string",
        ),
    ],
)
def synthesize(draft: str, critique: str) -> str:
    """Synthesizer tool: Create final answer."""
    if _synthesizer_agent is None:
        return "[ERROR] Synthesizer agent not initialized"

    return _synthesizer_agent.run(draft, critique)


def get_tool_registry() -> ToolRegistry:
    """Get the tool registry with all agent tools."""
    from agentic_studio.core.tools import get_tool_registry as get_registry
    return get_registry()
