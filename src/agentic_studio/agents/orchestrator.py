from __future__ import annotations

from agentic_studio.agents.analyst import AnalystAgent
from agentic_studio.agents.planner import PlannerAgent
from agentic_studio.agents.retriever import RetrieverAgent
from agentic_studio.agents.skeptic import SkepticAgent
from agentic_studio.agents.synthesizer import SynthesizerAgent
from agentic_studio.core.config import settings
from agentic_studio.core.guardrails import (
    detect_prompt_injection,
    sanitize_user_text,
    trusted_context_prefix,
)
from agentic_studio.core.llm import LLM
from agentic_studio.core.memory import WorkingMemory
from agentic_studio.core.models import AgentResult, OrchestrationOutput
from agentic_studio.evals.metrics import evaluate_output
from agentic_studio.rag.index import HybridRAGIndex


class AgenticOrchestrator:
    def __init__(self, index: HybridRAGIndex, memory: WorkingMemory | None = None) -> None:
        llm = LLM()
        self.memory = memory or WorkingMemory()
        self.planner = PlannerAgent(llm)
        self.retriever = RetrieverAgent(index)
        self.analyst = AnalystAgent(llm)
        self.skeptic = SkepticAgent(llm)
        self.synthesizer = SynthesizerAgent(llm)

    def run(self, question: str, objective: str) -> OrchestrationOutput:
        safe_question = sanitize_user_text(question)
        if detect_prompt_injection(safe_question):
            safe_question = (
                "Potential prompt injection detected and blocked. "
                "Please rephrase your request as a domain question only."
            )

        hardened_objective = f"{trusted_context_prefix()} Objective: {objective}"

        plan = self.planner.run(safe_question, hardened_objective)
        self.memory.add("planner", plan)

        hits = self.retriever.run(safe_question, top_k=settings.max_chunks)
        evidence_summary = "\n".join(
            f"- {h.chunk.title} ({h.chunk.source}) score={h.score:.3f}" for h in hits
        )
        self.memory.add("retriever", evidence_summary)

        draft = self.analyst.run(
            question=safe_question,
            plan=plan,
            hits=hits,
            memory_text=self.memory.render(),
        )
        self.memory.add("analyst", draft[:1200])

        critique = self.skeptic.run(draft=draft, question=safe_question)
        self.memory.add("skeptic", critique[:1200])

        final_answer = self.synthesizer.run(draft=draft, critique=critique)
        self.memory.add("synthesizer", final_answer[:1200])

        metrics = evaluate_output(final_answer, hits)

        trace = [
            AgentResult(name="planner", output=plan),
            AgentResult(name="retriever", output=evidence_summary, artifacts={"hits": hits}),
            AgentResult(name="analyst", output=draft),
            AgentResult(name="skeptic", output=critique),
            AgentResult(name="synthesizer", output=final_answer),
        ]

        return OrchestrationOutput(
            final_answer=final_answer,
            citations=hits,
            critique=critique,
            plan=plan,
            metrics=metrics,
            trace=trace,
        )