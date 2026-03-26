from __future__ import annotations

from agentic_studio.core.llm import LLM


class PlannerAgent:
    name = "planner"

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def run(self, question: str, objective: str) -> str:
        prompt = (
            "Create a concise analysis plan for the question below. Include: goals, required evidence, "
            "unknowns, and success criteria.\n\n"
            f"Objective: {objective}\n"
            f"Question: {question}"
        )
        return self.llm.generate(prompt).text