from __future__ import annotations

from agentic_studio.core.llm import LLM


class SkepticAgent:
    name = "skeptic"

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def run(self, draft: str, question: str) -> str:
        prompt = (
            "Critique the draft answer for hallucination risk, weak evidence, blind spots, and overconfidence. "
            "Suggest targeted fixes and missing tests.\n\n"
            f"Question: {question}\n\n"
            f"Draft:\n{draft}"
        )
        return self.llm.generate(prompt).text