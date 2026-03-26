from __future__ import annotations

from agentic_studio.core.llm import LLM


class SynthesizerAgent:
    name = "synthesizer"

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def run(self, draft: str, critique: str) -> str:
        prompt = (
            "Produce a final executive-grade answer that integrates critique. "
            "Keep it evidence-grounded and actionable.\n\n"
            f"Draft Answer:\n{draft}\n\n"
            f"Critique:\n{critique}\n\n"
            "Output sections: Executive Summary, What We Know, What We Don't Know, Recommended Actions."
        )
        return self.llm.generate(prompt).text