from __future__ import annotations

from agentic_studio.core.llm import LLM
from agentic_studio.core.models import RetrievalHit


class AnalystAgent:
    name = "analyst"

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def run(
        self,
        question: str,
        plan: str,
        hits: list[RetrievalHit] | None = None,
        memory_text: str = "",
    ) -> str:
        """Analyze question using plan and evidence."""
        hits = hits or []

        evidence = "\n\n".join(
            f"[{i}] {h.chunk.title} ({h.chunk.source})\n{h.chunk.text[:500]}"
            for i, h in enumerate(hits, start=1)
        )

        if not evidence:
            evidence = "[No evidence provided]"

        prompt = (
            "Answer the strategy question using only provided evidence where possible. "
            "Be explicit about assumptions and uncertainty.\n\n"
            f"Plan:\n{plan}\n\n"
            f"Working memory:\n{memory_text}\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"Question:\n{question}\n\n"
            "Return sections: Thesis, Supporting Evidence, Risks, Experiments."
        )
        return self.llm.generate(prompt).text