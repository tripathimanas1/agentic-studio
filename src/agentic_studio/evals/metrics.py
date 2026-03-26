from __future__ import annotations

import re

from agentic_studio.core.models import RetrievalHit


def evaluate_output(answer: str, hits: list[RetrievalHit]) -> dict[str, float]:
    if not hits:
        return {
            "citation_coverage": 0.0,
            "context_precision": 0.0,
            "risk_score": 1.0,
        }

    citation_mentions = len(re.findall(r"\[[0-9]+\]", answer))
    citation_coverage = min(1.0, citation_mentions / max(1, len(hits)))

    avg_relevance = sum(h.score for h in hits) / len(hits)
    context_precision = min(1.0, avg_relevance * 100)

    hedges = len(re.findall(r"\b(maybe|might|uncertain|assume|estimate)\b", answer.lower()))
    overclaim = len(re.findall(r"\b(always|definitely|guaranteed|certain)\b", answer.lower()))
    risk_score = min(1.0, (overclaim + 1) / (hedges + 2))

    return {
        "citation_coverage": round(citation_coverage, 3),
        "context_precision": round(context_precision, 3),
        "risk_score": round(risk_score, 3),
    }