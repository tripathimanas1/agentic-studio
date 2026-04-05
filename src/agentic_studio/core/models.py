from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    source: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalHit:
    chunk: DocumentChunk
    score: float
    channel: str


@dataclass(slots=True)
class AgentMessage:
    role: str
    content: str


@dataclass(slots=True)
class ToolCall:
    """A tool invocation by the orchestrator."""
    tool_name: str
    arguments: dict[str, Any]
    reasoning: str = ""


@dataclass(slots=True)
class SkepticFinding:
    """A single finding from the skeptic analysis."""
    category: str  # hallucination, attack, weak_point, missing_evidence, overconfidence, logical_flaw
    severity: Literal["critical", "high", "medium", "low"]
    description: str
    evidence: str = ""
    recommendation: str = ""


@dataclass(slots=True)
class StructuredCritique:
    """Structured output from the skeptic agent."""
    hallucinations: list[SkepticFinding] = field(default_factory=list)
    attack_vectors: list[SkepticFinding] = field(default_factory=list)
    weak_points: list[SkepticFinding] = field(default_factory=list)
    missing_evidence: list[SkepticFinding] = field(default_factory=list)
    overconfidence: list[SkepticFinding] = field(default_factory=list)
    logical_flaws: list[SkepticFinding] = field(default_factory=list)
    overall_confidence: float = 0.0  # 0-1 scale
    summary: str = ""

    def to_text(self) -> str:
        """Convert to readable text format."""
        lines = [f"Overall Confidence: {self.overall_confidence:.0%}"]
        if self.summary:
            lines.append(f"\nSummary:\n{self.summary}")

        for category, findings in [
            ("Hallucinations", self.hallucinations),
            ("Attack Vectors", self.attack_vectors),
            ("Weak Points", self.weak_points),
            ("Missing Evidence", self.missing_evidence),
            ("Overconfidence Issues", self.overconfidence),
            ("Logical Flaws", self.logical_flaws),
        ]:
            if findings:
                lines.append(f"\n{category}:")
                for i, finding in enumerate(findings, 1):
                    lines.append(f"  {i}. [{finding.severity.upper()}] {finding.description}")
                    if finding.evidence:
                        lines.append(f"     Evidence: {finding.evidence}")
                    if finding.recommendation:
                        lines.append(f"     Recommendation: {finding.recommendation}")

        return "\n".join(lines)


@dataclass(slots=True)
class AgentResult:
    name: str
    output: str
    reasoning: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrchestrationOutput:
    final_answer: str
    citations: list[RetrievalHit]
    critique: str | StructuredCritique
    plan: str
    metrics: dict[str, float]
    trace: list[AgentResult]
    tool_calls: list[ToolCall] = field(default_factory=list)