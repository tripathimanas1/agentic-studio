from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
class AgentResult:
    name: str
    output: str
    reasoning: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrchestrationOutput:
    final_answer: str
    citations: list[RetrievalHit]
    critique: str
    plan: str
    metrics: dict[str, float]
    trace: list[AgentResult]