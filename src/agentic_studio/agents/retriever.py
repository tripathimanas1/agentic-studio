from __future__ import annotations

from agentic_studio.core.models import RetrievalHit
from agentic_studio.rag.index import HybridRAGIndex


class RetrieverAgent:
    name = "retriever"

    def __init__(self, index: HybridRAGIndex) -> None:
        self.index = index

    def run(self, query: str, top_k: int = 6) -> list[RetrievalHit]:
        return self.index.search(query, top_k=top_k)