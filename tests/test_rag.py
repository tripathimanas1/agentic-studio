from agentic_studio.core.models import DocumentChunk
from agentic_studio.rag.index import HybridRAGIndex


def test_hybrid_rag_returns_relevant_hit() -> None:
    chunks = [
        DocumentChunk(chunk_id="1", source="a", title="A", text="Agentic systems use planning and tools."),
        DocumentChunk(chunk_id="2", source="b", title="B", text="Cooking pasta requires boiling water."),
    ]
    idx = HybridRAGIndex(chunks=chunks)
    hits = idx.search("How do agentic tools improve planning?", top_k=1)
    assert hits
    assert hits[0].chunk.source == "a"