from __future__ import annotations

from agentic_studio.core.models import DocumentChunk
from agentic_studio.rag.chunker import chunk_text


class Ingestor:
    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []

    @property
    def chunks(self) -> list[DocumentChunk]:
        return self._chunks

    def add_text(self, source: str, title: str, text: str) -> int:
        parts = list(chunk_text(text))
        for i, chunk in enumerate(parts, start=1):
            chunk_id = f"{source}:{i}"
            self._chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source=source,
                    title=title,
                    text=chunk,
                    metadata={"part": i, "parts": len(parts)},
                )
            )
        return len(parts)