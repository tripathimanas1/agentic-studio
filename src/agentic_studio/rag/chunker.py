from __future__ import annotations

from typing import Iterable


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> Iterable[str]:
    clean = " ".join(text.split())
    if not clean:
        return []

    chunks: list[str] = []
    start = 0
    n = len(clean)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(clean[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks