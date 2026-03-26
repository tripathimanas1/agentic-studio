from __future__ import annotations

import math
import re
from collections import Counter

from agentic_studio.core.models import DocumentChunk, RetrievalHit


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class HybridRAGIndex:
    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        self._doc_tokens = [_tokens(c.text) for c in chunks]
        self._doc_tf = [Counter(toks) for toks in self._doc_tokens]
        self._df = Counter()
        for toks in self._doc_tokens:
            for t in set(toks):
                self._df[t] += 1
        self._n_docs = max(1, len(chunks))

    def search(self, query: str, top_k: int = 6) -> list[RetrievalHit]:
        if not self.chunks:
            return []

        q_tokens = _tokens(query)
        q_tf = Counter(q_tokens)

        semantic_scores = [self._tfidf_cosine(q_tf, d_tf) for d_tf in self._doc_tf]
        lexical_scores = [self._keyword_overlap(q_tokens, d_toks) for d_toks in self._doc_tokens]

        sem_rank = sorted(range(len(self.chunks)), key=lambda i: semantic_scores[i], reverse=True)
        lex_rank = sorted(range(len(self.chunks)), key=lambda i: lexical_scores[i], reverse=True)
        fused = self._rrf(sem_rank, lex_rank)

        top_ids = sorted(range(len(self.chunks)), key=lambda i: fused[i], reverse=True)[:top_k]
        return [
            RetrievalHit(chunk=self.chunks[i], score=float(fused[i]), channel="hybrid")
            for i in top_ids
        ]

    def _tfidf_cosine(self, q_tf: Counter[str], d_tf: Counter[str]) -> float:
        vocab = set(q_tf) | set(d_tf)
        if not vocab:
            return 0.0

        q_vec = []
        d_vec = []
        for term in vocab:
            idf = math.log((self._n_docs + 1) / (1 + self._df.get(term, 0))) + 1.0
            q_vec.append(q_tf.get(term, 0) * idf)
            d_vec.append(d_tf.get(term, 0) * idf)

        dot = sum(a * b for a, b in zip(q_vec, d_vec))
        q_norm = math.sqrt(sum(a * a for a in q_vec))
        d_norm = math.sqrt(sum(b * b for b in d_vec))
        if q_norm == 0 or d_norm == 0:
            return 0.0
        return dot / (q_norm * d_norm)

    @staticmethod
    def _keyword_overlap(q_tokens: list[str], d_tokens: list[str]) -> float:
        if not q_tokens or not d_tokens:
            return 0.0
        q_set = set(q_tokens)
        d_set = set(d_tokens)
        intersection = len(q_set & d_set)
        union = len(q_set | d_set)
        return intersection / max(1, union)

    @staticmethod
    def _rrf(rank_a: list[int], rank_b: list[int], k: int = 60) -> list[float]:
        n = len(rank_a)
        pos_a = [0] * n
        pos_b = [0] * n
        for i, doc_id in enumerate(rank_a):
            pos_a[doc_id] = i
        for i, doc_id in enumerate(rank_b):
            pos_b[doc_id] = i
        return [1.0 / (k + pos_a[i] + 1) + 1.0 / (k + pos_b[i] + 1) for i in range(n)]