"""BM25 lexical retriever over function text and docstrings.

Implements BM25 (Best Match 25) ranking, the standard probabilistic information
retrieval function.  Operates on (id, text) pairs — typically function bodies
with optional docstring content.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _BM25Doc:
    doc_id: str
    text: str
    tokens: list[str] = field(default_factory=list)


def _tokenize(text: str) -> list[str]:
    """Simple lowercase tokenizer splitting on whitespace and punctuation."""
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    """Lexical retriever using BM25 scoring over function text."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[_BM25Doc] = []
        self._doc_index: dict[str, _BM25Doc] = {}
        self._avgdl: float = 0.0
        self._df: dict[str, int] = defaultdict(int)
        self._N: int = 0

    def index(self, items: list[dict[str, Any]]) -> None:
        """Index documents from a list of dicts with 'id' and 'text' keys."""
        self._docs = []
        self._doc_index = {}
        self._df = defaultdict(int)
        self._N = len(items)

        total_len = 0
        for item in items:
            doc_id = item["id"]
            text = item.get("text", "")
            tokens = _tokenize(text)
            self._docs.append(_BM25Doc(doc_id=doc_id, text=text, tokens=tokens))
            self._doc_index[doc_id] = self._docs[-1]
            total_len += len(tokens)
            for t in set(tokens):
                self._df[t] += 1

        self._avgdl = total_len / self._N if self._N > 0 else 1.0

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Score and rank documents against a query string."""
        if self._N == 0:
            return []

        query_tokens = _tokenize(query)
        scores: list[tuple[float, str, str]] = []

        for doc in self._docs:
            score = self._bm25_score(query_tokens, doc)
            scores.append((score, doc.doc_id, doc.text))

        scores.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        for score, doc_id, text in scores[:top_k]:
            results.append(
                {
                    "id": doc_id,
                    "text": text,
                    "score": round(score, 4),
                    "source": "bm25",
                }
            )
        return results

    def _bm25_score(self, query_tokens: list[str], doc: _BM25Doc) -> float:
        dl = len(doc.tokens)
        score = 0.0
        for qt in set(query_tokens):
            tf = doc.tokens.count(qt)
            if tf == 0:
                continue
            idf = math.log(
                (self._N - self._df.get(qt, 0) + 0.5) / (self._df.get(qt, 0) + 0.5) + 1.0
            )
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            score += idf * numerator / denominator
        return score
