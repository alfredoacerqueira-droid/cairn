"""Hybrid retrieval package — multi-strategy code search.

Provides factory function and exports for BM25 lexical, AST-graph PageRank,
embedding-based semantic, and hybrid (reciprocal-rank fusion) retrievers.
"""

from pipeline.retrieval.ast_rank import ASTRankRetriever
from pipeline.retrieval.bm25 import BM25Retriever
from pipeline.retrieval.embeddings import EmbeddingRetriever
from pipeline.retrieval.hybrid import HybridRetriever

__all__ = ["BM25Retriever", "ASTRankRetriever", "EmbeddingRetriever", "HybridRetriever"]
