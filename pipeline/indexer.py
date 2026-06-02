"""ChromaDB vector indexer for semantic code search."""

import time
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from server.ollama_client import OllamaClient


class VectorIndexer:
    # Placeholder vector stored when embeddings are disabled (iac/shell profiles).
    # ChromaDB requires *some* embedding per upsert, but these profiles never run
    # a vector query (no EmbeddingRetriever is built) — ChromaDB is used only as a
    # document/metadata store for the BM25 + structural legs. A constant 1-dim
    # vector avoids ~1 Ollama embed call per block (and loading the embed model
    # into VRAM at all), which is the whole point of embeddings=OFF.
    _PLACEHOLDER_EMBEDDING = [0.0]

    def __init__(
        self,
        chroma_path: str | Path,
        ollama_client: Optional[OllamaClient] = None,
        embedding_model: Optional[str] = None,
        cache=None,
        embeddings_enabled: bool = True,
    ):
        self.chroma_path = str(chroma_path)
        self.ollama = ollama_client or OllamaClient()
        # Resolution precedence: explicit arg > OllamaClient's configured model
        # OllamaClient already resolves OLLAMA_EMBED_MODEL env or defaults to
        # "nomic-embed-text". This ensures indexing and querying use the same
        # model consistently.
        self.embedding_model = embedding_model or self.ollama.embed_model
        self.cache = cache
        # When False (iac/shell profiles), skip Ollama embedding at index time and
        # store a placeholder vector instead — see _PLACEHOLDER_EMBEDDING.
        self.embeddings_enabled = embeddings_enabled

        self.client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="functions",
            metadata={"hnsw:space": "cosine"},
        )

    def embed_function(self, code: str) -> list[float]:
        """Generate embedding for a code snippet."""
        if not self.embeddings_enabled:
            return list(self._PLACEHOLDER_EMBEDDING)
        return self.ollama.embed(code, model=self.embedding_model)

    def index_function(
        self,
        filepath: str,
        function_name: str,
        code: str,
        line_start: int,
        line_end: int,
    ):
        """Index a single function with metadata."""
        embedding = self.embed_function(code)
        doc_id = f"{filepath}:{function_name}:{line_start}"

        self.collection.upsert(  # type: ignore[arg-type]  # ChromaDB v1.x stubs
            ids=[doc_id],
            embeddings=[embedding],  # type: ignore[arg-type]
            metadatas=[  # type: ignore[arg-type]
                {
                    "filepath": filepath,
                    "function": function_name,
                    "line_start": line_start,
                    "line_end": line_end,
                    "indexed_at": time.time(),
                }
            ],
            documents=[code],
        )

    def index_ast(self, ast_result, batch_size: int = 8):
        """Index all functions from an AST result using batched embeddings."""
        items = []

        for func in ast_result.functions:
            items.append(
                {
                    "filepath": ast_result.filepath,
                    "function_name": func.name,
                    "code": func.code,
                    "line_start": func.line_start,
                    "line_end": func.line_end,
                }
            )

        for cls in ast_result.classes:
            for method in cls.methods:
                items.append(
                    {
                        "filepath": ast_result.filepath,
                        "function_name": f"{cls.name}.{method.name}",
                        "code": method.code,
                        "line_start": method.line_start,
                        "line_end": method.line_end,
                    }
                )

        if not items:
            return

        codes = [item["code"] for item in items]
        if self.embeddings_enabled:
            embeddings = self.ollama.embed_batch(codes, model=self.embedding_model)
        else:
            # embeddings=OFF: store placeholders, never call Ollama (saves the
            # embed model load + one call per block on iac/shell profiles).
            embeddings = [list(self._PLACEHOLDER_EMBEDDING) for _ in codes]

        ids = []
        all_embeddings = []
        metadatas = []
        documents = []

        for item, embedding in zip(items, embeddings):
            doc_id = f"{item['filepath']}:{item['function_name']}:{item['line_start']}"
            ids.append(doc_id)
            all_embeddings.append(embedding)
            metadatas.append(
                {
                    "filepath": item["filepath"],
                    "function": item["function_name"],
                    "line_start": item["line_start"],
                    "line_end": item["line_end"],
                    "indexed_at": time.time(),
                }
            )
            documents.append(item["code"])

        self.collection.upsert(  # type: ignore[arg-type]  # ChromaDB v1.x stubs
            ids=ids,
            embeddings=all_embeddings,  # type: ignore[arg-type]
            metadatas=metadatas,  # type: ignore[arg-type]
            documents=documents,
        )

    def remove_file(self, filepath: str):
        """Remove all functions for a file."""
        results = self.collection.get(where={"filepath": filepath})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])

    def search(
        self,
        query: str,
        top_k: int = 5,
        filepath_prefix: Optional[str] = None,
        metrics=None,
    ) -> list[dict]:
        """Semantic search for relevant functions (with embedding cache)."""
        start = time.perf_counter()

        # Check cache for query embedding
        if self.cache:
            cached_embedding = self.cache.get("embedding", query, self.embedding_model)
            if cached_embedding is not None:
                query_embedding = cached_embedding
            else:
                query_embedding = self.ollama.embed(query, model=self.embedding_model)
                self.cache.set(query_embedding, "embedding", query, self.embedding_model)
        else:
            query_embedding = self.ollama.embed(query, model=self.embedding_model)

        where_filter = None
        if filepath_prefix:
            where_filter = {"filepath": {"$startswith": filepath_prefix}}

        results = self.collection.query(  # type: ignore[arg-type]  # ChromaDB v1.x stubs
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,  # type: ignore[arg-type]
        )

        formatted = self._format_results(results)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if metrics:
            metrics.record_search(query, len(formatted), elapsed_ms)

        return formatted

    def _format_results(self, results) -> list[dict]:
        """Format ChromaDB results into a usable list."""
        formatted = []

        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            document = documents[i] if i < len(documents) else ""
            distance = distances[i] if i < len(distances) else 0.0

            # Convert distance to similarity (cosine)
            similarity = 1.0 - (distance / 2.0) if distance else 1.0

            formatted.append(
                {
                    "id": doc_id,
                    "filepath": metadata.get("filepath", ""),
                    "function": metadata.get("function", ""),
                    "line_start": metadata.get("line_start", 0),
                    "line_end": metadata.get("line_end", 0),
                    "code": document,
                    "similarity": similarity,
                }
            )

        return formatted

    def count(self) -> int:
        """Get total number of indexed functions."""
        return self.collection.count()

    def clear(self):
        """Delete all vectors from the collection."""
        self.client.delete_collection("functions")
        self.collection = self.client.get_or_create_collection(
            name="functions",
            metadata={"hnsw:space": "cosine"},
        )
