"""ChromaDB vector indexer for semantic code search."""

import logging
import time
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from core.repo import project_id
from server.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# Maximum batch size for ChromaDB upserts. The client reports 5461, but we use
# a conservative 5000 to leave headroom.
_MAX_UPSERT_BATCH = 5000


def _derive_project_root(chroma_path: str | Path) -> Path | None:
    """Recover the project root from a standard ``<project>/.cairn/chroma`` path.

    Returns None for non-standard chroma paths (e.g. bare tmp dirs used by some
    low-level unit tests), which keeps the legacy un-namespaced collection.
    """
    p = Path(chroma_path).resolve()
    if p.name == "chroma" and p.parent.name == ".cairn":
        return p.parent.parent
    return None


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
        project_root: Optional[str | Path] = None,
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

        # Multi-repo isolation: namespace the collection per project and stamp
        # provenance metadata on every record. project_root may be passed
        # explicitly (reader path), but we also DERIVE it from a standard
        # "<project>/.cairn/chroma" chroma_path so that every indexing call site
        # (CLI reindex/init, sync engine, janitor) lands in the SAME namespaced
        # collection the reader queries — without having to touch all of them.
        if project_root is None:
            project_root = _derive_project_root(self.chroma_path)
        if project_root is not None:
            self.project_root: str | None = str(Path(project_root).resolve())
            self.project_id: str | None = project_id(project_root)
            collection_name = f"functions_{self.project_id}"
        else:
            self.project_root = None
            self.project_id = None
            collection_name = "functions"

        self.client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
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

        metadata = {
            "filepath": filepath,
            "function": function_name,
            "line_start": line_start,
            "line_end": line_end,
            "indexed_at": time.time(),
        }
        # Add project provenance if project_root is set
        if self.project_id is not None:
            metadata["project_id"] = self.project_id
            metadata["project_root"] = self.project_root

        self.collection.upsert(  # type: ignore[arg-type]  # ChromaDB v1.x stubs
            ids=[doc_id],
            embeddings=[embedding],  # type: ignore[arg-type]
            metadatas=[metadata],  # type: ignore[arg-type]
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
            metadata = {
                "filepath": item["filepath"],
                "function": item["function_name"],
                "line_start": item["line_start"],
                "line_end": item["line_end"],
                "indexed_at": time.time(),
            }
            # Add project provenance if project_root is set
            if self.project_id is not None:
                metadata["project_id"] = self.project_id
                metadata["project_root"] = self.project_root
            metadatas.append(metadata)
            documents.append(item["code"])

        # Determine the ChromaDB max batch size defensively, falling back to 5000.
        try:
            client_max = self.client.get_max_batch_size()
        except Exception:
            client_max = None
        max_batch = min(_MAX_UPSERT_BATCH, client_max) if client_max else _MAX_UPSERT_BATCH

        # Split into sub-batches and upsert each chunk separately.
        num_items = len(ids)
        for i in range(0, num_items, max_batch):
            end_idx = min(i + max_batch, num_items)
            self.collection.upsert(  # type: ignore[arg-type]  # ChromaDB v1.x stubs
                ids=ids[i:end_idx],
                embeddings=all_embeddings[i:end_idx],  # type: ignore[arg-type]
                metadatas=metadatas[i:end_idx],  # type: ignore[arg-type]
                documents=documents[i:end_idx],
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
        """Semantic search for relevant functions (with embedding cache + multi-repo isolation)."""
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

        # NOTE: we deliberately do NOT add a {"project_id": ...} metadata filter to
        # the query. The collection is already namespaced per project
        # (functions_<id>), and on large repos a project_id where-clause makes
        # Chroma build a SQL `IN (...)` over every id, hitting SQLite's
        # "too many SQL variables" limit. Isolation is instead enforced by the
        # per-collection namespace + the in-memory drop of foreign ids below.

        results = self.collection.query(  # type: ignore[arg-type]  # ChromaDB v1.x stubs
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,  # type: ignore[arg-type]
        )

        formatted = self._format_results(results)

        # Final belt-and-suspenders: drop any results that don't match project_id
        # (should never happen, but if ChromaDB filter fails, we catch it here).
        if self.project_id is not None:
            filtered = []
            for result in formatted:
                result_pid = result.get("project_id")
                # Drop only a provably foreign record. A None id means a legacy /
                # un-stamped record in this (namespaced) collection — the where
                # filter already scoped the query, so keep it.
                if result_pid is not None and result_pid != self.project_id:
                    logger.warning(
                        "Dropped cross-project result %s (got project_id=%s, expected %s)",
                        result.get("filepath"),
                        result_pid,
                        self.project_id,
                    )
                else:
                    filtered.append(result)
            formatted = filtered

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
                    # Provenance: propagate so the isolation drop below (and the
                    # assembler) can verify each result's owning project.
                    "project_id": metadata.get("project_id"),
                    "project_root": metadata.get("project_root"),
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
