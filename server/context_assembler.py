"""Context assembler - stitches semantic search, repo map, and memory into surgical prompts.

Supports configurable hybrid retrieval (BM25, AST-graph PageRank, embeddings,
RRF fusion) via pipeline/retrieval/. Includes RTK-style token compression.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from core.cache import SessionCache
from core.config import load_config
from core.persistent_cache import PersistentCache
from core.profiles import get_profile
from core.repo import RepoManager, project_id
from pipeline.indexer import VectorIndexer
from pipeline.retrieval.ast_rank import ASTRankRetriever
from pipeline.retrieval.bm25 import BM25Retriever
from pipeline.retrieval.embeddings import EmbeddingRetriever
from pipeline.retrieval.hybrid import HybridRetriever
from pipeline.retrieval.reranker import Reranker
from server.ollama_client import OllamaClient, make_llm_client
from server.token_compressor import FilterLevel, Language, TokenCompressor

logger = logging.getLogger(__name__)


def _is_foreign_path(filepath: str, project_root: str) -> bool:
    """True if an ABSOLUTE filepath points outside ``project_root``.

    Result filepaths may carry a ``:symbol`` suffix (e.g. ``values.yaml:replicaCount``);
    we look only at the path portion. Relative paths are assumed in-repo (the
    legs that emit them are already scoped to this project).
    """
    if not filepath:
        return False
    head = filepath.split(":", 1)[0]
    if not os.path.isabs(head):
        return False
    head = os.path.realpath(head)
    return head != project_root and not head.startswith(project_root + os.sep)


class ContextAssembler:
    def __init__(
        self,
        project_path: Optional[Path] = None,
        ollama_client: Optional[OllamaClient] = None,
        top_k: int = 5,
        cache: Optional[SessionCache] = None,
    ):
        self.project_path = project_path or Path.cwd()
        self.repo = RepoManager(self.project_path)
        self.top_k = top_k

        # In-memory embedding cache (fast, resets per process)
        self.cache: SessionCache | None = None
        # Persistent cache for cross-process warmth (assemble_context results)
        self.persistent_cache: PersistentCache | None = None

        cfg = load_config(self.project_path)
        if cache is not None:
            self.cache = cache
        elif cfg.cache.enabled:
            self.cache = SessionCache(
                max_entries=cfg.cache.max_entries,
                ttl_seconds=cfg.cache.ttl_seconds,
            )
            # Wire persistent cache for assemble_context results
            cache_dir = self.project_path / ".cairn" / "cache"
            self.persistent_cache = PersistentCache(
                cache_dir=cache_dir,
                max_entries=cfg.cache.max_entries,
                ttl_seconds=cfg.cache.ttl_seconds,
            )

        # Use factory to build client based on local_llm config
        self.ollama = ollama_client or make_llm_client(cfg.local_llm)

        # Effective embeddings flag: only enable if both config and local LLM are enabled
        emb_enabled = cfg.embeddings_enabled and cfg.local_llm.enabled
        self.vector_indexer = VectorIndexer(
            chroma_path=self.repo.get_chroma_path(),
            ollama_client=self.ollama,
            cache=self.cache,
            embeddings_enabled=emb_enabled,
            project_root=self.project_path,
        )
        self.project_id: str | None = project_id(self.project_path)

        self._retrieval_mode = cfg.retrieval.mode
        self._retrieval_weights = cfg.retrieval.weights
        self._retriever: HybridRetriever | None = None
        self._retriever_commit: str = ""

    def _git_commit(self) -> str:
        import subprocess

        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    def _get_retriever(self) -> HybridRetriever:
        commit = self._git_commit()
        if self._retriever is not None and self._retriever_commit == commit:
            return self._retriever

        cfg = load_config(self.project_path)
        profile = get_profile(cfg.profile)

        # Only build embeddings retriever if config, profile, AND local LLM all enable embeddings
        emb = None
        emb_enabled = cfg.embeddings_enabled and cfg.local_llm.enabled and profile.embedding_enabled
        if emb_enabled:
            emb = EmbeddingRetriever(self.vector_indexer, cache=self.cache)

        bm25 = BM25Retriever()
        ast_rank = ASTRankRetriever()

        bm25_items, ast_items = self._load_function_texts()
        bm25.index(bm25_items)
        ast_rank.index(ast_items, repo_map=self.repo.load_repo_map())

        # Choose reranker by config. cross_encoder=FlashRank (CPU, ms, default);
        # llm=local-model scoring (opt-in, slow); none=disabled.
        # Typed Any: FlashRank Reranker and LLMReranker share a duck-typed
        # .rerank(query, candidates, top_k) interface.
        reranker: Any = None
        if cfg.retrieval.rerank_enabled and not cfg.retrieval.offline:
            rtype = getattr(cfg.retrieval, "reranker_type", "cross_encoder")
            if rtype == "llm":
                from pipeline.retrieval.llm_reranker import LLMReranker

                reranker = LLMReranker(ollama_client=self.ollama)
            elif rtype != "none":
                reranker = Reranker(
                    ca_bundle=cfg.retrieval.ca_bundle, offline=cfg.retrieval.offline
                )

        # Lexical leg: ripgrep over the live working tree (fresh, exact-match),
        # falling back to in-memory BM25 over the loaded function texts when rg
        # is not installed. Replaces the AST keyword-graph leg in hybrid fusion.
        from pipeline.retrieval.ripgrep import RipgrepRetriever

        lexical = RipgrepRetriever(
            project_path=self.project_path,
            file_patterns=cfg.indexing.file_patterns,
            exclude_patterns=cfg.indexing.exclude_patterns,
            source_roots=cfg.indexing.source_roots,
            fallback_items=bm25_items,
        )

        # Structural leg: exact block-identity + reference matching.
        # Excels at config file retrieval (Terraform, Kubernetes, etc.)
        # where embeddings conflate resource types.
        from pipeline.retrieval.structural import StructuralRetriever

        structural = StructuralRetriever()
        structural.index(bm25_items)

        self._retriever = HybridRetriever(
            bm25=bm25,
            ast_rank=ast_rank,
            embeddings=emb,
            weights=self._retrieval_weights,
            mode=self._retrieval_mode,
            reranker=reranker,
            rerank_enabled=cfg.retrieval.rerank_enabled,
            lexical=lexical,
            structural=structural,
            profile_legs=profile.legs,
        )
        self._retriever_commit = commit
        return self._retriever

    def _load_function_texts(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Load all indexed functions as (id, text, name) pairs from ChromaDB.

        Scoped to this project: when a project_id is bound, only this project's
        records are loaded so the lexical/structural legs can never surface a
        foreign record that happens to share the collection.
        """
        # Fetch ALL rows in PAGES. A single unbounded get(include=["metadatas",...])
        # makes Chroma bind one SQL variable per row and blows SQLite's
        # "too many SQL variables" limit (~32k) on large repos — silently
        # returning nothing (retrieval dies). We also do NOT pass a
        # where={"project_id": ...} filter (same blow-up); the collection is
        # already namespaced per project, and we drop any provably-foreign record
        # in-memory below.
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        documents: list[str] = []
        page = 2000
        offset = 0
        while True:
            try:
                data = self.vector_indexer.collection.get(
                    include=["metadatas", "documents"], limit=page, offset=offset
                )
            except Exception:
                break
            batch_ids = list(data.get("ids") or [])
            if not batch_ids:
                break
            ids.extend(batch_ids)
            metadatas.extend(dict(m) for m in (data.get("metadatas") or []))  # type: ignore[arg-type]
            documents.extend(data.get("documents") or [])
            offset += len(batch_ids)
            if len(batch_ids) < page:
                break

        bm25_items: list[dict[str, Any]] = []
        ast_items: list[dict[str, Any]] = []

        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc = documents[i] if i < len(documents) else ""
            doc_id = ids[i]

            # Project isolation (in-memory): skip a record only if it is provably
            # from a different project. None == legacy/un-stamped → keep.
            rec_pid = meta.get("project_id")
            if self.project_id is not None and rec_pid is not None and rec_pid != self.project_id:
                continue

            bm25_items.append({"id": doc_id, "text": doc})
            ast_items.append(
                {
                    "id": doc_id,
                    "text": doc,
                    "name": meta.get("function", ""),
                    "filepath": meta.get("filepath", ""),
                }
            )

        return bm25_items, ast_items

    def _passes_confidence_guard(self, functions: list[dict]) -> bool:
        """True if the top result is confident enough to inject/return.

        Shared by assemble_context AND semantic_search so EVERY path (CLI search,
        MCP search_code, proxy) honors the same relevance gate — not just the
        assemble path. When rerank is active, gates on the cross-encoder absolute
        score; otherwise on raw embedding cosine. (The normalized 'similarity' is
        min-max scaled so the top result is always ~1.0 — useless as a gate.)
        Threshold 0 disables the guard.
        """
        cfg = load_config(self.project_path)
        if cfg.retrieval.rerank_enabled:
            # Prefer the PROFILE's rerank threshold (iac=0.15 for terse HCL,
            # code=0.47 for prose-y code) over the global config value, so the
            # gate matches the cross-encoder score distribution of the repo type.
            profile = get_profile(cfg.profile)
            threshold = getattr(profile, "rerank_min_score", cfg.retrieval.rerank_min_score)
            top_score = functions[0].get("rerank_score", 0.0) if functions else 0.0
        else:
            threshold = cfg.retrieval.min_confidence
            top_score = functions[0].get("raw_cosine", 0.0) if functions else 0.0

        if threshold <= 0:
            return True
        return bool(functions) and top_score >= threshold

    def semantic_search(
        self, query: str, top_k: Optional[int] = None, apply_guard: bool = False
    ) -> list[dict]:
        """Find relevant functions using configured retrieval strategy (cached).

        apply_guard=True drops results when the top match fails the confidence
        guard (returns []), so CLI `search` and the MCP `search_code` tool reject
        off-topic queries instead of returning low-confidence noise.
        """
        if top_k is None:
            top_k = self.top_k

        commit = self._git_commit()
        cache_key = ("search", query, str(top_k), commit, self._retrieval_mode)

        cached = self.cache.get(*cache_key) if self.cache else None
        if cached is not None:
            results = cached
        else:
            if self._retrieval_mode == "embeddings":
                results = self.vector_indexer.search(query, top_k=top_k)
                # VectorIndexer.search returns real cosine in 'similarity'; mirror
                # it into 'raw_cosine' so the confidence guard has one field.
                for r in results:
                    r.setdefault("raw_cosine", r.get("similarity", 0.0))
            else:
                retriever = self._get_retriever()
                hybrid_results = retriever.search(query, top_k=top_k, commit=commit)
                results = self._hybrid_results_to_legacy(hybrid_results)

            if self.cache:
                self.cache.set(results, *cache_key)

        # Final assertion: reject any result provably from a DIFFERENT project,
        # then normalize provenance on what we keep. Two leg-agnostic checks:
        #   1. a NON-None project_id that mismatches (foreign vector record), or
        #   2. an absolute filepath outside this repo (foreign file from any leg).
        # Lexical/structural/ripgrep results are repo-scoped by construction and
        # carry no project_id, so we keep them and stamp the bound id.
        if self.project_id is not None:
            proj_root = str(self.project_path.resolve())
            filtered = []
            for result in results:
                result_pid = result.get("project_id")
                fp = result.get("filepath") or ""
                if (result_pid is not None and result_pid != self.project_id) or _is_foreign_path(
                    fp, proj_root
                ):
                    logger.warning(
                        "Dropped cross-project result: %s (got %s, expected %s)",
                        fp,
                        result_pid,
                        self.project_id,
                    )
                    continue
                result["project_id"] = self.project_id  # normalize provenance
                filtered.append(result)
            results = filtered

        # Apply the confidence guard so CLI search + MCP search_code reject
        # off-topic queries (not just the assemble path). Cache stores the raw
        # results; the guard is applied on return.
        if apply_guard and not self._passes_confidence_guard(results):
            return []

        return results

    def _hybrid_results_to_legacy(self, hybrid_results: list[dict[str, Any]]) -> list[dict]:
        """Convert hybrid retriever results to the format expected by _format_functions.

        The ID format is 'filepath:function:line_start'.  We split it back out.
        Ensures similarity is always in [0, 1] range:
        - If result has normalized 'similarity' in [0,1], use it (upstream normalized).
        - Otherwise, normalize the raw 'score' across the result set (min-max).
        This guarantees consistent scale across all retrieval modes and cache hits.
        """
        legacy: list[dict] = []

        # Check if upstream already normalized similarity values
        has_normalized = any(
            isinstance(hr.get("similarity"), (int, float))
            and 0.0 <= hr.get("similarity", 0.0) <= 1.0
            for hr in hybrid_results
        )

        # If not normalized by upstream, normalize raw scores ourselves
        if not has_normalized and hybrid_results:
            raw_scores = [float(hr.get("score", 0.0)) for hr in hybrid_results]
            min_score = min(raw_scores)
            max_score = max(raw_scores)
            if min_score == max_score or len(hybrid_results) == 1:
                normalized_sims = [1.0] * len(hybrid_results)
            else:
                normalized_sims = [
                    (score - min_score) / (max_score - min_score) for score in raw_scores
                ]
        else:
            normalized_sims = None

        for i, hr in enumerate(hybrid_results):
            doc_id = hr["id"]
            parts = doc_id.rsplit(":", 2)
            filepath = parts[0] if len(parts) > 0 else doc_id
            function = parts[1] if len(parts) > 1 else ""
            try:
                line_start = int(parts[2]) if len(parts) > 2 else 0
            except ValueError:
                line_start = 0

            # Use upstream normalized similarity if available, else use self-normalized
            if has_normalized:
                similarity = float(hr.get("similarity", 0.0))
            else:
                similarity = normalized_sims[i] if normalized_sims else 1.0

            legacy.append(
                {
                    "filepath": filepath,
                    "function": function,
                    "line_start": line_start,
                    "line_end": line_start + 1,
                    "code": hr.get("text", ""),
                    "similarity": similarity,
                    # Absolute embedding cosine (0 if this doc had no embedding hit).
                    # Used by the confidence guard; 'similarity' above is normalized
                    # for ranking/display and is always ~1.0 for the top result.
                    "raw_cosine": float(hr.get("raw_cosine", 0.0)),
                    # Cross-encoder rerank score (0..1, absolute quality signal).
                    # Used by the guard when rerank_enabled is True.
                    "rerank_score": float(hr.get("rerank_score", 0.0)),
                }
            )
        return legacy

    def get_repo_map(self) -> dict:
        return self.repo.load_repo_map()

    def get_memory(self, last_n: int = 10) -> str:
        return self.repo.load_memory(last_n)

    def _maybe_compress(self, text: str) -> str:
        """Compress context using RTK-style token compressor.

        Respects config.compression.enabled and COMPRESSION_LEVEL env override.
        Marks compressed output with a header comment so proxy can detect and
        skip re-compression (avoid double-compression).

        Note: context is markdown-formatted with headers (##). The compressor is
        code-aware and will strip lines starting with # thinking they're comments.
        We preserve markdown headers by protecting them before compression.
        """
        cfg = load_config(self.project_path)

        if not cfg.compression.enabled:
            return text

        try:
            level_str = os.environ.get("COMPRESSION_LEVEL", cfg.compression.level)
            try:
                level = FilterLevel(level_str)
            except ValueError:
                level = FilterLevel.MINIMAL

            # Protect markdown headers before compression (replace ## with placeholder)
            headers: list[tuple[str, str]] = []
            protected = text
            for match in re.finditer(r"^(#{1,6}\s+.*)$", text, re.MULTILINE):
                header = match.group(1)
                placeholder = f"__MARKDOWN_HEADER_{len(headers)}__"
                headers.append((placeholder, header))
                protected = protected.replace(header, placeholder, 1)

            compressor = TokenCompressor(level=level)
            result = compressor.compress(protected, Language.PYTHON)

            # Restore markdown headers
            for placeholder, header in headers:
                result = result.replace(placeholder, header)

            stats = compressor.get_stats()
            logger.info(
                "Context compression: %s → %s tokens (%.1f%%)",
                stats["original_tokens"],
                stats["compressed_tokens"],
                stats["reduction_pct"],
            )
            # Mark as already-compressed so proxy doesn't re-compress
            marker = "# [already-compressed-by-gateway]\n"
            return marker + result
        except Exception as e:
            logger.warning("Compression failed, returning unmodified: %s", e)
            return text

    def _format_functions(self, functions: list[dict]) -> str:
        if not functions:
            return "*No relevant functions found.*"

        parts = []
        for f in functions:
            parts.append(
                f"### {f['filepath']}:{f['function']} "
                f"(lines {f['line_start']}-{f['line_end']}, "
                f"similarity: {f['similarity']:.2f})\n"
                f"```\n{f['code']}\n```"
            )
        return "\n\n".join(parts)

    def _format_repo_map(self, repo_map: dict, max_items: int = 20) -> str:
        if not repo_map:
            return "*No repo map available.*"

        parts = []
        for filepath, data in list(repo_map.items())[:max_items]:
            items = []
            for cls in data.get("classes", []):
                items.append(cls["name"])
                items.extend(f"  {m['name']}" for m in cls.get("methods", []))
            items.extend(f["name"] for f in data.get("functions", []))
            parts.append(f"{filepath}:\n  " + "\n  ".join(items))

        return "\n".join(parts)

    def assemble(self, user_prompt: str) -> str:
        commit = self._git_commit()
        cache_key = ("assemble", user_prompt, commit)

        if self.cache:
            cached = self.cache.get(*cache_key)
            if cached is not None:
                return cached

        ctx = self.assemble_context(user_prompt)
        prompt = f"{ctx}\n\n## Your Task\n{user_prompt}"
        if self.cache:
            self.cache.set(prompt, *cache_key)

        return prompt

    def assemble_context(self, user_prompt: str) -> str:
        commit = self._git_commit()
        cache_key = ("context", user_prompt, commit)

        # Try persistent cache first (cross-process warmth)
        if self.persistent_cache:
            cached = self.persistent_cache.get(*cache_key)
            if cached is not None:
                logger.debug("Persistent cache hit for context query")
                return cached

        # Fallback to in-memory cache
        if self.cache:
            cached = self.cache.get(*cache_key)
            if cached is not None:
                return cached

        functions = self.semantic_search(user_prompt)

        if not self._passes_confidence_guard(functions):
            # Confidence-guard rejection: do not compress
            return "*No confident matches found for this query.*"
        repo_map = self.get_repo_map()
        memory = self.get_memory()

        func_text = self._format_functions(functions)
        map_text = self._format_repo_map(repo_map)

        prompt = f"""# Codebase Context

## Relevant Functions (Semantic Match)
{func_text}

## Repository Structure
{map_text}

## Recent Changes
{memory if memory else '*No recent changes recorded.*'}"""

        # Compress context before caching/returning so all consumers
        # (CLI, MCP, proxy) get the compressed form
        compressed = self._maybe_compress(prompt)

        # Cache compressed result in both in-memory and persistent
        if self.cache:
            self.cache.set(compressed, *cache_key)
        if self.persistent_cache:
            self.persistent_cache.set(compressed, *cache_key)

        return compressed
