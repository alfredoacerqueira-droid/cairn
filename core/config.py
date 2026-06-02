"""Configuration loader for cairn per-project settings."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class EnabledConfig(BaseModel):
    file_watcher: bool = True
    vector_indexing: bool = True
    memory_summarizer: bool = True
    auto_start: bool = False


class ResourceConfig(BaseModel):
    max_cpu_percent: int = 50
    max_memory_mb: int = 4096
    vram_priority: str = "gateway"


class IndexingConfig(BaseModel):
    # Default patterns are conservative source-code languages. JS/TS are NOT in the
    # default set: indexing *.js on a Python repo pulls in vendored/minified bundles
    # (jquery, select2) that pollute retrieval. Add them explicitly per-project if you
    # actually want to index JS/TS source.
    file_patterns: list[str] = [
        "*.py",
        "*.rs",
        "*.go",
        "*.c",
        "*.h",
        "*.cpp",
        "*.hpp",
        "*.cs",
        "*.java",
        "*.rb",
        "*.sh",
        "*.bash",
        "*.tf",
        "*.tfvars",
        "*.toml",
    ]
    exclude_patterns: list[str] = [
        "**/node_modules/**",
        "**/.git/**",
        # CI/CD config (workflow YAML, etc.) — not application source; pollutes
        # retrieval with build-pipeline blocks. Add back per-project if you index it.
        "**/.github/**",
        "**/.gitlab/**",
        "**/__pycache__/**",
        "**/.venv/**",
        "**/venv/**",
        "**/.cairn/**",
        "**/tests/**",
        "**/test/**",
        # Demo/sample code: like tests, it's not the source of truth and pollutes
        # retrieval (e.g. a Terraform module repo's examples/ duplicates resource
        # types). Excluded by default; add the dir back per-project if you want it.
        "**/examples/**",
        "**/example/**",
        "**/samples/**",
        "**/benchmarks/**",
        "**/build/**",
        "**/dist/**",
        "**/*.egg-info/**",
        "**/.mypy_cache/**",
        "**/.pytest_cache/**",
        "**/.ruff_cache/**",
        # Vendored / generated / static assets — never useful as retrieval targets
        # and a major source of noise (e.g. Django admin's bundled jQuery/Select2).
        "**/vendor/**",
        "**/static/**",
        "**/migrations/**",
        "**/*.min.js",
        "**/*.min.css",
    ]
    source_roots: list[str] = ["."]
    batch_size: int = 50
    delay_ms: int = 500
    # Maximum file size in KB before skipping parse. Set to 0 to disable the limit.
    # Files exceeding this limit are skipped with a warning log.
    max_file_kb: int = 0
    # Maximum time in seconds to spend parsing a single file. If exceeded, the file
    # is skipped with a warning log. Set to 0 to disable (no timeout).
    parse_timeout_s: float = 10.0
    # Embedding model for semantic code search via Ollama.
    # Default "nomic-embed-text" is a general-purpose embedder.
    # For better code retrieval discrimination, use a code-trained model
    # (e.g. a specialized Ollama code-embedding model).
    #
    # UPGRADE PATH to a code-trained model:
    #   1. Set this field to your new model name (e.g. "code-model-v1")
    #   2. Ensure the model is available in Ollama: ollama pull code-model-v1
    #   3. Run: cairn reindex --mode full  (MUST reindex; old vectors incompatible)
    #   4. Re-measure RetrievalConfig.min_confidence on the new model's cosine scale
    #      (different models have different output distributions)
    embedding_model: str = "nomic-embed-text"


class StaleDBConfig(BaseModel):
    auto_quick_reindex: bool = True
    quick_reindex_threshold: int = 1000
    full_reindex_threshold: int = 10000


class MemoryConfig(BaseModel):
    trigger: str = "manual"  # "manual" | "post-commit" | "periodic"
    max_entries: int = 50
    compaction_model: str = "qwen2.5-coder:1.5b"
    period_minutes: int = 5


class RoutingConfig(BaseModel):
    mode: str = "cloud_only"  # "cloud_only" | "conservative" | "aggressive"
    require_user_confirm: bool = True


class CacheConfig(BaseModel):
    enabled: bool = True
    ttl_seconds: int = 300
    max_entries: int = 100


class RetrievalConfig(BaseModel):
    # Default "hybrid" = lexical (ripgrep, fresh/exact) + embeddings (semantic),
    # fused via RRF, then reordered by the cross-encoder reranker. The AST
    # keyword-graph leg is NOT used by default (it degraded ranking on large
    # repos). "embeddings" / "bm25" / "ast" force a single leg (no rerank fusion).
    mode: str = "hybrid"  # hybrid | embeddings | bm25 | ast
    weights: list[float] = [0.4, 0.3, 0.3]  # bm25, ast, embeddings
    rerank_enabled: bool = True
    # Which reranker to use when rerank_enabled:
    #   "cross_encoder" (default) — FlashRank, CPU, ~milliseconds. Recommended.
    #   "llm"  — score candidates with the local Ollama worker model. OPT-IN ONLY:
    #            MEASURED ~19-39s PER generation on a 6GB GPU, so reranking N
    #            candidates is minutes/query — impractical here. Only enable on
    #            fast local inference (Apple Silicon / big VRAM).
    #   "none" — skip reranking.
    reranker_type: str = "cross_encoder"  # cross_encoder | llm | none
    # Minimum cross-encoder rerank score (0..1 scale) for the top match
    # before context is injected. Only used when rerank_enabled=True.
    # Cross-encoder scores are absolute quality metrics (unlike RRF/cosine
    # which are relative), so thresholds transfer across repos.
    # Set 0.0 to disable the guard entirely (accept all reranked results).
    # MEASURED (2026-06-01) on Django (8331 functions, ms-marco-MiniLM-L-12-v2):
    #   - Relevant queries (5): [0.9918, 0.9981, 0.9887, 0.9199, 0.9992] mean=0.9795
    #   - Nonsense queries (3): [0.0000, 0.0000, 0.0211] mean=0.0070
    #   - Separation gap: 0.8988
    # Recommended threshold 0.4705 achieves perfect separation:
    #   - All 5/5 relevant queries pass (min=0.9199 > 0.4705)
    #   - All 3/3 nonsense queries fail (max=0.0211 < 0.4705)
    rerank_min_score: float = 0.47
    # Legacy: Minimum RAW EMBEDDING COSINE for the top match before context
    # is injected (fallback when rerank is off). This is an absolute quality
    # gate, NOT the normalized display 'similarity' (which is min-max scaled
    # so the top result is always ~1.0 and can't gate).
    # VALUE IS MODEL-SPECIFIC:
    # Updated 2026-06-01: measured pure EMBEDDINGS mode on Django (9616 functions,
    # qwen3-embedding:0.6b) with 5 relevant + 3 nonsense queries:
    #   - Relevant: [0.8920, 0.8700, 0.8378, 0.8550, 0.8514] mean=0.8612
    #   - Nonsense: [0.7931, 0.7839, 0.7818] mean=0.7863
    #   - Gap: 0.0750
    # Threshold 0.82 achieves perfect separation:
    #   - All 5/5 relevant queries pass (min=0.8378 > 0.82)
    #   - All 3/3 nonsense queries fail (max=0.7931 < 0.82)
    # Retrieved mode changed from "hybrid" to "embeddings" because:
    #   - Pure embeddings outperforms hybrid (RRF+BM25+AST) on large repos
    #   - Hybrid fusion degrades ranking on diverse codebases like Django
    #   - Embeddings provide a cleaner, more reliable confidence signal
    # For small/specialized repos, hybrid mode may help; override via config.
    # Set 0.0 to disable the guard entirely.
    min_confidence: float = 0.82
    # Offline mode: disable reranker entirely (skip FlashRank load).
    # Useful behind corporate proxies where model download fails/hangs.
    offline: bool = False
    # Custom CA bundle path for HTTPS certificate validation (e.g., corporate proxies).
    # If set, overrides CAIRN_CA_BUNDLE / REQUESTS_CA_BUNDLE / SSL_CERT_FILE.
    # Passed to flashrank via os.environ before model load.
    ca_bundle: str | None = None


class CompressionConfig(BaseModel):
    # Enable token compression on assembled context (retrieval path).
    # Compresses the assembled context (functions + repo map + memory) so all
    # consumers (CLI, MCP, proxy) benefit. Honors COMPRESSION_LEVEL env var.
    enabled: bool = True
    # Compression level: "none" (0%), "minimal" (20-40%), "aggressive" (60-90%)
    level: str = "minimal"


class LocalLLMConfig(BaseModel):
    # Whether to use a local LLM for embeddings and text generation.
    # When disabled, Cairn operates without any LLM (lexical/structural + cross-encoder only).
    enabled: bool = False
    # Backend type: "ollama" (Ollama) or "openai_compatible" (LM Studio, llama.cpp, etc.)
    backend: str = "ollama"  # ollama | openai_compatible
    # Base URL for the backend (e.g., http://127.0.0.1:11434 for Ollama,
    # http://127.0.0.1:8000 for LM Studio). If None, uses backend defaults.
    base_url: str | None = None
    # Model name for text generation (summarization, etc.).
    # For Ollama: qwen2.5-coder:1.5b (default). For OpenAI-compatible: model ID.
    model: str | None = None
    # Model name for embeddings (semantic search).
    # For Ollama: nomic-embed-text (default). For OpenAI-compatible: model ID.
    embed_model: str | None = None


class Config(BaseModel):
    # Repository profile: determines retrieval strategy
    # "auto" = detect at init; "iac"/"dotnet"/"python"/"code"/"shell" = explicit
    profile: str = "code"
    # Whether to load and use embedding models (controls VRAM usage)
    embeddings_enabled: bool = True
    enabled: EnabledConfig = EnabledConfig()
    resources: ResourceConfig = ResourceConfig()
    indexing: IndexingConfig = IndexingConfig()
    stale_db: StaleDBConfig = StaleDBConfig()
    memory: MemoryConfig = MemoryConfig()
    routing: RoutingConfig = RoutingConfig()
    cache: CacheConfig = CacheConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    compression: CompressionConfig = CompressionConfig()
    local_llm: LocalLLMConfig = LocalLLMConfig()


def load_config(project_path: Optional[Path] = None) -> Config:
    """Load configuration from .cairn/config.yaml in the project.

    If a stale config is loaded (old exclude_patterns without new excludes),
    merges in the new standard excludes to prevent index pollution.
    """
    if project_path is None:
        project_path = Path.cwd()

    config_file = project_path / ".cairn" / "config.yaml"

    if not config_file.exists():
        return Config()

    with open(config_file) as f:
        data = yaml.safe_load(f) or {}

    config = Config(**data)

    # Migration: if persisted config is missing new exclude patterns,
    # merge them in without overwriting user-added custom patterns.
    # Sentinel: check if **/.venv/** is in the loaded exclude_patterns.
    has_venv_exclude = "**/.venv/**" in config.indexing.exclude_patterns
    if not has_venv_exclude:
        # Old config detected; merge in the new standard excludes
        default_config = Config()
        old_patterns = set(config.indexing.exclude_patterns)
        new_patterns = set(default_config.indexing.exclude_patterns)
        # Union: preserve user patterns, add all defaults
        config.indexing.exclude_patterns = sorted(list(old_patterns | new_patterns))

    # Ensure source_roots exists (in case old config lacks it)
    if not config.indexing.source_roots:
        config.indexing.source_roots = ["."]

    return config


def save_config(config: Config, project_path: Optional[Path] = None):
    """Save configuration to .cairn/config.yaml in the project."""
    if project_path is None:
        project_path = Path.cwd()

    config_dir = project_path / ".cairn"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"

    with open(config_file, "w") as f:
        yaml.safe_dump(config.model_dump(), f, default_flow_style=False)
