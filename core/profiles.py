"""Repository profile system: detect repo type and select retrieval strategy.

A profile maps repository type (e.g., "iac", "dotnet", "python") to a
configuration that optimizes retrieval legs and embedding behavior for that
stack.
"""

from dataclasses import dataclass, field


@dataclass
class ProfileSpec:
    """Profile specification: defines retrieval strategy for a repo type."""

    name: str
    """Profile name: e.g., 'iac', 'dotnet', 'python', 'code', 'shell'."""

    file_patterns: list[str]
    """File patterns to index for this profile."""

    embedding_enabled: bool
    """Whether to load and use embedding models (expensive VRAM)."""

    retrieval_mode: str
    """Retrieval strategy: 'hybrid' | 'embeddings' | 'bm25' | 'ast'."""

    legs: list[str] = field(default_factory=list)
    """Retrieval legs to use: e.g., ['structural', 'lexical'] or
    ['structural', 'lexical', 'embeddings']."""

    embedding_model: str = "nomic-embed-text"
    """Which embedding model to use (only if embedding_enabled=True)."""

    worker_model: str = "qwen2.5-coder:1.5b"
    """Local generative model for optional reranking/summarization.
    1.5b (vs 3b): warm ~0.6s/gen at half the VRAM → less model-swap eviction
    on a 6GB GPU, which is the real bottleneck (embedder+worker thrash)."""

    rerank_min_score: float = 0.47
    """Confidence-guard threshold on the cross-encoder rerank score.
    Code profiles use 0.47 (calibrated on Django prose-y code). IaC overrides
    to 0.15: cross-encoder scores run LOWER on terse HCL blocks, so 0.47
    false-negatives valid queries (e.g. KMS at 0.42) while real nonsense still
    scores ~0.02 — measured on tf-eks."""

    description: str = ""
    """Human-readable description of this profile."""


# Profile definitions: optimized for each repo type
PROFILES: dict[str, ProfileSpec] = {
    "iac": ProfileSpec(
        name="iac",
        file_patterns=[
            "*.tf",
            "*.tfvars",
            "*.hcl",
            "*.yaml",
            "*.yml",
            "*.sh",
            "*.bash",
        ],
        embedding_enabled=False,
        retrieval_mode="hybrid",
        legs=["structural", "lexical"],
        embedding_model="",  # Not used
        rerank_min_score=0.15,  # terse HCL → lower cross-encoder scores (measured)
        description=(
            "Infrastructure-as-Code (Terraform, Helm, Kubernetes). "
            "Relies on structural + lexical (ripgrep) retrieval; embeddings "
            "OFF to save VRAM and avoid conflating resource types."
        ),
    ),
    "dotnet": ProfileSpec(
        name="dotnet",
        file_patterns=[
            "*.cs",
        ],
        embedding_enabled=True,
        retrieval_mode="hybrid",
        legs=["embeddings", "lexical", "structural"],
        embedding_model="qwen3-embedding:0.6b",
        # ms-marco cross-encoder scores terse C# code far lower than prose even for
        # perfect matches (MEASURED on MediatR: a correct top-1 "send a request
        # through mediator"→Mediator.Send scored 0.17, while nonsense maxed at
        # 0.0007). The Python-tuned 0.47 default would suppress valid C# hits, so
        # use a lower threshold. 0.10 passes all measured relevant, rejects nonsense.
        rerank_min_score=0.10,
        description=(
            ".NET / C# (MediatR, Roslyn, etc.). "
            "Full hybrid: embeddings + lexical + structural. "
            "Uses qwen3-embedding:0.6b (639MB) so embedder + 1.5b worker both "
            "fit in 6GB VRAM without swap thrash; 4b is better but evicts the "
            "worker on small GPUs — override embedding_model if you have VRAM."
        ),
    ),
    "python": ProfileSpec(
        name="python",
        file_patterns=[
            "*.py",
        ],
        embedding_enabled=True,
        retrieval_mode="hybrid",
        legs=["embeddings", "lexical", "structural"],
        embedding_model="nomic-embed-text",
        description=(
            "Python (Django, FastAPI, etc.). "
            "Full hybrid: embeddings + lexical + structural. "
            "Uses standard nomic-embed-text."
        ),
    ),
    "shell": ProfileSpec(
        name="shell",
        file_patterns=[
            "*.sh",
            "*.bash",
        ],
        embedding_enabled=False,
        retrieval_mode="hybrid",
        legs=["structural", "lexical"],
        embedding_model="",  # Not used
        rerank_min_score=0.15,  # terse scripts → lower cross-encoder scores
        description=("Shell scripts / bash. " "Lexical + structural; embeddings OFF."),
    ),
    "code": ProfileSpec(
        name="code",
        file_patterns=[
            "*.js",
            "*.ts",
            "*.tsx",
            "*.jsx",
            "*.go",
            "*.rs",
            "*.java",
            "*.cpp",
            "*.c",
            "*.h",
            "*.hpp",
            "*.rb",
            "*.toml",
        ],
        embedding_enabled=True,
        retrieval_mode="hybrid",
        legs=["embeddings", "lexical", "structural"],
        embedding_model="nomic-embed-text",
        description=(
            "Generic code (JS/TS, Go, Rust, Java, C++, Ruby, etc.). "
            "Full hybrid: embeddings + lexical + structural."
        ),
    ),
}


def detect_profile(extension_counts: dict[str, int]) -> str:
    """Detect the best profile for a repo given a census of file extensions.

    Heuristic logic:
    1. If terraform/HCL files present, OR yaml/yml dominant (>50% of code files
       in an infra context) -> "iac"
    2. Else if C# files present -> "dotnet"
    3. Else if Python dominant (>50% of code files) -> "python"
    4. Else if shell scripts dominant (>50% of code files) -> "shell"
    5. Else -> "code" (catch-all generic programming languages)

    The heuristic is deterministic: it checks in a fixed order and picks the
    first match. This makes detection predictable across runs.

    Args:
        extension_counts: dict mapping extension (with dot) to count,
                          e.g., {'.py': 150, '.tf': 50, '.yaml': 20}

    Returns:
        Profile name: 'iac' | 'dotnet' | 'python' | 'shell' | 'code'
    """
    if not extension_counts:
        # Empty codebase; default to generic code
        return "code"

    # Total code files (for computing percentages)
    total_files = sum(extension_counts.values())

    # IaC detection: tf/hcl files OR (yaml/yml dominant in infra context)
    has_tf = extension_counts.get(".tf", 0) > 0
    has_hcl = extension_counts.get(".hcl", 0) > 0
    yaml_count = extension_counts.get(".yaml", 0) + extension_counts.get(".yml", 0)
    yaml_pct = yaml_count / total_files if total_files > 0 else 0

    # If terraform/HCL explicitly present, always pick iac
    # (even if it's a multi-language repo with some iac)
    if has_tf or has_hcl:
        return "iac"

    # If yaml/yml dominant (>50%) and no code files yet, tentatively iac
    # (e.g., a k8s repo with only yamls). But only if *no* other code is
    # detected, to avoid misclassifying a Python+YAML repo as iac.
    has_code = any(
        ext in extension_counts and extension_counts[ext] > 0
        for ext in [".py", ".cs", ".sh", ".bash", ".js", ".ts", ".go", ".rs"]
    )
    if yaml_pct > 0.5 and not has_code and yaml_count > 0:
        return "iac"

    # C# detection: .cs files present
    if extension_counts.get(".cs", 0) > 0:
        return "dotnet"

    # Python detection: python dominant
    py_count = extension_counts.get(".py", 0)
    py_pct = py_count / total_files if total_files > 0 else 0
    if py_pct > 0.5:
        return "python"

    # Shell detection: shell scripts dominant
    shell_count = extension_counts.get(".sh", 0) + extension_counts.get(".bash", 0)
    shell_pct = shell_count / total_files if total_files > 0 else 0
    if shell_pct > 0.5:
        return "shell"

    # Default to generic code
    return "code"


def get_profile(name: str) -> ProfileSpec:
    """Retrieve a profile by name, falling back to 'code' if unknown.

    Args:
        name: Profile name (e.g., 'iac', 'dotnet', 'python').

    Returns:
        ProfileSpec for the profile.
    """
    return PROFILES.get(name, PROFILES["code"])


def profile_to_config_dict(profile: ProfileSpec) -> dict:
    """Convert a ProfileSpec to config dict entries (for YAML serialization).

    Returns a dict suitable for merging into config.yaml:
      {
        'profile': profile.name,
        'indexing': {
          'file_patterns': [...],
          'embedding_model': '...' (if enabled, else omit),
        },
        'retrieval': {
          'mode': 'hybrid' | 'embeddings' | ...,
        },
      }
    """
    d: dict = {
        "profile": profile.name,
        "indexing": {
            "file_patterns": profile.file_patterns,
        },
        "retrieval": {
            "mode": profile.retrieval_mode,
        },
    }

    # Only set embedding_model if embeddings are enabled
    if profile.embedding_enabled:
        d["indexing"]["embedding_model"] = profile.embedding_model
    else:
        # Embeddings OFF: indicate via a flag or by setting mode
        # We use an explicit flag in the config (new field, with default True)
        # so profiles can disable it without breaking existing behavior.
        if "embeddings_enabled" not in d:
            d["embeddings_enabled"] = profile.embedding_enabled

    return d
