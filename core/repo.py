"""Per-repo data management."""

import fnmatch
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def project_id(project_root: str | Path) -> str:
    """Compute a short stable hash of the resolved absolute path.

    Used to namespace ChromaDB collections and add provenance metadata to indexed
    records, enabling multi-repo isolation. The hash is deterministic: the same
    project root always yields the same ID.

    Args:
            project_root: Path to the project (can be relative or absolute).

    Returns:
            First 12 chars of SHA1 hash of the resolved absolute path.
    """
    resolved = str(Path(project_root).resolve())
    return hashlib.sha1(resolved.encode()).hexdigest()[:12]


def detect_infra_markers(
    project_path: Path,
    source_roots: list[str] | None = None,
) -> bool:
    """Detect if a project is explicitly marked as infrastructure/IaC.

    Checks for:
    1. Chart.yaml (Helm)
    2. kustomization.yaml or kustomization.yml (Kustomize)
    3. YAML files containing Kubernetes manifest markers (kind: + apiVersion:)

    This is CHEAP: only scans ~200 YAML files, reads first 4KB of each, and
    returns early on first match.

    Args:
        project_path: Root directory of the project.
        source_roots: Subdirectories to scan (e.g., ["src"]).
                      If None, scans the entire project.

    Returns:
        True if infrastructure markers are found, False otherwise.
    """
    project_path = Path(project_path)

    if source_roots is None:
        source_roots = ["."]

    # Check for explicit chart/kustomization files
    for root_str in source_roots:
        root_path = project_path / root_str
        if not root_path.exists():
            continue

        # Helm: Chart.yaml
        chart_path = root_path / "Chart.yaml"
        if chart_path.exists():
            return True

        # Kustomize: kustomization.yaml or kustomization.yml
        kustomize_yaml = root_path / "kustomization.yaml"
        kustomize_yml = root_path / "kustomization.yml"
        if kustomize_yaml.exists() or kustomize_yml.exists():
            return True

    # Scan YAML files for Kubernetes manifest markers (kind: + apiVersion:)
    yaml_files_scanned = 0
    max_yaml_files_to_scan = 200

    for root_str in source_roots:
        root_path = project_path / root_str
        if not root_path.exists():
            continue

        for yaml_file in root_path.rglob("*.yaml"):
            if yaml_files_scanned >= max_yaml_files_to_scan:
                # Bounded scan; if we haven't found a marker by now, assume it's not there
                return False

            if not yaml_file.is_file():
                continue

            yaml_files_scanned += 1

            try:
                # Read first 4KB to look for markers
                content = yaml_file.read_text(errors="ignore")[:4096]

                # Check for Kubernetes manifest markers: both kind: and apiVersion:
                has_kind = "kind:" in content
                has_api_version = "apiVersion:" in content

                if has_kind and has_api_version:
                    return True
            except Exception:
                # Silently skip unreadable files
                continue

        # Also check .yml files
        for yml_file in root_path.rglob("*.yml"):
            if yaml_files_scanned >= max_yaml_files_to_scan:
                return False

            if not yml_file.is_file():
                continue

            yaml_files_scanned += 1

            try:
                content = yml_file.read_text(errors="ignore")[:4096]
                has_kind = "kind:" in content
                has_api_version = "apiVersion:" in content

                if has_kind and has_api_version:
                    return True
            except Exception:
                continue

    return False


def census_extensions(
    project_path: Path,
    source_roots: list[str] | None = None,
) -> dict[str, int]:
    """Census of file extensions in the project's source roots.

    Returns a dict mapping extension (with dot, e.g., '.py') to the count of
    files with that extension. Used by profile detection to decide retrieval
    strategy.

    Args:
        project_path: Root directory of the project.
        source_roots: Subdirectories to scan (e.g., ["src", "app"]).
                      If None, scans the entire project.

    Returns:
        Dict like {'.py': 150, '.tf': 50, '.yaml': 20}.
    """
    project_path = Path(project_path)
    from pipeline.ast_parser import EXTENSION_MAP

    if source_roots is None:
        source_roots = ["."]

    extension_counts: dict[str, int] = {}

    for root_str in source_roots:
        root_path = project_path / root_str
        if not root_path.exists():
            continue

        for filepath in root_path.rglob("*"):
            if not filepath.is_file():
                continue
            ext = filepath.suffix
            if ext not in EXTENSION_MAP:
                continue
            extension_counts[ext] = extension_counts.get(ext, 0) + 1

    return extension_counts


def detect_source_layout(
    project_path: Path,
) -> tuple[list[str], list[str]]:
    """Auto-detect source roots and file patterns for a project.

    Heuristic approach:
    1. Look for standard source dirs (src/, app/, lib/, etc.).
    2. Look for any top-level dir with __init__.py or multiple source files.
    3. Exclude obvious non-source dirs (tests/, benchmarks/, .venv/, etc.).
    4. If a single clear package dir exists, prefer it; else fall back to ["."].
    5. Detect file patterns from extensions actually present.

    Args:
        project_path: Root directory of the project.

    Returns:
        A tuple of (source_roots, file_patterns) where:
        - source_roots: list of relative paths, e.g. ["app"] or ["src", "lib"] or ["."]
        - file_patterns: list of glob patterns, e.g. ["*.py"] or ["*.py", "*.js"]
    """
    project_path = Path(project_path)

    # Known non-source directories to exclude
    non_source_dirs = {
        "tests",
        "test",
        "benchmarks",
        "docs",
        "examples",
        ".venv",
        "venv",
        "env",
        "node_modules",
        ".git",
        ".github",
        ".vscode",
        ".idea",
        ".opencode",
        "build",
        "dist",
        ".cairn",
        "semantic_gateway.egg-info",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }

    # Candidate source root directories (in preference order)
    candidate_source_dirs = ["src", "app", "lib", "packages"]

    # Extension map (from ast_parser)
    from pipeline.ast_parser import EXTENSION_MAP

    # 1. Collect all code files at project root and in subdirectories
    all_files: dict[str, list[Path]] = {}  # ext -> list of paths

    for filepath in project_path.rglob("*"):
        if not filepath.is_file():
            continue
        if filepath.relative_to(project_path).as_posix().startswith(".git/"):
            continue
        ext = filepath.suffix
        if ext not in EXTENSION_MAP:
            continue
        if ext not in all_files:
            all_files[ext] = []
        all_files[ext].append(filepath)

    # 2. Detect candidate source roots
    candidate_roots = []

    # Check standard directories first
    for candidate_dir in candidate_source_dirs:
        candidate_path = project_path / candidate_dir
        if candidate_path.exists() and candidate_path.is_dir():
            # Count code files in this dir
            code_count = sum(
                len([f for f in paths if candidate_path in f.parents or f.parent == candidate_path])
                for paths in all_files.values()
            )
            if code_count > 0:
                candidate_roots.append((candidate_dir, code_count))

    # Check top-level directories for __init__.py or multiple code files
    try:
        for item in project_path.iterdir():
            if not item.is_dir():
                continue
            if item.name in non_source_dirs or item.name.startswith("."):
                continue
            if item.name in candidate_source_dirs:
                # Already checked
                continue

            # Check for __init__.py (Python package)
            if (item / "__init__.py").exists():
                code_count = sum(
                    len([f for f in paths if item in f.parents or f.parent == item])
                    for paths in all_files.values()
                )
                if code_count > 0:
                    candidate_roots.append((item.name, code_count))

            # Check for multiple code files (other languages)
            else:
                code_count = sum(
                    len([f for f in paths if item in f.parents or f.parent == item])
                    for paths in all_files.values()
                )
                if code_count >= 3:  # At least 3 source files
                    candidate_roots.append((item.name, code_count))
    except (OSError, PermissionError):
        pass

    # Does the project root itself hold primary source directly? Use the DOMINANT
    # code extension (the most common across the tree) as "primary" — config files
    # (.json/.toml/.ini) sitting at root don't make the root a source root, but
    # e.g. Terraform module repos keep the main module's *.tf at root next to a
    # modules/ dir of submodules. Without this, narrowing to the dominant subdir
    # (["modules"]) silently drops the root module — the most-used code.
    root_holds_primary = False
    if all_files:
        dominant_ext = max(all_files.items(), key=lambda kv: len(kv[1]))[0]
        root_holds_primary = any(
            f.parent == project_path and f.suffix == dominant_ext for f in all_files[dominant_ext]
        )

    # 3. Decide on source_roots
    source_roots: list[str] = ["."]
    if candidate_roots and not root_holds_primary:
        # Sort by code count (descending)
        candidate_roots.sort(key=lambda x: x[1], reverse=True)
        best_dir = candidate_roots[0][0]

        # If the best dir has significantly more code than others, prefer it alone
        if len(candidate_roots) == 1 or candidate_roots[0][1] > candidate_roots[1][1] * 2:
            source_roots = [best_dir]
    # If root_holds_primary, keep source_roots = ["."] so the root module/package
    # is indexed alongside subdirs; exclude_patterns keep tests/examples out.

    # 4. Detect file patterns from extensions present in chosen roots
    detected_extensions: set[str] = set()

    for root_str in source_roots:
        root_path = project_path / root_str
        if root_path.exists():
            for ext, paths in all_files.items():
                for path in paths:
                    is_in_root = (
                        root_path in path.parents or path.parent == root_path or (root_str == ".")
                    )
                    if is_in_root:
                        detected_extensions.add(ext)

    # Convert extensions to patterns
    if detected_extensions:
        file_patterns = sorted([f"*{ext}" for ext in detected_extensions])
    else:
        # Default to Python if nothing detected
        file_patterns = ["*.py"]

    return source_roots, file_patterns


def _matches_exclude_pattern(relative_path: str, pattern: str) -> bool:
    """Check if relative_path matches glob pattern (handles ** correctly).

    Uses segment-based matching to robustly handle patterns like:
      - **/tests/** (matches tests/ at any level)
      - tests/** (matches tests/ at top level or any level)
      - **/tests (matches tests/ at any level)
      - app/secret/** (matches app/secret/ at top level)

    Args:
        relative_path: Relative POSIX-style path (e.g., "tests/test_x.py").
        pattern: Glob pattern (e.g., "**/tests/**", "tests/**").

    Returns:
        True if the path matches the pattern.
    """
    # Strip ** markers to extract the core path
    core = pattern
    has_leading_glob = core.startswith("**/")
    if has_leading_glob:
        core = core[3:]  # Strip leading **/
    if core.endswith("/**"):
        core = core[:-3]  # Strip trailing /**
    elif core.endswith("/**/**"):
        core = core[:-6]  # Strip trailing /**/**
    if core.endswith("/"):
        core = core[:-1]  # Strip trailing /

    # Check if the core contains glob metacharacters
    has_glob_chars = any(c in core for c in ["*", "?", "["])

    path_parts = relative_path.split("/")
    core_parts = core.split("/")

    if not has_glob_chars:
        # Simple case: core is a plain path with no glob metachars
        if "/" in core:
            # Multi-segment core like "app/secret"
            if has_leading_glob:
                # **/app/secret — match this sequence anywhere in path
                for i in range(len(path_parts) - len(core_parts) + 1):
                    if path_parts[i : i + len(core_parts)] == core_parts:
                        return True
                return False
            else:
                # app/secret (no leading **) — must match from the start
                if len(path_parts) >= len(core_parts):
                    return path_parts[: len(core_parts)] == core_parts
                return False
        else:
            # Single segment like "tests"
            # Both **/tests and tests match at any level
            # (fnmatch doesn't handle ** so treat as matching anywhere)
            return core in path_parts
    else:
        # Complex case: core still has glob metachars
        if "/" in core:
            # Pattern like app/*.pyc — try to match as segment sequence
            for i in range(len(path_parts) - len(core_parts) + 1):
                match = True
                for j, core_seg in enumerate(core_parts):
                    if not fnmatch.fnmatch(path_parts[i + j], core_seg):
                        match = False
                        break
                if match:
                    return True
            return False
        else:
            # Single segment with glob like *.pyc
            # Match any segment or the basename
            for segment in path_parts:
                if fnmatch.fnmatch(segment, core):
                    return True
            return False


def collect_source_files(
    project_path: Path,
    file_patterns: list[str],
    exclude_patterns: list[str],
    source_roots: list[str] | None = None,
) -> list[Path]:
    """Collect source files matching patterns, excluding those matching exclude patterns.

    Args:
        project_path: Root directory to search from.
        file_patterns: List of glob patterns (e.g., ["*.py"]) to include.
        exclude_patterns: List of glob patterns (e.g., ["**/tests/**"]) to exclude.
            Patterns are matched against relative POSIX-style paths.
        source_roots: Subdirectories under project_path to index (e.g., ["app", "lib"]).
            Defaults to ["."] (entire project).

    Returns:
        A deduplicated, sorted list of absolute Path objects.
    """
    project_path = Path(project_path)
    if source_roots is None:
        source_roots = ["."]

    all_files: set[Path] = set()

    for root_str in source_roots:
        root = project_path / root_str
        if not root.exists():
            continue

        for pattern in file_patterns:
            for filepath in root.rglob(pattern):
                if filepath.is_file():
                    all_files.add(filepath)

    # Filter by exclusion patterns (relative path, POSIX-style)
    filtered = []
    for filepath in all_files:
        relative_posix = filepath.relative_to(project_path).as_posix()
        if not any(_matches_exclude_pattern(relative_posix, exc) for exc in exclude_patterns):
            filtered.append(filepath)

    return sorted(filtered)


class RepoManager:
    def __init__(self, project_path: Optional[Path] = None):
        if project_path is None:
            project_path = Path.cwd()
        self.project_path = Path(project_path)
        self.data_dir = self.project_path / ".cairn"

    def get_chroma_path(self) -> Path:
        return self.data_dir / "chroma"

    def get_lance_path(self) -> Path:
        return self.data_dir / "lance"

    def get_index_meta_path(self) -> Path:
        return self.data_dir / "index_meta.json"

    def write_index_meta(self) -> None:
        """Stamp the index with the gateway + schema version that built it.

        Called after every index build (init/reindex) so a later upgrade can tell
        the on-disk index is stale and prompt a reindex.
        """
        from core.version import CAIRN_VERSION, INDEX_SCHEMA_VERSION

        self.data_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "gateway_version": CAIRN_VERSION,
            "schema_version": INDEX_SCHEMA_VERSION,
            "built_at": time.time(),
        }
        self.get_index_meta_path().write_text(json.dumps(meta, indent=2))

    def read_index_meta(self) -> Optional[dict]:
        """Read the index stamp, or None if absent/unreadable."""
        path = self.get_index_meta_path()
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def index_is_stale(self) -> bool:
        """True if the index predates the current schema (or is unstamped).

        An unstamped index was built before stamping existed (schema < current),
        so it is treated as stale.
        """
        from core.version import INDEX_SCHEMA_VERSION

        meta = self.read_index_meta()
        if meta is None:
            return True
        return int(meta.get("schema_version", 0)) < INDEX_SCHEMA_VERSION

    def get_repo_map_path(self) -> Path:
        return self.data_dir / "repo_map.json"

    def get_memory_path(self) -> Path:
        return self.data_dir / "memory.md"

    def get_learning_db_path(self) -> Path:
        return self.data_dir / "learning.db"

    def ensure_directories(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.get_chroma_path().mkdir(exist_ok=True)

    def load_repo_map(self) -> dict:
        path = self.get_repo_map_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Corrupted repo_map.json, returning empty: %s", e)
            return {}

    def save_repo_map(self, data: dict):
        self.ensure_directories()
        self.get_repo_map_path().write_text(json.dumps(data, indent=2))

    def load_memory(self, last_n: int = 10, max_tokens: int | None = None) -> str:
        path = self.get_memory_path()
        if not path.exists():
            return ""
        text = path.read_text()
        lines = text.split("\n")
        result = "\n".join(lines[-last_n:])

        # If max_tokens is specified, trim to fit within budget
        if max_tokens is not None:
            from core.tokens import count_tokens, truncate_to_tokens

            if count_tokens(result) <= max_tokens:
                return result

            # Entry format: lines starting with "[YYYY-MM-DD" are timestamped entries.
            # Split on timestamp markers and drop oldest entries first.
            import re

            entries = re.split(r"(?=\n\[)", result)
            # Entries may start with newline; clean up
            entries = [e.lstrip("\n") for e in entries if e.strip()]

            # Keep dropping oldest (first) entries until we fit the budget
            for i in range(len(entries)):
                trimmed = "\n".join(entries[i:])
                if count_tokens(trimmed) <= max_tokens:
                    return trimmed

            # If even a single newest entry exceeds the budget, hard-truncate it
            if entries:
                return truncate_to_tokens(entries[-1], max_tokens)
            return ""

        return result

    def append_memory(self, entry: str):
        from datetime import datetime

        self.ensure_directories()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(self.get_memory_path(), "a") as f:
            f.write(f"\n[{timestamp}] {entry}")
