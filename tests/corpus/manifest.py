"""Public repo corpus manifest for integration testing."""

from pathlib import Path
from typing import NamedTuple


class CorpusRepo(NamedTuple):
    """A public repository to index and test against."""

    name: str
    url: str
    queries: list[tuple[str, str]]  # (query, expected_filename_substring)


# List of large public repos + known-answer queries
CORPUS_REPOS = [
    CorpusRepo(
        name="helm",
        url="https://github.com/helm/helm.git",
        queries=[
            ("kubernetes chart template rendering", "template.go"),
            ("helm values processing", "values.go"),
        ],
    ),
    CorpusRepo(
        name="terraform-aws-modules",
        url="https://github.com/terraform-aws-modules/terraform-aws-vpc.git",
        queries=[
            ("AWS VPC resource creation", "main.tf"),
            ("subnet configuration", "variables.tf"),
        ],
    ),
]


def get_cache_dir() -> Path:
    """Get corpus cache directory (under ~/.cache/cairn-corpus or tmp)."""
    cache_dir = Path.home() / ".cache" / "cairn-corpus"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
