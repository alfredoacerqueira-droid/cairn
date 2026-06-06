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
        name="terragrunt-live",
        url="https://github.com/gruntwork-io/terragrunt-infrastructure-live-example.git",
        queries=[
            ("remote state backend config", "root.hcl"),
            ("mysql module dependency", "mysql/terragrunt.hcl"),
            ("environment account id", "account.hcl"),
        ],
    ),
    CorpusRepo(
        name="terraform-eks",
        url="https://github.com/terraform-aws-modules/terraform-aws-eks.git",
        queries=[
            ("managed node group configuration", "node_groups.tf"),
            ("eks cluster iam role", "main.tf"),
            ("openid connect provider issuer", "outputs.tf"),
        ],
    ),
    CorpusRepo(
        name="helm-charts",
        url="https://github.com/prometheus-community/helm-charts.git",
        queries=[
            ("grafana dashboard configuration", "kube-prometheus-stack/values.yaml"),
            ("prometheus deployment template", "prometheus/templates/deploy.yaml"),
            ("alertmanager service monitor", "alertmanager/servicemonitor.yaml"),
        ],
    ),
    CorpusRepo(
        name="cert-manager",
        url="https://github.com/cert-manager/cert-manager.git",
        queries=[
            ("certificate request validation", "types_certificaterequest.go"),
            ("self signed issuer implementation", "types_issuer.go"),
            ("certificate custom resource definition", "cert-manager.io_certificates.yaml"),
        ],
    ),
    CorpusRepo(
        name="django",
        url="https://github.com/django/django.git",
        queries=[
            ("queryset filter operations", "query.py"),
            ("form field validation clean", "fields.py"),
            ("url path routing configuration", "urls/conf.py"),
        ],
    ),
    CorpusRepo(
        name="efcore",
        url="https://github.com/dotnet/efcore.git",
        queries=[
            ("change tracker entity tracking", "ChangeTracking/ChangeTracker.cs"),
            ("db context options configuration", "DbContextOptions.cs"),
            ("db set entity queries", "DbSet.cs"),
        ],
    ),
]

# Expected profile per repo (used in hard_test.py)
EXPECTED_PROFILE = {
    "efcore": "dotnet",
    "terragrunt-live": "iac",
    "terraform-eks": "iac",
    "terraform-aws-modules": "iac",
    "helm-charts": "iac",
    "helm": "iac",
    "cert-manager": "code",
    "django": "python",
}


def get_cache_dir() -> Path:
    """Get corpus cache directory (under ~/.cache/cairn-corpus or tmp)."""
    cache_dir = Path.home() / ".cache" / "cairn-corpus"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
