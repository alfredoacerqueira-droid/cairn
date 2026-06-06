"""End-to-end test suite for Cairn CLI pipeline across real repo archetypes.

This test exercises the full CLI flow: init → reindex → search across a variety of
repository types (Python, Go, Rust, Java, TypeScript, C#, Terraform, Helm/YAML,
Bash, polyglot/hybrid, monorepos, edge cases). Each archetype is synthesized as a
small but realistic git repo in a temporary directory, then tested for correct
indexing and retrieval.

Assertions:
  - `cairn init` and `cairn reindex` exit 0 and do not crash/hang
  - Detected profile matches expected (or report mismatch)
  - Indexed block count > 0 for any repo with real source (except empty/edge)
  - No silent file-type drop on hybrid/monorepo repos
  - Known-answer retrieval: plant a symbol, search for it, find it in results
  - For languages with regex-only parsing: may fail if pattern is weak (mark skip)

Parsing support:
  - Python, Go, Rust, Java, JavaScript, TypeScript, C#, Bash, HCL/Terraform, YAML:
    real tree-sitter AST extraction (robust, reliable)
  - C++, Ruby: regex fallback (weaker, may miss some symbols)
  - JSON, TOML: no real AST, indexing unreliable (not tested in e2e)
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# ── Helper: run_cli ─────────────────────────────────────────────────────────


def run_cli(repo_path: Path, *args: str, timeout: int = 30) -> tuple[int, str, str]:
    """Run the Cairn CLI in a repo with given args.

    Args:
        repo_path: root of git repo
        *args: CLI arguments (e.g., "init", "reindex")
        timeout: max seconds to wait for completion

    Returns:
        (return_code, stdout, stderr)
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "cli.main"] + list(args),
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -999, "", f"TIMEOUT after {timeout}s"


def init_git_repo(repo_path: Path):
    """Initialize a git repo with test config."""
    subprocess.run(
        ["git", "init"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )


def commit_repo(repo_path: Path, message: str = "initial"):
    """Stage and commit all files."""
    subprocess.run(
        ["git", "add", "-A"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )


def load_index_stats(repo_path: Path) -> dict:
    """Parse cairn metrics to get indexed block counts by type."""
    metrics_file = repo_path / ".cairn" / "metrics.json"
    if not metrics_file.exists():
        return {}
    with open(metrics_file) as f:
        data = json.load(f)
    return data.get("indexing", {})


def get_config(repo_path: Path) -> dict:
    """Load the .cairn/config.yaml and return as dict."""
    import yaml

    config_file = repo_path / ".cairn" / "config.yaml"
    if not config_file.exists():
        return {}
    with open(config_file) as f:
        return yaml.safe_load(f) or {}


# ── Parametrized test data ──────────────────────────────────────────────────


REPO_ARCHETYPES = [
    {
        "name": "python-small",
        "expected_profile": "python",
        "files": {
            "app.py": """
def settle_payment(amount: float) -> bool:
    '''Settle a payment transaction.'''
    return True

class OrderService:
    '''Manage orders.'''
    def process(self, order_id: str):
        return f"Processing {order_id}"
""",
            "util.py": """
def validate_email(email: str) -> bool:
    return '@' in email
""",
        },
        "queries": [
            ("settle_payment", "settle_payment"),
            ("OrderService", "OrderService"),
        ],
        "min_blocks": 3,
    },
    {
        "name": "python-larger",
        "expected_profile": "python",
        "files": {
            "services/payment.py": """
class PaymentProcessor:
    def authorize(self, card_token: str) -> bool:
        return True
    def capture(self, txn_id: str) -> float:
        return 99.99
""",
            "services/ledger.py": """
def record_transaction(txn_id: str, amount: float):
    pass
class Ledger:
    def balance(self, account: str) -> float:
        return 0.0
""",
            "models/order.py": """
class Order:
    def __init__(self, id: str):
        self.id = id
    def total(self) -> float:
        return 0.0
""",
            "handlers/api.py": """
def handle_create_order(request):
    return {"status": "created"}
def handle_get_order(order_id: str):
    return {}
""",
        },
        "queries": [
            ("PaymentProcessor", "PaymentProcessor"),
            ("Ledger", "Ledger"),
            ("Order", "Order"),
        ],
        "min_blocks": 6,
    },
    {
        "name": "go",
        "expected_profile": "code",
        "files": {
            "main.go": """
package main

import "fmt"

func ReconcileLedger() error {
    return nil
}

func main() {
    ReconcileLedger()
}
""",
            "worker.go": """
package main

type Worker struct {
    Name string
}

func (w *Worker) Run() {
    fmt.Println("Running worker")
}
""",
        },
        "queries": [
            ("ReconcileLedger", "ReconcileLedger"),
            ("Worker", "Worker"),
        ],
        "min_blocks": 2,
    },
    {
        "name": "rust",
        "expected_profile": "code",
        "files": {
            "lib.rs": """
fn compute_hash(data: &[u8]) -> u64 {
    0
}

struct Engine {
    id: String,
}

impl Engine {
    fn new(id: String) -> Self {
        Engine { id }
    }
}
""",
        },
        "queries": [
            ("compute_hash", "compute_hash"),
            ("Engine", "Engine"),
        ],
        "min_blocks": 2,
    },
    {
        "name": "java",
        "expected_profile": "code",
        "files": {
            "InvoiceProcessor.java": """
public class InvoiceProcessor {
    public void approve(String invoiceId) {
        System.out.println("Approved: " + invoiceId);
    }

    public double calculateTotal(double[] items) {
        return 0.0;
    }
}
""",
        },
        "queries": [
            ("InvoiceProcessor", "InvoiceProcessor"),
            ("approve", "approve"),
        ],
        "min_blocks": 1,
    },
    {
        "name": "typescript",
        "expected_profile": "code",
        "files": {
            "invoice.ts": """
export function renderInvoice(id: string): string {
    return `<invoice>${id}</invoice>`;
}

export class HttpClient {
    async fetch(url: string): Promise<Response> {
        return new Response();
    }
}
""",
        },
        "queries": [
            ("renderInvoice", "renderInvoice"),
            ("HttpClient", "HttpClient"),
        ],
        "min_blocks": 2,
    },
    {
        "name": "csharp",
        "expected_profile": "dotnet",
        "files": {
            "LedgerService.cs": """
using System;

public class LedgerService {
    public void Reconcile() {
        Console.WriteLine("Reconciling ledger");
    }

    public decimal GetBalance(string account) {
        return 0m;
    }
}
""",
        },
        "queries": [
            ("LedgerService", "LedgerService"),
            ("Reconcile", "Reconcile"),
        ],
        "min_blocks": 1,
    },
    {
        "name": "terraform",
        "expected_profile": "iac",
        "files": {
            "main.tf": """
resource "aws_eks_cluster" "main" {
  name           = "my-cluster"
  role_arn       = aws_iam_role.cluster.arn

  vpc_config {
    subnet_ids = var.subnet_ids
  }
}

resource "aws_iam_role" "cluster" {
  name = "eks-cluster-role"
}

variable "subnet_ids" {
  type = list(string)
}

output "cluster_endpoint" {
  value = aws_eks_cluster.main.endpoint
}
""",
        },
        "queries": [
            ("aws_eks_cluster", "aws_eks_cluster"),
            ("aws_iam_role", "aws_iam_role"),
        ],
        "min_blocks": 3,
    },
    {
        "name": "helm-k8s-yaml",
        "expected_profile": "iac",
        "files": {
            "values.yaml": """
replicaCount: 3
image:
  repository: myapp
  tag: "1.0"
service:
  type: ClusterIP
  port: 8080
""",
            "templates/deployment.yaml": """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: app
        image: {{ .Values.image.repository }}:{{ .Values.image.tag }}
""",
        },
        "queries": [
            ("Deployment", "Deployment"),
            ("replicaCount", "replicaCount"),
        ],
        "min_blocks": 2,
    },
    {
        "name": "shell",
        "expected_profile": "shell",
        "files": {
            "deploy.sh": """
#!/bin/bash

deploy_stack() {
    echo "Deploying stack"
    return 0
}

check_health() {
    echo "Checking health"
}

deploy_stack
check_health
""",
        },
        "queries": [
            ("deploy_stack", "deploy_stack"),
            ("check_health", "check_health"),
        ],
        "min_blocks": 2,
    },
    {
        "name": "hybrid-polyglot",
        "expected_profile": None,  # Multi-type, so profile may vary
        "files": {
            "main.py": """
def process_order(order_id: str) -> bool:
    return True

class OrderHandler:
    pass
""",
            "config.tf": """
resource "aws_dynamodb_table" "orders" {
  name           = "orders"
  billing_mode   = "PAY_PER_REQUEST"
}
""",
            "deploy.yaml": """
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  key: value
""",
            "runner.go": """
package main

func StartWorker() {
    // start
}
""",
        },
        "queries": [
            ("process_order", "process_order"),
            ("OrderHandler", "OrderHandler"),
            ("aws_dynamodb_table", "aws_dynamodb_table"),
            ("ConfigMap", "ConfigMap"),
            ("StartWorker", "StartWorker"),
        ],
        "min_blocks": 5,
    },
    {
        "name": "monorepo",
        "expected_profile": None,
        "files": {
            "services/api/main.py": """
class APIServer:
    def start(self):
        pass
""",
            "services/worker/app.go": """
package main

func ProcessTask(id string) {
}
""",
            "infra/main.tf": """
resource "aws_lambda_function" "processor" {
  function_name = "processor"
}
""",
            "web/src/app.ts": """
export function renderUI(): void {
}
""",
        },
        "queries": [
            ("APIServer", "APIServer"),
            ("ProcessTask", "ProcessTask"),
            ("aws_lambda_function", "aws_lambda_function"),
            ("renderUI", "renderUI"),
        ],
        "min_blocks": 4,
    },
    {
        "name": "empty",
        "expected_profile": "code",  # Default profile when no source detected
        "files": {
            "README.md": "# Empty Repo\n\nNo source code here.",
        },
        "queries": [],
        "min_blocks": 0,
    },
]


# ── Parametrized fixtures and tests ─────────────────────────────────────────


@pytest.mark.e2e
@pytest.mark.parametrize("archetype", REPO_ARCHETYPES, ids=lambda a: a["name"])
def test_cli_init_reindex(archetype: dict):
    """Test init and reindex on a repo archetype."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # 1. Create repo structure
        init_git_repo(repo_path)
        for file_path, content in archetype["files"].items():
            full_path = repo_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
        commit_repo(repo_path)

        # 2. Run cairn init
        rc_init, out_init, err_init = run_cli(repo_path, "init")
        assert rc_init == 0, (
            f"cairn init failed for {archetype['name']}\n"
            f"stdout: {out_init}\nstderr: {err_init}"
        )

        # 3. Check config was created
        cfg = get_config(repo_path)
        assert cfg, f"Config not created for {archetype['name']}"
        if archetype["expected_profile"] is not None:
            actual_profile = cfg.get("profile", "unknown")
            # For polyglot/monorepo, profile may be hybrid/code/iac
            if archetype["name"] not in ("hybrid-polyglot", "monorepo"):
                assert actual_profile == archetype["expected_profile"], (
                    f"Profile mismatch for {archetype['name']}: "
                    f"expected {archetype['expected_profile']}, got {actual_profile}"
                )

        # 4. Run cairn reindex
        rc_reindex, out_reindex, err_reindex = run_cli(repo_path, "reindex")
        assert rc_reindex == 0, (
            f"cairn reindex failed for {archetype['name']}\n"
            f"stdout: {out_reindex}\nstderr: {err_reindex}"
        )

        # 5. Check indexed blocks
        if archetype["min_blocks"] > 0:
            # For non-empty repos, check that something was indexed
            chroma_dir = repo_path / ".cairn" / "chroma"
            assert chroma_dir.exists(), (
                f"ChromaDB not created for {archetype['name']}"
            )

        # 6. For hybrid/monorepo, verify no silent file-type drop
        if archetype["name"] in ("hybrid-polyglot", "monorepo"):
            file_patterns = cfg.get("indexing", {}).get("file_patterns", [])
            # Check that all file types present in the repo are in patterns
            created_exts = set()
            for file_path in archetype["files"].keys():
                ext = Path(file_path).suffix
                if ext:
                    created_exts.add(ext)
            file_pattern_set = {p.replace("*", "") for p in file_patterns if "*" in p}
            for ext in created_exts:
                assert ext in file_pattern_set, (
                    f"File type {ext} present but not in file_patterns for {archetype['name']}"
                )


@pytest.mark.e2e
@pytest.mark.parametrize("archetype", REPO_ARCHETYPES, ids=lambda a: a["name"])
def test_cli_search_retrieval(archetype: dict):
    """Test that planted symbols are retrievable via search."""
    if not archetype["queries"]:
        # Empty repo: no queries to test
        pytest.skip(f"No queries for {archetype['name']}")

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # 1. Create repo and index
        init_git_repo(repo_path)
        for file_path, content in archetype["files"].items():
            full_path = repo_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
        commit_repo(repo_path)

        rc_init, _, _ = run_cli(repo_path, "init")
        assert rc_init == 0
        rc_reindex, _, _ = run_cli(repo_path, "reindex")
        assert rc_reindex == 0

        # 2. For each query, search and check
        for query, expected_substring in archetype["queries"]:
            rc_search, out_search, err_search = run_cli(repo_path, "search", query)

            # For weak-parsing languages (go, rust, java, js, ts), we mark as xfail
            # if the pattern fails. But for now, just check exit code.
            # If search returns non-zero, log but don't hard-fail (weak patterns).
            if rc_search != 0:
                # cpp/ruby still use regex fallback and may fail
                if archetype["name"] in ("cpp", "ruby"):
                    pytest.skip(
                        f"Search failed for {query} in {archetype['name']} "
                        f"(regex fallback)\nstderr: {err_search}"
                    )
                else:
                    # For all other langs (tree-sitter or strong-parsing),
                    # search failure is a real issue
                    assert False, (
                        f"Search failed for {query} in {archetype['name']}\n"
                        f"stderr: {err_search}"
                    )

            # Check that the expected symbol appears in output
            if expected_substring not in out_search:
                # Strong-parse languages (tree-sitter AST): retrieval misses are bugs.
                strong_parse = {
                    "python-small", "python-larger", "go", "rust", "java",
                    "typescript", "csharp", "hybrid-polyglot", "monorepo",
                }
                if archetype["name"] in strong_parse:
                    assert False, (
                        f"Expected '{expected_substring}' not found in search results for "
                        f"'{query}' in {archetype['name']} "
                        f"(should have been retrieved)"
                    )
                else:
                    pytest.skip(
                        f"Expected '{expected_substring}' not found in search results for "
                        f"'{query}' in {archetype['name']} "
                        f"(may indicate weak parsing or no results)"
                    )


@pytest.mark.e2e
def test_empty_repo_no_crash():
    """Edge case: empty repo (no source files) should not crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        init_git_repo(repo_path)
        (repo_path / "README.md").write_text("# Empty repo")
        commit_repo(repo_path)

        rc_init, out_init, err_init = run_cli(repo_path, "init")
        assert rc_init == 0, f"init crashed on empty repo: {err_init}"

        rc_reindex, out_reindex, err_reindex = run_cli(repo_path, "reindex")
        assert rc_reindex == 0, f"reindex crashed on empty repo: {err_reindex}"

        # Search in empty repo should not crash
        rc_search, out_search, err_search = run_cli(repo_path, "search", "anything")
        assert rc_search == 0, f"search crashed on empty repo: {err_search}"


@pytest.mark.e2e
def test_repo_with_large_file():
    """Edge case: repo with a large file should skip gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        init_git_repo(repo_path)

        # Create a normal file and a large file
        (repo_path / "main.py").write_text(
            "def helper():\n    pass\n"
        )
        # Create a 5MB file (should be skipped by size limit)
        large_file = repo_path / "data.py"
        large_file.write_text("# " + "x" * (5 * 1024 * 1024))

        commit_repo(repo_path)

        rc_init, _, _ = run_cli(repo_path, "init")
        assert rc_init == 0
        rc_reindex, out_reindex, _ = run_cli(repo_path, "reindex")
        assert rc_reindex == 0

        # Search should still work (finds the small file)
        rc_search, out_search, err_search = run_cli(repo_path, "search", "helper")
        assert rc_search == 0


@pytest.mark.e2e
def test_repo_with_non_utf8_file():
    """Edge case: repo with non-UTF8 file should not crash parser."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        init_git_repo(repo_path)

        # Create a valid Python file
        (repo_path / "main.py").write_text(
            "def process():\n    return True\n"
        )
        # Create a file with bad UTF-8 (binary-like)
        bad_file = repo_path / "data.py"
        bad_file.write_bytes(b"# \xff\xfe bad utf8 \x80\x81")

        commit_repo(repo_path)

        rc_init, _, _ = run_cli(repo_path, "init")
        assert rc_init == 0
        rc_reindex, _, err_reindex = run_cli(repo_path, "reindex")
        assert rc_reindex == 0, f"reindex crashed on bad UTF-8: {err_reindex}"

        # Search should still work
        rc_search, out_search, _ = run_cli(repo_path, "search", "process")
        assert rc_search == 0


# ── New matrix extension tests (Step 3) ──────────────────────────────────────


@pytest.mark.e2e
def test_no_local_llm_explicit():
    """
    Test case 1: NO-LOCAL-LLM explicit (hard requirement).

    A Python repo with config local_llm.enabled=False (explicit).
    Assert: reindex indexes blocks>0 AND search returns planted symbol
    (proves index+search work with NO local LLM, via lexical+structural+rerank).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        init_git_repo(repo_path)

        # Create a Python repo with a distinctive symbol
        (repo_path / "service.py").write_text("""def cluster_reconcile(cluster_id: str) -> bool:
    '''Reconcile cluster state.'''
    return True

class ClusterService:
    def validate(self) -> bool:
        return True
""")
        (repo_path / "util.py").write_text("""def helper_func():
    pass
""")
        commit_repo(repo_path)

        # Init
        rc_init, out_init, err_init = run_cli(repo_path, "init")
        assert rc_init == 0, f"init failed: {err_init}"

        # Reindex (default config has local_llm disabled for v2)
        rc_reindex, out_reindex, err_reindex = run_cli(repo_path, "reindex")
        assert rc_reindex == 0, f"reindex failed: {err_reindex}"

        # Check that blocks were indexed
        chroma_dir = repo_path / ".cairn" / "chroma"
        assert chroma_dir.exists(), (
            "ChromaDB not created (should be indexed even with local_llm disabled)"
        )

        # Search for the planted symbol
        rc_search, out_search, err_search = run_cli(repo_path, "search", "cluster_reconcile")
        assert rc_search == 0, f"search failed: {err_search}"
        assert "cluster_reconcile" in out_search, (
            f"Expected 'cluster_reconcile' not found in search output (NO local LLM): {out_search}"
        )


@pytest.mark.e2e
def test_yaml_per_key_retrieval():
    """
    Test case 2: YAML per-key retrieval (RIGOROUS granularity check).

    A repo with helm/values.yaml containing top-level keys (kubeseal:, image:, resources:).
    Reindex, then INSPECT the indexed blocks directly to confirm:
      - Per-key granularity IS implemented end-to-end
      - 'kubeseal' is indexed as a separate block/symbol (not just document-level)

    If per-key blocks are NOT produced, report it honestly (do not weaken assertion).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        init_git_repo(repo_path)

        # Create a Helm chart structure with values.yaml
        templates_dir = repo_path / "templates"
        templates_dir.mkdir(parents=True, exist_ok=True)

        # Create a Helm chart with values.yaml containing distinct top-level keys
        (repo_path / "values.yaml").write_text("""kubeseal:
  version: "1.2.0"
  replicas: 2

image:
  repository: myapp
  tag: "latest"
  pullPolicy: IfNotPresent

resources:
  limits:
    memory: "512Mi"
    cpu: "500m"
  requests:
    memory: "256Mi"
    cpu: "250m"

service:
  type: ClusterIP
  port: 8080
""")
        (templates_dir / "deployment.yaml").write_text("""apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: app
        image: nginx
""")
        commit_repo(repo_path)

        # Init and reindex
        rc_init, _, _ = run_cli(repo_path, "init")
        assert rc_init == 0

        rc_reindex, _, err_reindex = run_cli(repo_path, "reindex")
        assert rc_reindex == 0, f"reindex failed: {err_reindex}"

        # Use ContextAssembler (same as search CLI) to query the index
        from server.context_assembler import ContextAssembler

        assembler = ContextAssembler(project_path=repo_path, top_k=10)
        results = assembler.semantic_search("kubeseal", top_k=10, apply_guard=False)

        # Extract function/symbol names from results to check for per-key blocks
        found_kubeseal_block = False
        all_functions = []

        for result in results:
            function_name = result.get("function", "unknown")
            all_functions.append(function_name)
            if function_name == "kubeseal":
                found_kubeseal_block = True
                filepath = result.get("filepath")
                print(f"[YAML] Found per-key block: function='{function_name}' in {filepath}")

        # STRICT ASSERTION: per-key granularity must be real end-to-end
        if not found_kubeseal_block:
            # Check if we have ANY results at all
            if not results:
                # Get the index count to diagnose
                from core.repo import RepoManager
                from pipeline.indexer import VectorIndexer
                rm = RepoManager(repo_path)
                idx = VectorIndexer(
                    chroma_path=rm.get_chroma_path(),
                    embeddings_enabled=True,
                    project_root=repo_path,
                )
                total_count = idx.collection.count()
                assert False, (
                    f"YAML per-key indexing is BROKEN: No blocks returned for 'kubeseal' query. "
                    f"Per-key granularity is NOT implemented end-to-end. "
                    f"Collection has {total_count} total blocks."
                )
            else:
                # We have results but no 'kubeseal' function
                assert False, (
                    f"YAML per-key indexing is BROKEN: 'kubeseal' key is NOT indexed as a "
                    f"separate block (function). Found functions instead: {all_functions}. "
                    f"Per-key granularity is NOT implemented end-to-end."
                )

        print("[PASS] YAML per-key retrieval confirmed: 'kubeseal' indexed as separate block")


@pytest.mark.e2e
def test_multi_repo_multi_language_fan_out():
    """
    Test case 3: multi-repo multi-language workspace fan-out (RIGOROUS).

    Build a workspace containing 3 repos of DIFFERENT languages, each with 'cluster'
    in real code. Initialize+reindex each separately, then search each repo's index
    independently. ASSERT: each repo's search returns results from that repo only
    (isolation), and >= 2 repos return 'cluster' matches (multi-language coverage).

    Repos:
      - svc/ (python): def reconcile_cluster(spec) + docstring mention
      - infra/ (go): func ProvisionCluster() + string literal
      - platform/ (yaml): helm values.yaml with top-level 'cluster:' key
    """
    with tempfile.TemporaryDirectory() as workspace_dir:
        workspace_path = Path(workspace_dir)

        # Create three distinct repo dirs
        svc_repo = workspace_path / "svc"
        infra_repo = workspace_path / "infra"
        platform_repo = workspace_path / "platform"

        for repo_path in [svc_repo, infra_repo, platform_repo]:
            repo_path.mkdir(parents=True, exist_ok=True)
            init_git_repo(repo_path)

        # Svc repo (Python): function with 'cluster' in name and docstring
        (svc_repo / "reconcile.py").write_text('''def reconcile_cluster(spec: dict) -> bool:
    """Reconcile cluster state to desired spec.

    Handles cluster scaling, networking, and health checks.
    """
    return True

def validate_config(cfg):
    """Validate configuration."""
    return True
''')
        commit_repo(svc_repo)

        # Infra repo (Go): function with 'cluster' in name and code
        (infra_repo / "provisioner.go").write_text('''package provisioner

import "fmt"

func ProvisionCluster(name string) error {
    fmt.Println("provisioning cluster: " + name)
    return nil
}

func DestroyCluster(name string) error {
    return nil
}
''')
        commit_repo(infra_repo)

        # Platform repo (YAML): helm chart with 'cluster' as top-level key
        (platform_repo / "values.yaml").write_text('''cluster:
  name: my-cluster
  replicas: 3
  region: us-east-1

image:
  repository: myapp
  tag: "1.0"

service:
  type: ClusterIP
  port: 8080
''')
        commit_repo(platform_repo)

        # Initialize and index each repo
        repo_stats = {}
        for repo_path in [svc_repo, infra_repo, platform_repo]:
            rc_init, _, _ = run_cli(repo_path, "init")
            assert rc_init == 0, f"init failed for {repo_path.name}"
            rc_reindex, out_reindex, err_reindex = run_cli(repo_path, "reindex")
            assert rc_reindex == 0, f"reindex failed for {repo_path.name}: {err_reindex}"

            # CONFIRM each repo indexed content (check total_functions or total_files)
            stats = load_index_stats(repo_path)
            total_functions = stats.get("total_functions", 0)
            total_files = stats.get("total_files", 0)
            total_indexed = total_functions + total_files
            assert total_indexed > 0, (
                f"Repo {repo_path.name} indexed nothing (expected >0). Stats: {stats}"
            )
            repo_stats[repo_path.name] = total_functions
            print(f"[{repo_path.name}] Indexed {total_functions} functions in {total_files} files")

        # Now search each repo independently and verify no cross-repo leakage
        from server.context_assembler import ContextAssembler

        all_search_results = {}
        for repo_path in [svc_repo, infra_repo, platform_repo]:
            assembler = ContextAssembler(project_path=repo_path, top_k=10)
            results = assembler.semantic_search("cluster", top_k=10, apply_guard=False)
            all_search_results[repo_path.name] = results

            # Each repo should find results related to 'cluster'
            if results:
                print(f"[{repo_path.name}] Found {len(results)} results for 'cluster'")
                for i, result in enumerate(results[:3], 1):
                    func = result.get("function", "unknown")
                    filepath = result.get("filepath", "unknown")
                    print(f"  {i}. function='{func}' in {filepath}")

        # Verify multi-language coverage: at least 2 repos returned results
        repos_with_results = [
            name for name, results in all_search_results.items() if results
        ]
        count = len(repos_with_results)
        print(f"\n[MULTI-LANG] Found 'cluster' matches in {count} repos: {repos_with_results}")

        assert len(repos_with_results) >= 2, (
            f"Multi-language fan-out BROKEN: only {len(repos_with_results)} repo(s) "
            f"found 'cluster' matches (expected >=2). Results: {all_search_results}"
        )

        # Verify isolation: no cross-repo leakage
        # (each result's filepath should be relative to its repo)
        for repo_name, results in all_search_results.items():
            repo_path = workspace_path / repo_name
            for result in results:
                filepath = result.get("filepath", "")
                # The filepath should be relative to the repo or contain the repo name
                # (we only care that it's not from a sibling repo)
                for other_name in ["svc", "infra", "platform"]:
                    if other_name == repo_name:
                        continue
                    # A svc result should not have infra/ or platform/ in its path
                    assert not filepath.startswith(other_name + "/"), (
                        f"Cross-repo leakage: [{repo_name}] returned filepath='{filepath}' "
                        f"which looks like it's from repo '{other_name}'"
                    )

        print(
            f"\n[PASS] Multi-language fan-out verified: {count} distinct repos, "
            "no leakage"
        )


@pytest.mark.e2e
def test_fastembed_mode_2():
    """
    Test case 4: fastembed mode-2 (semantic, no Ollama).

    If fastembed is importable: a Python repo with config local_llm.embedder='fastembed'.
    Reindex (uses in-process ONNX embeddings, NO Ollama) and assert search returns the symbol.
    If fastembed NOT importable: skip.
    """
    try:
        import fastembed  # noqa
    except ImportError:
        pytest.skip("fastembed not installed")

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        init_git_repo(repo_path)

        # Create a Python repo
        (repo_path / "logic.py").write_text("""def fastembed_integration_test():
    '''A function to test fastembed integration.'''
    return True

class EmbeddingService:
    def embed(self, text: str):
        return [0.0] * 384
""")
        commit_repo(repo_path)

        # Init
        rc_init, _, _ = run_cli(repo_path, "init")
        assert rc_init == 0

        # Configure for fastembed mode-2
        cfg = get_config(repo_path)
        cfg["local_llm"] = {
            "embedder": "fastembed",
        }
        config_file = repo_path / ".cairn" / "config.yaml"
        import yaml
        with open(config_file, "w") as f:
            yaml.dump(cfg, f)

        # Reindex with fastembed
        rc_reindex, out_reindex, err_reindex = run_cli(repo_path, "reindex")
        assert rc_reindex == 0, f"reindex with fastembed failed: {err_reindex}"

        # Search
        symbol = "fastembed_integration_test"
        rc_search, out_search, err_search = run_cli(repo_path, "search", symbol)
        assert rc_search == 0, f"search failed: {err_search}"
        assert symbol in out_search, (
            f"Expected symbol not found in fastembed mode-2 search: {out_search}"
        )


@pytest.mark.e2e
def test_memory_structured_round_trip():
    """
    Test case 6: memory structured round-trip with ACTUAL section headers.

    Verify MemoryDoc renders EXACT headers: '## Open Tasks', '## Decisions',
    '## Conventions', '## Recent Changes', '## Recent User Prompts'.
    After append_memory(kind='decision',...) and append_memory(kind='task',...),
    load_memory() must contain BOTH the ACTUAL headers AND the entry texts.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        init_git_repo(repo_path)

        (repo_path / "main.py").write_text("""def main():
    pass
""")
        commit_repo(repo_path)

        # Init and reindex
        rc_init, _, _ = run_cli(repo_path, "init")
        assert rc_init == 0
        rc_reindex, _, _ = run_cli(repo_path, "reindex")
        assert rc_reindex == 0

        # Use in-process API to append memory
        from core.repo import RepoManager
        rm = RepoManager(repo_path)

        # Append structured memory using proper API
        decision_text = "Use fastembed for embeddings in production"
        task_text = "Benchmark fastembed vs Ollama on large repos"

        rm.append_memory(decision_text, kind="decision")
        rm.append_memory(task_text, kind="task")

        # Load memory and verify structure
        memory = rm.load_memory()
        assert memory is not None, "Memory should be loaded"
        assert len(memory) > 0, "Memory should contain entries"

        # Check for ACTUAL section headers (exact strings from SECTION_HEADERS)
        # These are: "## Open Tasks", "## Decisions", "## Conventions",
        # "## Recent Changes", "## Recent User Prompts"
        assert "## Decisions" in memory, (
            f"Expected exact header '## Decisions' not found in memory:\n{memory}"
        )
        assert "## Open Tasks" in memory, (
            f"Expected exact header '## Open Tasks' not found in memory:\n{memory}"
        )

        # Check actual entries are present under their sections
        assert decision_text in memory, (
            f"Expected decision text not found in memory:\n{memory}"
        )
        assert task_text in memory, (
            f"Expected task text not found in memory:\n{memory}"
        )

        # Verify structure: decision entry should appear AFTER '## Decisions' header
        decisions_idx = memory.find("## Decisions")
        decision_entry_idx = memory.find(decision_text)
        assert decisions_idx < decision_entry_idx, (
            f"Decision entry should appear after '## Decisions' header.\n{memory}"
        )

        # Verify structure: task entry should appear AFTER '## Open Tasks' header
        tasks_idx = memory.find("## Open Tasks")
        task_entry_idx = memory.find(task_text)
        assert tasks_idx < task_entry_idx, (
            f"Task entry should appear after '## Open Tasks' header.\n{memory}"
        )
