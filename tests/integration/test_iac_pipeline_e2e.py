"""Hermetic end-to-end test of the IaC indexing+retrieval pipeline.

Builds a synthetic Terraform module repo on disk and runs the real path:
detect_source_layout -> collect_source_files -> VectorIndexer -> StructuralRetriever.

No Ollama, no flashrank, no network: the iac profile disables embeddings, so the
indexer must never call the embedder (asserted via a fake client that raises).

This is the test class that would have caught this release's real bugs:
  - embeddings firing on the iac profile (wasted calls + dropped files),
  - the root module being dropped from source-root detection,
  - examples/tests/.github polluting the index,
  - structural retrieval not surfacing the right config block.
"""

from pathlib import Path

from core.config import Config
from core.repo import RepoManager, collect_source_files, detect_source_layout
from pipeline.ast_parser import ASTParser
from pipeline.indexer import VectorIndexer
from pipeline.retrieval.structural import StructuralRetriever


class _ExplodingOllama:
    """Stand-in OllamaClient that fails loudly if anything tries to embed."""

    embed_model = "should-never-be-called"

    def embed(self, *a, **k):  # pragma: no cover - must not run
        raise AssertionError("embeddings disabled, but embed() was called")

    def embed_batch(self, *a, **k):  # pragma: no cover - must not run
        raise AssertionError("embeddings disabled, but embed_batch() was called")


def _write_tf_module_repo(root: Path) -> None:
    """A repo shaped like terraform-aws-eks: root module + submodules + noise."""
    (root / "main.tf").write_text(
        'resource "aws_eks_cluster" "this" {\n'
        "  name = var.cluster_name\n"
        "  encryption_config {\n"
        "    provider { key_arn = module.kms.key_arn }\n"
        "  }\n"
        "}\n"
    )
    (root / "variables.tf").write_text(
        'variable "create_kms_key" {\n'
        '  description = "Whether to create a KMS key for cluster encryption"\n'
        "  type        = bool\n"
        "  default     = true\n"
        "}\n"
    )

    mod = root / "modules" / "karpenter"
    mod.mkdir(parents=True)
    (mod / "main.tf").write_text(
        'resource "aws_sqs_queue" "this" {\n'
        "  name                              = local.queue_name\n"
        "  kms_data_key_reuse_period_seconds = 300\n"
        "}\n"
    )

    # Noise that must be excluded from the index.
    ex = root / "examples" / "demo"
    ex.mkdir(parents=True)
    (ex / "main.tf").write_text('resource "aws_eks_cluster" "demo" {}\n')
    tst = root / "tests" / "unit"
    tst.mkdir(parents=True)
    (tst / "main.tf").write_text('resource "aws_eks_cluster" "fixture" {}\n')
    gh = root / ".github" / "workflows"
    gh.mkdir(parents=True)
    (gh / "ci.yml").write_text("name: CI\non: [push]\n")


class TestIacPipelineEndToEnd:
    def test_detection_keeps_root_and_excludes_noise(self, tmp_path):
        _write_tf_module_repo(tmp_path)

        roots, patterns = detect_source_layout(tmp_path)
        assert roots == ["."], f"root module dropped: {roots}"
        assert "*.tf" in patterns

        cfg = Config()  # default exclude_patterns
        files = collect_source_files(tmp_path, ["*.tf"], cfg.indexing.exclude_patterns, roots)
        rel = {f.relative_to(tmp_path).as_posix() for f in files}
        assert "main.tf" in rel  # root module
        assert "modules/karpenter/main.tf" in rel  # submodule
        assert not any(p.startswith("examples/") for p in rel), rel
        assert not any(p.startswith("tests/") for p in rel), rel
        assert not any(p.startswith(".github/") for p in rel), rel

    def test_index_without_embeddings_then_structural_retrieval(self, tmp_path):
        _write_tf_module_repo(tmp_path)
        repo = RepoManager(tmp_path)

        # embeddings_enabled=False is the iac contract: must not touch Ollama.
        indexer = VectorIndexer(
            chroma_path=repo.get_chroma_path(),
            ollama_client=_ExplodingOllama(),
            embeddings_enabled=False,
        )

        parser = ASTParser()
        cfg = Config()
        files = collect_source_files(tmp_path, ["*.tf"], cfg.indexing.exclude_patterns, ["."])
        for fp in files:
            indexer.index_ast(parser.parse_file(fp))

        # Root + submodule blocks indexed; examples/tests excluded.
        data = indexer.collection.get(include=["metadatas", "documents"])
        names = {m["function"] for m in data["metadatas"]}
        assert any("aws_eks_cluster.this" in n for n in names), names
        assert any("create_kms_key" in n for n in names), names
        assert any("aws_sqs_queue.this" in n for n in names), names
        assert not any("demo" in n or "fixture" in n for n in names), names

        # Structural retrieval (no embeddings, no rerank) finds the cluster KMS
        # control variable for a config query.
        items = [{"id": i, "text": t} for i, t in zip(data["ids"], data["documents"])]
        structural = StructuralRetriever()
        structural.index(items)
        hits = structural.search("create_kms_key cluster encryption", top_k=5)
        assert hits, "structural retrieval returned nothing"
        assert any("create_kms_key" in h["id"] for h in hits), [h["id"] for h in hits]

    def test_index_meta_stamp_and_staleness(self, tmp_path):
        repo = RepoManager(tmp_path)
        assert repo.read_index_meta() is None
        assert repo.index_is_stale() is True  # unstamped -> stale

        repo.write_index_meta()
        meta = repo.read_index_meta()
        assert meta is not None
        from core.version import CAIRN_VERSION, INDEX_SCHEMA_VERSION

        assert meta["gateway_version"] == CAIRN_VERSION
        assert meta["schema_version"] == INDEX_SCHEMA_VERSION
        assert repo.index_is_stale() is False
