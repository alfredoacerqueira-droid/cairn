"""Unit tests for structural retriever — block-identity and reference match."""

from pipeline.retrieval.structural import StructuralRetriever

# Synthetic Terraform-like test data
TERRAFORM_BLOCKS = [
    {
        "id": "infra/main.tf:resource.aws_iam_role.this:1",
        "text": (
            'resource "aws_iam_role" "this" {\n'
            '  name = "eks-cluster-role"\n'
            "  assume_role_policy = jsonencode({\n"
            '    Version = "2012-10-17"\n'
            "    Statement = [{\n"
            '      Action = "sts:AssumeRole"\n'
            '      Effect = "Allow"\n'
            "      Principal = {\n"
            '        Service = "eks.amazonaws.com"\n'
            "      }\n"
            "    }]\n"
            "  })\n"
            "}"
        ),
    },
    {
        "id": "infra/main.tf:resource.aws_kms_key.this:10",
        "text": (
            'resource "aws_kms_key" "this" {\n'
            '  description = "KMS key for cluster encryption"\n'
            "  enable_key_rotation = true\n"
            "  tags = {\n"
            '    Name = "eks-key"\n'
            "  }\n"
            "}"
        ),
    },
    {
        "id": "infra/variables.tf:variable.cluster_version:1",
        "text": (
            'variable "cluster_version" {\n'
            "  type        = string\n"
            '  description = "Kubernetes version for the cluster"\n'
            '  default     = "1.27"\n'
            "}"
        ),
    },
    {
        "id": "infra/main.tf:resource.aws_eks_node_group.main:5",
        "text": (
            'resource "aws_eks_node_group" "main" {\n'
            "  cluster_name    = aws_eks_cluster.main.name\n"
            '  node_group_name = "managed-nodes"\n'
            "  node_role_arn   = aws_iam_role.node.arn\n"
            "  subnet_ids      = var.subnet_ids\n"
            "  scaling_config {\n"
            "    desired_size = 3\n"
            "    max_size     = 5\n"
            "    min_size     = 1\n"
            "  }\n"
            "  depends_on = [\n"
            "    aws_iam_role_policy_attachment.node_policy,\n"
            "  ]\n"
            "}"
        ),
    },
    {
        "id": "infra/main.tf:resource.aws_eks_cluster.main:50",
        "text": (
            'resource "aws_eks_cluster" "main" {\n'
            "  name            = var.cluster_name\n"
            "  version         = var.cluster_version\n"
            "  role_arn        = aws_iam_role.this.arn\n"
            "  vpc_config {\n"
            "    subnet_ids              = var.private_subnets\n"
            "    security_groups         = [aws_security_group.cluster.id]\n"
            "  }\n"
            '  enabled_cluster_log_types = ["api", "audit"]\n'
            "  encryption_config {\n"
            '    resources = ["secrets"]\n'
            "    provider = {\n"
            "      key_arn = aws_kms_key.this.arn\n"
            "    }\n"
            "  }\n"
            "}"
        ),
    },
]


class TestStructuralRetriever:
    def test_index_and_basic_search(self):
        """Test basic indexing and search."""
        retriever = StructuralRetriever()
        retriever.index(TERRAFORM_BLOCKS)

        results = retriever.search("iam role", top_k=5)
        assert len(results) > 0
        assert all("id" in r for r in results)
        assert all("text" in r for r in results)
        assert all("score" in r for r in results)
        assert all(r["source"] == "structural" for r in results)

    def test_iam_role_priority(self):
        """Test that 'iam role' query prioritizes aws_iam_role.

        This is the key differentiator from embeddings (17% top-1).
        """
        retriever = StructuralRetriever()
        retriever.index(TERRAFORM_BLOCKS)

        results = retriever.search("iam role", top_k=5)
        assert len(results) > 0
        # The aws_iam_role should rank first
        assert (
            "aws_iam_role" in results[0]["id"]
        ), f"Expected aws_iam_role #1, got {results[0]['id']}"

    def test_kms_key_encryption(self):
        """Test 'kms key encryption' prioritizes aws_kms_key."""
        retriever = StructuralRetriever()
        retriever.index(TERRAFORM_BLOCKS)

        results = retriever.search("kms key encryption", top_k=5)
        assert len(results) > 0
        # The aws_kms_key should rank first
        assert "aws_kms_key" in results[0]["id"], f"Expected aws_kms_key #1, got {results[0]['id']}"

    def test_cluster_version_variable(self):
        """Test 'cluster version variable' query."""
        retriever = StructuralRetriever()
        retriever.index(TERRAFORM_BLOCKS)

        results = retriever.search("cluster version variable", top_k=5)
        assert len(results) > 0
        # The variable.cluster_version should rank high
        assert (
            "variable.cluster_version" in results[0]["id"]
        ), f"Expected variable.cluster_version #1, got {results[0]['id']}"

    def test_explicit_var_reference(self):
        """Test explicit var.X reference query."""
        retriever = StructuralRetriever()
        retriever.index(TERRAFORM_BLOCKS)

        results = retriever.search("var.cluster_version", top_k=5)
        assert len(results) > 0
        # Should find the variable block
        top_ids = [r["id"] for r in results[:2]]
        found = any("variable.cluster_version" in rid for rid in top_ids)
        assert found, f"Expected variable.cluster_version in top-2, got {top_ids}"

    def test_node_group_query(self):
        """Test 'eks managed node group' query."""
        retriever = StructuralRetriever()
        retriever.index(TERRAFORM_BLOCKS)

        results = retriever.search("eks managed node group", top_k=5)
        assert len(results) > 0
        # Should prioritize aws_eks_node_group
        top_ids = [r["id"] for r in results[:3]]
        found = any("aws_eks_node_group" in rid for rid in top_ids)
        assert found, f"Expected aws_eks_node_group in top-3, got {top_ids}"

    def test_empty_index(self):
        """Test search on empty index."""
        retriever = StructuralRetriever()
        retriever.index([])
        results = retriever.search("query", top_k=5)
        assert results == []

    def test_result_format(self):
        """Test that result dicts have the right shape."""
        retriever = StructuralRetriever()
        retriever.index(TERRAFORM_BLOCKS)

        results = retriever.search("resource", top_k=3)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, dict)
            assert "id" in r
            assert "text" in r
            assert "score" in r
            assert "source" in r
            assert r["source"] == "structural"
            assert isinstance(r["score"], (int, float))
            assert r["score"] > 0

    def test_scoring_order(self):
        """Test that scores are sorted descending."""
        retriever = StructuralRetriever()
        retriever.index(TERRAFORM_BLOCKS)

        results = retriever.search("cluster", top_k=5)
        assert len(results) > 0
        scores = [r["score"] for r in results]
        # Verify descending order
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limit(self):
        """Test that top_k is respected."""
        retriever = StructuralRetriever()
        retriever.index(TERRAFORM_BLOCKS)

        for k in [1, 2, 3, 10]:
            results = retriever.search("resource", top_k=k)
            assert len(results) <= k

    def test_no_match_returns_empty(self):
        """Test that unrelated query returns empty."""
        retriever = StructuralRetriever()
        retriever.index(TERRAFORM_BLOCKS)

        results = retriever.search("zzzz_nonexistent_thing", top_k=5)
        # Nothing matches, should return empty
        assert results == []
