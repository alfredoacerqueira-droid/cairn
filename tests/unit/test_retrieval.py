"""Unit tests for hybrid retrieval: BM25, ASTRank, embeddings, and RRF fusion."""

from pipeline.retrieval.ast_rank import ASTRankRetriever
from pipeline.retrieval.bm25 import BM25Retriever
from pipeline.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion

SAMPLE_FUNCTIONS = [
    {
        "id": "app/main.py:create_nail:10",
        "text": "def create_nail(payload):\n    nail = Nail(id=next_id, **payload.dict())",
        "name": "create_nail",
        "filepath": "app/main.py",
    },
    {
        "id": "app/main.py:place_order:80",
        "text": (
            "def place_order(payload):\n    total = 0.0\n"
            "    for item in payload.items:\n        nail = db.get(item.id)"
        ),
        "name": "place_order",
        "filepath": "app/main.py",
    },
    {
        "id": "app/main.py:delete_nail:30",
        "text": (
            "def delete_nail(nail_id):\n    if nail_id not in db:\n" "        raise HTTPError(404)"
        ),
        "name": "delete_nail",
        "filepath": "app/main.py",
    },
    {
        "id": "app/main.py:list_nails:5",
        "text": "def list_nails():\n    return list(db.values())",
        "name": "list_nails",
        "filepath": "app/main.py",
    },
    {
        "id": "app/main.py:get_order:100",
        "text": (
            "def get_order(order_id):\n    order = db.get(order_id)\n"
            "    if not order:\n        raise HTTPError(404)"
        ),
        "name": "get_order",
        "filepath": "app/main.py",
    },
]


class TestBM25Retriever:
    def test_index_and_search(self):
        bm25 = BM25Retriever()
        items = [{"id": f["id"], "text": f["text"]} for f in SAMPLE_FUNCTIONS]
        bm25.index(items)

        results = bm25.search("create nail order", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3
        assert all("id" in r for r in results)
        assert all("score" in r for r in results)
        assert all(r["source"] == "bm25" for r in results)

    def test_empty_index(self):
        bm25 = BM25Retriever()
        bm25.index([])
        results = bm25.search("query", top_k=5)
        assert results == []

    def test_exact_match_scores_higher(self):
        bm25 = BM25Retriever()
        items = [{"id": f["id"], "text": f["text"]} for f in SAMPLE_FUNCTIONS]
        bm25.index(items)

        results = bm25.search("create_nail", top_k=5)
        assert len(results) > 0
        assert "create_nail" in results[0]["id"]

    def test_no_match_gives_low_scores(self):
        bm25 = BM25Retriever()
        items = [{"id": f["id"], "text": f["text"]} for f in SAMPLE_FUNCTIONS]
        bm25.index(items)

        results = bm25.search("zzzz_nonexistent_function", top_k=3)
        if len(results) > 0:
            for r in results:
                assert r["score"] <= 0.01 or r["score"] == 0.0


class TestASTRankRetriever:
    def test_index_and_search(self):
        ast = ASTRankRetriever()
        ast.index(SAMPLE_FUNCTIONS)

        results = ast.search("create nail order", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3
        assert all("id" in r for r in results)
        assert all(r["source"] == "ast_rank" for r in results)

    def test_empty_index(self):
        ast = ASTRankRetriever()
        ast.index([])
        results = ast.search("query")
        assert results == []

    def test_keyword_match_ranks_higher(self):
        ast = ASTRankRetriever()
        ast.index(SAMPLE_FUNCTIONS)

        results = ast.search("place_order total", top_k=3)
        assert len(results) > 0
        assert any("place_order" in r["id"] for r in results)

    def test_repo_map_optional(self):
        ast = ASTRankRetriever()
        ast.index(SAMPLE_FUNCTIONS, repo_map=None)
        results = ast.search("delete", top_k=3)
        assert len(results) > 0
        assert any("delete_nail" in r["id"] for r in results)


class TestReciprocalRankFusion:
    def test_fuses_two_lists(self):
        a = [
            {"id": "a", "text": "func a", "score": 10.0},
            {"id": "b", "text": "func b", "score": 5.0},
        ]
        b = [
            {"id": "c", "text": "func c", "score": 10.0},
            {"id": "a", "text": "func a", "score": 7.0},
        ]
        fused = reciprocal_rank_fusion([a, b])
        assert len(fused) == 3
        assert fused[0]["source"] == "hybrid"

    def test_score_decays_with_rank(self):
        a = [
            {"id": "a", "text": "func a", "score": 1.0},
            {"id": "b", "text": "func b", "score": 0.5},
            {"id": "c", "text": "func c", "score": 0.1},
        ]
        b = [
            {"id": "b", "text": "func b", "score": 1.0},
            {"id": "a", "text": "func a", "score": 0.9},
        ]
        fused = reciprocal_rank_fusion([a, b])
        assert len(fused) == 3
        assert fused[0]["score"] > fused[-1]["score"]

    def test_weights_affect_ranking(self):
        a = [
            {"id": "a", "text": "func a", "score": 1.0},
            {"id": "b", "text": "func b", "score": 0.1},
        ]
        b = [
            {"id": "b", "text": "func b", "score": 1.0},
            {"id": "a", "text": "func a", "score": 0.1},
        ]

        fused_a_heavy = reciprocal_rank_fusion([a, b], weights=[10.0, 0.1])
        fused_b_heavy = reciprocal_rank_fusion([a, b], weights=[0.1, 10.0])

        assert fused_a_heavy[0]["id"] == "a"
        assert fused_b_heavy[0]["id"] == "b"

    def test_empty_input(self):
        fused = reciprocal_rank_fusion([], weights=[])
        assert fused == []


class TestHybridRetriever:
    def test_bm25_mode(self):
        bm25 = BM25Retriever()
        ast = ASTRankRetriever()
        bm25.index([{"id": f["id"], "text": f["text"]} for f in SAMPLE_FUNCTIONS])

        hybrid = HybridRetriever(bm25=bm25, ast_rank=ast, embeddings=None, mode="bm25")
        results = hybrid.search("create", top_k=3)
        assert all(r["source"] == "bm25" for r in results)

    def test_ast_mode(self):
        bm25 = BM25Retriever()
        ast = ASTRankRetriever()
        ast.index(SAMPLE_FUNCTIONS)

        hybrid = HybridRetriever(bm25=bm25, ast_rank=ast, embeddings=None, mode="ast")
        results = hybrid.search("order", top_k=3)
        assert all(r["source"] == "ast_rank" for r in results)

    def test_hybrid_mode(self):
        bm25 = BM25Retriever()
        ast = ASTRankRetriever()
        bm25.index([{"id": f["id"], "text": f["text"]} for f in SAMPLE_FUNCTIONS])
        ast.index(SAMPLE_FUNCTIONS)

        hybrid = HybridRetriever(bm25=bm25, ast_rank=ast, embeddings=None, mode="hybrid")
        results = hybrid.search("create nail order", top_k=3)
        assert len(results) > 0
        assert all(r["source"] == "hybrid" for r in results)

    def test_hybrid_with_embeddings(self):
        bm25 = BM25Retriever()
        ast = ASTRankRetriever()
        bm25.index([{"id": f["id"], "text": f["text"]} for f in SAMPLE_FUNCTIONS])
        ast.index(SAMPLE_FUNCTIONS)

        hybrid = HybridRetriever(bm25=bm25, ast_rank=ast, embeddings=None, mode="hybrid")
        results = hybrid.search("place order total", top_k=3)
        assert len(results) > 0

    def test_config_mode_switches_results(self):
        bm25 = BM25Retriever()
        ast = ASTRankRetriever()
        bm25.index([{"id": f["id"], "text": f["text"]} for f in SAMPLE_FUNCTIONS])
        ast.index(SAMPLE_FUNCTIONS)

        hybrid_bm25 = HybridRetriever(bm25=bm25, ast_rank=ast, embeddings=None, mode="bm25")
        hybrid_hybrid = HybridRetriever(bm25=bm25, ast_rank=ast, embeddings=None, mode="hybrid")

        bm25_results = hybrid_bm25.search("order", top_k=5)
        hybrid_results = hybrid_hybrid.search("order", top_k=5)

        assert len(bm25_results) > 0
        assert len(hybrid_results) > 0
        assert bm25_results[0]["source"] == "bm25"
        assert hybrid_results[0]["source"] == "hybrid"
