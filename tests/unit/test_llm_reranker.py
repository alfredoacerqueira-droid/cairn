"""Unit tests for the opt-in LLM reranker (offline; mocked Ollama)."""

from pipeline.retrieval.llm_reranker import LLMReranker


class _FakeOllama:
    """Scores by block text so ordering is deterministic and testable."""

    def __init__(self, keyword: str = "iam_role"):
        self.keyword = keyword
        self.calls = 0

    def generate(self, prompt: str, model=None) -> str:
        self.calls += 1
        block = prompt.split("Block:")[-1]
        return "9" if self.keyword in block else "1"


def _cands():
    return [
        {"id": "a:kms:1", "text": "resource aws_kms_key this", "similarity": 0.5},
        {"id": "b:iam:1", "text": "resource aws_iam_role this", "similarity": 0.5},
    ]


def test_rerank_orders_by_llm_score():
    r = LLMReranker(ollama_client=_FakeOllama(keyword="iam_role"))
    out = r.rerank("iam role for cluster", _cands(), top_k=2)
    assert out[0]["id"] == "b:iam:1"
    assert out[0]["rerank_score"] == 0.9
    assert out[1]["rerank_score"] == 0.1


def test_rerank_scores_normalized_0_1():
    r = LLMReranker(ollama_client=_FakeOllama())
    out = r.rerank("x", _cands())
    for c in out:
        assert 0.0 <= c["rerank_score"] <= 1.0


def test_rerank_graceful_on_error():
    class Broken:
        def generate(self, prompt, model=None):
            raise RuntimeError("ollama down")

    r = LLMReranker(ollama_client=Broken())
    out = r.rerank("x", _cands(), top_k=2)
    # Falls back to each candidate's similarity, never raises.
    assert all(c["rerank_score"] == 0.5 for c in out)


def test_rerank_empty():
    r = LLMReranker(ollama_client=_FakeOllama())
    assert r.rerank("x", []) == []


def test_rerank_latency_cap():
    """Only the first _MAX_CANDIDATES are LLM-scored; rest keep similarity."""
    from pipeline.retrieval import llm_reranker

    fake = _FakeOllama()
    cands = [
        {"id": f"c{i}", "text": "block", "similarity": 0.3}
        for i in range(llm_reranker._MAX_CANDIDATES + 5)
    ]
    llm_reranker.LLMReranker(ollama_client=fake).rerank("q", cands)
    assert fake.calls == llm_reranker._MAX_CANDIDATES
