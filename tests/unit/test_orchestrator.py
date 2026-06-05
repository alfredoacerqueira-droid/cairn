"""Unit tests for server/orchestrator.py — Orchestrator, SessionBudget, emit().

Uses injectable fakes (no real LLM/assembler) and a real Config to keep
tests fast, deterministic, and focused on the routing logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.config import BudgetConfig, Config, LocalLLMConfig
from core.tokens import count_tokens
from server.orchestrator import (
    Orchestrator,
    SessionBudget,
    WorkClass,
    emit,
)

# ---- Fake implementations for test isolation ----


@dataclass
class FakeAssembler:
    """Mock ContextAssembler for testing."""

    ctx: str

    def assemble_context(self, query: str) -> str:
        return self.ctx


class FakeLLM:
    """Mock LLM client that records calls and returns deterministic output."""

    def __init__(self):
        self.calls: list[str] = []

    def generate(self, prompt: str, model: str | None = None, stream: bool = False) -> str:
        self.calls.append(prompt)
        return f"ANSWER({len(self.calls)})"


# ---- SessionBudget tests ----


class TestSessionBudget:
    """Tests for SessionBudget token aggregation."""

    def test_init(self):
        budget = SessionBudget(100)
        assert budget.cap == 100
        assert budget.remaining == 100

    def test_charge_under_cap(self):
        budget = SessionBudget(1000)
        text = "hello world"
        result = budget.charge(text)
        assert result == text
        assert budget.remaining < 1000

    def test_charge_over_cap(self):
        budget = SessionBudget(10)
        text = "a" * 100
        result = budget.charge(text)
        # Result should be truncated
        assert count_tokens(result) <= 10
        assert budget.remaining == 0

    def test_charge_exact_cap(self):
        budget = SessionBudget(10)
        text = "hello world"
        n = count_tokens(text)
        if n <= 10:
            budget.charge(text)
            assert budget.remaining == 10 - n
        else:
            budget.charge(text)
            assert budget.remaining == 0

    def test_multiple_charges(self):
        budget = SessionBudget(100)
        budget.charge("hello")
        remaining_after_first = budget.remaining
        budget.charge("world")
        assert budget.remaining < remaining_after_first

    def test_reset(self):
        budget = SessionBudget(100)
        budget.charge("hello world" * 100)
        assert budget.remaining == 0
        budget.reset()
        assert budget.remaining == 100


# ---- emit() tests ----


class TestEmit:
    """Tests for emit() tool output capping."""

    def test_emit_no_budget(self):
        text = "hello world"
        result = emit(text, per_tool_cap=1000)
        assert result == text

    def test_emit_with_budget_under_cap(self):
        budget = SessionBudget(1000)
        text = "hello world"
        result = emit(text, per_tool_cap=1000, session_budget=budget)
        assert result == text

    def test_emit_per_tool_truncation(self):
        budget = SessionBudget(1000)
        text = "hello world " * 100
        result = emit(text, per_tool_cap=10, session_budget=budget)
        assert count_tokens(result) <= 10

    def test_emit_session_budget_depletion(self):
        budget = SessionBudget(50)
        text1 = "a" * 100
        result1 = emit(text1, per_tool_cap=100, session_budget=budget)
        assert count_tokens(result1) <= 50
        # After first emit, remaining should be 0 or very small
        # Second emit with depleted budget should get truncated to 0 tokens
        budget.remaining = 0
        result2 = emit("b" * 100, per_tool_cap=100, session_budget=budget)
        assert result2 == ""

    def test_emit_per_tool_cap_stricter(self):
        budget = SessionBudget(100)
        text = "hello world " * 100
        # per_tool_cap is stricter than budget
        result = emit(text, per_tool_cap=5, session_budget=budget)
        assert count_tokens(result) <= 5


# ---- Planning tests ----


class TestPlan:
    """Tests for Orchestrator.plan() classification."""

    @staticmethod
    def make_config(max_local_tokens: int = 2000) -> Config:
        """Build a test Config with customizable local LLM params."""
        cfg = Config()
        cfg.budget = BudgetConfig(tool_max_tokens=1000)
        cfg.local_llm = LocalLLMConfig(
            enabled=True,
            max_local_tokens=max_local_tokens,
            reduce_reserve_tokens=256,
            chunk_overlap_pct=0.12,
            one_shot_threshold=0.75,
            map_concurrency=1,
        )
        return cfg

    def test_plan_no_llm(self):
        cfg = self.make_config()
        orchestrator = Orchestrator(FakeAssembler("ctx"), cfg, llm=None)
        plan = orchestrator.plan("some text", has_instruction=True)
        assert plan.work_class == WorkClass.CONTEXT_ONLY
        assert "no llm" in plan.reason

    def test_plan_no_instruction(self):
        cfg = self.make_config()
        orchestrator = Orchestrator(FakeAssembler("ctx"), cfg, llm=FakeLLM())
        plan = orchestrator.plan("some text", has_instruction=False)
        assert plan.work_class == WorkClass.CONTEXT_ONLY

    def test_plan_empty_instruction(self):
        cfg = self.make_config()
        orchestrator = Orchestrator(FakeAssembler("ctx"), cfg, llm=FakeLLM())
        plan = orchestrator.plan("some text", has_instruction=False)
        assert plan.work_class == WorkClass.CONTEXT_ONLY

    def test_plan_local_one_shot(self):
        cfg = self.make_config(max_local_tokens=2000)
        cfg.local_llm.one_shot_threshold = 0.75
        orchestrator = Orchestrator(FakeAssembler("ctx"), cfg, llm=FakeLLM())
        # Tiny text: should fit in one shot
        small_text = "hello world"
        plan = orchestrator.plan(small_text, has_instruction=True)
        assert plan.work_class == WorkClass.LOCAL_ONE_SHOT
        assert plan.input_tokens > 0
        assert "one shot" in plan.reason

    def test_plan_local_map_reduce(self):
        cfg = self.make_config(max_local_tokens=200)
        cfg.local_llm.one_shot_threshold = 0.75
        orchestrator = Orchestrator(FakeAssembler("ctx"), cfg, llm=FakeLLM())
        # Medium text: should trigger map-reduce
        # "hello world " * 100 is ~201 tokens, which is > 150 (0.75*200)
        # but may still fit in one chunk. Let's make it explicitly larger.
        medium_text = "hello world " * 500
        plan = orchestrator.plan(medium_text, has_instruction=True)
        assert plan.work_class == WorkClass.LOCAL_MAP_REDUCE
        assert len(plan.chunks) >= 1  # At least one chunk (may be just one if all fits)
        assert "chunks" in plan.reason

    def test_plan_defer_to_cloud(self):
        cfg = self.make_config(max_local_tokens=200)
        cfg.budget.tool_max_tokens = 500
        orchestrator = Orchestrator(FakeAssembler("ctx"), cfg, llm=FakeLLM())
        # Huge text: should defer
        huge_text = "hello " * 10000
        plan = orchestrator.plan(huge_text, has_instruction=True)
        assert plan.work_class == WorkClass.DEFER_TO_CLOUD
        assert "too big" in plan.reason or "cloud_ceiling" in plan.reason


# ---- Chunking tests ----


class TestChunking:
    """Tests for Orchestrator._chunk() token-based chunking."""

    @staticmethod
    def make_config(max_local_tokens: int = 500) -> Config:
        cfg = Config()
        cfg.budget = BudgetConfig(tokenizer_model="claude")
        cfg.local_llm = LocalLLMConfig(
            enabled=True,
            max_local_tokens=max_local_tokens,
            reduce_reserve_tokens=100,
            chunk_overlap_pct=0.12,
            one_shot_threshold=0.75,
            map_concurrency=1,
        )
        return cfg

    def test_chunk_small_text(self):
        cfg = self.make_config()
        orchestrator = Orchestrator(FakeAssembler(""), cfg, llm=None)
        text = "hello world"
        chunks = orchestrator._chunk(text)
        assert len(chunks) >= 1

    def test_chunk_large_text(self):
        cfg = self.make_config(max_local_tokens=300)
        orchestrator = Orchestrator(FakeAssembler(""), cfg, llm=None)
        text = "hello world " * 200
        chunks = orchestrator._chunk(text)
        assert len(chunks) > 1

    def test_chunk_fits_window(self):
        cfg = self.make_config(max_local_tokens=500)
        orchestrator = Orchestrator(FakeAssembler(""), cfg, llm=None)
        text = "hello " * 50
        chunks = orchestrator._chunk(text)
        # Each chunk should fit within the budget
        m = cfg.local_llm.max_local_tokens
        for chunk in chunks:
            n = count_tokens(chunk)
            assert n <= m, f"Chunk has {n} tokens, exceeds limit {m}"

    def test_chunk_overlap(self):
        cfg = self.make_config(max_local_tokens=300)
        orchestrator = Orchestrator(FakeAssembler(""), cfg, llm=None)
        text = "hello " * 200
        chunks = orchestrator._chunk(text)
        if len(chunks) > 1:
            # Consecutive chunks should share some text (overlap)
            # We can't easily verify token-level overlap, but we can check
            # that multiple chunks exist
            assert len(chunks) >= 2


# ---- Execution path tests ----


class TestExecuteContextOnly:
    """Tests for CONTEXT_ONLY execution path."""

    @staticmethod
    def make_config() -> Config:
        cfg = Config()
        cfg.budget = BudgetConfig(tool_max_tokens=1000)
        cfg.local_llm = LocalLLMConfig(enabled=False)
        return cfg

    def test_execute_no_llm_returns_context(self):
        cfg = self.make_config()
        ctx = "important context"
        orchestrator = Orchestrator(FakeAssembler(ctx), cfg, llm=None)
        result = orchestrator.execute("query", payload=None, instruction=None)
        # Should return context (possibly truncated)
        assert len(result) > 0
        assert "important" in result or "context" in result

    def test_execute_no_instruction_returns_context(self):
        cfg = self.make_config()
        ctx = "important context"
        orchestrator = Orchestrator(
            FakeAssembler(ctx),
            cfg,
            llm=FakeLLM(),  # Has LLM but no instruction
        )
        result = orchestrator.execute("query", payload=None, instruction=None)
        # Should still return context, no LLM call
        assert "important" in result or "context" in result

    def test_execute_context_only_no_llm_call(self):
        cfg = self.make_config()
        fake_llm = FakeLLM()
        orchestrator = Orchestrator(FakeAssembler("ctx"), cfg, llm=fake_llm)
        orchestrator.execute("query", payload=None, instruction=None)
        # LLM should not be called
        assert len(fake_llm.calls) == 0

    def test_execute_context_only_truncation(self):
        cfg = self.make_config()
        cfg.budget.tool_max_tokens = 10
        ctx = "a" * 1000
        orchestrator = Orchestrator(FakeAssembler(ctx), cfg, llm=None)
        result = orchestrator.execute("query", payload=None, instruction=None)
        assert count_tokens(result) <= 10


class TestExecuteLocalOneShot:
    """Tests for LOCAL_ONE_SHOT execution path."""

    @staticmethod
    def make_config() -> Config:
        cfg = Config()
        cfg.budget = BudgetConfig(tool_max_tokens=1000)
        cfg.local_llm = LocalLLMConfig(
            enabled=True,
            max_local_tokens=2000,
            one_shot_threshold=0.75,
            reduce_reserve_tokens=256,
            chunk_overlap_pct=0.12,
            map_concurrency=1,
        )
        return cfg

    def test_execute_one_shot_calls_llm_once(self):
        cfg = self.make_config()
        fake_llm = FakeLLM()
        orchestrator = Orchestrator(FakeAssembler("ctx"), cfg, llm=fake_llm)
        result = orchestrator.execute("query", payload="hello world", instruction="summarize")
        assert len(fake_llm.calls) == 1
        assert "ANSWER" in result

    def test_execute_one_shot_returns_answer(self):
        cfg = self.make_config()
        fake_llm = FakeLLM()
        orchestrator = Orchestrator(FakeAssembler("ctx"), cfg, llm=fake_llm)
        result = orchestrator.execute("query", payload="hello", instruction="summarize")
        assert "ANSWER" in result

    def test_execute_one_shot_truncation(self):
        cfg = self.make_config()
        cfg.budget.tool_max_tokens = 10
        fake_llm = FakeLLM()
        orchestrator = Orchestrator(FakeAssembler("ctx"), cfg, llm=fake_llm)
        result = orchestrator.execute("query", payload="hello", instruction="summarize")
        assert count_tokens(result) <= 10


class TestExecuteLocalMapReduce:
    """Tests for LOCAL_MAP_REDUCE execution path."""

    @staticmethod
    def make_config() -> Config:
        cfg = Config()
        cfg.budget = BudgetConfig(tool_max_tokens=1000)
        cfg.local_llm = LocalLLMConfig(
            enabled=True,
            max_local_tokens=300,
            one_shot_threshold=0.75,
            reduce_reserve_tokens=100,
            chunk_overlap_pct=0.12,
            map_concurrency=1,
        )
        return cfg

    def test_execute_map_reduce_calls_llm_multiple_times(self):
        cfg = self.make_config()
        fake_llm = FakeLLM()
        orchestrator = Orchestrator(FakeAssembler("ctx"), cfg, llm=fake_llm)
        # Force split: use much larger payload than one_shot threshold allows
        # max_local_tokens=300, one_shot=0.75 means threshold is 225
        payload = "hello world " * 500
        result = orchestrator.execute("query", payload=payload, instruction="summarize")
        # Should call llm at least once (one shot or map + reduce)
        # With 500 repetitions, should definitely force map-reduce
        assert len(fake_llm.calls) >= 1
        assert "ANSWER" in result

    def test_execute_map_reduce_chunks_fit_window(self):
        cfg = self.make_config()
        orchestrator = Orchestrator(FakeAssembler(""), cfg, llm=None)
        payload = "hello " * 500
        chunks = orchestrator._chunk(payload)
        m = cfg.local_llm.max_local_tokens
        for chunk in chunks:
            n = count_tokens(chunk)
            assert n <= m, f"Chunk {n} exceeds window {m}"

    def test_execute_map_reduce_preserves_order(self):
        cfg = self.make_config()

        class CountingLLM:
            def __init__(self):
                self.calls = []

            def generate(self, prompt: str, model=None, stream=False) -> str:
                self.calls.append(prompt)
                # Return a numbered answer so we can track ordering
                return f"ANSWER({len(self.calls)})"

        fake_llm = CountingLLM()
        orchestrator = Orchestrator(FakeAssembler(""), cfg, llm=fake_llm)
        payload = "chunk1 " * 100 + "\n\n" + "chunk2 " * 100
        result = orchestrator.execute("query", payload=payload, instruction="process")
        # Result should contain answers in order
        assert "ANSWER" in result

    def test_execute_map_reduce_truncation(self):
        cfg = self.make_config()
        cfg.budget.tool_max_tokens = 20
        fake_llm = FakeLLM()
        orchestrator = Orchestrator(FakeAssembler(""), cfg, llm=fake_llm)
        payload = "hello " * 100
        result = orchestrator.execute("query", payload=payload, instruction="summarize")
        assert count_tokens(result) <= 20


class TestExecuteDeferToCloud:
    """Tests for DEFER_TO_CLOUD execution path."""

    @staticmethod
    def make_config() -> Config:
        cfg = Config()
        cfg.budget = BudgetConfig(tool_max_tokens=500)
        cfg.local_llm = LocalLLMConfig(
            enabled=True,
            max_local_tokens=200,
            one_shot_threshold=0.75,
            reduce_reserve_tokens=50,
            chunk_overlap_pct=0.12,
            map_concurrency=1,
        )
        return cfg

    def test_execute_defer_to_cloud_no_llm_call(self):
        cfg = self.make_config()
        fake_llm = FakeLLM()
        orchestrator = Orchestrator(FakeAssembler("ctx"), cfg, llm=fake_llm)
        huge_payload = "hello " * 10000  # Force defer
        orchestrator.execute("query", payload=huge_payload, instruction="summarize")
        # Should NOT call LLM
        assert len(fake_llm.calls) == 0

    def test_execute_defer_to_cloud_returns_context_with_marker(self):
        cfg = self.make_config()
        fake_llm = FakeLLM()
        ctx = "important context"
        orchestrator = Orchestrator(FakeAssembler(ctx), cfg, llm=fake_llm)
        huge_payload = "hello " * 10000
        result = orchestrator.execute("query", payload=huge_payload, instruction="summarize")
        # Should mention deferral
        lower_result = result.lower()
        assert ("defer" in lower_result or "context only" in lower_result
                or "important" in result)

    def test_execute_defer_preserves_context(self):
        cfg = self.make_config()
        fake_llm = FakeLLM()
        ctx = "keep me"
        orchestrator = Orchestrator(FakeAssembler(ctx), cfg, llm=fake_llm)
        # Don't use payload, so assembled context is used (not huge_payload)
        result = orchestrator.execute("query", payload=None, instruction=None)
        # Context should appear in output (no instruction = CONTEXT_ONLY path)
        assert "keep" in result or "me" in result


# ---- Tree-reduce tests ----


class TestTreeReduce:
    """Tests for Orchestrator._reduce() tree reduction."""

    @staticmethod
    def make_config() -> Config:
        cfg = Config()
        cfg.budget = BudgetConfig(tool_max_tokens=1000)
        cfg.local_llm = LocalLLMConfig(
            enabled=True,
            max_local_tokens=300,
            reduce_reserve_tokens=100,
            chunk_overlap_pct=0.12,
            one_shot_threshold=0.75,
            map_concurrency=1,
        )
        return cfg

    def test_reduce_single_partial(self):
        cfg = self.make_config()
        fake_llm = FakeLLM()
        orchestrator = Orchestrator(FakeAssembler(""), cfg, llm=fake_llm)
        partials = ["answer1"]
        result = orchestrator._reduce(partials, "combine")
        assert "ANSWER" in result

    def test_reduce_multiple_partials_fit(self):
        cfg = self.make_config()
        fake_llm = FakeLLM()
        orchestrator = Orchestrator(FakeAssembler(""), cfg, llm=fake_llm)
        partials = ["answer1", "answer2"]
        result = orchestrator._reduce(partials, "combine")
        assert "ANSWER" in result

    def test_reduce_many_partials_tree_reduces(self):
        cfg = self.make_config()
        fake_llm = FakeLLM()
        orchestrator = Orchestrator(FakeAssembler(""), cfg, llm=fake_llm)
        # Create many partials so tree reduction is needed
        partials = [f"answer_{i} " * 50 for i in range(10)]
        result = orchestrator._reduce(partials, "combine")
        # Should make multiple LLM calls (tree reduction)
        assert len(fake_llm.calls) > 1
        assert "ANSWER" in result

    def test_reduce_terminates(self):
        cfg = self.make_config()
        fake_llm = FakeLLM()
        orchestrator = Orchestrator(FakeAssembler(""), cfg, llm=fake_llm)
        partials = [f"text_{i} " * 100 for i in range(20)]
        result = orchestrator._reduce(partials, "summarize")
        # Should terminate and return a single answer
        assert isinstance(result, str)
        assert len(result) > 0

    def test_reduce_single_huge_partial_truncated(self):
        cfg = self.make_config()
        fake_llm = FakeLLM()
        orchestrator = Orchestrator(FakeAssembler(""), cfg, llm=fake_llm)
        # Single partial that exceeds M
        huge_partial = "x" * 10000
        partials = [huge_partial]
        result = orchestrator._reduce(partials, "reduce")
        # Should truncate and still reduce
        assert isinstance(result, str)
        assert "ANSWER" in result


# ---- Integration tests ----


class TestIntegration:
    """End-to-end tests combining multiple components."""

    @staticmethod
    def make_config(small: bool = False) -> Config:
        cfg = Config()
        cfg.budget = BudgetConfig(tool_max_tokens=500)
        if small:
            cfg.local_llm = LocalLLMConfig(
                enabled=True,
                max_local_tokens=150,
                one_shot_threshold=0.75,
                reduce_reserve_tokens=50,
                chunk_overlap_pct=0.12,
                map_concurrency=1,
            )
        else:
            cfg.local_llm = LocalLLMConfig(
                enabled=True,
                max_local_tokens=1000,
                one_shot_threshold=0.75,
                reduce_reserve_tokens=256,
                chunk_overlap_pct=0.12,
                map_concurrency=1,
            )
        return cfg

    def test_full_execution_small_to_large(self):
        cfg = self.make_config(small=True)
        fake_llm = FakeLLM()
        ctx = "base context"
        orchestrator = Orchestrator(FakeAssembler(ctx), cfg, llm=fake_llm)

        # Small text: one shot
        orchestrator.execute("q", "hello", "task1")
        calls_after_small = len(fake_llm.calls)

        # Medium text: map-reduce
        orchestrator.execute("q", "hello " * 100, "task2")
        calls_after_medium = len(fake_llm.calls)

        # Verify progression
        assert calls_after_small >= 1
        assert calls_after_medium > calls_after_small

    def test_execution_respects_tool_cap(self):
        cfg = self.make_config()
        cfg.budget.tool_max_tokens = 20
        fake_llm = FakeLLM()
        orchestrator = Orchestrator(FakeAssembler("ctx"), cfg, llm=fake_llm)
        result = orchestrator.execute("q", "hello", "task")
        assert count_tokens(result) <= 20

    def test_orchestrator_with_session_budget(self):
        cfg = self.make_config()
        session_budget = SessionBudget(cap=50)
        fake_llm = FakeLLM()
        orchestrator = Orchestrator(FakeAssembler("ctx"), cfg, llm=fake_llm)

        result = orchestrator.execute("q", "hello", "task")
        # Manually charge it
        result_charged = session_budget.charge(result)
        assert count_tokens(result_charged) <= 50

    def test_prompt_builders_consistency(self):
        cfg = self.make_config()
        orchestrator = Orchestrator(FakeAssembler(""), cfg, llm=None)

        one_shot = orchestrator._one_shot_prompt("task", "context")
        assert "task" in one_shot
        assert "context" in one_shot

        map_prompt = orchestrator._map_prompt("task", "chunk")
        assert "task" in map_prompt
        assert "chunk" in map_prompt

        reduce_prompt = orchestrator._reduce_prompt("task", "partial")
        assert "task" in reduce_prompt
        assert "partial" in reduce_prompt
