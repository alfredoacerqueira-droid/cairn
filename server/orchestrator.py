"""Smart local-LLM task router with map-reduce splitting and session budgeting.

The Orchestrator routes work (user queries + instruction + assembled context) to
one of four execution paths based on token budget and LLM availability:
  - CONTEXT_ONLY: return enriched context (no LLM call)
  - LOCAL_ONE_SHOT: single local-LLM call
  - LOCAL_MAP_REDUCE: split + map-reduce execution
  - DEFER_TO_CLOUD: too big for local → context-only with marker

Uses SessionBudget to track per-session MCP tool output and emit() to cap each
tool's contribution.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum

from core.tokens import count_tokens, get_encoder, truncate_to_tokens  # noqa: I001

CLOUD_CEILING_FACTOR = 4


class WorkClass(Enum):
    """Execution strategy for a given workload."""

    CONTEXT_ONLY = "context_only"
    LOCAL_ONE_SHOT = "local_one_shot"
    LOCAL_MAP_REDUCE = "local_map_reduce"
    DEFER_TO_CLOUD = "defer_to_cloud"


@dataclass
class Plan:
    """Output of the Orchestrator's planning phase."""

    work_class: WorkClass
    input_tokens: int
    reason: str
    chunks: list[str] = field(default_factory=list)


class SessionBudget:
    """Aggregate token cap across all MCP tool outputs in a session."""

    def __init__(self, cap: int):
        """Initialize budget with a total token cap.

        Args:
            cap: Total tokens available for all tool outputs in this session.
        """
        self.cap = cap
        self.remaining = cap

    def charge(self, text: str) -> str:
        """Truncate text if needed and charge against the session budget.

        Args:
            text: Text to charge for.

        Returns:
            Possibly-truncated text, always <= remaining budget.
            Sets remaining to 0 if text exceeded it.
        """
        n = count_tokens(text)
        if n <= self.remaining:
            self.remaining -= n
            return text
        out = truncate_to_tokens(text, self.remaining)
        self.remaining = 0
        return out

    def reset(self) -> None:
        """Reset remaining to cap for a new session."""
        self.remaining = self.cap


def emit(
    text: str,
    per_tool_cap: int,
    session_budget: SessionBudget | None = None,
) -> str:
    """Cap a tool output: truncate to per_tool_cap, then charge session budget.

    Args:
        text: Text to emit.
        per_tool_cap: Hard per-tool limit (always enforced).
        session_budget: Session-level budget (optional). If present, text is
            charged against it after per-tool truncation.

    Returns:
        Text truncated to per_tool_cap, then charged to session_budget if given.
    """
    text = truncate_to_tokens(text, per_tool_cap)
    return session_budget.charge(text) if session_budget is not None else text


class Orchestrator:
    """Smart router for local-LLM work with token-budget-aware map-reduce.

    Injects a ContextAssembler (to fetch semantic context), a config object
    (with token budgets and local LLM parameters), and an LLM client
    (for local generation or None for context-only mode).

    Main entry point: execute(query, payload, instruction) -> str
    """

    def __init__(self, assembler, cfg, llm=None):
        """Initialize the orchestrator.

        Args:
            assembler: ContextAssembler-like with assemble_context(query) method.
            cfg: Config object with .budget and .local_llm attributes.
            llm: LLM client with .generate(prompt, model=None, stream=False)
                method, or None for context-only operation.
        """
        self.assembler = assembler
        self.cfg = cfg
        self.llm = llm

    # ---- Planning ----

    def plan(self, work_text: str, has_instruction: bool) -> Plan:
        """Classify work into one of four execution strategies.

        Args:
            work_text: The full text to process (payload or assembled context).
            has_instruction: Whether an instruction was provided.

        Returns:
            Plan with work_class and reason.
        """
        n = count_tokens(work_text)
        m = self.cfg.local_llm.max_local_tokens
        one = self.cfg.local_llm.one_shot_threshold * m
        cloud_ceiling = self.cfg.budget.tool_max_tokens * CLOUD_CEILING_FACTOR

        if self.llm is None or not has_instruction:
            return Plan(
                WorkClass.CONTEXT_ONLY,
                n,
                "no llm or no instruction",
            )
        if n <= one:
            return Plan(
                WorkClass.LOCAL_ONE_SHOT,
                n,
                f"fits one shot ({n} <= {one:.0f})",
            )
        if n <= cloud_ceiling:
            chunks = self._chunk(work_text)
            return Plan(
                WorkClass.LOCAL_MAP_REDUCE,
                n,
                f"split into {len(chunks)} chunks",
                chunks,
            )
        return Plan(
            WorkClass.DEFER_TO_CLOUD,
            n,
            f"too big for local ({n} > {cloud_ceiling})",
        )

    # ---- Execution ----

    def execute(self, query: str, payload: str | None, instruction: str | None) -> str:
        """Route work to one of four execution paths.

        Args:
            query: The semantic search query.
            payload: Optional explicit payload (overrides assembled context).
            instruction: Task instruction (e.g., "summarize", "extract").

        Returns:
            LLM output (capped to tool_max_tokens) or context-only output.
        """
        ctx = self.assembler.assemble_context(query)
        work_text = payload if payload else ctx
        plan = self.plan(work_text, instruction is not None and instruction != "")
        tool_cap = self.cfg.budget.tool_max_tokens

        if plan.work_class == WorkClass.CONTEXT_ONLY:
            return truncate_to_tokens(ctx, tool_cap)

        if plan.work_class == WorkClass.LOCAL_ONE_SHOT:
            out = self._generate(self._one_shot_prompt(instruction, work_text))
            return truncate_to_tokens(out, tool_cap)

        if plan.work_class == WorkClass.LOCAL_MAP_REDUCE:
            partials = self._map(plan.chunks, instruction)
            out = self._reduce(partials, instruction)
            return truncate_to_tokens(out, tool_cap)

        # DEFER_TO_CLOUD
        return (
            truncate_to_tokens(ctx, tool_cap)
            + "\n\n<!-- cairn: input too large for local LLM; "
            "returning context only -->"
        )

    # ---- Chunking ----

    def _chunk(self, text: str) -> list[str]:
        """TOKEN-based sliding-window chunking.

        Splits text into overlapping chunks that each fit within the local
        model's context window, accounting for prompt overhead and output
        reserve.

        Args:
            text: Full text to chunk.

        Returns:
            List of chunks, each <= chunk_tokens tokens.
        """
        enc = get_encoder(self.cfg.budget.tokenizer_model)
        toks = enc.encode(text)

        m = self.cfg.local_llm.max_local_tokens
        prompt_overhead = 64
        chunk_tokens = max(256, m - self.cfg.local_llm.reduce_reserve_tokens
                          - prompt_overhead)
        overlap = int(chunk_tokens * self.cfg.local_llm.chunk_overlap_pct)
        stride = max(1, chunk_tokens - overlap)

        chunks = []
        i = 0
        while i < len(toks):
            end = min(i + chunk_tokens, len(toks))
            chunk_toks = toks[i:end]
            try:
                chunk_text = enc.decode(chunk_toks)
                chunks.append(chunk_text)
            except Exception:
                # Fallback: decode what we can
                try:
                    chunk_text = enc.decode(chunk_toks[:-1]) if chunk_toks else ""
                    if chunk_text:
                        chunks.append(chunk_text)
                except Exception:
                    pass

            if end >= len(toks):
                break
            i += stride

        return chunks if chunks else [text]

    # ---- Compression (optional, graceful fallback) ----

    def _compress_chunk(self, chunk: str) -> str:
        """Compress chunk via TokenCompressor, gracefully.

        Args:
            chunk: Text to compress.

        Returns:
            Compressed chunk, or original if compression unavailable.
        """
        try:
            from server.token_compressor import FilterLevel, Language, TokenCompressor

            compressor = TokenCompressor(level=FilterLevel.MINIMAL)
            return compressor.compress(chunk, Language.PYTHON)
        except Exception:
            return chunk

    # ---- Map phase ----

    def _map(self, chunks: list[str], instruction: str) -> list[str]:
        """Apply instruction to each chunk in parallel (map phase).

        Args:
            chunks: List of text chunks.
            instruction: Task instruction.

        Returns:
            List of LLM outputs, one per chunk, in order.
        """
        concurrency = max(1, self.cfg.local_llm.map_concurrency)

        if concurrency <= 1:
            # Sequential execution for determinism in tests
            return [
                self._generate(self._map_prompt(instruction, self._compress_chunk(chunk)))
                for chunk in chunks
            ]

        # Parallel execution
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    lambda c=chunk: self._generate(
                        self._map_prompt(instruction, self._compress_chunk(c))
                    )
                )
                for chunk in chunks
            ]
            return [f.result() for f in futures]

    # ---- Reduce phase (tree-reduce for large batches) ----

    def _reduce(self, partials: list[str], instruction: str) -> str:
        """Reduce map outputs via tree-reduction if needed.

        If joined partials fit in one local call, return their reduction.
        Otherwise, greedily batch and recurse.

        Args:
            partials: List of map-phase outputs.
            instruction: Task instruction.

        Returns:
            Single reduced string.
        """
        m = self.cfg.local_llm.max_local_tokens

        joined = "\n\n".join(partials)
        if count_tokens(joined) <= m:
            return self._generate(self._reduce_prompt(instruction, joined))

        # Tree-reduce: greedily batch partials
        batches = []
        current_batch = []
        current_tokens = 0

        for partial in partials:
            n = count_tokens(partial)
            if current_tokens + n > m and current_batch:
                batches.append("\n\n".join(current_batch))
                current_batch = []
                current_tokens = 0
            current_batch.append(partial)
            current_tokens += n

        if current_batch:
            batches.append("\n\n".join(current_batch))

        # Reduce each batch
        reduced = []
        for batch in batches:
            # Guard: if a single batch exceeds m, truncate it
            if count_tokens(batch) > m:
                batch = truncate_to_tokens(batch, m)
            reduced.append(self._generate(self._reduce_prompt(instruction, batch)))

        # Recurse: if we still have multiple reduced items, reduce again
        if len(reduced) > 1:
            return self._reduce(reduced, instruction)

        return reduced[0] if reduced else ""

    # ---- Generation ----

    def _generate(self, prompt: str) -> str:
        """Call the local LLM.

        Args:
            prompt: Prompt to send.

        Returns:
            LLM output.

        Raises:
            RuntimeError: If llm is None (shouldn't happen given plan gating).
        """
        if self.llm is None:
            raise RuntimeError("Tried to generate without an LLM client")
        return self.llm.generate(prompt)

    # ---- Prompt builders ----

    def _one_shot_prompt(self, instruction: str, work_text: str) -> str:
        """Build a one-shot prompt.

        Args:
            instruction: The task.
            work_text: The context/work to process.

        Returns:
            Formatted prompt.
        """
        return f"{instruction}\n\nContext:\n{work_text}"

    def _map_prompt(self, instruction: str, chunk: str) -> str:
        """Build a map-phase prompt.

        Args:
            instruction: The task.
            chunk: One chunk of the work.

        Returns:
            Formatted prompt.
        """
        return f"{instruction}\n\nProcess this section:\n{chunk}"

    def _reduce_prompt(self, instruction: str, partials: str) -> str:
        """Build a reduce-phase prompt.

        Args:
            instruction: The task.
            partials: Joined partial results to combine.

        Returns:
            Formatted prompt.
        """
        return (
            f"Combine these partial findings into a single coherent answer "
            f"to: {instruction}\n\nPartial results:\n{partials}"
        )
