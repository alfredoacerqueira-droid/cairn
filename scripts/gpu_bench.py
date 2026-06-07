#!/usr/bin/env python3
"""GPU vs CPU fastembed embedding throughput benchmark.

Measures wall-clock embedding throughput for a representative corpus.
GPU leg only runs if onnxruntime reports a CUDA provider.
Exit 0 even if GPU is skipped (CPU-only is a valid validation run).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

_SOURCE_EXTS = {".py", ".go", ".rs", ".java", ".ts", ".js", ".cs", ".cpp", ".rb", ".tf", ".yaml"}
_SKIP_NAMES = {".git", "node_modules", "vendor", "dist", "build", "__pycache__"}


def _should_skip_dir(parts: tuple[str, ...]) -> bool:
    for part in parts:
        if part in _SKIP_NAMES:
            return True
        if part.startswith(".venv"):
            return True
    return False


def _chunk_text(text: str, size: int = 512) -> list[str]:
    chunks: list[str] = []
    for i in range(0, len(text), size):
        piece = text[i : i + size].strip()
        if piece:
            chunks.append(piece)
    return chunks


def _read_file_safe(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def build_corpus(repo_path: Path, target_n: int) -> list[str]:
    """Walk --repo, collect *source* files, chunk to ~512-char pieces, deterministic order."""
    files: list[Path] = []
    for item in sorted(repo_path.rglob("*")):
        if item.is_file() and item.suffix in _SOURCE_EXTS:
            parts = item.relative_to(repo_path).parts
            if not _should_skip_dir(parts):
                files.append(item)

    chunks: list[str] = []
    for fp in files:
        text = _read_file_safe(fp)
        if text is None:
            continue
        for ch in _chunk_text(text):
            chunks.append(ch)
            if len(chunks) >= target_n:
                return chunks

    return chunks


def run_leg(embedder, corpus: list[str], batch: int) -> tuple[float, float, int]:
    """Warm up, then time embedding corpus in batch slices. Returns (wall_s, texts/sec, dim)."""
    embedder(["warmup"])

    dim = len(embedder(["dim_probe"])[0])

    t0 = time.perf_counter()
    total = len(corpus)
    for i in range(0, total, batch):
        batch_texts = corpus[i : i + batch]
        embedder(batch_texts)
    wall_s = time.perf_counter() - t0

    return wall_s, total / wall_s if wall_s > 0 else 0.0, dim


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU vs CPU fastembed embedding throughput benchmark"
    )
    parser.add_argument(
        "--repo",
        default="/mnt/c/Users/alfre/Projects/django",
        help="Path to source repo for corpus building (default: django)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2000,
        help="Target number of ~512-char corpus chunks (default: 2000)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=256,
        help="Batch size for embedding slices (default: 256)",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-en-v1.5",
        help="fastembed model ID (default: BAAI/bge-small-en-v1.5)",
    )
    parser.add_argument(
        "--report",
        default="scripts/gpu_bench_report.md",
        help="Output markdown report path (default: scripts/gpu_bench_report.md)",
    )

    args = parser.parse_args()
    repo = Path(args.repo)
    target_n = args.n
    batch = args.batch
    model = args.model
    report_path = Path(args.report)

    # Build corpus
    corpus = build_corpus(repo, target_n)
    actual_n = len(corpus)
    print(f"Corpus built: {actual_n} chunks from {args.repo}")

    # GPU detection
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
    except Exception:
        providers = []
    gpu_ok = "CUDAExecutionProvider" in providers

    # GPU name (if any)
    gpu_name: str | None = None
    try:
        from core.resources import get_system_resources

        res = get_system_resources()
        gpu_name = res.get("gpu_name")
    except Exception:
        pass

    # CPU leg (always)
    from pipeline.store.embedders import FastEmbedEmbedder

    print("CPU leg: initializing...")
    cpu_embedder = FastEmbedEmbedder(model=model, device="cpu", threads=0)
    cpu_s, cpu_tps, cpu_dim = run_leg(cpu_embedder, corpus, batch)
    print(f"CPU: {cpu_s:.2f}s  {cpu_tps:.1f} texts/sec  dim={cpu_dim}")

    # GPU leg
    gpu_s: float | None = None
    gpu_tps: float | None = None
    gpu_dim: int | None = None
    gpu_status: str
    gpu_note: str | None = None
    gpu_active_real: bool = False

    if not gpu_ok:
        gpu_status = "GPU provider unavailable in this env"
        print(gpu_status)
    else:
        print("GPU leg: initializing...")
        gpu_embedder = FastEmbedEmbedder(model=model, device="cuda", threads=0)
        gpu_s, gpu_tps, gpu_dim = run_leg(gpu_embedder, corpus, batch)

        gpu_active_real = gpu_embedder.gpu_active
        gpu_active_providers = gpu_embedder._active_providers

        if gpu_active_real:
            print(f"GPU: {gpu_s:.2f}s  {gpu_tps:.1f} texts/sec  dim={gpu_dim}")
            gpu_status = "OK"
        else:
            gpu_note = (
                "WARNING: device='cuda' but onnxruntime fell back to CPU "
                f"(active providers: {gpu_active_providers}) — "
                "GPU benchmark ran on CPU (speedup would be bogus)"
            )
            print(gpu_note)
            gpu_status = "GPU fell back to CPU"

        # Sanity
        if cpu_dim != gpu_dim:
            raise AssertionError(f"Dimension mismatch: CPU dim={cpu_dim} vs GPU dim={gpu_dim}")
        assert cpu_dim == 384, f"Expected dim=384, got {cpu_dim}"

    # Build report
    speedup_line: str
    gpu_label: str
    if gpu_s is not None and gpu_s > 0:
        if gpu_active_real:
            speedup = cpu_s / gpu_s
            speedup_line = f"{speedup:.1f}x"
            gpu_label = "GPU"
        else:
            speedup_line = (
                "GPU fell back to CPU — not a valid GPU measurement "
                "(install CUDA 12.x/cuDNN 9.x)"
            )
            gpu_label = "GPU (FELL BACK TO CPU — not a real GPU run)"
    else:
        speedup_line = "GPU unavailable"
        gpu_label = "GPU"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    report_lines: list[str] = [
        "# GPU vs CPU fastembed Embedding Throughput Benchmark\n",
        f"**Timestamp:** {ts}\n",
        f"**Repo:** {args.repo}\n",
        f"**Corpus size:** {actual_n} chunks\n",
        f"**Batch:** {batch}\n",
        f"**Model:** {model}\n",
        f"**ONNX Runtime providers:** {', '.join(providers) if providers else 'unknown'}\n",
        f"**GPU name:** {gpu_name or 'not detected'}\n",
        "",
        "## Results\n",
        "| Leg | Wall time (s) | Throughput (texts/sec) | Dim |\n",
        "|-----|--------------|------------------------|-----|\n",
        f"| CPU | {cpu_s:.2f} | {cpu_tps:.1f} | {cpu_dim} |\n",
    ]
    if gpu_s is not None:
        report_lines.append(f"| {gpu_label} | {gpu_s:.2f} | {gpu_tps:.1f} | {gpu_dim} |\n")
    else:
        report_lines.append("| GPU | — | — | — |\n")

    report_lines.append(f"\n**Speedup (CPU / GPU):** {speedup_line}\n")

    if gpu_note:
        report_lines.append(f"\n**{gpu_note}**\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(report_lines))

    print(f"\nReport written to {report_path}")
    print(f"\nSpeedup (CPU / GPU): {speedup_line}")

    sys.exit(0)


if __name__ == "__main__":
    main()
