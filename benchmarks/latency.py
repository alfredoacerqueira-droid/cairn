"""Latency benchmarks — p50/p95/p99 with warmup, cold vs warm cache.

Uses time.perf_counter for high-precision timing.  Supports pytest-benchmark
integration if available, falling back to manual measurement.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any


def measure_latency(
    fn: Callable[[], Any],
    iters: int = 50,
    warmup: int = 5,
) -> dict[str, float]:
    """Measure p50, p95, p99 latencies in milliseconds.

    Runs warmup iterations (discarded), then ``iters`` measured iterations.
    Uses perf_counter_ns for nanosecond precision.
    """
    for _ in range(warmup):
        fn()

    measurements: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000
        measurements.append(elapsed_ms)

    sorted_ms = sorted(measurements)
    n = len(sorted_ms)

    def percentile(p: float) -> float:
        idx = int(p * (n - 1) / 100)
        return round(sorted_ms[idx], 3)

    return {
        "p50_ms": percentile(50),
        "p95_ms": percentile(95),
        "p99_ms": percentile(99),
        "mean_ms": round(statistics.mean(measurements), 3),
        "min_ms": round(min(measurements), 3),
        "max_ms": round(max(measurements), 3),
        "warmup_iters": warmup,
        "measured_iters": iters,
    }


def measure_cold_vs_warm(
    cold_fn: Callable[[], Any],
    warm_fn: Callable[[], Any],
    iters: int = 30,
) -> dict[str, Any]:
    """Compare cold-cache vs warm-cache latencies."""
    cold = measure_latency(cold_fn, iters=iters, warmup=0)
    warm = measure_latency(warm_fn, iters=iters, warmup=5)

    return {"cold_cache": cold, "warm_cache": warm}


def latency_summary(name: str, stats: dict[str, float]) -> str:
    """Format a latency summary as a human-readable line."""
    return (
        f"{name:20s}: p50={stats['p50_ms']:.1f}ms  "
        f"p95={stats['p95_ms']:.1f}ms  p99={stats['p99_ms']:.1f}ms  "
        f"(n={stats['measured_iters']})"
    )
