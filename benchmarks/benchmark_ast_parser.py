"""Benchmarks for the semantic code gateway."""

import json
import statistics
import sys
import time
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.ast_parser import ASTParser


def generate_test_code(num_functions: int = 100) -> str:
    """Generate a Python file with N functions."""
    lines = []
    for i in range(num_functions):
        lines.append(f"def function_{i}(arg1, arg2=None):")
        lines.append(f'    """Function number {i}."""')
        lines.append("    result = arg1 + arg2")
        lines.append("    return result")
        lines.append("")
    return "\n".join(lines)


def benchmark_ast_parser():
    """Benchmark AST parsing at different file sizes."""
    parser = ASTParser()
    results = {}

    sizes = {"xsmall": 10, "small": 100, "medium": 500, "large": 2000}

    for size_name, num_funcs in sizes.items():
        code = generate_test_code(num_funcs)
        times = []

        for _ in range(10):
            start = time.perf_counter()
            parser.parse_string(code)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        results[size_name] = {
            "functions": num_funcs,
            "mean_ms": round(statistics.mean(times) * 1000, 2),
            "median_ms": round(statistics.median(times) * 1000, 2),
            "p95_ms": round(sorted(times)[int(len(times) * 0.95)] * 1000, 2),
            "min_ms": round(min(times) * 1000, 2),
            "max_ms": round(max(times) * 1000, 2),
        }

    return results


def run_benchmarks():
    """Run all benchmarks and save results."""
    print("Running benchmarks...")

    results = {
        "ast_parser": benchmark_ast_parser(),
    }

    # Save results
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "current.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\nAST Parser Benchmark Results:")
    print("-" * 60)
    print(f"{'Size':<8} {'Functions':<10} {'Mean (ms)':<12} {'P95 (ms)':<12}")
    print("-" * 60)

    ast = results["ast_parser"]
    for size_name, data in ast.items():
        print(
            f"{size_name:<8} {data['functions']:<10} "
            f"{data['mean_ms']:<12.2f} {data['p95_ms']:<12.2f}"
        )

    print("-" * 60)

    # Check baseline
    baseline_path = output_dir / "baseline.json"
    if not baseline_path.exists():
        print("\nNo baseline found. Creating baseline from current results.")
        baseline_path.write_text(json.dumps(results, indent=2))
    else:
        print(f"\nBaseline exists at {baseline_path}")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    run_benchmarks()
