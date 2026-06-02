"""Benchmark dataset loaders — ground-truth queries for retrieval evaluation.

Each loader returns a list of BenchmarkQuery tuples:
  (query_text, relevant_function_ids)
"""

from benchmarks.datasets.demo import load_demo_queries

__all__ = ["load_demo_queries"]
