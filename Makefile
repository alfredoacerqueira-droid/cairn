.PHONY: test lint format typecheck bench clean

test:
	pytest

lint:
	ruff check server/ pipeline/ core/ throttle/ cli/

format:
	black server/ pipeline/ core/ throttle/ cli/ tests/

typecheck:
	mypy server/ pipeline/ core/ throttle/ cli/ --ignore-missing-imports

bench:
	python3 -m benchmarks.run_all

bench-ast:
	python3 benchmarks/benchmark_ast_parser.py

bench-e2e:
	python3 benchmarks/end_to_end.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
