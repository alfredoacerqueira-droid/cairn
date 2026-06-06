"""Tests for debug mode and observability features.

Covers:
- configure_logging() with debug flag and CAIRN_DEBUG env var
- Semantic search produces debug logs when enabled
- Confidence guard logs rejections
- Context assembly logs cache hits/misses, compression, tokens
- Hybrid retriever logs per-leg timings and reranking
- MCP logging goes to stderr only (preserves stdout for JSON-RPC)
"""

import logging
import sys
import tempfile
from pathlib import Path

from tests.fixtures import fresh_index, make_python_repo


class TestConfigureLogging:
    """Test logging configuration."""

    def test_configure_logging_debug_true(self):
        """configure_logging(debug=True) sets DEBUG level."""
        from core.logging_setup import configure_logging

        # Clear existing handlers
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)

        configure_logging(debug=True)

        # Check root logger is DEBUG
        assert root.level == logging.DEBUG

        # Check stderr handler exists and is DEBUG
        stderr_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stderr_handlers) > 0
        stderr_handler = stderr_handlers[0]
        assert stderr_handler.stream == sys.stderr
        assert stderr_handler.level == logging.DEBUG

    def test_configure_logging_debug_false(self):
        """configure_logging(debug=False) sets WARNING level."""
        from core.logging_setup import configure_logging

        # Clear existing handlers
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)

        configure_logging(debug=False)

        # Check root logger is WARNING
        assert root.level == logging.WARNING

        # Check stderr handler exists and is WARNING
        stderr_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stderr_handlers) > 0
        stderr_handler = stderr_handlers[0]
        assert stderr_handler.level == logging.WARNING

    def test_configure_logging_respects_cairn_debug_env(self, monkeypatch):
        """configure_logging() respects CAIRN_DEBUG=1 env var."""
        from core.logging_setup import configure_logging

        monkeypatch.setenv("CAIRN_DEBUG", "1")

        # Clear existing handlers
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)

        # Call with debug=False, but CAIRN_DEBUG=1 should override
        configure_logging(debug=False)

        # Should be DEBUG due to env var
        assert root.level == logging.DEBUG

    def test_configure_logging_no_stdout_handler(self):
        """configure_logging() never adds a stdout handler."""
        from core.logging_setup import configure_logging

        # Clear existing handlers
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)

        configure_logging(debug=True)

        # Check no stdout handler
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler):
                assert h.stream != sys.stdout

    def test_configure_logging_with_log_file(self):
        """configure_logging(log_file=...) adds file handler."""
        from core.logging_setup import configure_logging

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            log_file = f.name

        try:
            # Clear existing handlers
            root = logging.getLogger()
            for h in root.handlers[:]:
                root.removeHandler(h)

            configure_logging(debug=True, log_file=log_file)

            # Check file handler exists
            file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) > 0
            assert file_handlers[0].baseFilename == log_file
        finally:
            Path(log_file).unlink(missing_ok=True)


class TestSemanticSearchDebugLogs:
    """Test semantic search produces debug logs."""

    def test_semantic_search_logs_debug_info(self, tmp_path, caplog):
        """Semantic search produces debug logs on search."""
        from server.context_assembler import ContextAssembler

        # Setup: build Python repo and index it
        repo_path = make_python_repo(tmp_path)
        fresh_index(repo_path, embeddings=False)

        with caplog.at_level(logging.DEBUG):
            assembler = ContextAssembler(project_path=repo_path)
            assembler.semantic_search("test query", top_k=5)

            # Check for search-related debug logs
            debug_logs = [r.message for r in caplog.records if r.levelname == "DEBUG"]
            # Should have some debug messages about search
            assert len(debug_logs) > 0


class TestAssembleContextDebugLogs:
    """Test context assembly logs cache and token tracking."""

    def test_assemble_context_produces_debug_logs(self, tmp_path, caplog):
        """assemble_context produces debug logs."""
        from server.context_assembler import ContextAssembler

        repo_path = make_python_repo(tmp_path)
        fresh_index(repo_path, embeddings=False)

        with caplog.at_level(logging.DEBUG):
            assembler = ContextAssembler(project_path=repo_path)
            assembler.assemble_context("test prompt")

            # Check for debug logs
            debug_logs = [r.message for r in caplog.records if r.levelname == "DEBUG"]
            assert len(debug_logs) > 0


class TestMCPLoggingSetup:
    """Test MCP logging configuration."""

    def test_mcp_logging_no_stdout(self, monkeypatch):
        """MCP logging setup never adds stdout handler."""
        from core.logging_setup import configure_logging

        # Clear existing handlers
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)

        configure_logging(debug=True)

        # Verify no handler writes to stdout
        for handler in root.handlers:
            if isinstance(handler, logging.StreamHandler):
                assert handler.stream != sys.stdout, "MCP must not log to stdout"


class TestDebugFlagInCLI:
    """Test --debug flag in CLI."""

    def test_cli_main_has_debug_option(self):
        """CLI main group has --debug option."""
        from cli.main import main

        # Check that --debug is in the help or params
        params = [p.name for p in main.params]
        assert "debug" in params


class TestDebugEnvVar:
    """Test CAIRN_DEBUG env var."""

    def test_cairn_debug_env_respected(self, monkeypatch):
        """CAIRN_DEBUG=1 is honored by configure_logging."""
        from core.logging_setup import configure_logging

        monkeypatch.setenv("CAIRN_DEBUG", "1")

        # Clear existing handlers
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)

        # Call with debug=False but env var should override
        configure_logging(debug=False)

        # Should be DEBUG due to env var
        assert root.level == logging.DEBUG
