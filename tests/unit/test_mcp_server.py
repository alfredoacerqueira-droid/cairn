"""Unit tests for MCP server."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from cli.main import main


class TestMCPServerImport:
    def test_mcp_server_imports(self):
        """Test that server.mcp_server module imports without error."""
        from server.mcp_server import run_stdio, mcp

        assert run_stdio is not None
        assert mcp is not None
        assert mcp.name == "cairn"


class TestMCPServerTools:
    def test_search_code_tool_exists(self):
        """Test that search_code tool is decorated as an MCP tool."""
        from server.mcp_server import search_code

        # Check that the function exists and is callable
        assert callable(search_code)
        assert search_code.__doc__ is not None

    def test_assemble_context_tool_exists(self):
        """Test that assemble_context tool is decorated as an MCP tool."""
        from server.mcp_server import assemble_context

        # Check that the function exists and is callable
        assert callable(assemble_context)
        assert assemble_context.__doc__ is not None

    @patch("server.mcp_server._get_assembler")
    def test_search_code_with_results(self, mock_get_assembler):
        """Test search_code returns formatted results."""
        from server.mcp_server import search_code

        # Mock the assembler
        mock_assembler = MagicMock()
        mock_assembler.semantic_search.return_value = [
            {
                "filepath": "src/module.py",
                "function": "my_function",
                "line_start": 42,
                "similarity": 0.95,
                "code": "def my_function():\n    pass",
            }
        ]
        mock_get_assembler.return_value = mock_assembler

        result = search_code("test query")

        assert "src/module.py" in result
        assert "my_function" in result
        assert "42" in result
        assert "0.95" in result

    @patch("server.mcp_server._get_assembler")
    def test_search_code_no_results(self, mock_get_assembler):
        """Test search_code with no results."""
        from server.mcp_server import search_code

        mock_assembler = MagicMock()
        mock_assembler.semantic_search.return_value = []
        mock_get_assembler.return_value = mock_assembler

        result = search_code("test query")

        assert "No confident matches found" in result

    @patch("server.mcp_server._get_assembler")
    def test_search_code_exception_handling(self, mock_get_assembler):
        """Test search_code gracefully handles exceptions."""
        from server.mcp_server import search_code

        mock_assembler = MagicMock()
        mock_assembler.semantic_search.side_effect = Exception("Test error")
        mock_get_assembler.return_value = mock_assembler

        result = search_code("test query")

        assert "Search error" in result
        assert "Test error" in result

    @patch("server.mcp_server._get_assembler")
    def test_assemble_context_with_result(self, mock_get_assembler):
        """Test assemble_context returns assembled context."""
        from server.mcp_server import assemble_context

        mock_assembler = MagicMock()
        mock_assembler.assemble_context.return_value = "# Assembled Context\n\nSome code..."
        mock_get_assembler.return_value = mock_assembler

        result = assemble_context("test query")

        assert "Assembled Context" in result
        assert "Some code" in result

    @patch("server.mcp_server._get_assembler")
    def test_assemble_context_exception_handling(self, mock_get_assembler):
        """Test assemble_context gracefully handles exceptions."""
        from server.mcp_server import assemble_context

        mock_assembler = MagicMock()
        mock_assembler.assemble_context.side_effect = Exception("Assembly failed")
        mock_get_assembler.return_value = mock_assembler

        result = assemble_context("test query")

        assert "Context assembly error" in result
        assert "Assembly failed" in result


class TestMCPCLICommand:
    def test_mcp_command_registered(self):
        """Test that the 'mcp' command is registered in the CLI."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "mcp" in result.output

    def test_mcp_command_help(self):
        """Test that 'mcp --help' works."""
        runner = CliRunner()
        result = runner.invoke(main, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "MCP server" in result.output or "mcp" in result.output.lower()
