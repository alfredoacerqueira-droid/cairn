"""Unit tests for resource managers."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest  # noqa: F401

from throttle.cpu import CPUThrottler
from throttle.memory import MemoryManager
from throttle.vram import VRAMPriority


class TestGetSystemResources:
    def test_detects_ram_cpu_from_psutil(self):
        """get_system_resources uses psutil for RAM/CPU."""
        from core.resources import get_system_resources

        mock_mem = MagicMock()
        mock_mem.total = 16 * 1024**3
        mock_mem.available = 8 * 1024**3

        with patch("psutil.virtual_memory", return_value=mock_mem), \
             patch("psutil.cpu_count", return_value=8), \
             patch("subprocess.run", side_effect=FileNotFoundError):
            result = get_system_resources()

        assert result["ram_total_gb"] == 16.0
        assert result["ram_available_gb"] == 8.0
        assert result["cpu_count"] == 8
        assert result["vram_total_gb"] is None
        assert result["vram_free_gb"] is None
        assert result["gpu_name"] is None

    def test_detects_vram_from_nvidia_smi(self):
        """Parses nvidia-smi output for VRAM."""
        from core.resources import get_system_resources

        mock_mem = MagicMock()
        mock_mem.total = 32 * 1024**3
        mock_mem.available = 20 * 1024**3

        smi_result = MagicMock()
        smi_result.returncode = 0
        smi_result.stdout = "NVIDIA GeForce RTX 4090, 24564, 20480\n"
        smi_result.stderr = ""

        with patch("psutil.virtual_memory", return_value=mock_mem), \
             patch("psutil.cpu_count", return_value=16), \
             patch("subprocess.run", return_value=smi_result):
            result = get_system_resources()

        assert result["ram_total_gb"] == 32.0
        assert result["ram_available_gb"] == 20.0
        assert result["cpu_count"] == 16
        assert result["vram_total_gb"] == 24.0
        assert result["vram_free_gb"] == 20.0
        assert result["gpu_name"] == "NVIDIA GeForce RTX 4090"

    def test_nvidia_smi_segfault_no_exception(self):
        """Subprocess raises OSError (segfault) -> vram_* is None, no exception."""
        from core.resources import get_system_resources

        mock_mem = MagicMock()
        mock_mem.total = 16 * 1024**3
        mock_mem.available = 8 * 1024**3

        with patch("psutil.virtual_memory", return_value=mock_mem), \
             patch("psutil.cpu_count", return_value=4), \
             patch("subprocess.run", side_effect=OSError("segfault")):
            result = get_system_resources()

        assert result["ram_total_gb"] == 16.0
        assert result["vram_total_gb"] is None
        assert result["vram_free_gb"] is None
        assert result["gpu_name"] is None

    def test_nvidia_smi_timeout_graceful(self):
        """Subprocess timeout -> vram_* is None, no exception."""
        from core.resources import get_system_resources

        mock_mem = MagicMock()
        mock_mem.total = 16 * 1024**3
        mock_mem.available = 8 * 1024**3

        with patch("psutil.virtual_memory", return_value=mock_mem), \
             patch("psutil.cpu_count", return_value=4), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)):
            result = get_system_resources()

        assert result["ram_total_gb"] == 16.0
        assert result["vram_total_gb"] is None
        assert result["vram_free_gb"] is None

    def test_nvidia_smi_nonzero_exit(self):
        """nvidia-smi returns non-zero -> vram_* is None."""
        from core.resources import get_system_resources

        mock_mem = MagicMock()
        mock_mem.total = 16 * 1024**3
        mock_mem.available = 8 * 1024**3

        smi_result = MagicMock()
        smi_result.returncode = 1
        smi_result.stdout = ""
        smi_result.stderr = "No devices found"

        with patch("psutil.virtual_memory", return_value=mock_mem), \
             patch("psutil.cpu_count", return_value=4), \
             patch("subprocess.run", return_value=smi_result):
            result = get_system_resources()

        assert result["vram_total_gb"] is None

    def test_psutil_import_error_graceful(self):
        """psutil not available -> returns zeros."""
        from core.resources import get_system_resources

        with patch.dict("sys.modules", {"psutil": None}):
            result = get_system_resources()

        assert result["ram_total_gb"] == 0.0
        assert result["ram_available_gb"] == 0.0
        assert result["cpu_count"] == 0

    def test_rounds_to_one_decimal(self):
        """GiB values are rounded to 1 decimal."""
        from core.resources import get_system_resources

        mock_mem = MagicMock()
        mock_mem.total = int(15.678 * 1024**3)
        mock_mem.available = int(7.234 * 1024**3)

        with patch("psutil.virtual_memory", return_value=mock_mem), \
             patch("psutil.cpu_count", return_value=4), \
             patch("subprocess.run", side_effect=FileNotFoundError):
            result = get_system_resources()

        assert result["ram_total_gb"] == 15.7
        assert result["ram_available_gb"] == 7.2


class TestRecommendLocalModels:
    def test_picks_largest_worker_that_fits_budget(self):
        """With ram=12, vram=5 -> budget=15, picks the largest fitting worker."""
        from core.resources import recommend_local_models

        resources = {
            "ram_total_gb": 16.0,
            "ram_available_gb": 12.0,
            "cpu_count": 8,
            "vram_total_gb": 6.0,
            "vram_free_gb": 5.0,
            "gpu_name": "RTX 2060",
        }
        installed = [
            {"name": "gemma4:latest", "size_gb": 9.6},
            {"name": "qwen2.5-coder:3b", "size_gb": 1.9},
            {"name": "nomic-embed-text:latest", "size_gb": 0.3},
            {"name": "qwen3-embedding:4b", "size_gb": 2.5},
        ]

        rec = recommend_local_models(resources, installed)

        assert rec["worker"]["model"] == "gemma4:latest"
        assert rec["embed"]["model"] == "nomic-embed-text:latest"
        assert rec["budget_gb"] == 15.0
        assert "largest worker fitting budget" in rec["worker"]["reason"]
        assert "smallest embedder fitting budget" in rec["embed"]["reason"]

    def test_picks_small_worker_when_none_fit_budget(self):
        """ram=4, no vram -> budget=2, 9.6GB worker doesn't fit (9.6*1.15=11.04 > 2)."""
        from core.resources import recommend_local_models

        resources = {
            "ram_total_gb": 8.0,
            "ram_available_gb": 4.0,
            "cpu_count": 4,
            "vram_total_gb": None,
            "vram_free_gb": None,
            "gpu_name": None,
        }
        installed = [
            {"name": "gemma4:latest", "size_gb": 9.6},
            {"name": "qwen2.5-coder:3b", "size_gb": 1.9},
            {"name": "nomic-embed-text:latest", "size_gb": 0.3},
        ]

        rec = recommend_local_models(resources, installed)

        assert rec["worker"]["model"] == "qwen2.5-coder:3b"
        assert "no worker fits" in rec["worker"]["reason"]
        assert rec["budget_gb"] == 2.0

    def test_falls_back_to_suggested_models_when_none_installed(self):
        """No installed models -> suggests defaults."""
        from core.resources import recommend_local_models

        resources = {
            "ram_total_gb": 16.0,
            "ram_available_gb": 12.0,
            "cpu_count": 8,
            "vram_total_gb": None,
            "vram_free_gb": None,
            "gpu_name": None,
        }
        rec = recommend_local_models(resources, [])

        assert rec["worker"]["model"] == "qwen2.5-coder:3b"
        assert rec["embed"]["model"] == "nomic-embed-text"
        assert "no worker models installed" in rec["worker"]["reason"]
        assert "no embed models installed" in rec["embed"]["reason"]

    def test_num_ctx_tiers(self):
        """suggested_num_ctx follows tiered heuristic."""
        from core.resources import recommend_local_models

        resources = {"ram_available_gb": 30.0, "vram_free_gb": 10.0}
        installed = [{"name": "gemma4:latest", "size_gb": 9.6}]
        rec = recommend_local_models(resources, installed)
        assert rec["suggested_num_ctx"] == 65536

        resources = {"ram_available_gb": 14.0, "vram_free_gb": 2.0}
        rec = recommend_local_models(resources, installed)
        assert rec["suggested_num_ctx"] == 32768

        resources = {"ram_available_gb": 10.0, "vram_free_gb": 2.0}
        rec = recommend_local_models(resources, installed)
        assert rec["suggested_num_ctx"] == 16384

        resources = {"ram_available_gb": 4.0, "vram_free_gb": None}
        rec = recommend_local_models(resources, installed)
        assert rec["suggested_num_ctx"] == 8192

    def test_no_negative_budget(self):
        """ram=1, no vram -> budget floors at 0."""
        from core.resources import recommend_local_models

        resources = {"ram_available_gb": 1.0, "vram_free_gb": None}
        rec = recommend_local_models(resources, [])
        assert rec["budget_gb"] == 0.0

    def test_vram_none_uses_ram_only_minus_headroom(self):
        """When vram is None, budget = ram_available - 2."""
        from core.resources import recommend_local_models

        resources = {
            "ram_available_gb": 12.0,
            "vram_free_gb": None,
            "vram_total_gb": None,
        }
        rec = recommend_local_models(resources, [])
        assert rec["budget_gb"] == 10.0


class TestListInstalledOllamaModels:
    def test_parses_ollama_list_output(self):
        """Parses standard `ollama list` output to name+size dicts."""
        from core.resources import _parse_ollama_list

        output = (
            "NAME                     ID              SIZE      MODIFIED      \n"
            "gemma4:latest            abc123          9.6 GB    2 days ago     \n"
            "qwen2.5-coder:3b         def456          1.9 GB    5 days ago     \n"
            "nomic-embed-text:latest  ghi789          274 MB    1 week ago     \n"
        )
        models = _parse_ollama_list(output)

        assert len(models) == 3
        assert models[0] == {"name": "gemma4:latest", "size_gb": 9.6}
        assert models[1] == {"name": "qwen2.5-coder:3b", "size_gb": 1.9}
        assert models[2] == {"name": "nomic-embed-text:latest", "size_gb": 0.3}

    def test_returns_empty_on_subprocess_failure(self):
        """list_installed_ollama_models returns [] when subprocess fails."""
        from core.resources import list_installed_ollama_models

        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = list_installed_ollama_models()
        assert result == []

    def test_returns_empty_on_nonzero_exit(self):
        """Returns [] when ollama list exits non-zero."""
        from core.resources import list_installed_ollama_models

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "ollama not running"

        with patch("subprocess.run", return_value=mock_result):
            result = list_installed_ollama_models()
        assert result == []

    def test_returns_empty_on_empty_stdout(self):
        """Returns [] when ollama list produces no output."""
        from core.resources import list_installed_ollama_models

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = list_installed_ollama_models()
        assert result == []

    def test_parses_mb_sizes(self):
        """MB sizes are converted to GB."""
        from core.resources import _parse_ollama_list

        output = (
            "NAME              ID       SIZE      MODIFIED\n"
            "tiny-model:latest abc123   500 MB    1 day ago\n"
        )
        models = _parse_ollama_list(output)
        assert models[0]["size_gb"] == 0.5


class TestSuggestLocalAndDoctorCli:
    def test_suggest_local_shows_resources(self):
        """suggest-local command shows system resources."""
        from unittest.mock import patch

        from click.testing import CliRunner

        from cli.main import main

        runner = CliRunner()
        with runner.isolated_filesystem():
            # Simulate a git repo and config
            subprocess.run(["git", "init"], check=True, capture_output=True)
            Path("main.py").write_text("def main(): pass")
            # Create minimal cairn config
            import os
            os.makedirs(".cairn", exist_ok=True)
            (Path(".cairn") / "config.yaml").write_text(
                "profile: code\n"
            )

            mock_resources = {
                "ram_total_gb": 16.0,
                "ram_available_gb": 12.0,
                "cpu_count": 8,
                "vram_total_gb": 6.0,
                "vram_free_gb": 5.0,
                "gpu_name": "NVIDIA GeForce RTX 4090",
            }
            mock_models = [
                {"name": "gemma4:latest", "size_gb": 9.6},
                {"name": "nomic-embed-text:latest", "size_gb": 0.3},
            ]

            with patch("core.resources.get_system_resources", return_value=mock_resources), \
                 patch("core.resources.list_installed_ollama_models", return_value=mock_models):
                result = runner.invoke(main, ["suggest-local", "add docstring"])
                assert result.exit_code == 0
                assert "System Resources" in result.output
                assert "16.0 GB total" in result.output
                assert "NVIDIA GeForce RTX 4090" in result.output
                assert "Worker:" in result.output or "worker" in result.output.lower()
                assert "gemma4:latest" in result.output

    def test_doctor_shows_resources_and_load_test_skipped_when_disabled(self):
        """doctor shows resources info even when local LLM disabled."""
        from unittest.mock import MagicMock, patch

        from click.testing import CliRunner

        from cli.main import main

        runner = CliRunner()
        with runner.isolated_filesystem():
            subprocess.run(["git", "init"], check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                check=True,
                capture_output=True,
            )
            Path("main.py").write_text("def main(): pass")
            import os
            os.makedirs(".cairn", exist_ok=True)
            (Path(".cairn") / "config.yaml").write_text(
                "profile: code\n"
                "local_llm:\n"
                "  enabled: false\n"
            )

            mock_resources = {
                "ram_total_gb": 16.0,
                "ram_available_gb": 12.0,
                "cpu_count": 8,
                "vram_total_gb": None,
                "vram_free_gb": None,
                "gpu_name": None,
            }
            mock_models = []

            with patch("server.ollama_client.OllamaClient") as mock_ollama_class:
                mock_ollama = MagicMock()
                mock_ollama.health_check.return_value = False
                mock_ollama_class.return_value = mock_ollama

                with patch("core.resources.get_system_resources", return_value=mock_resources), \
                     patch("core.resources.list_installed_ollama_models", return_value=mock_models):
                    result = runner.invoke(main, ["doctor"])
                    assert "System Resources" in result.output
                    assert "16.0 GB total" in result.output
                    assert "no GPU detected" in result.output.lower() or "no GPU" in result.output
                    assert "local LLM disabled" in result.output

    def test_suggest_local_handles_no_gpu(self):
        """suggest-local prints 'no GPU detected' when vram is None."""
        from unittest.mock import patch

        from click.testing import CliRunner

        from cli.main import main

        runner = CliRunner()
        with runner.isolated_filesystem():
            subprocess.run(["git", "init"], check=True, capture_output=True)
            Path("main.py").write_text("def main(): pass")
            import os
            os.makedirs(".cairn", exist_ok=True)
            (Path(".cairn") / "config.yaml").write_text("profile: code\n")

            mock_resources = {
                "ram_total_gb": 8.0,
                "ram_available_gb": 4.0,
                "cpu_count": 4,
                "vram_total_gb": None,
                "vram_free_gb": None,
                "gpu_name": None,
            }

            with patch("core.resources.get_system_resources", return_value=mock_resources), \
                 patch("core.resources.list_installed_ollama_models", return_value=[]):
                result = runner.invoke(main, ["suggest-local", "fix typo"])
                assert result.exit_code == 0
                assert "no GPU" in result.output


class TestCPUThrottler:
    def test_default_max_cpu(self):
        throttler = CPUThrottler()
        assert throttler.max_cpu_percent == 50

    def test_custom_max_cpu(self):
        throttler = CPUThrottler(max_cpu_percent=30)
        assert throttler.max_cpu_percent == 30

    def test_is_safe_to_proceed(self):
        throttler = CPUThrottler(max_cpu_percent=100)
        # Should always be safe with 100% limit
        assert throttler.is_safe_to_proceed() is True


class TestMemoryManager:
    def test_default_max_memory(self):
        manager = MemoryManager()
        assert manager.max_memory_mb == 4096

    def test_current_usage_returns_float(self):
        manager = MemoryManager()
        usage = manager.current_usage_mb()
        assert isinstance(usage, float)
        assert usage > 0

    def test_system_memory_info(self):
        manager = MemoryManager()
        info = manager.system_memory_info()
        assert "total_gb" in info
        assert "available_gb" in info
        assert "percent" in info


class TestVRAMPriority:
    def test_gateway_always_granted(self):
        vram = VRAMPriority()
        assert vram.request("gateway") is True
        vram.release("gateway")

    def test_janitor_denied_when_gateway_active(self):
        vram = VRAMPriority()
        vram.request("gateway")
        assert vram.request("janitor") is False
        vram.release("gateway")

    def test_janitor_granted_without_gateway(self):
        vram = VRAMPriority()
        assert vram.request("janitor") is True
        vram.release("janitor")

    def test_release_clears_state(self):
        vram = VRAMPriority()
        vram.request("gateway")
        vram.release("gateway")
        assert vram.gateway_active is False

    def test_gateway_active_property(self):
        vram = VRAMPriority()
        vram.request("gateway")
        assert vram.gateway_active is True
        vram.release("gateway")
        assert vram.gateway_active is False
