"""Unit tests for _select_worker_model (pure deterministic function)."""

from cli.main import _select_worker_model


class TestSelectWorkerModel:
    """Test VRAM-aware worker model selection logic."""

    def test_7b_chosen_when_sufficient_vram_and_installed(self):
        """When VRAM >= 6000 MiB and 7b is installed, prefer 7b."""
        installed = [
            "nomic-embed-text:latest",
            "qwen2.5-coder:1.5b",
            "qwen2.5-coder:7b",
        ]
        selected = _select_worker_model(installed, 8000, "qwen2.5-coder:1.5b")
        assert selected == "qwen2.5-coder:7b"

    def test_3b_chosen_when_vram_sufficient_and_7b_missing(self):
        """When VRAM >= 6000 MiB but 7b missing, choose largest (3b)."""
        installed = [
            "nomic-embed-text:latest",
            "qwen2.5-coder:1.5b",
            "qwen2.5-coder:3b",
        ]
        selected = _select_worker_model(installed, 6500, "qwen2.5-coder:1.5b")
        assert selected == "qwen2.5-coder:3b"

    def test_fallback_to_largest_when_vram_insufficient(self):
        """When VRAM < 6000 MiB, choose largest available (even if 7b exists)."""
        installed = [
            "nomic-embed-text:latest",
            "qwen2.5-coder:1.5b",
            "qwen2.5-coder:7b",
        ]
        selected = _select_worker_model(installed, 4000, "qwen2.5-coder:1.5b")
        assert selected == "qwen2.5-coder:7b"  # Largest available

    def test_fallback_to_largest_when_vram_unknown(self):
        """When VRAM is None (unknown), choose largest available."""
        installed = [
            "nomic-embed-text:latest",
            "qwen2.5-coder:1.5b",
            "qwen2.5-coder:3b",
        ]
        selected = _select_worker_model(installed, None, "qwen2.5-coder:1.5b")
        assert selected == "qwen2.5-coder:3b"

    def test_keep_current_when_no_coder_models(self):
        """When no qwen2.5-coder models installed, keep current."""
        installed = [
            "nomic-embed-text:latest",
            "llama:7b",
            "mistral:7b",
        ]
        current = "qwen2.5-coder:1.5b"
        selected = _select_worker_model(installed, 8000, current)
        assert selected == current

    def test_ignore_non_coder_models(self):
        """Non-coder models are ignored; select from coder models only."""
        installed = [
            "nomic-embed-text:latest",
            "llama:7b",
            "qwen2.5-coder:1.5b",
            "mistral:7b",
        ]
        selected = _select_worker_model(installed, 8000, "qwen2.5-coder:1.5b")
        # Should select from coder models only; no 7b coder, so 1.5b
        assert "qwen2.5-coder" in selected

    def test_keep_current_when_no_models_installed(self):
        """When installed list is empty, keep current."""
        current = "qwen2.5-coder:1.5b"
        selected = _select_worker_model([], 8000, current)
        assert selected == current

    def test_case_insensitive_model_matching(self):
        """Model matching should be case-insensitive."""
        installed = [
            "nomic-embed-text:latest",
            "Qwen2.5-Coder:1.5b",  # Different case
            "QWEN2.5-CODER:7b",
        ]
        selected = _select_worker_model(installed, 8000, "qwen2.5-coder:1.5b")
        # Should match and select 7b variant (case-insensitive)
        assert "7b" in selected.lower()

    def test_size_ordering_1_5b_vs_3b_vs_7b(self):
        """Verify correct ordering: 7b > 3b > 1.5b."""
        # When all sizes available and VRAM insufficient, choose largest
        installed = [
            "qwen2.5-coder:7b",
            "qwen2.5-coder:3b",
            "qwen2.5-coder:1.5b",
        ]
        selected = _select_worker_model(installed, 2000, "current")
        assert "7b" in selected

        # Repeat without 7b
        installed = [
            "qwen2.5-coder:3b",
            "qwen2.5-coder:1.5b",
        ]
        selected = _select_worker_model(installed, 2000, "current")
        assert "3b" in selected

    def test_vram_boundary_at_6000(self):
        """Test VRAM boundary: 6000 is the threshold for 7b preference."""
        installed = [
            "qwen2.5-coder:1.5b",
            "qwen2.5-coder:7b",
        ]

        # Just below threshold
        selected = _select_worker_model(installed, 5999, "current")
        assert "7b" in selected  # Still choose largest

        # At threshold
        selected = _select_worker_model(installed, 6000, "current")
        assert "7b" in selected  # Prefer 7b at threshold

        # Above threshold
        selected = _select_worker_model(installed, 6001, "current")
        assert "7b" in selected  # Prefer 7b above threshold

    def test_untagged_models_handled(self):
        """Models without version tags (just names) are handled."""
        installed = [
            "qwen2.5-coder",  # No version tag
            "nomic-embed-text",
        ]
        current = "qwen2.5-coder:1.5b"
        selected = _select_worker_model(installed, 8000, current)
        # Should recognize qwen2.5-coder (untagged) as a coder model
        assert "qwen2.5-coder" in selected
