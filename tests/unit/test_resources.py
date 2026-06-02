"""Unit tests for resource managers."""

from throttle.cpu import CPUThrottler
from throttle.memory import MemoryManager
from throttle.vram import VRAMPriority


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
