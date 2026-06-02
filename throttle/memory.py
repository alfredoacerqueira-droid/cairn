"""Memory (RAM) usage manager."""

import time

import psutil


class MemoryManager:
    def __init__(self, max_memory_mb: int = 4096):
        self.max_memory_mb = max_memory_mb

    def current_usage_mb(self) -> float:
        """Get current system memory usage in MB."""
        return psutil.virtual_memory().used / 1024 / 1024

    def check(self) -> bool:
        """Returns True if system memory usage is under the limit."""
        return self.current_usage_mb() < self.max_memory_mb

    def wait_for_memory(self, timeout: float = 30.0) -> bool:
        """Wait until memory is available or timeout."""
        start = time.time()
        while time.time() - start < timeout:
            if self.check():
                return True
            time.sleep(0.5)
        return False

    def throttle(self):
        """Block if memory exceeds threshold."""
        if not self.check():
            self.wait_for_memory()

    def system_memory_info(self) -> dict:
        """Get system-wide memory info."""
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "used_gb": mem.used / (1024**3),
            "percent": mem.percent,
        }
