"""CPU throttling to prevent resource exhaustion."""

import time

import psutil


class CPUThrottler:
    def __init__(self, max_cpu_percent: int = 50):
        self.max_cpu_percent = max_cpu_percent
        self.process = psutil.Process()

    def throttle_if_needed(self):
        """Sleep if CPU usage exceeds threshold."""
        cpu_percent = self.process.cpu_percent(interval=0.1)
        if cpu_percent > self.max_cpu_percent:
            sleep_time = (cpu_percent - self.max_cpu_percent) / 100
            time.sleep(sleep_time)

    def is_safe_to_proceed(self) -> bool:
        """Check if CPU is below threshold."""
        cpu_percent = self.process.cpu_percent(interval=0.1)
        return cpu_percent < self.max_cpu_percent

    def wait_until_safe(self, timeout: float = 10.0):
        """Block until CPU usage is safe or timeout."""
        start = time.time()
        while time.time() - start < timeout:
            if self.is_safe_to_proceed():
                return True
            time.sleep(0.5)
        return False
