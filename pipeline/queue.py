"""Priority-based job queue for background indexing tasks."""

import queue
import threading
import time
from typing import Callable

from throttle.cpu import CPUThrottler
from throttle.memory import MemoryManager
from throttle.vram import VRAMPriority


class PriorityJobQueue:
    def __init__(
        self,
        cpu_throttler: CPUThrottler,
        memory_manager: MemoryManager,
        vram_priority: VRAMPriority,
    ):
        self.queue: queue.PriorityQueue = queue.PriorityQueue()
        self.cpu = cpu_throttler
        self.memory = memory_manager
        self.vram = vram_priority

        self._running = False
        self._worker_thread: threading.Thread | None = None

    def add_job(
        self,
        func: Callable,
        args: tuple = (),
        requester: str = "janitor",
        priority: int = 2,
    ):
        """
        Add a job to the queue.

        Priority: 0 = highest (gateway), 1 = medium, 2 = lowest (janitor)
        """
        self.queue.put((priority, {"func": func, "args": args, "requester": requester}))

    def start(self):
        """Start the worker thread."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def stop(self):
        """Stop the worker thread."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

    def _worker(self):
        """Main worker loop."""
        while self._running:
            try:
                priority, job = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                # Check resources
                if not self.cpu.is_safe_to_proceed():
                    time.sleep(0.5)
                    self.queue.put((priority + 1, job))
                    continue

                if not self.memory.check():
                    time.sleep(1.0)
                    self.queue.put((priority + 1, job))
                    continue

                # Check VRAM availability
                if job["requester"] == "janitor":
                    if not self.vram.request("janitor"):
                        time.sleep(0.5)
                        self.queue.put((priority, job))
                        continue

                # Execute job
                try:
                    job["func"](*job["args"])
                finally:
                    if job["requester"] == "janitor":
                        self.vram.release("janitor")

            except Exception:
                pass  # Log errors but don't crash the worker
            finally:
                self.queue.task_done()

    @property
    def pending_jobs(self) -> int:
        return self.queue.qsize()

    @property
    def is_running(self) -> bool:
        return self._running
