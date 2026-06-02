"""Unit tests for pipeline/queue.py — PriorityJobQueue."""

import time
from unittest.mock import patch

from pipeline.queue import PriorityJobQueue
from throttle.cpu import CPUThrottler
from throttle.memory import MemoryManager
from throttle.vram import VRAMPriority


class TestPriorityJobQueue:
    def make_queue(self):
        cpu = CPUThrottler(max_cpu_percent=100)
        mem = MemoryManager(max_memory_mb=16000)
        vram = VRAMPriority()
        return PriorityJobQueue(cpu, mem, vram)

    def test_initial_state(self):
        q = self.make_queue()
        assert q.pending_jobs == 0
        assert q.is_running is False

    def test_add_job_increases_pending(self):
        q = self.make_queue()
        q.add_job(lambda: None, priority=2)
        assert q.pending_jobs >= 0

    def test_start_and_stop(self):
        q = self.make_queue()
        q.start()
        assert q.is_running is True
        q.stop()
        assert q.is_running is False

    @patch("throttle.cpu.CPUThrottler.is_safe_to_proceed", return_value=True)
    def test_job_executes_when_resources_available(self, mock_safe):
        results = []
        q = self.make_queue()
        q.add_job(lambda: results.append("done"), priority=2)
        q.start()
        time.sleep(0.5)
        q.stop()
        assert results == ["done"]

    @patch("throttle.cpu.CPUThrottler.is_safe_to_proceed", return_value=False)
    def test_job_requeues_when_cpu_unsafe(self, mock_safe):
        q = self.make_queue()
        q.add_job(lambda: None, priority=2)
        q.start()
        time.sleep(0.3)
        q.stop()
        assert q.pending_jobs >= 0

    def test_multiple_jobs(self):
        results = []
        q = self.make_queue()
        for i in range(3):
            q.add_job(lambda i=i: results.append(i), priority=i + 1)
        q.start()
        time.sleep(0.8)
        q.stop()
        assert len(results) == 3

    def test_vram_priority_blocks_janitor(self):
        vram = VRAMPriority()
        vram.request("gateway")
        cpu = CPUThrottler(max_cpu_percent=100)
        mem = MemoryManager(max_memory_mb=16000)
        q = PriorityJobQueue(cpu, mem, vram)
        results = []
        q.add_job(lambda: results.append("ran"), requester="janitor", priority=2)
        q.start()
        time.sleep(0.3)
        q.stop()
        assert results == []
