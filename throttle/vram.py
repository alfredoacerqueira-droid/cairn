"""VRAM priority system. Ensures live gateway has priority over background janitor."""

import threading
import time
from typing import Literal


class VRAMPriority:
    def __init__(self):
        self._lock = threading.Lock()
        self._gateway_active = False
        self._janitor_count = 0

    def request(self, requester: Literal["gateway", "janitor"]) -> bool:
        """Request VRAM access. Returns True if granted. Priority: gateway > janitor."""
        if requester not in ("gateway", "janitor"):
            raise ValueError(f"Invalid requester: {requester}")

        with self._lock:
            if requester == "gateway":
                self._gateway_active = True
                return True
            elif requester == "janitor":
                if self._gateway_active:
                    return False
                self._janitor_count += 1
                return True
        return False

    def release(self, requester: Literal["gateway", "janitor"]):
        """Release VRAM for the given requester."""
        with self._lock:
            if requester == "gateway":
                self._gateway_active = False
            elif requester == "janitor":
                self._janitor_count = max(0, self._janitor_count - 1)

    def wait_for_vram(self, requester: Literal["janitor"], timeout: float = 10.0) -> bool:
        """Wait until VRAM is available for janitor."""
        start = time.time()
        while time.time() - start < timeout:
            if self.request(requester):
                return True
            time.sleep(0.5)
        return False

    @property
    def gateway_active(self) -> bool:
        with self._lock:
            return self._gateway_active

    @property
    def janitor_active(self) -> bool:
        with self._lock:
            return self._janitor_count > 0
