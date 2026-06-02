"""Token tracking system (RTK-style).

Tracks compression metrics in a JSON file for analytics.
Returns compression statistics and historical data.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional


class TokenTracker:
    """JSON-based token tracking for compression analytics.

    Tracks: original tokens, compressed tokens, savings, strategies used.
    Data retained for 90 days.
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path.home() / ".cairn" / "token_history.json"
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_store()

    def _init_store(self) -> None:
        """Initialize the data store."""
        if not self.db_path.exists():
            self.db_path.write_text(json.dumps({"history": []}))

    def _load(self) -> dict[str, Any]:
        """Load the data store."""
        try:
            return json.loads(self.db_path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return {"history": []}

    def _save(self, data: dict[str, Any]) -> None:
        """Save the data store."""
        self.db_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def track(
        self,
        query: str,
        original_tokens: int,
        compressed_tokens: int,
        strategies: list[str],
        exec_time_ms: int = 0,
    ) -> None:
        """Record token compression metrics."""
        saved_tokens = original_tokens - compressed_tokens
        savings_pct = (
            round((saved_tokens / original_tokens) * 100, 1) if original_tokens > 0 else 0.0
        )

        data = self._load()
        data["history"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "query": query[:100],  # Truncate long queries
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "saved_tokens": saved_tokens,
                "savings_pct": savings_pct,
                "strategies": strategies,
                "exec_time_ms": exec_time_ms,
            }
        )

        # Keep only last 90 days
        cutoff = (datetime.now() - timedelta(days=90)).isoformat()
        data["history"] = [h for h in data["history"] if h["timestamp"] > cutoff]
        # Limit to 10000 entries max
        if len(data["history"]) > 10000:
            data["history"] = data["history"][-10000:]

        self._save(data)

    def get_stats(self, days: int = 90) -> dict[str, Any]:
        """Get token savings statistics."""
        data = self._load()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        history = [h for h in data["history"] if h["timestamp"] > cutoff]

        if not history:
            return {
                "total_requests": 0,
                "total_saved_tokens": 0,
                "avg_savings_pct": 0.0,
                "total_time_ms": 0,
                "avg_time_ms": 0.0,
                "period_days": days,
            }

        total = len(history)
        total_saved = sum(h["saved_tokens"] for h in history)
        avg_savings = round(sum(h["savings_pct"] for h in history) / total, 1)
        total_time = sum(h["exec_time_ms"] for h in history)
        avg_time = round(total_time / total, 0) if total > 0 else 0.0

        return {
            "total_requests": total,
            "total_saved_tokens": total_saved,
            "avg_savings_pct": avg_savings,
            "total_time_ms": total_time,
            "avg_time_ms": avg_time,
            "period_days": days,
        }

    def get_recent_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent compression history."""
        data = self._load()
        history = data.get("history", [])
        return sorted(history, key=lambda h: h["timestamp"], reverse=True)[:limit]
