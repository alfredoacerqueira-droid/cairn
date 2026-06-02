"""Metrics tracking for observability."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class Metrics:
    """Lightweight metrics tracker with rolling history."""

    MAX_HISTORY = 500

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = Path.cwd() / ".cairn"
        self.data_dir = Path(data_dir)
        self.metrics_file = self.data_dir / "metrics.json"
        self._data: Optional[dict] = None

    def _load(self) -> dict:
        if self._data is not None:
            return self._data

        if self.metrics_file.exists():
            try:
                self._data = json.loads(self.metrics_file.read_text())
            except (json.JSONDecodeError, ValueError):
                self._data = self._empty()
        else:
            self._data = self._empty()

        return self._data

    def _empty(self) -> dict:
        return {
            "indexing": {
                "total_files": 0,
                "total_functions": 0,
                "total_index_time_ms": 0,
                "events": [],
            },
            "search": {
                "total_queries": 0,
                "total_latency_ms": 0,
                "avg_results": 0,
                "events": [],
            },
            "server": {
                "total_requests": 0,
                "total_errors": 0,
                "total_latency_ms": 0,
                "events": [],
            },
            "janitor": {
                "file_changes_detected": 0,
                "reindexes_triggered": 0,
                "events": [],
            },
            "system": {
                "snapshots": [],
            },
            "last_updated": None,
        }

    def _save(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        data = self._load()
        data["last_updated"] = datetime.now().isoformat()

        for category in ["indexing", "search", "server", "janitor"]:
            events = data[category].get("events", [])
            if len(events) > self.MAX_HISTORY:
                data[category]["events"] = events[-self.MAX_HISTORY :]

            snapshots = data.get("system", {}).get("snapshots", [])
            if len(snapshots) > self.MAX_HISTORY:
                data["system"]["snapshots"] = snapshots[-self.MAX_HISTORY :]

        self.metrics_file.write_text(json.dumps(data, indent=2))

    def record_index(
        self,
        files: int,
        functions: int,
        time_ms: float,
        mode: str = "quick",
    ):
        """Record an indexing operation."""
        data = self._load()
        data["indexing"]["total_files"] += files
        data["indexing"]["total_functions"] += functions
        data["indexing"]["total_index_time_ms"] += time_ms
        data["indexing"]["events"].append(
            {
                "ts": datetime.now().isoformat(),
                "files": files,
                "functions": functions,
                "time_ms": round(time_ms, 2),
                "mode": mode,
            }
        )
        self._save()

    def record_search(
        self,
        query: str,
        results: int,
        time_ms: float,
    ):
        """Record a search operation."""
        data = self._load()
        data["search"]["total_queries"] += 1
        data["search"]["total_latency_ms"] += time_ms
        data["search"]["events"].append(
            {
                "ts": datetime.now().isoformat(),
                "query": query[:50],
                "results": results,
                "time_ms": round(time_ms, 2),
            }
        )
        total = data["search"]["total_queries"]
        if total > 0:
            data["search"]["avg_results"] = round(
                sum(e["results"] for e in data["search"]["events"][-50:])
                / min(len(data["search"]["events"]), 50),
                1,
            )
        self._save()

    def record_request(
        self,
        latency_ms: float,
        error: bool = False,
    ):
        """Record a server request."""
        data = self._load()
        data["server"]["total_requests"] += 1
        data["server"]["total_latency_ms"] += latency_ms
        if error:
            data["server"]["total_errors"] += 1
        data["server"]["events"].append(
            {
                "ts": datetime.now().isoformat(),
                "latency_ms": round(latency_ms, 2),
                "error": error,
            }
        )
        self._save()

    def record_file_change(self, filepath: str):
        """Record a file change detected by the janitor."""
        data = self._load()
        data["janitor"]["file_changes_detected"] += 1
        data["janitor"]["events"].append(
            {
                "ts": datetime.now().isoformat(),
                "filepath": filepath,
            }
        )
        self._save()

    def record_reindex(self, source: str = "watcher"):
        """Record a re-index trigger."""
        data = self._load()
        data["janitor"]["reindexes_triggered"] += 1
        data["janitor"]["events"].append(
            {
                "ts": datetime.now().isoformat(),
                "type": "reindex",
                "source": source,
            }
        )
        self._save()

    def record_snapshot(self, cpu: float, ram_mb: float, functions: int):
        """Record a system health snapshot."""
        data = self._load()
        data["system"]["snapshots"].append(
            {
                "ts": datetime.now().isoformat(),
                "cpu_percent": round(cpu, 1),
                "ram_mb": round(ram_mb, 1),
                "indexed_functions": functions,
            }
        )
        self._save()

    def get_summary(self) -> dict:
        """Get a summary of all metrics."""
        data = self._load()

        idx = data["indexing"]
        srch = data["search"]
        srv = data["server"]
        jan = data["janitor"]

        idx_events = idx.get("events", [])
        srch_events = srch.get("events", [])
        srv_events = srv.get("events", [])
        snapshots = data.get("system", {}).get("snapshots", [])

        return {
            "indexing": {
                "total_files": idx["total_files"],
                "total_functions": idx["total_functions"],
                "avg_time_ms": (
                    round(idx["total_index_time_ms"] / len(idx_events), 1) if idx_events else 0
                ),
                "last_event": idx_events[-1] if idx_events else None,
            },
            "search": {
                "total_queries": srch["total_queries"],
                "avg_latency_ms": (
                    round(srch["total_latency_ms"] / srch["total_queries"], 1)
                    if srch["total_queries"] > 0
                    else 0
                ),
                "avg_results": srch["avg_results"],
                "recent_events": srch_events[-5:],
            },
            "server": {
                "total_requests": srv["total_requests"],
                "total_errors": srv["total_errors"],
                "avg_latency_ms": (
                    round(srv["total_latency_ms"] / srv["total_requests"], 1)
                    if srv["total_requests"] > 0
                    else 0
                ),
                "error_rate": (
                    round(srv["total_errors"] / srv["total_requests"] * 100, 1)
                    if srv["total_requests"] > 0
                    else 0
                ),
                "recent_events": srv_events[-5:],
            },
            "janitor": {
                "file_changes": jan["file_changes_detected"],
                "reindexes": jan["reindexes_triggered"],
                "recent_events": jan.get("events", [])[-5:],
            },
            "system": {
                "snapshots": snapshots[-10:],
            },
            "last_updated": data.get("last_updated"),
        }

    def get_latency_history(self, category: str = "search", limit: int = 20) -> list[float]:
        """Get recent latency values for graphing."""
        data = self._load()
        events = data.get(category, {}).get("events", [])
        return [e["latency_ms"] for e in events[-limit:] if "latency_ms" in e]

    def get_index_time_history(self, limit: int = 20) -> list[float]:
        """Get recent indexing time values for graphing."""
        data = self._load()
        events = data.get("indexing", {}).get("events", [])
        return [e["time_ms"] for e in events[-limit:]]
