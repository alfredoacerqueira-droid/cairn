"""Unit tests for core/metrics.py — Metrics."""

from core.metrics import Metrics


class TestMetrics:
    def test_init(self, tmp_path):
        data_dir = tmp_path / ".cairn"
        Metrics(data_dir=str(data_dir))

    def test_record_index(self, tmp_path):
        m = Metrics(data_dir=str(tmp_path))
        m.record_index(files=10, functions=50, time_ms=1500.0, mode="quick")
        summary = m.get_summary()
        assert summary["indexing"]["total_files"] == 10
        assert summary["indexing"]["total_functions"] == 50

    def test_record_search(self, tmp_path):
        m = Metrics(data_dir=str(tmp_path))
        m.record_search(query="test", results=3, time_ms=100.0)
        summary = m.get_summary()
        assert summary["search"]["total_queries"] == 1

    def test_record_request(self, tmp_path):
        m = Metrics(data_dir=str(tmp_path))
        m.record_request(latency_ms=250.0, error=False)
        m.record_request(latency_ms=300.0, error=True)
        summary = m.get_summary()
        assert summary["server"]["total_requests"] == 2
        assert summary["server"]["total_errors"] == 1

    def test_record_file_change(self, tmp_path):
        m = Metrics(data_dir=str(tmp_path))
        m.record_file_change("app/main.py")
        summary = m.get_summary()
        assert summary["janitor"]["file_changes"] == 1

    def test_record_reindex(self, tmp_path):
        m = Metrics(data_dir=str(tmp_path))
        m.record_reindex(source="watcher")
        summary = m.get_summary()
        assert summary["janitor"]["reindexes"] == 1

    def test_record_snapshot(self, tmp_path):
        m = Metrics(data_dir=str(tmp_path))
        m.record_snapshot(cpu=25.0, ram_mb=1024, functions=100)
        summary = m.get_summary()
        snapshots = summary["system"].get("snapshots", [])
        assert len(snapshots) > 0

    def test_get_latency_history(self, tmp_path):
        m = Metrics(data_dir=str(tmp_path))
        m.record_request(latency_ms=50.0)
        m.record_request(latency_ms=100.0)
        history = m.get_latency_history("server", limit=10)
        assert len(history) >= 2

    def test_get_index_time_history(self, tmp_path):
        m = Metrics(data_dir=str(tmp_path))
        m.record_index(files=1, functions=5, time_ms=200.0)
        history = m.get_index_time_history(limit=10)
        assert len(history) >= 1

    def test_persistence(self, tmp_path):
        m1 = Metrics(data_dir=str(tmp_path))
        m1.record_search(query="persist", results=1, time_ms=42.0)

        m2 = Metrics(data_dir=str(tmp_path))
        summary = m2.get_summary()
        assert summary["search"]["total_queries"] >= 1

    def test_empty_summary(self, tmp_path):
        m = Metrics(data_dir=str(tmp_path))
        summary = m.get_summary()
        for category in ["indexing", "search", "server", "janitor", "system"]:
            assert category in summary
