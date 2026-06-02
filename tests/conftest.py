"""Pytest configuration with integration marker."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires real services)"
    )
