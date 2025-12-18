# tests/test_prometheus.py

from ptrader.metrics import prometheus as prom
from unittest.mock import MagicMock


def test_make_prom_with_prometheus(monkeypatch):
    # Create mock instances that will be returned by Gauge/Counter calls
    mock_gauge_instances = {}
    mock_counter_instance = MagicMock()

    # Create a factory function for Gauge that returns different mocks for each metric
    def mock_gauge_factory(name, description, registry=None):
        mock = MagicMock()
        mock_gauge_instances[name] = mock
        return mock

    # Patch Gauge and Counter
    monkeypatch.setattr(prom, "Gauge", mock_gauge_factory)
    monkeypatch.setattr(prom, "Counter", lambda *a, **k: mock_counter_instance)

    # Force recreation so monkeypatched Gauge/Counter are used in this test
    gauges, counters = prom.make_prom(force=True)

    # Check gauges dict keys
    assert set(gauges.keys()) == {
        "equity",
        "position",
        "price",
        "signal",
        "daily_pnl",
        "realized_pnl",
        "unrealized_pnl",
        "prediction_latency",
    }

    # Check counters dict keys
    assert set(counters.keys()) == {"kills"}

    # Verify that the mocks were actually called and stored
    assert all(isinstance(v, MagicMock) for v in gauges.values())
    assert isinstance(counters["kills"], MagicMock)


def test_make_prom_without_prometheus(monkeypatch):
    # Simulate prometheus_client not being available
    monkeypatch.setattr(prom, "Gauge", None)
    monkeypatch.setattr(prom, "Counter", None)

    gauges, counters = prom.make_prom()

    assert gauges == {}
    assert counters == {}
