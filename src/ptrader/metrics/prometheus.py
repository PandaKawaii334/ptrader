# ptrader/metrics/prometheus.py

from prometheus_client import Gauge, Counter, start_http_server as _start_http_server


start_http_server = _start_http_server

# Cache created metrics so repeated calls won't attempt to re-register the same names
_PROM_GAUGES = None
_PROM_COUNTERS = None


def make_prom(force: bool = False):
    """Create or return cached Prometheus metric objects.

    If `force=True`, metrics will be recreated even if previously cached (useful for tests
    that monkeypatch Gauge/Counter). If `Gauge`/`Counter` is not available, returns empty dicts.
    """
    global _PROM_GAUGES, _PROM_COUNTERS

    # If prometheus client not available in this runtime, return empty without caching
    if Gauge is None or Counter is None:
        return {}, {}

    if not force and _PROM_GAUGES is not None and _PROM_COUNTERS is not None:
        return _PROM_GAUGES, _PROM_COUNTERS

    # Create and cache gauges/counters once per process
    g = {
        "equity": Gauge("ptr_equity", "Equity"),
        "position": Gauge("ptr_position", "Position qty"),
        "price": Gauge("ptr_price", "Price"),
        "signal": Gauge("ptr_signal", "Signal prob"),
        "daily_pnl": Gauge("ptr_daily_pnl", "Daily PnL"),
        "realized_pnl": Gauge("ptr_realized_pnl", "Realized PnL"),
        "unrealized_pnl": Gauge("ptr_unrealized_pnl", "Unrealized PnL"),
        "prediction_latency": Gauge("ptr_pred_latency", "Prediction latency"),
    }
    c = {"kills": Counter("ptr_kills", "Kill-switches")}

    _PROM_GAUGES, _PROM_COUNTERS = g, c
    return g, c
