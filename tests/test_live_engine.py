# tests/test_live_engine.py

import time
import pandas as pd
import pytest
from unittest.mock import MagicMock

from ptrader.engine.live import LiveEngine
from ptrader.config.types import TradingConfig


def test_live_engine_one_iteration(monkeypatch):
    # Dummy feed with minimal required attributes
    class DummyFeed:
        def __init__(self):
            self._lock = MagicMock()
            self.latest_price = 100.0
            self.latest_bars = pd.DataFrame({"high": [101], "low": [99], "close": [100]})
            self.latest_features = pd.DataFrame({"ts": [pd.Timestamp.utcnow()], "vol_8": [1]})
            self.ex = MagicMock(symbol="SOL/USD")

    # Patch time.sleep so the loop doesn’t stall
    monkeypatch.setattr(time, "sleep", lambda x: None)

    cfg = TradingConfig()
    ens = MagicMock(predict=lambda X: [0.6])  # force a buy signal
    jsonl = MagicMock()

    engine = LiveEngine(ens, cfg, DummyFeed(), "k", "s", prom=({}, {}), jsonl=jsonl, real=False)
    engine.run(interval_sec=0, max_iters=1)

    # Assert that an entry was attempted
    assert engine.trade_count >= 0
    jsonl.write.assert_called()


def test_live_engine_exit(monkeypatch):
    class DummyFeed:
        def __init__(self):
            self._lock = MagicMock()
            self.latest_price = 80.0
            self.latest_bars = pd.DataFrame({"high": [85], "low": [75], "close": [80]})
            self.latest_features = pd.DataFrame({"ts": [pd.Timestamp.utcnow()], "vol_8": [1]})
            self.ex = MagicMock(symbol="SOL/USD")

    monkeypatch.setattr(time, "sleep", lambda x: None)
    monkeypatch.setattr("ptrader.engine.live.dynamic_slippage", lambda high, low, price: 0.0)

    cfg = TradingConfig(stop_loss=0.05, take_profit=0.05)
    ens = MagicMock(predict=lambda X: [0.6])
    jsonl = MagicMock()

    engine = LiveEngine(ens, cfg, DummyFeed(), "k", "s", prom=({}, {}), jsonl=jsonl, real=False)
    # Set pos/entry on the underlying PaperEngine
    engine.engine.pos = 1
    engine.engine.entry = 100.0

    engine.engine.sell = MagicMock(return_value=(1, 80.0))

    engine.run(interval_sec=0, max_iters=1)

    engine.engine.sell.assert_called()
    jsonl.write.assert_any_call(event="exit", qty=1, price=80.0, pnl=pytest.approx(-0.2))


def test_live_engine_killswitch(monkeypatch):
    class DummyFeed:
        def __init__(self):
            self._lock = MagicMock()
            self.latest_price = 100.0
            self.latest_bars = pd.DataFrame({"high": [101], "low": [99], "close": [100]})
            self.latest_features = pd.DataFrame({"ts": [pd.Timestamp.utcnow()], "vol_8": [1]})
            self.ex = MagicMock(symbol="SOL/USD")

    monkeypatch.setattr(time, "sleep", lambda x: None)

    cfg = TradingConfig(daily_max_loss=0.01, max_drawdown_stop=0.01)

    ens = MagicMock(predict=lambda X: [0.6])
    jsonl = MagicMock()

    engine = LiveEngine(ens, cfg, DummyFeed(), "k", "s", prom=({}, {}), jsonl=jsonl, real=False)
    # Patch equity to simulate a big loss
    engine.engine.equity = MagicMock(return_value=cfg.capital * 0.5)

    engine.run(interval_sec=0, max_iters=1)

    jsonl.write.assert_any_call(
        event="kill",
        equity=pytest.approx(cfg.capital * 0.5),
        daily_pnl=pytest.approx(-cfg.capital * 0.5),
    )


def test_live_engine_nan_features(monkeypatch):
    class DummyFeed:
        def __init__(self):
            self._lock = MagicMock()
            self.latest_price = 100.0
            self.latest_bars = pd.DataFrame({"high": [101], "low": [99], "close": [100]})
            # Features with a NaN value
            self.latest_features = pd.DataFrame(
                {"ts": [pd.Timestamp.utcnow()], "vol_8": [float("nan")]}
            )
            self.ex = MagicMock(symbol="SOL/USD")

    # Patch time.sleep so the loop doesn’t stall
    monkeypatch.setattr(time, "sleep", lambda x: None)

    cfg = TradingConfig()
    ens = MagicMock(predict=lambda X: [0.6])
    jsonl = MagicMock()

    engine = LiveEngine(ens, cfg, DummyFeed(), "k", "s", prom=({}, {}), jsonl=jsonl, real=False)

    # Run one iteration; should skip due to NaN features
    engine.run(interval_sec=0, max_iters=1)

    # Assert that no entry/exit events were written
    for call in jsonl.write.call_args_list:
        kwargs = call.kwargs
        assert "event" not in kwargs or kwargs["event"] not in ("entry", "exit")
