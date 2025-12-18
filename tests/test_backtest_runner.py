# tests/test_backtest_runner.py

import pandas as pd
import pytest
from unittest.mock import MagicMock
from ptrader.engine.backtest import BacktestRunner
from ptrader.engine.paper import PaperEngine
from ptrader.risk.sizing import RiskSizer
from ptrader.models.sk import RFProb


@pytest.fixture(autouse=True)
def patch_loggers(monkeypatch):
    # Patch HumanLogger and JSONLLogger globally for all tests
    monkeypatch.setattr("ptrader.cli.main.HumanLogger", lambda path: MagicMock())
    monkeypatch.setattr("ptrader.cli.main.JSONLLogger", lambda path: MagicMock())


def test_backtest_runner_entry_exit():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2025-01-01", periods=5),
            "close": [100, 105, 110, 90, 95],
            "vol_8": [1] * 5,
            "gate": [1] * 5,
        }
    )
    ens = MagicMock()
    ens.predict.return_value = [1, 1, 1, 1, 1]

    class Paper:
        pos = 0
        entry = None

        def equity(self, px):
            return 10000

        def buy(self, px, size, slip):
            self.pos = 1
            self.entry = px
            return (1, px)

        def sell(self, px, qty, slip):
            self.pos = 0
            return (1, px)

    cfg = MagicMock()
    cfg.entry_threshold = 0.5
    cfg.stop_loss = 0.05
    cfg.take_profit = 0.05
    cfg.cooldown_minutes = 0

    runner = BacktestRunner(
        ens, cfg, MagicMock(), Paper(), prom_gauges={}, human_log=None, json_log=None
    )
    runner.sizer.kelly_size_usd = MagicMock(return_value=1.5)
    results = runner.run(df)
    assert any(r["action"] == "entry" for r in results)
    assert any(r["action"] == "exit" for r in results)


def test_backtest_runner_respects_gate():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=20, freq="30min"),
            "open": range(20),
            "high": range(1, 21),
            "low": range(20),
            "close": range(20),
            "volume": [100] * 20,
            "vol_8": [0.1] * 20,
            "gate": [0] * 20,  # all gated off
        }
    )
    X = df.drop(columns=["ts", "open", "high", "low", "close", "volume", "gate"])
    y = pd.Series([0, 1] * 10)
    model = RFProb().fit(X, y)
    runner = BacktestRunner(model, cfg=None, sizer=RiskSizer(), paper=PaperEngine())
    results = runner.run(df)
    assert results == []  # no trades because gate=0
