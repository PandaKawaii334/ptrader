# tests/test_cli.py

import sys
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from ptrader.cli import main as cli_main
from ptrader.cli.main import _run_symbol, _safe_corr, _evaluate_ensemble_cv
from ptrader.config.types import TradingConfig


@pytest.fixture(autouse=True)
def patch_loggers(monkeypatch):
    # Patch HumanLogger and JSONLLogger globally for all tests
    monkeypatch.setattr("ptrader.cli.main.HumanLogger", lambda path: MagicMock())
    monkeypatch.setattr("ptrader.cli.main.JSONLLogger", lambda path: MagicMock())


class DummyExchange:
    def __init__(self, *args, **kwargs):
        self.symbol = "SOL/USD"

    def bars(self, tf, start, end):
        return pd.DataFrame(
            {
                "ts": pd.date_range("2024-01-01", periods=50, freq="30min"),
                "open": range(50),
                "high": range(1, 51),
                "low": range(50),
                "close": range(50),
                "volume": [100] * 50,
            }
        )

    def latest_price(self):
        return 50.0


def test_run_symbol_backtest_real(monkeypatch):
    monkeypatch.setattr("ptrader.cli.main.AlpacaExchange", DummyExchange)
    cfg = TradingConfig()
    _run_symbol(
        "SOL/USD",
        "30Min",
        "key",
        "secret",
        cfg,
        prom_port=8000,
        backtest=True,
        real=False,
    )


def test_safe_corr_nan_handling():
    import numpy as np

    arr = np.array([1, 2, 3])
    nan_arr = np.array([np.nan, 2, 3])
    assert np.isnan(_safe_corr(arr, nan_arr))


@patch("ptrader.cli.main.AlpacaExchange")
@patch("ptrader.cli.main.MarketFeed")
@patch("ptrader.cli.main.FeatureEngineer")
@patch("ptrader.cli.main.adaptive_labels")
@patch("ptrader.cli.main.Ensemble")
def test_run_symbol_backtest(mock_ens, mock_labels, mock_feat, mock_feed, mock_ex, tmp_path):
    # Mocks
    bars = MagicMock()
    bars.empty = False
    bars.__len__.return_value = 500
    mock_ex.return_value.bars.return_value = bars
    mock_labels.return_value = bars
    mock_feat.return_value.create_features.return_value = bars

    mock_ens_inst = mock_ens.return_value
    mock_ens_inst.predict.return_value = [0.5] * len(bars)

    cfg = MagicMock()
    cfg.default_timeframe = "1Hour"
    cfg.max_position_pct = 0.1
    cfg.max_leverage = 2
    cfg.capital = 10000
    cfg.fee = 0
    cfg.slippage = 0
    cfg.entry_threshold = 0.5
    cfg.stop_loss = 0.05
    cfg.take_profit = 0.05
    cfg.cooldown_minutes = 1
    cfg.max_drawdown_stop = 0.5
    cfg.daily_max_loss = 0.05

    _run_symbol(
        "SOL/USD",
        tf="auto",
        api_key="x",
        api_secret="y",
        cfg=cfg,
        prom_port="8000",
        backtest=True,
        real=False,
    )
    mock_ens_inst.fit.assert_called()


def test_main_missing_keys(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prog"])
    monkeypatch.setenv("ALPACA_API_KEY", "")
    monkeypatch.setenv("ALPACA_API_SECRET", "")
    cli_main.main()
    out = capsys.readouterr().out
    assert "Missing ALPACA_API_KEY" in out


def test_main_with_symbol(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--symbol", "SOL/USD", "--alpaca-key", "k", "--alpaca-secret", "s", "--backtest"],
    )
    with patch("ptrader.cli.main._run_symbol") as run_sym:
        cli_main.main()
        run_sym.assert_called_once()


def test_main_with_symbols(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--symbols",
            "SOL/USD,ETH/USD",
            "--alpaca-key",
            "k",
            "--alpaca-secret",
            "s",
            "--backtest",
        ],
    )
    with patch("ptrader.cli.main._run_symbol") as run_sym:
        cli_main.main()
        assert run_sym.call_count == 2


def test_prometheus_error(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["prog", "--symbol", "SOL/USD", "--alpaca-key", "k", "--alpaca-secret", "s"]
    )
    with patch("ptrader.cli.main.start_http_server", side_effect=Exception("fail")):
        with patch("ptrader.cli.main._run_symbol") as run_sym:
            cli_main.main()
            run_sym.assert_called()


def test_run_symbol_no_bars(monkeypatch):
    class EmptyEx(DummyExchange):
        def bars(self, tf, start, end):
            return pd.DataFrame()

    monkeypatch.setattr("ptrader.cli.main.AlpacaExchange", EmptyEx)
    cfg = TradingConfig()
    _run_symbol("SOL/USD", "30Min", "k", "s", cfg, prom_port=8000, backtest=True, real=False)


def test_run_symbol_insufficient_features(monkeypatch):
    class FewBars(DummyExchange):
        def bars(self, tf, start, end):
            return pd.DataFrame(
                {
                    "ts": pd.date_range("2024-01-01", periods=5, freq="30min"),
                    "open": range(5),
                    "high": range(5),
                    "low": range(5),
                    "close": range(5),
                    "volume": [100] * 5,
                }
            )

    monkeypatch.setattr("ptrader.cli.main.AlpacaExchange", FewBars)
    cfg = TradingConfig()
    _run_symbol("SOL/USD", "30Min", "k", "s", cfg, prom_port=8000, backtest=True, real=False)


def test_evaluate_ensemble_cv():
    import numpy as np

    X = pd.DataFrame({"f": [1, 2, 3, 4, 5]})
    y = pd.Series([0, 1, 0, 1, 0])
    ens = MagicMock()
    ens.fit.return_value = None
    # Return predictions matching the length of the test set
    ens.predict.side_effect = lambda X_test: [0] * len(X_test)
    scores = _evaluate_ensemble_cv(ens, X, y, n_splits=2)
    assert len(scores) == 2
    assert all(np.isnan(scores) | (scores == 0))


def test_run_symbol_live(monkeypatch):
    class ManyBars(DummyExchange):
        def bars(self, tf, start, end):
            return pd.DataFrame(
                {
                    "ts": pd.date_range("2024-01-01", periods=100, freq="30min"),
                    "open": range(100),
                    "high": range(1, 101),
                    "low": range(100),
                    "close": range(100),
                    "volume": [100] * 100,
                }
            )

    monkeypatch.setattr("ptrader.cli.main.AlpacaExchange", ManyBars)

    # Patch FeatureEngineer to return enough features
    monkeypatch.setattr(
        "ptrader.cli.main.FeatureEngineer",
        lambda tf=None: MagicMock(
            create_features=lambda bars: pd.DataFrame({"f": range(len(bars))})
        ),
    )

    # Patch make_prom to avoid duplicate Gauge registration
    monkeypatch.setattr("ptrader.cli.main.make_prom", lambda: ({}, {}))

    cfg = TradingConfig()
    with patch("ptrader.cli.main.LiveEngine") as live_cls:
        live_inst = live_cls.return_value
        live_inst.run.side_effect = lambda **kwargs: None
        _run_symbol("SOL/USD", "30Min", "k", "s", cfg, prom_port=8000, backtest=False, real=False)
        live_inst.run.assert_called()
