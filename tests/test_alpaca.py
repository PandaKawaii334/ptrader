# tests/test_alpaca.py

import pandas as pd
import pytest
from ptrader.data.alpaca import AlpacaExchange
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def patch_loggers(monkeypatch):
    # Patch HumanLogger and JSONLLogger globally for all tests
    monkeypatch.setattr("ptrader.cli.main.HumanLogger", lambda path: MagicMock())
    monkeypatch.setattr("ptrader.cli.main.JSONLLogger", lambda path: MagicMock())


def test_tf_conversion():
    ex = AlpacaExchange("k", "s")
    assert ex._tf("15Min").amount == 15
    assert ex._tf("1Hour").amount == 1


def test_bars_empty(monkeypatch):
    ex = AlpacaExchange("k", "s")
    monkeypatch.setattr(
        ex.data_client,
        "get_crypto_bars",
        lambda req: type(
            "Resp",
            (),
            {"df": pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])},
        )(),
    )
    df = ex.bars("15Min", None, None)
    assert df.empty


def test_latest_price_exception(monkeypatch):
    ex = AlpacaExchange("k", "s")
    monkeypatch.setattr(ex.data_client, "get_crypto_latest_trade", lambda req: 1 / 0)
    assert ex.latest_price() == 0.0
