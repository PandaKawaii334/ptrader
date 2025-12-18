# tests/test_labels.py

import pandas as pd
from ptrader.labels.adaptive import adaptive_labels
from ptrader.labels.gates import regime_gate
import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def patch_loggers(monkeypatch):
    # Patch HumanLogger and JSONLLogger globally for all tests
    monkeypatch.setattr("ptrader.cli.main.HumanLogger", lambda path: MagicMock())
    monkeypatch.setattr("ptrader.cli.main.JSONLLogger", lambda path: MagicMock())


def test_adaptive_labels_binary():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=50, freq="30min"),
            "open": range(50),
            "high": range(1, 51),
            "low": range(50),
            "close": range(50),
            "volume": [100] * 50,
        }
    )
    labels = adaptive_labels(df)
    assert set(labels.unique()) <= {0, 1}


def test_regime_gate_returns_binary_series():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=100, freq="30min"),
            "open": range(100),
            "high": range(1, 101),
            "low": range(100),
            "close": range(100),
            "volume": [100] * 100,
        }
    )
    gate = regime_gate(df)
    assert set(gate.unique()) <= {0, 1}
