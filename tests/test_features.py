# tests/test_features.py

import pandas as pd
import pytest
from ptrader.features.engineer import FeatureEngineer
from ptrader.features.schema import assert_ohlcv
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def patch_loggers(monkeypatch):
    # Patch HumanLogger and JSONLLogger globally for all tests
    monkeypatch.setattr("ptrader.cli.main.HumanLogger", lambda path: MagicMock())
    monkeypatch.setattr("ptrader.cli.main.JSONLLogger", lambda path: MagicMock())


def test_feature_engineer_outputs_expected_columns():
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
    fe = FeatureEngineer(tf="30Min")
    out = fe.create_features(df)
    assert "ret_2" in out.columns
    assert "vol_8" in out.columns


def test_schema_validation_missing_column():
    df = pd.DataFrame({"ts": [1], "open": [1], "close": [1]})
    with pytest.raises(ValueError):
        assert_ohlcv(df)
