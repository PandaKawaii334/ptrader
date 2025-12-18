# tests/test_validate.py

import pytest
from types import SimpleNamespace
from ptrader.config.validate import validate, ConfigError
from ptrader.config.types import ALLOWLIST
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def patch_loggers(monkeypatch):
    # Patch HumanLogger and JSONLLogger globally for all tests
    monkeypatch.setattr("ptrader.cli.main.HumanLogger", lambda path: MagicMock())
    monkeypatch.setattr("ptrader.cli.main.JSONLLogger", lambda path: MagicMock())


# Helper to generate a valid config
def make_cfg(**overrides):
    cfg = SimpleNamespace(
        capital=10000,
        fee=0.001,
        slippage=0.01,
        max_position_pct=0.5,
        max_leverage=2.0,
        max_drawdown_stop=0.5,
        daily_max_loss=0.05,
        cooldown_minutes=10,
        entry_threshold=0.5,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_validate_happy_path():
    cfg = make_cfg()
    symbol = ALLOWLIST[0]  # Use a valid symbol
    # Should not raise
    validate(cfg, symbol)


@pytest.mark.parametrize(
    "attr,value,msg",
    [
        ("capital", 0, "capital must be > 0"),
        ("fee", -0.1, "fee must be between 0 and 1%"),
        ("fee", 0.02, "fee must be between 0 and 1%"),
        ("slippage", -0.01, "slippage must be between 0 and 5%"),
        ("slippage", 0.1, "slippage must be between 0 and 5%"),
        ("max_position_pct", 0, "max_position_pct ∈ \\(0,1\\]"),
        ("max_position_pct", 1.5, "max_position_pct ∈ \\(0,1\\]"),
        ("max_leverage", 0.5, "max_leverage ∈ \\[1,5\\]"),
        ("max_leverage", 6, "max_leverage ∈ \\[1,5\\]"),
        ("max_drawdown_stop", -0.1, "max_drawdown_stop ≤ 90%"),
        ("max_drawdown_stop", 1.0, "max_drawdown_stop ≤ 90%"),
        ("daily_max_loss", -0.1, "daily_max_loss ≤ 90%"),
        ("daily_max_loss", 1.0, "daily_max_loss ≤ 90%"),
        ("cooldown_minutes", 0, "cooldown_minutes ≤ 120"),
        ("cooldown_minutes", 200, "cooldown_minutes ≤ 120"),
        ("entry_threshold", -0.1, "entry_threshold ∈ \\[0,1\\]"),
        ("entry_threshold", 1.5, "entry_threshold ∈ \\[0,1\\]"),
    ],
)
def test_validate_numeric_fail(attr, value, msg):
    cfg = make_cfg(**{attr: value})
    symbol = ALLOWLIST[0]
    with pytest.raises(ConfigError, match=msg):
        validate(cfg, symbol)


def test_validate_symbol_fail():
    cfg = make_cfg()
    symbol = "INVALID_SYMBOL"
    with pytest.raises(ConfigError, match="symbol must be in allowlist"):
        validate(cfg, symbol)
