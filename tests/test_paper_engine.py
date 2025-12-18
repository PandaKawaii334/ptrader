# tests/test_paper_engine.py

import time
import pytest
from ptrader.engine.paper import PaperEngine
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def patch_loggers(monkeypatch):
    # Patch HumanLogger and JSONLLogger globally for all tests
    monkeypatch.setattr("ptrader.cli.main.HumanLogger", lambda path: MagicMock())
    monkeypatch.setattr("ptrader.cli.main.JSONLLogger", lambda path: MagicMock())


def test_paper_engine_buy_sell_cycle():
    engine = PaperEngine(capital=10000, fee=0.001, slippage=0.0005, max_leverage=2.0)

    # Initial state
    assert engine.balance == 10000
    assert engine.pos == 0.0
    assert engine.debt == 0.0

    # Buy $1000 notional at $100
    qty, px = engine.buy(price=100, size_usd=1000, slip_pct=0.0005)
    assert qty > 0
    assert engine.pos == pytest.approx(qty)
    assert engine.entry == pytest.approx(px)

    # Equity should reflect position value minus debt
    equity = engine.equity(price=100)
    assert equity <= 10000  # fees/slippage reduce equity slightly

    # Sell entire position at profit
    sell_qty, sell_px = engine.sell(price=110, qty=qty, slip_pct=0.0005)
    assert sell_qty == pytest.approx(qty)
    assert engine.pos == 0.0
    assert engine.entry is None
    assert engine.balance > 10000  # profit realized


def test_paper_engine_fee_and_balance_adjustment():
    engine = PaperEngine(capital=10000, fee=0.001, slippage=0, max_leverage=1.0)

    # Buy $1000 worth at $100
    qty, px = engine.buy(price=100, size_usd=1000, slip_pct=0)
    # Expected quantity accounts for fee
    expected_qty = 1000 / (100 * (1 + 0.001))
    assert abs(qty - expected_qty) < 1e-6

    # Balance reduced by cost minus any borrowed amount
    fee = qty * px * 0.001
    cost = qty * px + fee
    shortfall = max(0.0, cost - 10000)
    expected_balance = 10000 - (cost - shortfall)
    assert pytest.approx(engine.balance, rel=1e-6) == expected_balance
    # Debt should equal shortfall
    assert pytest.approx(engine.debt, rel=1e-6) == shortfall


def test_paper_engine_interest_accrual():
    engine = PaperEngine(capital=1000, fee=0.001, slippage=0, max_leverage=2.0)
    # Force debt
    qty, px = engine.buy(price=100, size_usd=2000, slip_pct=0)
    assert engine.debt > 0

    # Simulate time passing
    old_debt = engine.debt
    time.sleep(0.01)  # small delay
    _ = engine.equity(price=100)  # triggers _accrue
    assert engine.debt >= old_debt  # debt should increase slightly
