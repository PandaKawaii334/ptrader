# tests/test_real_engine.py

import pytest
from unittest.mock import MagicMock
import time
from ptrader.engine.real import RealEngine


class DummyClient:
    """A dummy Alpaca client for testing RealEngine."""

    def __init__(self):
        self.raise_exc = False
        self.side_effects = []
        self.order_status = "filled"
        self.filled_qty = 1
        self.filled_avg_price = 101

    def get_account(self):
        class A:
            equity = "12345.67"
            buying_power = "99999"

        return A()

    def get_open_position(self, symbol):
        if getattr(self, "raise_exc", False):
            raise Exception("fail")

        class P:
            avg_entry_price = "100"
            qty = "1"

        return P()

    def submit_order(self, req):
        class Os:
            id = "order123"

        return Os()

    def get_order_by_id(self, oid):
        if self.side_effects:
            return self.side_effects.pop(0)

        class Oid:
            status = self.order_status
            filled_qty = self.filled_qty
            filled_avg_price = self.filled_avg_price

        return Oid()


def test_set_symbol_and_get_position():
    re = RealEngine("key", "secret")
    dummy = DummyClient()
    re.client = dummy

    # entry is None by default
    re.entry = None
    dummy.raise_exc = True
    re.set_symbol("SOL/USD")
    assert re.entry is None  # exception branch triggers

    # normal get position
    dummy.raise_exc = False
    assert re._get_position("SOL/USD") == 1

    # normal set_symbol
    re.set_symbol("SOL/USD")
    assert re.pos == 1
    assert re.entry == 100

    # exception in _get_position
    dummy.raise_exc = True
    assert re._get_position("SOL/USD") == 0.0

    # exception when fetching entry
    dummy.raise_exc = True
    re.entry = 100
    re.set_symbol("SOL/USD")
    assert re.entry is None


def test_wait_for_fill(monkeypatch):
    re = RealEngine("key", "secret")
    dummy = DummyClient()
    re.client = dummy
    monkeypatch.setattr(time, "sleep", lambda x: None)

    # normal filled order
    filled, avg = re._wait_for_fill("order123", timeout=1, poll=0.01)
    assert filled == 1
    assert avg == 101

    # partially filled
    class PartiallyFilled:
        status = "partially_filled"
        filled_qty = 0.5
        filled_avg_price = 50

    dummy.side_effects = [PartiallyFilled()]
    filled, avg = re._wait_for_fill("order123", timeout=1, poll=0.01)
    assert filled == 0.5
    assert avg == 50

    # canceled/rejected/expired triggers RuntimeError
    class Cancel:
        status = "canceled"
        filled_qty = 0
        filled_avg_price = 0

    dummy.side_effects = [Cancel()]
    with pytest.raises(RuntimeError):
        re._wait_for_fill("order123", timeout=0.01, poll=0.001)

    # timeout triggers TimeoutError
    dummy.side_effects = []
    dummy.order_status = "pending"
    with pytest.raises(TimeoutError):
        re._wait_for_fill("order123", timeout=0.01, poll=0.001)


def test_buy_and_sell():
    re = RealEngine("key", "secret")
    dummy = DummyClient()
    re.client = dummy
    re.set_symbol("SOL/USD")

    # patch _wait_for_fill to simulate realistic fills
    def wait_mock(order_id, timeout=30, poll=1.0):
        # simulate fill for requested quantity, max 2
        return min(2, max(0, 2)), 100

    re._wait_for_fill = wait_mock

    # buy 2 units
    qty, px = re.buy(price=100, size_usd=1000, slip_pct=0)
    assert qty == 2
    assert px == 100
    assert re.pos == 3  # previous pos 1 + 2
    assert re.entry == pytest.approx(100.0)

    # sell normal 2 units
    qty, px = re.sell(price=100, qty=2, slip_pct=0)
    assert qty == 2
    assert px == 100
    assert re.pos == 1
    assert re.entry is not None

    # sell more than remaining pos (should only sell 1)
    qty, px = re.sell(price=100, qty=10, slip_pct=0)
    assert qty == 1
    assert px == 100
    assert re.pos == 0
    assert re.entry is None


def test_equity():
    re = RealEngine("key", "secret")
    dummy = DummyClient()
    re.client = dummy

    # normal equity fetch
    assert re.equity() == pytest.approx(12345.67)

    # exception branch
    dummy.get_account = MagicMock(side_effect=Exception("fail"))
    re.human_log = MagicMock()
    re.json_log = MagicMock()
    eq = re.equity()
    assert eq == 0
    re.human_log.error.assert_called_once()
    re.json_log.write.assert_called_once()
