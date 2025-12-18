# src/ptrader/engine/real.py

import time
from typing import Optional, Tuple
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


class RealEngine:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        fee=0.001,
        slippage=0.0005,
        human_log=None,
        json_log=None,
    ):
        self.client = TradingClient(api_key, api_secret, paper=False)
        self.fee = fee
        self.slippage = slippage
        self.pos = 0.0
        self.entry: Optional[float] = None
        self.symbol: Optional[str] = None
        self.human_log = human_log
        self.json_log = json_log

    def set_symbol(self, symbol: str):
        self.symbol = symbol
        try:
            self.pos = self._get_position(symbol)
        except Exception:
            self.pos = 0.0
            self.entry = None
            return
        if self.pos > 0:
            try:
                pos = self.client.get_open_position(symbol)
                if isinstance(pos, dict):
                    self.entry = float(pos.get("avg_entry_price", 0))
                else:
                    self.entry = float(getattr(pos, "avg_entry_price", 0))
            except Exception:
                self.entry = None
        else:
            self.entry = None

    def _get_position(self, symbol: str) -> float:
        """Return current position size for the symbol."""
        try:
            pos = self.client.get_open_position(symbol)
            return float(getattr(pos, "qty", 0.0))
        except Exception:
            return 0.0

    def _wait_for_fill(
        self, order_id: str, timeout: int = 30, poll: float = 1.0
    ) -> Tuple[float, float]:
        """Wait until order is filled or timeout. Returns (filled_qty, avg_price)."""
        start = time.time()
        while time.time() - start < timeout:
            order = self.client.get_order_by_id(order_id)
            status = str(getattr(order, "status", "")).lower()
            filled_qty = float(getattr(order, "filled_qty", 0.0) or 0.0)

            if status in ("canceled", "rejected", "expired"):
                raise RuntimeError(f"Order {order_id} {status}")

            if filled_qty > 0 and status in ("partially_filled", "filled"):
                avg_px_raw = getattr(order, "filled_avg_price", None)
                limit_px_raw = getattr(order, "limit_price", None)
                avg_px = float(avg_px_raw) if avg_px_raw else float(limit_px_raw or 0.0)
                return filled_qty, avg_px

            time.sleep(poll)
        raise TimeoutError(f"Order {order_id} not filled within {timeout}s")

    def buy(self, price: float, size_usd: float, slip_pct: float) -> tuple[float, float]:
        if not self.symbol:
            raise ValueError("Symbol not set. Call set_symbol() first.")

        account = self.client.get_account()
        if isinstance(account, dict):
            bp_raw = account.get("buying_power", 0.0)
        else:
            bp_raw = getattr(account, "buying_power", 0.0)
        bp = float(bp_raw or 0.0)
        if bp < size_usd:
            raise ValueError(f"Insufficient buying power: {bp} < {size_usd}")

        # exec_px = price * (1 + slip_pct)
        req = MarketOrderRequest(
            symbol=self.symbol,
            notional=size_usd,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.IOC,
        )
        order = self.client.submit_order(req)
        order_id = str(order.get("id")) if isinstance(order, dict) else str(order.id)
        filled_qty, avg_px = self._wait_for_fill(order_id)

        self.pos += filled_qty
        if self.entry is None:
            self.entry = avg_px
        else:
            self.entry = ((self.pos - filled_qty) * self.entry + filled_qty * avg_px) / max(
                self.pos, 1e-12
            )
        return filled_qty, avg_px

    def sell(self, price: float, qty: float, slip_pct: float) -> tuple[float, float]:
        if not self.symbol:
            raise ValueError("Symbol not set. Call set_symbol() first.")

        if self.pos < qty:
            qty = self.pos
        if qty <= 0:
            return 0.0, price

        req = MarketOrderRequest(
            symbol=self.symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.IOC,
        )
        order = self.client.submit_order(req)
        order_id = str(order.get("id")) if isinstance(order, dict) else str(order.id)
        filled_qty, avg_px = self._wait_for_fill(order_id)
        # Clamp to requested qty and current pos
        filled_qty = min(filled_qty, qty, self.pos)
        self.pos -= filled_qty
        if self.pos <= 1e-12:
            self.pos = 0.0
            self.entry = None
        return filled_qty, avg_px

    def equity(self) -> float:
        try:
            account = self.client.get_account()
            eq_raw = getattr(account, "equity", 0.0)
            return float(eq_raw or 0.0)
        except Exception as e:
            if self.human_log:
                self.human_log.error(f"Equity fetch error: {e}")
            if self.json_log:
                self.json_log.write(event="equity_error", error=str(e))
            return 0.0
