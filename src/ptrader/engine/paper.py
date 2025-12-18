# ptrader/engine/paper.py

import time


class PaperEngine:
    def __init__(self, capital=10000.0, fee=0.001, slippage=0.0005, max_leverage=2.0):
        self.balance = capital
        self.fee = fee
        self.slippage = slippage
        self.pos = 0.0
        self.entry = None
        self.debt = 0.0
        self.mmr = 0.25
        self.borrow_rate_annual = 0.10
        self.last_interest_ts = time.time()
        self.max_leverage = max_leverage

    def _accrue(self):
        if self.debt <= 0:
            self.last_interest_ts = time.time()
            return
        now = time.time()
        dt_days = (now - self.last_interest_ts) / 86400.0
        if dt_days > 0:
            self.debt += self.debt * (self.borrow_rate_annual / 365.0) * dt_days
            self.last_interest_ts = now

    def equity(self, price: float) -> float:
        self._accrue()
        return self.balance + self.pos * price - self.debt

    def buy(self, price: float, size_usd: float, slip_pct: float) -> tuple[float, float]:
        self._accrue()
        exec_px = price * (1 + slip_pct)
        qty = size_usd / (exec_px * (1 + self.fee))
        qty = round(qty, 6)
        fee = qty * exec_px * self.fee
        cost = qty * exec_px + fee
        shortfall = max(0.0, cost - self.balance)
        self.debt += shortfall
        self.balance -= cost - shortfall
        if self.pos > 0 and self.entry is not None:
            self.entry = ((self.pos * self.entry) + (qty * exec_px)) / max(self.pos + qty, 1e-12)
        else:
            self.entry = exec_px
        self.pos += qty
        return qty, exec_px

    def sell(self, price: float, qty: float, slip_pct: float) -> tuple[float, float]:
        self._accrue()
        exec_px = price * (1 - slip_pct)
        fee = qty * exec_px * self.fee
        proceeds = qty * exec_px - fee
        repay = min(self.debt, proceeds)
        self.debt -= repay
        proceeds -= repay
        self.balance += proceeds
        self.pos -= qty
        if self.pos <= 1e-12:
            self.pos = 0.0
            self.entry = None
        return qty, exec_px
