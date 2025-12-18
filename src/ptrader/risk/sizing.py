# ptrader/risk/sizing.py


class RiskSizer:
    def __init__(self, max_pos_pct=0.10, max_leverage=1.0):
        self.max_pos_pct = max_pos_pct
        self.max_leverage = max_leverage

    def kelly_size_usd(self, prob: float, vol: float, equity: float) -> float:
        edge = max(0.0, prob - 0.5)
        kelly = min(edge * 0.25, 0.10)  # cap Kelly fraction
        vol_adj = 1 / (1 + abs(vol) * 10)  # throttle in high vol
        size_usd = equity * kelly * vol_adj * self.max_leverage
        return min(size_usd, equity * self.max_pos_pct * self.max_leverage)
