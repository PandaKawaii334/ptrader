# ptrader/risk/controls.py


class KillSwitch:
    def __init__(self, capital: float, daily_max_loss=0.10, max_drawdown=0.20):
        self.capital = capital
        self.daily_max_loss = daily_max_loss
        self.max_drawdown = max_drawdown
        self.peak = capital
        self.session_start = capital

    def check(self, equity: float, daily_pnl: float) -> bool:
        self.peak = max(self.peak, equity)
        drawdown = (self.peak - equity) / max(self.peak, 1e-12)
        if daily_pnl < -self.daily_max_loss * self.capital:
            return True
        if drawdown >= self.max_drawdown:
            return True
        return False
