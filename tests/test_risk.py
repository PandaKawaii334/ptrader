# tests/test_risk.py

from ptrader.risk.sizing import RiskSizer
from ptrader.risk.controls import KillSwitch


def test_risk_sizer_limits():
    sizer = RiskSizer(max_pos_pct=0.1, max_leverage=2.0)
    size = sizer.kelly_size_usd(prob=0.6, vol=0.05, equity=10000)
    assert 0 < size <= 2000  # 10% * 2 leverage


def test_kill_switch_triggers():
    ks = KillSwitch(capital=10000, daily_max_loss=0.1, max_drawdown=0.2)
    assert ks.check(equity=8000, daily_pnl=-1500)  # daily loss breach
    ks.peak = 10000
    assert ks.check(equity=7500, daily_pnl=-100)  # drawdown breach
