# ptrader/config/types.py

from dataclasses import dataclass, field
from typing import List

ALLOWLIST = ["SOL/USD", "BTC/USD", "ETH/USD"]


@dataclass(frozen=True)
class TradingConfig:
    capital: float = 10000.0
    fee: float = 0.001
    slippage: float = 0.0005
    max_position_pct: float = 0.10
    max_leverage: float = 1.0
    max_drawdown_stop: float = 0.20
    daily_max_loss: float = 0.10
    stop_loss: float = 0.01
    take_profit: float = 0.03
    cooldown_minutes: int = 8
    entry_threshold: float = 0.25
    default_timeframe: str = "30Min"
    symbols: List[str] = field(default_factory=lambda: ALLOWLIST)
