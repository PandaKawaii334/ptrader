# ptrader/config/validate.py

from .types import TradingConfig, ALLOWLIST


class ConfigError(Exception): ...


def validate(cfg: TradingConfig, symbol: str) -> None:
    if symbol not in ALLOWLIST:
        raise ConfigError(f"symbol must be in allowlist {ALLOWLIST}")
    if cfg.capital <= 0:
        raise ConfigError("capital must be > 0")
    if not (0 <= cfg.fee <= 0.01):
        raise ConfigError("fee must be between 0 and 1%")
    if not (0 <= cfg.slippage <= 0.05):
        raise ConfigError("slippage must be between 0 and 5%")
    if not (0 < cfg.max_position_pct <= 1.0):
        raise ConfigError("max_position_pct ∈ (0,1]")
    if not (1.0 <= cfg.max_leverage <= 5.0):
        raise ConfigError("max_leverage ∈ [1,5]")
    if not (0 <= cfg.max_drawdown_stop <= 0.9):
        raise ConfigError("max_drawdown_stop ≤ 90%")
    if not (0 <= cfg.daily_max_loss <= 0.9):
        raise ConfigError("daily_max_loss ≤ 90%")
    if not (0 < cfg.cooldown_minutes <= 120):
        raise ConfigError("cooldown_minutes ≤ 120")
    if not (0 <= cfg.entry_threshold <= 1):
        raise ConfigError("entry_threshold ∈ [0,1]")
