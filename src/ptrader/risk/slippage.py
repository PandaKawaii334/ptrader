# ptrader/risk/slippage.py


def dynamic_slippage(
    high: float, low: float, price: float, base: float = 0.0005, spread_bps: float = 0.0
) -> float:
    rng = max(0.0, high - low)
    spread = max(0.0, spread_bps / 10000.0 * price)
    return max(base, (0.25 * rng + spread) / max(price, 1e-9))
