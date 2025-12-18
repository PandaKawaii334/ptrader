# ptrader/features/schema.py

import pandas as pd


def assert_ohlcv(df: pd.DataFrame) -> None:
    req = {"ts", "open", "high", "low", "close", "volume"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    if df["close"].isna().any():
        raise ValueError("NaNs in close")
