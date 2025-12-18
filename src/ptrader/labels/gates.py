# ptrader/labels/gates.py

import pandas as pd


def regime_gate(df: pd.DataFrame) -> pd.Series:
    # Simple gate: trend and spread sanity
    sma = df["close"].rolling(48).mean()
    trend = (df["close"] - sma) / (sma + 1e-10)
    ret = df["close"].pct_change()
    vol_10 = ret.rolling(10).std()
    vol_48 = ret.rolling(48).std()
    vol_reg = (vol_10 / (vol_48 + 1e-10)).fillna(0)
    # Gate allows trading when trend magnitude is moderate and vol regime is not extreme chop
    gate = ((trend.abs() > 0.002) & (vol_reg < 1.8)).astype(int)
    return gate
