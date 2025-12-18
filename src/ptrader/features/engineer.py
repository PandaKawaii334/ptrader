# ptrader/features/engineer.py

import numpy as np
import pandas as pd
from .schema import assert_ohlcv


class FeatureEngineer:
    def __init__(self, tf: str = "30Min"):
        # window scales per timeframe
        self.tf = tf
        self.windows = [2, 4, 8, 16, 32, 64, 96] if tf == "30Min" else [5, 10, 20, 40, 80]

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values("ts").reset_index(drop=True)
        assert_ohlcv(df)
        for w in self.windows:
            df[f"ret_{w}"] = df["close"].pct_change(w).clip(-0.5, 0.5)
            df[f"ma_{w}"] = df["close"].rolling(w).mean()
            df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
        ret = df["close"].pct_change()
        for w in [2, 8, 48]:
            df[f"vol_{w}"] = ret.rolling(w).std()
        df["sv"] = np.where(df["close"] >= df["open"], df["volume"], -df["volume"])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.select_dtypes(include=["number"])
        df = df.ffill().fillna(0)
        first_valid = max(self.windows)
        return df.iloc[first_valid:].reset_index(drop=True)
