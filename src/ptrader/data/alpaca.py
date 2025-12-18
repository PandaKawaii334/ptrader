# ptrader/data/alpaca.py

import pandas as pd
from typing import Optional
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestTradeRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


class AlpacaExchange:
    def __init__(self, api_key: str, api_secret: str, symbol: str = "SOL/USD"):
        self.symbol = symbol
        self.data_client = CryptoHistoricalDataClient(api_key=api_key, secret_key=api_secret)

    def _tf(self, tf_str: str) -> TimeFrame:
        if tf_str.endswith("Min"):
            return TimeFrame(int(tf_str[:-3]), TimeFrameUnit.Minute)
        if tf_str.endswith("Hour"):
            return TimeFrame(int(tf_str[:-4]), TimeFrameUnit.Hour)
        return TimeFrame(30, TimeFrameUnit.Minute)

    def bars(
        self, tf: str, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]
    ) -> pd.DataFrame:
        req = CryptoBarsRequest(
            symbol_or_symbols=[self.symbol],
            timeframe=self._tf(tf),
            start=start,
            end=end,
        )
        # New SDK returns a BarsResponse, not an iterator
        resp = self.data_client.get_crypto_bars(req)
        # resp.df is a pandas DataFrame with OHLCV data
        if hasattr(resp, "df"):
            df = resp.df.reset_index()
            df.rename(columns={"timestamp": "ts"}, inplace=True)
            return df[["ts", "open", "high", "low", "close", "volume"]]
        return pd.DataFrame()

    def latest_price(self) -> float:
        try:
            req = CryptoLatestTradeRequest(symbol_or_symbols=[self.symbol])
            out = self.data_client.get_crypto_latest_trade(req)
            trade = out.get(self.symbol)
            px = float(getattr(trade, "price", 0.0))
            return px if px > 0 else 0.0
        except Exception:
            return 0.0
