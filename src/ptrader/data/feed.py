# ptrader/data/feed.py

from pathlib import Path
import time
import threading
import pandas as pd
from typing import Optional
from .alpaca import AlpacaExchange
from ..features.engineer import FeatureEngineer
from ..metrics.loggers import HumanLogger, JSONLLogger


class MarketFeed:
    def __init__(self, ex: AlpacaExchange, tf: str = "30Min", interval_sec: int = 1800):
        self.ex = ex
        # feed uses ex.symbol everywhere
        self.latest_price: float = 0.0
        self.latest_bars: Optional[pd.DataFrame] = None
        self.latest_features: Optional[pd.DataFrame] = None
        self.last_update_ts: float = 0.0
        self._fe = FeatureEngineer(tf=tf)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self.interval = interval_sec
        self.tf = tf
        self.human_log = HumanLogger(Path("logs/run.log"))
        self.json_log = JSONLLogger(Path("logs/run.jsonl"))

    def start(self) -> None:
        self._thr.start()

    def stop(self) -> None:
        self._stop.set()
        self._thr.join(timeout=3.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                end = pd.Timestamp.utcnow()
                start = end - pd.Timedelta(days=10)
                bars = self.ex.bars(self.tf, start, end)
                price = self.ex.latest_price()
                feats = self._fe.create_features(bars) if not bars.empty else None
                with self._lock:
                    self.latest_bars, self.latest_features, self.latest_price = (
                        bars,
                        (feats.iloc[-1:] if feats is not None and not feats.empty else None),
                        price,
                    )
                    self.last_update_ts = time.time()
            except Exception as e:
                self.human_log.error(f"Feed error: {e}")
                self.json_log.write(event="feed_error", error=str(e))
            # Respect configured interval between polls to avoid tight looping and excessive logs
            try:
                time.sleep(self.interval)
            except Exception:
                # In case the interval is invalid for some reason, back off a short fixed time
                time.sleep(1)
