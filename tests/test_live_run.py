import time
import pandas as pd
from pathlib import Path
from ptrader.engine.live import LiveEngine
from ptrader.metrics.loggers import JSONLLogger
from ptrader.config.types import TradingConfig


class FakeFeed:
    def __init__(self, updates=1):
        self._lock = type("L", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None})()
        self.latest_price = 1.0
        # a minimal one-row DataFrame with ts and OHLCV
        self.latest_bars = pd.DataFrame(
            {
                "ts": [pd.Timestamp.utcnow()],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
            }
        )
        self.latest_features = pd.DataFrame({"vol_8": [0.1]})
        self.last_update_ts = time.time()

    def set_stale(self, secs_ago=1000):
        self.last_update_ts = time.time() - secs_ago


class FakeEns:
    def predict(self, X):
        # return a constant high prob for entry
        return [0.9]


def test_live_engine_stops_after_max_iters(tmp_path: Path):
    jsonl = JSONLLogger(tmp_path / "run.jsonl")
    cfg = TradingConfig()
    feed = FakeFeed()
    ens = FakeEns()

    live = LiveEngine(ens, cfg, feed, api_key="k", api_secret="s", jsonl=jsonl)
    # small interval so the test runs fast
    live.run(interval_sec=1, max_iters=3)

    contents = (tmp_path / "run.jsonl").read_text()
    # Expect at least 3 per-iteration JSON lines
    assert contents.count("iteration") >= 3


def test_live_engine_accepts_prom_tuple(tmp_path: Path):
    from ptrader.metrics.prometheus import make_prom

    jsonl = JSONLLogger(tmp_path / "run.jsonl")
    cfg = TradingConfig()
    feed = FakeFeed()
    ens = FakeEns()

    prom_tuple = make_prom()
    live = LiveEngine(ens, cfg, feed, api_key="k", api_secret="s", prom=prom_tuple, jsonl=jsonl)
    live.run(interval_sec=1, max_iters=2)
    contents = (tmp_path / "run.jsonl").read_text()
    assert contents.count("iteration") >= 2


def test_live_engine_feed_timeout_exits(tmp_path: Path):
    jsonl = JSONLLogger(tmp_path / "run.jsonl")
    cfg = TradingConfig()
    feed = FakeFeed()
    # make feed stale
    feed.set_stale(secs_ago=10)
    ens = FakeEns()

    live = LiveEngine(ens, cfg, feed, api_key="k", api_secret="s", jsonl=jsonl)
    live.run(interval_sec=1, max_iters=None, feed_timeout=1)

    contents = (tmp_path / "run.jsonl").read_text()
    assert "feed_timeout" in contents
