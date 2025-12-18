import time
import pandas as pd
from ptrader.data.feed import MarketFeed


class FakeEx:
    def __init__(self):
        self._calls = 0

    def bars(self, tf, start, end):
        # return a minimal DataFrame used by FeatureEngineer
        self._calls += 1
        return pd.DataFrame(
            {
                "ts": [pd.Timestamp.utcnow()],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
            }
        )

    def latest_price(self):
        return 1.0


def test_marketfeed_respects_interval():
    ex = FakeEx()
    # Use a small interval for test speed
    mf = MarketFeed(ex, tf="30Min", interval_sec=0.05)

    mf.start()
    # Wait until initial data is set
    timeout = time.time() + 2.0
    while mf.latest_bars is None and time.time() < timeout:
        time.sleep(0.01)

    assert mf.latest_bars is not None

    # Collect a few timestamps of updates over a short window
    seen = set()
    start = time.time()
    while time.time() - start < 0.2:
        with mf._lock:
            ts = mf.last_update_ts
        if ts:
            seen.add(round(ts, 3))
        time.sleep(0.01)

    mf.stop()

    # We expect at least 2 updates (initial + one more) but not hundreds (not a tight loop)
    assert len(seen) >= 2
    assert len(seen) <= 20

    # And we expect updates to be spaced roughly by the configured interval
    seen_list = sorted(list(seen))
    if len(seen_list) >= 2:
        diffs = [b - a for a, b in zip(seen_list[:-1], seen_list[1:])]
        # at least one interval should be close to configured interval (allow slack)
        assert any(d > 0.02 for d in diffs)
