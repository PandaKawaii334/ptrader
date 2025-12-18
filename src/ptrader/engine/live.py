# ptrader/engine/live.py

import time
from typing import Optional, Union, Tuple, Dict, Any
import pandas as pd
from ..risk.sizing import RiskSizer
from ..risk.slippage import dynamic_slippage
from ..risk.controls import KillSwitch
from ..engine.paper import PaperEngine
from ..engine.real import RealEngine
from ..labels.gates import regime_gate  # for dynamic gate


class LiveEngine:
    def __init__(
        self,
        ensemble,
        config,
        feed,
        api_key: str,
        api_secret: str,
        feature_cols: Optional[list[str]] = None,
        prom=None,
        jsonl=None,
        real=False,
    ):
        self.ens = ensemble
        self.cfg = config
        self.feed = feed
        self.real = real
        self.human_log = None
        self.jsonl = jsonl
        self.engine: Union[RealEngine, PaperEngine]
        self.feature_cols = feature_cols

        if real:
            from .real import RealEngine

            self.engine = RealEngine(
                api_key,
                api_secret,
                config.fee,
                config.slippage,
                human_log=self.human_log,
                json_log=self.jsonl,
            )
            self.engine.set_symbol(feed.ex.symbol)
        else:
            self.engine = PaperEngine(
                config.capital, config.fee, config.slippage, config.max_leverage
            )

        self.sizer = RiskSizer(config.max_position_pct, config.max_leverage)
        self.kills = KillSwitch(config.capital, config.daily_max_loss, config.max_drawdown_stop)
        self.last_entry_bar_ts: Optional[float] = None
        self.prom: Tuple[Dict[str, Any], Dict[str, Any]] = prom or ({}, {})
        self.jsonl = jsonl

        # Optional running counters
        self.trade_count = 0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0

    def run(
        self,
        interval_sec: int = 1800,
        max_iters: Optional[int] = None,
        feed_timeout: Optional[int] = None,
    ):
        it = 0
        while True:
            if max_iters and it >= max_iters:
                break
            pred_start = time.time()

            with self.feed._lock:
                price = float(self.feed.latest_price or 0.0)
                bars = self.feed.latest_bars
                X = self.feed.latest_features
                last_update = float(getattr(self.feed, "last_update_ts", 0.0) or 0.0)

            # If configured, exit when feed has not produced data for feed_timeout seconds
            if (
                feed_timeout is not None
                and last_update > 0
                and (time.time() - last_update) > feed_timeout
            ):
                if self.jsonl:
                    self.jsonl.write(event="feed_timeout", last_update=last_update)
                if self.human_log:
                    self.human_log.error(f"Feed timeout: last update at {last_update}")
                break

            # Skip iteration if feed hasn’t produced usable data
            if price <= 0 or bars is None or X is None or X.empty:
                time.sleep(1)
                it += 1
                continue

            # Now safe to access bars and X
            ts_val = bars["ts"].iloc[-1] if "ts" in bars.columns else pd.Timestamp.utcnow()
            bar_ts = float(pd.to_datetime(ts_val).timestamp())

            # Guard against NaNs
            if X.isna().any().any():
                time.sleep(1)
                it += 1
                continue

            # Volatility feature
            vol = float(X["vol_8"].iloc[0]) if "vol_8" in X.columns and len(X) > 0 else 0.0

            # Align features
            if self.feature_cols is not None:
                for col in self.feature_cols:
                    if col not in X:
                        X[col] = 0.0
                X = X[self.feature_cols]

            prob = float(self.ens.predict(X)[0])
            latency = time.time() - pred_start

            # Unified state from engine
            pos = float(getattr(self.engine, "pos", 0.0))
            entry = getattr(self.engine, "entry", None)

            if isinstance(self.engine, PaperEngine):
                equity = self.engine.equity(price)
            else:
                equity = self.engine.equity()

            # Diagnostics: compute size estimate and reason for skipping entry
            vol = float(X["vol_8"].iloc[0]) if "vol_8" in X.columns and len(X) > 0 else 0.0
            try:
                size_usd = self.sizer.kelly_size_usd(prob, vol, equity)
            except Exception:
                size_usd = 0.0

            cooldown_ok = self.last_entry_bar_ts is None or (bar_ts - self.last_entry_bar_ts) >= (
                self.cfg.cooldown_minutes * 60
            )
            try:
                gate_ok = bool(regime_gate(bars).iloc[-1])
            except Exception:
                gate_ok = True

            entry_reasons = {
                "prob": prob,
                "threshold": self.cfg.entry_threshold,
                "prob_ok": prob >= self.cfg.entry_threshold,
                "gate_ok": gate_ok,
                "cooldown_ok": cooldown_ok,
                "size_usd": float(size_usd),
                "size_ok": float(size_usd) > 10,
            }
            if self.jsonl:
                self.jsonl.write(event="entry_diag", **entry_reasons)

            daily_pnl = equity - self.cfg.capital

            # Kill-switches
            if self.kills.check(equity, daily_pnl):
                if self.jsonl:
                    self.jsonl.write(event="kill", equity=equity, daily_pnl=daily_pnl)
                break

            # Dynamic regime gate
            try:
                gate_ok = bool(regime_gate(bars).iloc[-1])
            except Exception:
                gate_ok = True

            # Exit logic
            if pos > 0 and entry:
                pnl_pct = (price - entry) / max(entry, 1e-12)
                self.unrealized_pnl = pnl_pct * pos * price
                if pnl_pct <= -self.cfg.stop_loss or pnl_pct >= self.cfg.take_profit:
                    slip = dynamic_slippage(
                        float(bars.iloc[-1]["high"]), float(bars.iloc[-1]["low"]), price
                    )
                    qty, px = self.engine.sell(price, pos, slip)
                    self.trade_count += 1
                    # Realized PnL update (approximate)
                    realized = (px - entry) / max(entry, 1e-12)
                    self.realized_pnl += realized * qty * px
                    if self.jsonl:
                        self.jsonl.write(event="exit", qty=qty, price=px, pnl=pnl_pct)

            # Entry logic
            cooldown_ok = self.last_entry_bar_ts is None or (bar_ts - self.last_entry_bar_ts) >= (
                self.cfg.cooldown_minutes * 60
            )
            if pos == 0 and prob >= self.cfg.entry_threshold and cooldown_ok and gate_ok:
                slip = dynamic_slippage(
                    float(bars.iloc[-1]["high"]), float(bars.iloc[-1]["low"]), price
                )
                size_usd = self.sizer.kelly_size_usd(prob, vol, equity)
                if size_usd > 10:
                    qty, px = self.engine.buy(price, size_usd, slip)
                    self.last_entry_bar_ts = bar_ts
                    self.trade_count += 1
                    if self.jsonl:
                        self.jsonl.write(event="entry", qty=qty, price=px, signal=prob)

            # Per-iteration logs
            if self.jsonl:
                self.jsonl.write(
                    iteration=it,
                    equity=equity,
                    position=pos,
                    price=price,
                    signal=prob,
                    latency=latency,
                )

            # Prometheus gauges (optional)
            if self.prom:
                g = self.prom[0]
                if "equity" in g:
                    g["equity"].set(equity)
                if "realized_pnl" in g:
                    g["realized_pnl"].set(self.realized_pnl)
                if "unrealized_pnl" in g:
                    g["unrealized_pnl"].set(self.unrealized_pnl)
                if "position" in g:
                    g["position"].set(pos)
                if "price" in g:
                    g["price"].set(price)
                if "signal" in g:
                    g["signal"].set(prob)

            # Periodic summary
            if it % 50 == 0 and it > 0 and self.jsonl:
                self.jsonl.write(
                    event="summary",
                    iteration=it,
                    equity=equity,
                    trades=self.trade_count,
                    realized_pnl=self.realized_pnl,
                    unrealized_pnl=self.unrealized_pnl,
                )

            time.sleep(interval_sec)
            it += 1
