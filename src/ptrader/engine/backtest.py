# ptrader/engine/backtest.py

import pandas as pd
from ..risk.slippage import dynamic_slippage


class BacktestRunner:
    def __init__(self, ens, cfg, sizer, paper, prom_gauges=None, human_log=None, json_log=None):
        self.ens = ens
        self.cfg = cfg
        self.sizer = sizer
        self.paper = paper
        self.results = []
        self.last_entry_ts = None
        self.prom_gauges = prom_gauges or {}
        self.human_log = human_log
        self.json_log = json_log

    def run(self, df_feat: pd.DataFrame):
        X = df_feat.drop(
            columns=[
                c
                for c in ["ts", "open", "high", "low", "close", "volume", "gate"]
                if c in df_feat.columns
            ]
        )
        sig = self.ens.predict(X)
        for i in range(len(df_feat)):
            bar = df_feat.iloc[i]
            price = float(bar.get("close", 0.0))
            if price <= 0:
                continue
            signal = float(sig[i])
            equity = self.paper.equity(price)
            # Prometheus metrics
            if self.prom_gauges:
                if "equity" in self.prom_gauges:
                    self.prom_gauges["equity"].set(equity)
                if "position" in self.prom_gauges:
                    self.prom_gauges["position"].set(self.paper.pos)
                if "price" in self.prom_gauges:
                    self.prom_gauges["price"].set(price)
                if "signal" in self.prom_gauges:
                    self.prom_gauges["signal"].set(signal)
            # Exit
            if self.paper.pos > 0 and self.paper.entry:
                pnl = (price - self.paper.entry) / self.paper.entry
                if pnl <= -self.cfg.stop_loss or pnl >= self.cfg.take_profit:
                    slip = dynamic_slippage(
                        float(bar.get("high", price)),
                        float(bar.get("low", price)),
                        price,
                    )
                    qty, px = self.paper.sell(price, self.paper.pos, slip)
                    result = {
                        "ts": str(bar.get("ts")),
                        "action": "exit",
                        "price": px,
                        "qty": qty,
                        "pnl": pnl,
                        "equity": self.paper.equity(price),
                    }
                    self.results.append(result)
                    if self.human_log:
                        self.human_log.info(
                            f"EXIT: ts={bar.get('ts')} price={px:.2f} qty={qty:.4f} pnl={pnl:.4f} "
                            f"equity={self.paper.equity(price):.2f}"
                        )
                    if self.json_log:
                        self.json_log.write(event="exit", **result)
            gate_val = int(bar.get("gate", 1))
            if gate_val == 0:
                continue
            # Entry
            ts_val = bar.get("ts")
            if ts_val is not None:
                bar_ts = float(pd.to_datetime(ts_val).timestamp())
            else:
                bar_ts = None

            cooldown_ok = (
                True
                if self.last_entry_ts is None
                else (bar_ts - self.last_entry_ts) >= (self.cfg.cooldown_minutes * 60)
            )
            if self.paper.pos == 0 and signal >= self.cfg.entry_threshold and cooldown_ok:
                vol = float(bar.get("vol_8", 0.0))
                slip = dynamic_slippage(
                    float(bar.get("high", price)), float(bar.get("low", price)), price
                )
                size_usd = self.sizer.kelly_size_usd(signal, vol, equity)
                if size_usd > 1:
                    qty, px = self.paper.buy(price, size_usd, slip)
                    self.last_entry_ts = bar_ts
                    result = {
                        "ts": str(bar.get("ts")),
                        "action": "entry",
                        "signal": signal,
                        "price": px,
                        "qty": qty,
                        "pnl": 0.0,
                        "size_usd": size_usd,
                        "equity": self.paper.equity(price),
                    }
                    self.results.append(result)
                    if self.human_log:
                        self.human_log.info(
                            f"ENTRY: ts={bar.get('ts')} price={px:.2f} qty={qty:.4f} "
                            f"signal={signal:.4f} size_usd={size_usd:.2f} equity="
                            f"{self.paper.equity(price):.2f}"
                        )
                    if self.json_log:
                        self.json_log.write(event="entry", **result)

        return self.results
