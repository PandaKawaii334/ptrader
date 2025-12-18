# ptrader/cli/main.py

import os
import argparse
import time
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold  # type: ignore

# from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..config.types import TradingConfig, ALLOWLIST
from ..config.validate import validate
from ..data.alpaca import AlpacaExchange
from ..data.feed import MarketFeed
from ..features.engineer import FeatureEngineer
from ..labels.adaptive import adaptive_labels
from ..models.sk import RFProb
from ..models.lgb import LGBProb
from ..models.ensemble import Ensemble
from ..risk.sizing import RiskSizer
from ..engine.paper import PaperEngine
from ..engine.backtest import BacktestRunner
from ..metrics.prometheus import make_prom, start_http_server
from ..metrics.loggers import HumanLogger, JSONLLogger
from ..engine.live import LiveEngine
from ..labels.gates import regime_gate


# Silence common sklearn feature-name warnings once and for all
warnings.filterwarnings(
    "ignore",
    message="X has feature names, but .* was fitted without feature names",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but .* was fitted with feature names",
    category=UserWarning,
)

# Initialize loggers once
human_log = HumanLogger(Path("logs/run.log"))
json_log = JSONLLogger(Path("logs/run.jsonl"))


def _env(key: str, default: str = "") -> str:
    v = os.getenv(key, default).strip()
    return v


def _alpaca_keys_ok(key: str, secret: str) -> bool:
    """Validate Alpaca API key/secret for accidental newlines or shell prompt characters.

    Returns True if keys look OK; prints/logs and returns False otherwise.
    """
    bad_chars = ("\n", "\r", ">")
    for name, val in (("ALPACA_API_KEY", key), ("ALPACA_API_SECRET", secret)):
        if any(ch in (val or "") for ch in bad_chars):
            print(
                f"Invalid {name}: contains newline or illegal characters; "
                "set without newlines or extra characters."
            )
            human_log.error(f"Invalid {name} with embedded control characters: {repr(val)[:100]}")
            json_log.write(event="invalid_key", name=name, value_preview=str(val)[:64])
            return False
    return True


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _evaluate_ensemble_cv(
    ens: Ensemble, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 42
) -> np.ndarray:
    if y.nunique() >= 2:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splitter = kf.split(X, y)
    else:
        human_log.debug("CV: single-class labels detected; returning nan scores.")
        return np.array([np.nan] * n_splits, dtype=float)

    scores = []
    for i, (train_idx, test_idx) in enumerate(splitter):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            human_log.debug(f"Fold {i + 1}: skipping (single class in train/test)")
            scores.append(np.nan)
            continue
        ens.fit(X.iloc[train_idx], y_train)
        pred = ens.predict(X.iloc[test_idx])
        corr = _safe_corr(y_test.to_numpy(), np.asarray(pred))
        human_log.info(
            f"Fold {i + 1}: corr={corr:.3g}" if not np.isnan(corr) else f"Fold {i + 1}: corr=nan"
        )
        json_log.write(event="cv_fold", fold=i, corr=None if np.isnan(corr) else float(corr))

        scores.append(corr)
    return np.array(scores, dtype=float)


def _run_symbol(
    symbol: str,
    tf: str,
    api_key: str,
    api_secret: str,
    cfg: TradingConfig,
    prom_port: str,
    backtest: bool,
    real: bool,
    live_interval: int = 1800,
    max_iters: int | None = None,
    feed_timeout: int | None = None,
):
    validate(cfg, symbol)

    # Banner
    human_log.info(f"=== {symbol} ===")

    # Auto-select timeframe if requested
    if tf == "auto":
        candidates = ["15Min", "30Min", "1Hour"]
        tf_scores = []
        for t in candidates:
            try:
                ex = AlpacaExchange(api_key, api_secret, symbol=symbol)
                end = pd.Timestamp.utcnow()
                start = end - pd.Timedelta(days=10)
                bars = ex.bars(t, start, end)
                if bars.empty:
                    continue
                Xc = FeatureEngineer(tf=t).create_features(bars)
                yc = adaptive_labels(bars)
                min_len = min(len(Xc), len(yc))
                if min_len < 300:
                    continue
                Xc = Xc.iloc[-min_len:].reset_index(drop=True)
                yc = yc.iloc[-min_len:].reset_index(drop=True)
                ens_tmp = Ensemble()
                ens_tmp.add(RFProb())
                ens_tmp.add(LGBProb())
                ens_tmp.fit(Xc, yc)
                pred = ens_tmp.predict(Xc)
                score = _safe_corr(yc.to_numpy(), np.asarray(pred))
                tf_scores.append((t, score))
            except Exception:
                continue
        tf = (
            sorted(tf_scores, key=lambda x: (np.nan_to_num(x[1], nan=-1e9)), reverse=True)[0][0]
            if tf_scores
            else cfg.default_timeframe
        )

    # Acquire bars
    ex = AlpacaExchange(api_key, api_secret, symbol=symbol)
    feed = MarketFeed(ex, tf=tf, interval_sec=1800)
    feed.start()
    timeout = 30
    start_ts = time.time()
    while feed.latest_bars is None and (time.time() - start_ts) < timeout:
        time.sleep(0.1)
    with feed._lock:
        raw = feed.latest_bars
    feed.stop()

    if raw is None or raw.empty:
        human_log.error(f"No bars available for {symbol}, skipping.")
        json_log.write(event="no_bars", symbol=symbol)
        return

    human_log.info(f"{len(raw)} bars obtained for {symbol}")
    json_log.write(event="bars_obtained", symbol=symbol, count=len(raw))

    # Features / labels
    X = FeatureEngineer(tf=tf).create_features(raw)
    y = adaptive_labels(raw)
    min_len = min(len(X), len(y))
    X = X.iloc[-min_len:].reset_index(drop=True)
    y = y.iloc[-min_len:].reset_index(drop=True)

    # Drop raw OHLCV columns
    drop_cols = [
        c for c in ["ts", "open", "high", "low", "close", "volume", "gate"] if c in X.columns
    ]
    X = X.drop(columns=drop_cols)

    # skip training/backtest if there are no usable features
    if X.empty or len(X) < 10:
        human_log.warning(f"Insufficient features for {symbol}; skipping backtest.")
        json_log.write(event="skip", symbol=symbol, reason="insufficient_features")
        return

    # Add regime gate series aligned to features
    gate = regime_gate(raw).iloc[-min_len:].reset_index(drop=True)

    # Diagnostics
    human_log.debug(f"Label distribution: {y.value_counts().to_dict()}")
    json_log.write(event="labels", symbol=symbol, distribution=y.value_counts().to_dict())

    nulls = X.isna().sum()
    if int(nulls.sum()) > 0:
        nan_counts = {k: int(v) for k, v in nulls[nulls > 0].items()}
        human_log.debug(f"Feature NaNs detected: {nan_counts}")

    # Train ensemble
    ens = Ensemble()
    ens.add(RFProb())
    ens.add(LGBProb())
    ens.fit(X, y)

    # Store final feature list
    feature_cols = X.columns.tolist()
    print(feature_cols)

    print(f"\n{symbol}: cross‑validation stats")
    for model in getattr(ens, "models", []):
        if hasattr(model, "cv_stats"):
            print(f"{getattr(model, 'name', type(model).__name__)} CV:")
            for k, v in model.cv_stats.items():
                print(f"  - {k}: {v:.3g}")
            print()

    # Ensemble CV
    scores = _evaluate_ensemble_cv(ens, X, y)
    human_log.info(f"CV correlation mean: {np.nanmean(scores):.3g}")
    json_log.write(event="cv_summary", symbol=symbol, mean_corr=float(np.nanmean(scores)))

    # probs_full = ens.predict(X)

    # Backtest
    if backtest:
        sizer = RiskSizer(cfg.max_position_pct, cfg.max_leverage)
        paper = PaperEngine(cfg.capital, cfg.fee, cfg.slippage, cfg.max_leverage)
        df_feat = X.copy()
        df_feat.insert(0, "ts", raw["ts"].iloc[-len(X) :].reset_index(drop=True))
        df_feat["close"] = raw["close"].iloc[-len(X) :].reset_index(drop=True)

        df_feat["gate"] = gate.values  # aligned length
        json_log.write(event="gate", symbol=symbol, latest=int(gate.iloc[-1]))

        # Prometheus metrics
        prom_gauges, _ = make_prom() if "make_prom" in globals() else ({}, {})

        runner = BacktestRunner(ens, cfg, sizer, paper, prom_gauges, human_log, json_log)
        results = runner.run(df_feat)
        df_results = pd.DataFrame(results)
        human_log.info(f"Backtest complete for {symbol}: {len(df_results)} trades")
        json_log.write(
            event="backtest",
            symbol=symbol,
            trades=len(df_results),
            pnl=float(df_results["pnl"].sum()) if not df_results.empty else 0.0,
            final_equity=(
                float(df_results["equity"].iloc[-1]) if not df_results.empty else cfg.capital
            ),
        )
        if len(df_results) > 0:
            print(df_results)

    else:
        # Prometheus metrics (return tuple of (gauges, counters))
        prom_tuple = make_prom() if "make_prom" in globals() else ({}, {})
        # Start feed
        feed = MarketFeed(ex, tf=tf, interval_sec=live_interval)
        feed.start()
        try:
            live = LiveEngine(
                ens,
                cfg,
                feed,
                api_key,
                api_secret,
                prom=prom_tuple,
                jsonl=json_log,
                real=real,  # WARNING: real money can be used here
                feature_cols=feature_cols,
            )
            # Pass through live-run controls
            live.run(interval_sec=live_interval, max_iters=max_iters, feed_timeout=feed_timeout)
        finally:
            feed.stop()


def main():
    p = argparse.ArgumentParser()
    # Accept either a single symbol or a batch
    p.add_argument("--symbol", type=str, default=None, choices=ALLOWLIST)
    p.add_argument("--symbols", type=str, default=None, help="Comma-separated list of symbols")
    p.add_argument("--real", action="store_true")  # Dangerous: real money trading
    p.add_argument("--live", action="store_true")
    p.add_argument("--shadow", action="store_true")
    p.add_argument("--backtest", action="store_true")
    p.add_argument(
        "--tf",
        type=str,
        default="auto",
        choices=["auto", "5Min", "15Min", "30Min", "1Hour"],
    )
    p.add_argument("--prom-port", type=int, default=8000)
    p.add_argument("--alpaca-key", type=str, default=None)
    p.add_argument("--alpaca-secret", type=str, default=None)
    # Live-run control for testing/short runs
    p.add_argument(
        "--live-interval", type=int, default=1800, help="Interval seconds between live iterations"
    )
    p.add_argument(
        "--max-iters", type=int, default=None, help="Maximum iterations for live mode (testing)"
    )
    p.add_argument(
        "--feed-timeout",
        type=int,
        default=600,
        help="Timeout seconds to consider feed stale and exit",
    )
    # Safety: explicit confirmation required to enable real-money trading
    p.add_argument(
        "--confirm-real",
        action="store_true",
        help="Explicit confirmation to enable real trading (required in addition to --real).",
    )
    args = p.parse_args()

    # Safety: If real trading requested, require explicit confirmation via flag or env var
    def _confirm_real_ok() -> bool:
        if args.confirm_real:
            return True
        confirm_env = os.getenv("PTRADER_CONFIRM_REAL", "").lower()
        if confirm_env in ("1", "true", "yes"):
            return True
        return False

    if args.real and not _confirm_real_ok():
        msg = (
            "Real trading (--real) requires explicit confirmation. "
            "Re-run with --confirm-real or set PTRADER_CONFIRM_REAL=1 in the environment."
        )
        print(msg)
        human_log.error("Attempt to start real trading without confirmation")
        # Emit a lightweight confirm event (avoid referencing CLI symbols at this early stage)
        json_log.write(event="confirm_real_required")
        return

    # Resolve symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.symbol:
        symbols = [args.symbol]
    else:
        symbols = ["SOL/USD"]  # default

    # Env-first with CLI override
    api_key = args.alpaca_key if args.alpaca_key is not None else _env("ALPACA_API_KEY")
    api_secret = args.alpaca_secret if args.alpaca_secret is not None else _env("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        print("Missing ALPACA_API_KEY/ALPACA_API_SECRET. Set env or use CLI flags.")
        return

    # Validate keys for accidental newlines or shell prompt characters (common when copying keys)
    def _alpaca_keys_ok(key: str, secret: str) -> bool:
        bad_chars = ("\n", "\r", ">")
        for name, val in (("ALPACA_API_KEY", key), ("ALPACA_API_SECRET", secret)):
            if any(ch in val for ch in bad_chars):
                print(
                    f"Invalid {name}: contains newline or illegal characters; "
                    "set without newlines or extra characters."
                )
                human_log.error(
                    f"Invalid {name} with embedded control characters: {repr(val)[:100]}"
                )
                json_log.write(event="invalid_key", name=name, value_preview=str(val)[:64])
                return False
        return True

    if not _alpaca_keys_ok(api_key, api_secret):
        return

    # Config and banner (shown once)
    cfg = TradingConfig()
    run_type = (
        "Backtest"
        if args.backtest
        else (
            "Live Trading" if args.live and args.real else ("Paper Trading" if args.live else "Run")
        )
    )
    human_log.info(
        f"Starting {run_type} for {', '.join(symbols)} with "
        f"tf={args.tf}, prom_port={args.prom_port}"
    )
    json_log.write(
        event="start",
        run_type=run_type,
        symbols=symbols,
        tf=args.tf,
        prom_port=args.prom_port,
    )

    # Prometheus once
    try:
        make_prom()
        start_http_server(args.prom_port)
    except Exception as e:
        human_log.error(f"Prometheus server not started: {e}")
        json_log.write(event="prometheus_error", error=str(e))

    # Iterate symbols efficiently (single process, shared setup)
    for sym in symbols:
        try:
            _run_symbol(
                sym,
                args.tf,
                api_key,
                api_secret,
                cfg,
                prom_port=args.prom_port,
                backtest=args.backtest,
                real=args.real,
                live_interval=args.live_interval,
                max_iters=args.max_iters,
                feed_timeout=args.feed_timeout,
            )
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            # Log detailed traceback to human and json logs for post-mortem
            human_log.error(f"Error for {sym}: {e} (type={type(e)})")
            human_log.error(tb)
            json_log.write(event="error", symbol=sym, error=str(e), type=str(type(e)), traceback=tb)
            print(f"Error for {sym}: {e} (see logs for traceback)")
            print("---")
            continue


if __name__ == "__main__":
    main()
