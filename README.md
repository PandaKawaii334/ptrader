# `ptrader`

<div align="center">
  <a href="https://codecov.io/gh/dare-afolabi/ptrader">
    <img src="https://img.shields.io/codecov/c/github/dare-afolabi/ptrader?style=flat" alt="Coverage">
  </a>
  <a href="https://github.com/dare-afolabi/ptrader/releases/latest">
    <img src="https://img.shields.io/github/v/release/dare-afolabi/ptrader?style=flat" alt="Latest Release">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  </a>
  <a href="https://github.com/dare-afolabi/ptrader/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue?style=flat" alt="License">
  </a>
  <a href="https://github.com/sponsors/dare-afolabi">
    <img src="https://img.shields.io/badge/Sponsor-lightgrey?style=flat&logo=github-sponsors" alt="Sponsor">
  </a>
</div>


## Overview

`ptrader` is an **intraday crypto trading system** built for **economic fidelity** and **robust observability**. It simulates and executes trades with full treatment of leverage, margin, slippage, fees, and partial fills, while providing transparent analytics for both **backtesting** and **live trading** (paper or real).

This system is designed for **quants**, **researchers**, and **builders** who demand reproducible workflows and production‑ready controls.

---

## Key Features

### Risk & Controls

- Daily max loss and drawdown kill‑switches.
- Stop‑loss / take‑profit enforcement.
- Cooldown between entries.
- Circuit breakers for API/data/exec/feed errors.
- Dynamic slippage and Kelly‑based sizing with volatility throttling.

### Economic Fidelity

- Full leverage and margin accounting.
- Interest accrual on borrowed funds.
- Maintenance margin checks with liquidation buffer.
- Realized vs unrealized PnL separation.

### Models & Lifecycle

- Feature engineering (returns, volatility, moving averages, signed volume).
- Adaptive labeling with volatility‑scaled horizons.
- Ensemble of LightGBM and RandomForest models with logistic meta‑learner.
- Cross‑validation gates and candidate promotion.
- Periodic retraining with accuracy thresholds.

### Monitoring & Observability

- Structured JSONL logs for machine‑readable events (includes per‑iteration `entry_diag` events with `prob`, `size_usd`, `gate_ok`, `cooldown_ok` and related diagnostics).
- Human‑readable logs with detailed trade summaries.
- Prometheus gauges and counters (equity, PnL, fills, errors); registration is idempotent so the exporter/server can be started safely multiple times.
- External metrics push endpoint.
- State checkpointing for recovery.

### Concurrency & Resilience

Threaded market feed with locks for safe concurrent access.

---

## QuickStart

### Install

Clone the repo and install dependencies:

```bash
git clone https://github.com/dare-afolabi/ptrader.git
cd ptrader
pip install -e .[dev]
```

### Configure Environment

Set your Alpaca API keys in `.env`:

```bash
ALPACA_API_KEY="your_api_key_here"
ALPACA_API_SECRET="your_api_secret_here"
```

Or export them directly:

```bash
export ALPACA_API_KEY="your_api_key_here"
export ALPACA_API_SECRET="your_api_secret_here"
```

#### Run Backtest

```bash
ptrader --backtest --symbols "SOL/USD,BTC/USD,ETH/USD" --tf 30Min
```

#### Run Live Paper Trading

```bash
ptrader --live --symbols "SOL/USD,BTC/USD,ETH/USD" --tf 30Min
```

**Short test run** — useful for quick debugging and verifying `entry_diag` behavior:

```bash
ptrader --live --symbols "SOL/USD,BTC/USD,ETH/USD" --tf 30Min --max-iters 50 --live-interval 1 --feed-timeout 10
```

#### Run Live Real Trading (!)

```bash
# Explicit confirmation required (flag or env var):
ptrader --live --real --confirm-real --symbols "SOL/USD,BTC/USD,ETH/USD" --tf 30Min

# or set in your environment:
export PTRADER_CONFIRM_REAL=1
ptrader --live --real --symbols "SOL/USD,BTC/USD,ETH/USD" --tf 30Min
```

> **(!) WARNING**: You are advised against using `--real` as it executes trade with actual funds. The CLI will refuse to proceed without `--confirm-real` (or `PTRADER_CONFIRM_REAL=1`) and will emit a `confirm_real_required` event in `logs/run.jsonl` when confirmation is missing. If you absolutely must use real trading, do so with extreme caution and after extensive testing.

#### Diagnostics

```bash
ptrader --help

# NaN forensic check
ptrader --nan-check

# Custom retrain interval
ptrader --live --symbols "SOL/USD,BTC/USD,ETH/USD" --tf 30Min --retrain-every 60
```

Prometheus exporter runs by default on port `8000` (configurable via `--prom-port`).

---

## Troubleshooting

Quick checklist:

- Run a short test (`--max-iters`) and inspect `logs/run.jsonl` for `entry_diag` events to see why entries were skipped.
- Confirm `entry_diag` shows `prob`, `size_usd`, `prob_ok`, `size_ok`, `gate_ok`, `cooldown_ok` to isolate blocking conditions.
- If there are no `entry_diag` events, make sure the `logs/` directory exists and is writable by the running user (the CLI writes `logs/run.jsonl`).

- No trades are occurring: check `logs/run.jsonl` for `entry_diag` events and inspect the `prob`, `prob_ok`, `size_usd`, `size_ok`, `gate_ok`, and `cooldown_ok` fields to see which condition is blocking entries. By default **entry_threshold = 0.25**, sizing uses a Kelly baseline (`edge = max(0.0, prob - 0.5)`) so a probability < 0.5 produces zero size, and Live currently requires `size_usd > 10` USD to execute a trade. For quick experiments, lower `entry_threshold`, allow sizing for lower prob, or lower the minimum trade USD.

- Alpaca credentials: the CLI validates API keys for accidental newlines or shell prompt characters (e.g., `>`). If you copy/pasted keys and the CLI prints an `invalid_key` message, re-copy the keys without embedded newlines or extra characters.

- Real trading safety: enabling `--real` now **requires explicit confirmation** via the `--confirm-real` flag or the environment variable `PTRADER_CONFIRM_REAL=1` to avoid accidental execution with real funds.

- Old package in use: if you see behavior that doesn't match workspace code, reinstall editable package locally: `pip install -e .[dev]`.

- Feed behavior: `MarketFeed` has a polling interval and backoff on errors to avoid tight loops and runaway logging — use `--feed-timeout` to make live runs exit if the feed becomes stale.

### Example debugging workflow (quick commands)

1. Run a short live session (no real trades):

```bash
ptrader --live --symbols "SOL/USD,BTC/USD,ETH/USD" --tf 30Min --max-iters 50 --live-interval 1 --feed-timeout 10 > out.txt 2>&1
```

2. Inspect per-iteration diagnostics (`entry_diag` events):

```bash
# Quick grep
grep '"entry_diag"' logs/run.jsonl | tail -n 50

# Structured: show timestamp, prob, size and gate/cooldown flags
jq -r 'select(.event=="entry_diag") | "\(.ts) prob=\(.prob) size=\(.size_usd) prob_ok=\(.prob_ok) size_ok=\(.size_ok) gate_ok=\(.gate_ok) cooldown_ok=\(.cooldown_ok)"' logs/run.jsonl | tail -n 50
```

3. Check for actual entry/exit events and counts:

```bash
jq 'select(.event=="entry" or .event=="exit")' logs/run.jsonl | tail -n 50
jq 'select(.event=="entry")' logs/run.jsonl | wc -l
```

4. Look for key validation or feed-timeout issues:

```bash
jq 'select(.event=="invalid_key" or .event=="feed_timeout")' logs/run.jsonl | tail -n 10
```

> **Tip**: use `--max-iters` and `--live-interval` to run fast experiments and revert any temporary config changes (thresholds/sizing/minimum trade USD) after testing.

---

## System Architecture

```bash
(Historical Data)
Market Feed ──► Feature Engineering ──► Labeling ──► Ensemble Training
                                                  │
                                                  ▼
                                      Cross‑Validation / Pruning
                                                  │
                                                  ▼
                                       (In‑memory trained model)
                                                  │
                                                  ├──────────► BacktestRunner ──► Paper Engine
                                                  │
                                                  ▼
Live Market Feed ──► Feature Engineering ──► Live Engine ──► Paper Engine / Real Engine
                                                  │
                                                  ▼
                               Monitoring (Prometheus, JSONL, Human logs)
```

---

## Development

#### Code Style & Linting

- Format code with **Black**:

```bash
black src/ tests/ scripts/
```

- Lint with **Flake8**:

```bash
flake8 src/ tests/ scripts/
```

#### Testing

Run the full test suite with **pytest**:

```bash
pytest -q
```

Unit tests cover:

- **Engines**: Paper, Backtest, Live/Real via mocks.
- **Risk**: Kelly sizing and kill‑switch triggers.
- **Features**: schema validation and feature engineering.
- **Labels**: adaptive labels and regime gate.
- **Models**: RF, LGB, Ensemble predictions.
- **CLI**: wiring with dummy exchange and invalid key handling.
- **Feeds**: polling interval/backoff behavior and stale-feed detection.
- **Metrics**: Prometheus idempotence (safe re-registration).
- **Live-run**: max-iters, feed-timeout, and diagnostic `entry_diag` emission.

#### Continuous Integration

- Add `pytest`, `flake8`, and `black --check` to your CI pipeline.
- Ensure environment variables (`ALPACA_API_KEY`, `ALPACA_API_SECRET`) are set in CI for integration tests.
- Prometheus metrics can be disabled in CI by skipping `start_http_server`.

#### Contribution Workflow

1. Fork and clone the repo.
2. Create a feature branch:

```bash
git checkout -b feature/my-change
```

3. Make changes with tests.
4. Run linting and tests locally.
5. Push and open a pull request.

---

## Future Directions

1. Multi‑exchange fallback (Binance, Coinbase).
2. Automated retraining pipeline with drift monitoring.
3. Portfolio‑level risk management (VaR, Expected Shortfall).
4. CI/CD integration with containerized deployment.

---

## Legal Disclaimer ⚖️

- The `ptrader` project is provided **for research, education, and experimentation only**. It is **not** financial, investment, tax, or legal advice and is not intended to be used as a trading recommendation or service.
- There is **no guarantee of profitability**; any examples, backtests, or live paper runs are for demonstration purposes. Past performance does not predict future returns.
- Using real-money mode (`--real`) executes live orders and exposes you to financial risk — do not enable `--real` unless you fully understand the risks and have performed thorough testing and due diligence.
- The maintainers, contributors, and authors expressly disclaim all liability for any losses, damages, or claims resulting from the use of this project. Users assume all responsibility and risk when running or modifying the software.
- If you intend to trade with real funds, ensure compliance with applicable laws and regulations and consider seeking professional advice.

---

## License

MIT

---

***Generated**: December 18, 2025*