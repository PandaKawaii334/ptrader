---
name: Bug report
about: Report a problem or unexpected behavior when using `ptrader`
title: ''
labels: bug
assignees: ''
---

**Description**

Provide a clear and concise description of the issue or unexpected behavior observed when using **ptrader** from the command line.

---

**Steps to Reproduce**

Include a minimal reproducible command and any relevant log output:

```bash
ptrader --live --symbols "SOL/USD,BTC/USD,ETH/USD" --tf 30Min

# sample output
# ...
# ...

tail -n 100 logs/run.log

# sample log output
# ...
# ...

```

Attach relevant log files (`logs/run.log` and `logs/run.jsonl`) if available.

---

**Expected Behavior**

Describe what you expected the command to do (e.g., log structured events to `logs/run.jsonl`, update Prometheus metrics, etc.).

---

**Actual Behavior**

Describe what actually happened (e.g., error message, unexpected trade behavior, missing logs).

Paste the full error traceback here if available.

---

**Environment**

Please provide:
- **OS**: \[e.g. macOS 14, Ubuntu 22.04, Windows 11]
- **Console**: \[e.g. terminal, VS Code]
- **Python version**: \[e.g. 3.11]
- **ptrader version**: output of `ptrader --version`
- **Installation method**: \[e.g. `pip install .`, `pip install -e .`]

---

**Configuration**

If you modified defaults, include your `TradingConfig` values (capital, fee, slippage, leverage, etc.).

**Additional Context**

Add any other context that might help diagnose the issue:
- Symbol(s) (e.g. `SOL/USD`, `"SOL/USD,ETH/USD"`)
- Run type involved (e.g. `--backtest`, `--live`, `--live --real`)
- Other CLI arguments
- Whether it happens consistently or only under certain data conditions
- Approximate UTC time of occurrence