# scripts/sample_run.py

import sys
from ptrader.cli.main import main


if __name__ == "__main__":
    # EXAMPLE: run a backtest on three symbols
    sys.argv = [
        "ptrader",
        "--backtest",
        "--symbols",
        "SOL/USD,BTC/USD,ETH/USD",
        "--tf",
        "30Min",
    ]
    main()
