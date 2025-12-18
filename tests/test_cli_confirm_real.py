import os
from ptrader.cli import main as cli_main


def run_main_with_args(monkeypatch, args_list):
    monkeypatch.setattr("sys.argv", ["ptrader"] + args_list)
    # Capture prints via capsys in tests where needed
    try:
        cli_main.main()
        return 0
    except SystemExit as e:
        return e.code


def test_real_without_confirm_logs_and_exits(tmp_path, monkeypatch, capsys):
    # Ensure env has no confirm
    monkeypatch.delenv("PTRADER_CONFIRM_REAL", raising=False)
    # Use dummy Alpaca env keys to get past key check
    monkeypatch.setenv("ALPACA_API_KEY", "K123")
    monkeypatch.setenv("ALPACA_API_SECRET", "S123")

    # Run with --live and --real but without --confirm-real
    _ = run_main_with_args(
        monkeypatch, ["--live", "--real", "--symbols", "SOL/USD", "--max-iters", "1"]
    )
    # Should not raise other errors but simply return (no further run), rc is None or 0
    captured = capsys.readouterr()
    assert "Real trading (--real) requires explicit confirmation" in captured.out

    # Check logs file for confirm_real_required event (default logs/run.jsonl)
    default_log = "logs/run.jsonl"
    if os.path.exists(default_log):
        import json

        found = False
        with open(default_log) as f:
            for ln in f:
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                if obj.get("event") == "confirm_real_required":
                    found = True
                    break
        assert found


def test_real_with_confirm_allowed(monkeypatch):
    monkeypatch.setenv("PTRADER_CONFIRM_REAL", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "K123")
    monkeypatch.setenv("ALPACA_API_SECRET", "S123")

    # Running with --real and confirmation should proceed; limit to --max-iters=1 to avoid long runs
    monkeypatch.setattr(
        "sys.argv", ["ptrader", "--live", "--real", "--symbols", "SOL/USD", "--max-iters", "1"]
    )
    # Should not raise an exception
    cli_main.main()
