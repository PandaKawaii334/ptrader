import os


def test_alpaca_key_with_newline_detected(capsys):
    bad_key = "PK3WFA5BVDT\n> 7YB4UUD5EZ7QTWR"
    os.environ["ALPACA_API_KEY"] = bad_key
    os.environ["ALPACA_API_SECRET"] = "validsecret"

    # Import function under test
    from ptrader.cli.main import _alpaca_keys_ok

    ok = _alpaca_keys_ok(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"])
    captured = capsys.readouterr()
    assert ok is False
    assert "Invalid ALPACA_API_KEY" in captured.out


def test_alpaca_secret_with_carriage_return_detected(capsys):
    os.environ["ALPACA_API_KEY"] = "validkey"
    os.environ["ALPACA_API_SECRET"] = "secret\r"
    from ptrader.cli.main import _alpaca_keys_ok

    ok = _alpaca_keys_ok(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"])
    captured = capsys.readouterr()
    assert ok is False
    assert "Invalid ALPACA_API_SECRET" in captured.out
