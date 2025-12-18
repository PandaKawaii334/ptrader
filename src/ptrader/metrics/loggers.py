# ptrader/metrics/loggers.py

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class HumanLogger:
    """Human‑readable logger using Python's logging module."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("ptrader.human")
        self.logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(self.path, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fh_fmt)

        if not self.logger.handlers:
            self.logger.addHandler(fh)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)


class JSONLLogger:
    """Structured JSON Lines logger for machine‑readable events."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **record: Any) -> None:
        rec: Dict[str, Any] = {"ts": datetime.now(timezone.utc).isoformat(), **record}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
