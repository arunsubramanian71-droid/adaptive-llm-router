"""Structured logging.

Emits one JSON object per line — easy to grep/parse, and safe to redirect
into a run directory alongside the response records. Never log secrets:
`_REDACT_KEYS` is scrubbed from any `extra=` payload before it's written.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_REDACT_KEYS = {"api_key", "anthropic_api_key", "authorization", "x-api-key", "token"}

_STANDARD_LOG_RECORD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


def _redact(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        k: ("<redacted>" if k.lower() in _REDACT_KEYS else v)
        for k, v in payload.items()
    }


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extra = {
            k: v for k, v in record.__dict__.items() if k not in _STANDARD_LOG_RECORD_ATTRS
        }
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **_redact(extra),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", run_dir: Path | None = None) -> None:
    root = logging.getLogger("router")
    root.setLevel(level)
    root.handlers.clear()

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(JsonFormatter())
    root.addHandler(console_handler)

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)

    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"router.{name}")
