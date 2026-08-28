from __future__ import annotations

import json
import logging

from router.logging.setup import JsonFormatter


def _format(extra: dict) -> dict:
    logger = logging.getLogger("test-json-formatter")
    record = logger.makeRecord(
        name="test", level=logging.INFO, fn="", lno=0, msg="hello", args=(), exc_info=None, extra=extra
    )
    formatter = JsonFormatter()
    return json.loads(formatter.format(record))


def test_json_formatter_basic_fields():
    payload = _format({"model_id": "claude-haiku-4-5"})
    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["model_id"] == "claude-haiku-4-5"


def test_json_formatter_redacts_api_key():
    payload = _format({"api_key": "sk-ant-secret-value", "anthropic_api_key": "sk-ant-secret-value"})
    assert payload["api_key"] == "<redacted>"
    assert payload["anthropic_api_key"] == "<redacted>"
    assert "sk-ant-secret-value" not in json.dumps(payload)
