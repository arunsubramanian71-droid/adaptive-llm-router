from __future__ import annotations

from types import SimpleNamespace

import anthropic
import pytest

from router.models.anthropic_adapter import (
    AnthropicModelClient,
    build_generation_config,
    build_request_kwargs,
    classify_error,
)
from router.models.schemas import CompletionStatus


class FakeHttpResponse:
    """Minimal stand-in for httpx2.Response, enough for anthropic's
    exception classes to read status_code/headers without a real network
    stack."""

    def __init__(self, status_code: int, headers: dict | None = None):
        self.status_code = status_code
        self.headers = headers or {}
        self.request = SimpleNamespace()


def make_status_error(cls, status_code: int, headers: dict | None = None):
    return cls(f"error {status_code}", response=FakeHttpResponse(status_code, headers), body=None)


def make_connection_error():
    return anthropic.APIConnectionError(request=SimpleNamespace())


def make_timeout_error():
    return anthropic.APITimeoutError(request=SimpleNamespace())


def make_fake_message(text="pong", stop_reason="end_turn", model_id="claude-haiku-4-5"):
    usage = SimpleNamespace(
        input_tokens=12,
        output_tokens=3,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
        model_dump=lambda: {"input_tokens": 12, "output_tokens": 3},
    )
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=usage,
        stop_reason=stop_reason,
        model=model_id,
        _request_id="req_test_abc",
        model_dump=lambda: {"id": "msg_test", "model": model_id},
    )
    return message


# ---------------------------------------------------------------------------
# build_request_kwargs — pure function, no network/API key needed
# ---------------------------------------------------------------------------


def test_build_request_kwargs_minimal(haiku_entry):
    kwargs = build_request_kwargs("hello", haiku_entry, system_prompt=None)
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["max_tokens"] == 1024
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert "thinking" not in kwargs
    assert "output_config" not in kwargs
    assert "system" not in kwargs


def test_build_request_kwargs_with_thinking_and_effort(sonnet_entry):
    kwargs = build_request_kwargs("hello", sonnet_entry, system_prompt="be terse")
    assert kwargs["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert kwargs["output_config"] == {"effort": "high"}
    assert kwargs["system"] == "be terse"


def test_build_generation_config_captures_thinking(sonnet_entry):
    gen_config = build_generation_config(sonnet_entry, system_prompt=None)
    assert gen_config.thinking_type == "adaptive"
    assert gen_config.effort == "high"
    assert gen_config.provider == "anthropic"


# ---------------------------------------------------------------------------
# classify_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc,expected",
    [
        (make_status_error(anthropic.RateLimitError, 429), "retryable"),
        (make_connection_error(), "retryable"),
        (make_timeout_error(), "retryable"),
        (make_status_error(anthropic.InternalServerError, 500), "retryable"),
        (make_status_error(anthropic.BadRequestError, 400), "non_retryable"),
        (make_status_error(anthropic.AuthenticationError, 401), "non_retryable"),
        (make_status_error(anthropic.NotFoundError, 404), "non_retryable"),
        (ValueError("unexpected"), "non_retryable"),
    ],
)
def test_classify_error(exc, expected):
    assert classify_error(exc) == expected


# ---------------------------------------------------------------------------
# AnthropicModelClient.complete
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return AnthropicModelClient(api_key="test-key", max_retries=3)


def test_complete_success(client, haiku_entry, monkeypatch):
    monkeypatch.setattr(client._client.messages, "create", lambda **kw: make_fake_message())

    result = client.complete("ping", haiku_entry, sample_index=0)

    assert result.status == CompletionStatus.OK
    assert result.text == "pong"
    assert result.request_id == "req_test_abc"
    assert result.served_model_id == "claude-haiku-4-5"
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 3
    assert result.usage.reasoning_tokens is None  # not billed separately by Anthropic
    assert result.truncated is False
    assert result.retries == 0


def test_complete_marks_truncated_on_max_tokens(client, haiku_entry, monkeypatch):
    monkeypatch.setattr(
        client._client.messages, "create", lambda **kw: make_fake_message(stop_reason="max_tokens")
    )
    result = client.complete("ping", haiku_entry, sample_index=0)
    assert result.truncated is True


def test_complete_retries_then_succeeds(client, haiku_entry, monkeypatch):
    sleeps = []
    monkeypatch.setattr("router.models.anthropic_adapter.time.sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}

    def fake_create(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise make_status_error(anthropic.RateLimitError, 429)
        return make_fake_message()

    monkeypatch.setattr(client._client.messages, "create", fake_create)

    result = client.complete("ping", haiku_entry, sample_index=0)

    assert result.status == CompletionStatus.OK
    assert result.retries == 1
    assert calls["n"] == 2
    assert len(sleeps) == 1  # exactly one recorded backoff sleep


def test_complete_non_retryable_fails_immediately(client, haiku_entry, monkeypatch):
    sleeps = []
    monkeypatch.setattr("router.models.anthropic_adapter.time.sleep", lambda s: sleeps.append(s))

    def fake_create(**kwargs):
        raise make_status_error(anthropic.BadRequestError, 400)

    monkeypatch.setattr(client._client.messages, "create", fake_create)

    result = client.complete("ping", haiku_entry, sample_index=0)

    assert result.status == CompletionStatus.ERROR
    assert result.error_type == "BadRequestError"
    assert result.retries == 0
    assert sleeps == []  # no retry attempted


def test_complete_exhausts_retries_and_records_count(haiku_entry, monkeypatch):
    monkeypatch.setattr("router.models.anthropic_adapter.time.sleep", lambda s: None)
    client = AnthropicModelClient(api_key="test-key", max_retries=2)

    def always_rate_limited(**kwargs):
        raise make_status_error(anthropic.RateLimitError, 429)

    monkeypatch.setattr(client._client.messages, "create", always_rate_limited)

    result = client.complete("ping", haiku_entry, sample_index=0)

    assert result.status == CompletionStatus.RATE_LIMITED
    assert result.retries == 2  # max_retries honored, not retried indefinitely


def test_complete_uses_retry_after_header(client, haiku_entry, monkeypatch):
    delays = []
    monkeypatch.setattr("router.models.anthropic_adapter.time.sleep", lambda s: delays.append(s))

    calls = {"n": 0}

    def fake_create(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise make_status_error(anthropic.RateLimitError, 429, headers={"retry-after": "7"})
        return make_fake_message()

    monkeypatch.setattr(client._client.messages, "create", fake_create)

    client.complete("ping", haiku_entry, sample_index=0)
    assert delays == [7.0]
