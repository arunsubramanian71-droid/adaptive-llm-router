from __future__ import annotations

from types import SimpleNamespace

import openai
import pytest

from router.config import ModelEntry
from router.models.openai_adapter import (
    OpenAIModelClient,
    build_generation_config,
    build_request_kwargs,
    classify_error,
)
from router.models.schemas import CompletionStatus


@pytest.fixture
def gpt_cheap_entry() -> ModelEntry:
    return ModelEntry(id="gpt-5.6-luna", display_name="GPT-5.6 Luna", max_output_tokens=1024)


@pytest.fixture
def gpt_strong_entry() -> ModelEntry:
    return ModelEntry(id="gpt-5.6-sol", display_name="GPT-5.6 Sol", max_output_tokens=1024, effort="high")


class FakeHttpResponse:
    def __init__(self, status_code: int, headers: dict | None = None):
        self.status_code = status_code
        self.headers = headers or {}
        self.request = SimpleNamespace()


def make_status_error(cls, status_code: int, headers: dict | None = None):
    return cls(f"error {status_code}", response=FakeHttpResponse(status_code, headers), body=None)


def make_connection_error():
    return openai.APIConnectionError(request=SimpleNamespace())


def make_timeout_error():
    return openai.APITimeoutError(request=SimpleNamespace())


def make_fake_response(text="pong", status="completed", model_id="gpt-5.6-luna", reasoning_tokens=None):
    usage = SimpleNamespace(
        input_tokens=12,
        output_tokens=3,
        total_tokens=15,
        input_tokens_details=SimpleNamespace(cached_tokens=0),
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        model_dump=lambda: {"input_tokens": 12, "output_tokens": 3},
    )
    response = SimpleNamespace(
        id="resp_test_abc",
        model=model_id,
        status=status,
        output_text=text,
        usage=usage,
        model_dump=lambda: {"id": "resp_test", "model": model_id},
    )
    return response


# ---------------------------------------------------------------------------
# build_request_kwargs / build_generation_config — pure functions
# ---------------------------------------------------------------------------


def test_build_request_kwargs_minimal(gpt_cheap_entry):
    kwargs = build_request_kwargs("hello", gpt_cheap_entry, system_prompt=None)
    assert kwargs["model"] == "gpt-5.6-luna"
    assert kwargs["input"] == "hello"
    assert kwargs["max_output_tokens"] == 1024
    assert "instructions" not in kwargs
    assert "reasoning" not in kwargs


def test_build_request_kwargs_with_system_prompt_and_effort(gpt_strong_entry):
    kwargs = build_request_kwargs("hello", gpt_strong_entry, system_prompt="be terse")
    assert kwargs["instructions"] == "be terse"
    assert kwargs["reasoning"] == {"effort": "high"}


def test_build_generation_config_provider_is_openai(gpt_strong_entry):
    gen_config = build_generation_config(gpt_strong_entry, system_prompt=None)
    assert gen_config.provider == "openai"
    assert gen_config.effort == "high"


# ---------------------------------------------------------------------------
# classify_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc,expected",
    [
        (make_status_error(openai.RateLimitError, 429), "retryable"),
        (make_connection_error(), "retryable"),
        (make_timeout_error(), "retryable"),
        (make_status_error(openai.InternalServerError, 500), "retryable"),
        (make_status_error(openai.BadRequestError, 400), "non_retryable"),
        (make_status_error(openai.AuthenticationError, 401), "non_retryable"),
        (make_status_error(openai.NotFoundError, 404), "non_retryable"),
        (ValueError("unexpected"), "non_retryable"),
    ],
)
def test_classify_error(exc, expected):
    assert classify_error(exc) == expected


# ---------------------------------------------------------------------------
# OpenAIModelClient.complete — zero network calls, SDK method is monkeypatched
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return OpenAIModelClient(api_key="test-key", max_retries=3)


def test_complete_success(client, gpt_cheap_entry, monkeypatch):
    monkeypatch.setattr(client._client.responses, "create", lambda **kw: make_fake_response())

    result = client.complete("ping", gpt_cheap_entry, sample_index=0)

    assert result.status == CompletionStatus.OK
    assert result.text == "pong"
    assert result.request_id == "resp_test_abc"
    assert result.served_model_id == "gpt-5.6-luna"
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 3
    assert result.usage.reasoning_tokens is None  # no reasoning configured for the cheap tier
    assert result.retries == 0


def test_complete_exposes_reasoning_tokens_separately(client, gpt_strong_entry, monkeypatch):
    monkeypatch.setattr(
        client._client.responses, "create", lambda **kw: make_fake_response(reasoning_tokens=40)
    )
    result = client.complete("ping", gpt_strong_entry, sample_index=0)
    assert result.usage.reasoning_tokens == 40
    # NOTE: reasoning_tokens is a breakdown WITHIN output_tokens for OpenAI,
    # not additional -- see module docstring / ADR-0004. output_tokens is
    # still just 3 in the fake response, unaffected by reasoning_tokens.
    assert result.usage.output_tokens == 3


def test_complete_marks_truncated_on_incomplete_status(client, gpt_cheap_entry, monkeypatch):
    monkeypatch.setattr(
        client._client.responses, "create", lambda **kw: make_fake_response(status="incomplete")
    )
    result = client.complete("ping", gpt_cheap_entry, sample_index=0)
    assert result.truncated is True
    assert result.stop_reason == "incomplete"


def test_complete_retries_then_succeeds(client, gpt_cheap_entry, monkeypatch):
    sleeps = []
    monkeypatch.setattr("router.models.openai_adapter.time.sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}

    def fake_create(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise make_status_error(openai.RateLimitError, 429)
        return make_fake_response()

    monkeypatch.setattr(client._client.responses, "create", fake_create)

    result = client.complete("ping", gpt_cheap_entry, sample_index=0)

    assert result.status == CompletionStatus.OK
    assert result.retries == 1
    assert calls["n"] == 2
    assert len(sleeps) == 1


def test_complete_non_retryable_fails_immediately(client, gpt_cheap_entry, monkeypatch):
    sleeps = []
    monkeypatch.setattr("router.models.openai_adapter.time.sleep", lambda s: sleeps.append(s))

    def fake_create(**kwargs):
        raise make_status_error(openai.BadRequestError, 400)

    monkeypatch.setattr(client._client.responses, "create", fake_create)

    result = client.complete("ping", gpt_cheap_entry, sample_index=0)

    assert result.status == CompletionStatus.ERROR
    assert result.error_type == "BadRequestError"
    assert result.retries == 0
    assert sleeps == []


def test_complete_exhausts_retries_and_records_count(gpt_cheap_entry, monkeypatch):
    monkeypatch.setattr("router.models.openai_adapter.time.sleep", lambda s: None)
    client = OpenAIModelClient(api_key="test-key", max_retries=2)

    def always_rate_limited(**kwargs):
        raise make_status_error(openai.RateLimitError, 429)

    monkeypatch.setattr(client._client.responses, "create", always_rate_limited)

    result = client.complete("ping", gpt_cheap_entry, sample_index=0)

    assert result.status == CompletionStatus.RATE_LIMITED
    assert result.retries == 2
