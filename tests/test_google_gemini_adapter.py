from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from router.config import ModelEntry
from router.models.google_gemini_adapter import (
    GoogleGeminiModelClient,
    build_generation_config,
    build_request_kwargs,
    classify_error,
)
from router.models.schemas import CompletionStatus


@pytest.fixture
def gemini_cheap_entry() -> ModelEntry:
    return ModelEntry(id="gemini-2.5-flash-lite", display_name="Gemini 2.5 Flash-Lite", max_output_tokens=1024)


@pytest.fixture
def gemini_strong_entry() -> ModelEntry:
    return ModelEntry(
        id="gemini-3.1-pro-preview", display_name="Gemini 3.1 Pro", max_output_tokens=1024, effort="xhigh"
    )


def make_client_error(code: int, message: str = "error"):
    return genai_errors.ClientError(code, {"error": {"message": message}})


def make_server_error(code: int = 500, message: str = "server error"):
    return genai_errors.ServerError(code, {"error": {"message": message}})


def make_fake_usage(input_tokens=12, output_tokens=3, thought_tokens=None, cached_tokens=0):
    return SimpleNamespace(
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        total_thought_tokens=thought_tokens,
        total_cached_tokens=cached_tokens,
        total_tokens=input_tokens + output_tokens + (thought_tokens or 0),
        model_dump=lambda: {"total_input_tokens": input_tokens, "total_output_tokens": output_tokens},
    )


def make_fake_interaction(text="pong", status="completed", model_id="gemini-2.5-flash-lite", usage=None, errors=None):
    return SimpleNamespace(
        id="interaction_test_abc",
        model=model_id,
        status=status,
        output_text=text,
        usage=usage or make_fake_usage(),
        errors=errors,
        model_dump=lambda: {"id": "interaction_test", "model": model_id},
    )


# ---------------------------------------------------------------------------
# build_request_kwargs / build_generation_config — pure functions
# ---------------------------------------------------------------------------


def test_build_request_kwargs_minimal(gemini_cheap_entry):
    kwargs = build_request_kwargs("hello", gemini_cheap_entry, system_prompt=None)
    assert kwargs["model"] == "gemini-2.5-flash-lite"
    assert kwargs["input"] == "hello"
    assert "system_instruction" not in kwargs
    assert kwargs["generation_config"] == {"max_output_tokens": 1024}


def test_build_request_kwargs_with_system_prompt_and_effort(gemini_strong_entry):
    kwargs = build_request_kwargs("hello", gemini_strong_entry, system_prompt="be terse")
    assert kwargs["system_instruction"] == "be terse"
    # effort "xhigh" clamps to Gemini's highest thinking_level, "high"
    assert kwargs["generation_config"]["thinking_level"] == "high"


@pytest.mark.parametrize(
    "effort,expected_level", [("low", "low"), ("medium", "medium"), ("high", "high"), ("xhigh", "high"), ("max", "high")]
)
def test_effort_to_thinking_level_clamping(effort, expected_level):
    entry = ModelEntry(id="gemini-2.5-flash-lite", display_name="x", max_output_tokens=100, effort=effort)
    kwargs = build_request_kwargs("hello", entry, system_prompt=None)
    assert kwargs["generation_config"]["thinking_level"] == expected_level


def test_build_generation_config_provider_is_google(gemini_strong_entry):
    gen_config = build_generation_config(gemini_strong_entry, system_prompt=None)
    assert gen_config.provider == "google"
    assert gen_config.effort == "xhigh"  # generation_config preserves our own vocabulary, unclamped


# ---------------------------------------------------------------------------
# classify_error -- Gemini's hierarchy has no per-status-code subclasses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc,expected",
    [
        (make_client_error(429), "retryable"),
        (make_server_error(500), "retryable"),
        (make_server_error(503), "retryable"),
        (make_client_error(400), "non_retryable"),
        (make_client_error(401), "non_retryable"),
        (make_client_error(404), "non_retryable"),
        (ValueError("unexpected"), "non_retryable"),
    ],
)
def test_classify_error(exc, expected):
    assert classify_error(exc) == expected


# ---------------------------------------------------------------------------
# GoogleGeminiModelClient.complete — zero network calls, SDK method is monkeypatched
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return GoogleGeminiModelClient(api_key="test-key", max_retries=3)


def test_complete_success(client, gemini_cheap_entry, monkeypatch):
    monkeypatch.setattr(client._client.interactions, "create", lambda **kw: make_fake_interaction())

    result = client.complete("ping", gemini_cheap_entry, sample_index=0)

    assert result.status == CompletionStatus.OK
    assert result.text == "pong"
    assert result.request_id == "interaction_test_abc"
    assert result.served_model_id == "gemini-2.5-flash-lite"
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 3
    assert result.retries == 0


def test_complete_reasoning_tokens_are_separate_from_output(client, gemini_strong_entry, monkeypatch):
    usage = make_fake_usage(output_tokens=50, thought_tokens=200)
    monkeypatch.setattr(
        client._client.interactions, "create", lambda **kw: make_fake_interaction(usage=usage)
    )
    result = client.complete("ping", gemini_strong_entry, sample_index=0)
    # NOTE: unlike OpenAI, Gemini's thought tokens are ADDITIVE to output
    # tokens, not a breakdown within them -- see module docstring / ADR-0004.
    assert result.usage.output_tokens == 50
    assert result.usage.reasoning_tokens == 200


def test_complete_marks_truncated_on_incomplete_status(client, gemini_cheap_entry, monkeypatch):
    monkeypatch.setattr(
        client._client.interactions, "create", lambda **kw: make_fake_interaction(status="incomplete")
    )
    result = client.complete("ping", gemini_cheap_entry, sample_index=0)
    assert result.truncated is True
    assert result.status == CompletionStatus.OK  # incomplete is not a failure, just truncated


def test_complete_failed_status_without_exception_is_reported_as_error(client, gemini_cheap_entry, monkeypatch):
    # Gemini's Interactions API can report failure via interaction.status
    # WITHOUT the SDK raising a Python exception -- this is the case that
    # doesn't exist for the Anthropic/OpenAI adapters.
    monkeypatch.setattr(
        client._client.interactions,
        "create",
        lambda **kw: make_fake_interaction(status="failed", text=None, errors=["safety block"]),
    )
    result = client.complete("ping", gemini_cheap_entry, sample_index=0)

    assert result.status == CompletionStatus.ERROR
    assert result.error_type == "gemini_status_failed"
    assert "safety block" in result.error_message
    assert result.text is None


def test_complete_retries_then_succeeds(client, gemini_cheap_entry, monkeypatch):
    sleeps = []
    monkeypatch.setattr("router.models.google_gemini_adapter.time.sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}

    def fake_create(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise make_client_error(429)
        return make_fake_interaction()

    monkeypatch.setattr(client._client.interactions, "create", fake_create)

    result = client.complete("ping", gemini_cheap_entry, sample_index=0)

    assert result.status == CompletionStatus.OK
    assert result.retries == 1
    assert calls["n"] == 2
    assert len(sleeps) == 1


def test_complete_non_retryable_fails_immediately(client, gemini_cheap_entry, monkeypatch):
    sleeps = []
    monkeypatch.setattr("router.models.google_gemini_adapter.time.sleep", lambda s: sleeps.append(s))

    def fake_create(**kwargs):
        raise make_client_error(400)

    monkeypatch.setattr(client._client.interactions, "create", fake_create)

    result = client.complete("ping", gemini_cheap_entry, sample_index=0)

    assert result.status == CompletionStatus.ERROR
    assert result.error_type == "ClientError"
    assert result.retries == 0
    assert sleeps == []


def test_complete_exhausts_retries_and_records_count(gemini_cheap_entry, monkeypatch):
    monkeypatch.setattr("router.models.google_gemini_adapter.time.sleep", lambda s: None)
    client = GoogleGeminiModelClient(api_key="test-key", max_retries=2)

    def always_rate_limited(**kwargs):
        raise make_client_error(429)

    monkeypatch.setattr(client._client.interactions, "create", always_rate_limited)

    result = client.complete("ping", gemini_cheap_entry, sample_index=0)

    assert result.status == CompletionStatus.RATE_LIMITED
    assert result.retries == 2
