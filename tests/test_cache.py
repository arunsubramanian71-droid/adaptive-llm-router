from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from router.hashing import hash_text
from router.models.anthropic_adapter import build_generation_config
from router.models.schemas import CompletionStatus, NormalizedCompletion, TokenUsage
from router.storage.cache import ContentCache, compute_cache_key


def _completion(gen_config, text="hi") -> NormalizedCompletion:
    return NormalizedCompletion(
        provider="anthropic",
        requested_model_id=gen_config.requested_model_id,
        text=text,
        usage=TokenUsage(input_tokens=1, output_tokens=1),
        timestamp_utc=datetime.now(UTC),
        generation_config=gen_config,
        status=CompletionStatus.OK,
    )


def test_cache_key_is_deterministic(haiku_entry):
    gen_config = build_generation_config(haiku_entry, system_prompt=None)
    prompt_hash = hash_text("same prompt")
    key1 = compute_cache_key("anthropic", haiku_entry.id, prompt_hash, gen_config, 0)
    key2 = compute_cache_key("anthropic", haiku_entry.id, prompt_hash, gen_config, 0)
    assert key1 == key2


def test_cache_key_changes_with_sample_index(haiku_entry):
    gen_config = build_generation_config(haiku_entry, system_prompt=None)
    prompt_hash = hash_text("same prompt")
    key0 = compute_cache_key("anthropic", haiku_entry.id, prompt_hash, gen_config, 0)
    key1 = compute_cache_key("anthropic", haiku_entry.id, prompt_hash, gen_config, 1)
    assert key0 != key1


def test_cache_key_changes_with_prompt(haiku_entry):
    gen_config = build_generation_config(haiku_entry, system_prompt=None)
    key_a = compute_cache_key("anthropic", haiku_entry.id, hash_text("prompt a"), gen_config, 0)
    key_b = compute_cache_key("anthropic", haiku_entry.id, hash_text("prompt b"), gen_config, 0)
    assert key_a != key_b


def test_cache_key_changes_with_model(sonnet_entry, haiku_entry):
    prompt_hash = hash_text("same prompt")
    gen_sonnet = build_generation_config(sonnet_entry, system_prompt=None)
    gen_haiku = build_generation_config(haiku_entry, system_prompt=None)
    key_sonnet = compute_cache_key("anthropic", sonnet_entry.id, prompt_hash, gen_sonnet, 0)
    key_haiku = compute_cache_key("anthropic", haiku_entry.id, prompt_hash, gen_haiku, 0)
    assert key_sonnet != key_haiku


def test_cache_key_changes_with_generation_config(haiku_entry):
    prompt_hash = hash_text("same prompt")
    gen1 = build_generation_config(haiku_entry, system_prompt=None)
    gen2 = build_generation_config(haiku_entry, system_prompt="different system prompt")
    key1 = compute_cache_key("anthropic", haiku_entry.id, prompt_hash, gen1, 0)
    key2 = compute_cache_key("anthropic", haiku_entry.id, prompt_hash, gen2, 0)
    assert key1 != key2


def test_get_or_compute_hit_then_miss(tmp_path: Path, haiku_entry):
    cache = ContentCache(tmp_path / "cache")
    gen_config = build_generation_config(haiku_entry, system_prompt=None)
    key = compute_cache_key("anthropic", haiku_entry.id, hash_text("p"), gen_config, 0)

    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return _completion(gen_config)

    result1, hit1 = cache.get_or_compute(key, compute)
    result2, hit2 = cache.get_or_compute(key, compute)

    assert hit1 is False
    assert hit2 is True
    assert calls["n"] == 1  # compute() must not run again on the hit
    assert result1.text == result2.text


def test_errors_are_not_cached(tmp_path: Path, haiku_entry):
    cache = ContentCache(tmp_path / "cache")
    gen_config = build_generation_config(haiku_entry, system_prompt=None)
    key = compute_cache_key("anthropic", haiku_entry.id, hash_text("p"), gen_config, 0)

    def compute_error():
        return NormalizedCompletion(
            provider="anthropic",
            requested_model_id=haiku_entry.id,
            usage=TokenUsage(),
            timestamp_utc=datetime.now(UTC),
            generation_config=gen_config,
            status=CompletionStatus.ERROR,
            error_type="RateLimitError",
        )

    _result, hit = cache.get_or_compute(key, compute_error)
    assert hit is False
    assert cache.get(key) is None  # never written to disk
