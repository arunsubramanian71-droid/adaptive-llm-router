#!/usr/bin/env python
"""Stage 0 verification script.

Makes a SMALL number of real Anthropic API calls to prove the whole
pipeline works end-to-end: config -> adapter -> usage parsing -> cost
calculation -> response-level persistence -> content-addressed cache.

SAFETY: this script does NOT call the API by default. Pass --live to
actually spend money. Without --live it runs the identical pipeline against
a fake in-process completion so you can sanity-check everything except the
network call for free.

Example (after reviewing cost below and exporting ANTHROPIC_API_KEY):

    python scripts/verify_stage0.py --live --model claude-haiku-4-5 --num-prompts 2

This makes at most `num_prompts` real calls (each prompt is also requested a
second time to prove the cache avoids a second real call).
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from router.config import ConfigError, load_app_config
from router.cost.calculator import calculate_cost
from router.git_utils import get_git_dirty, get_git_sha
from router.hashing import hash_text
from router.logging import configure_logging, get_logger
from router.models.anthropic_adapter import (
    AnthropicModelClient,
    build_generation_config,
)
from router.models.schemas import (
    CompletionStatus,
    GenerationConfig,
    NormalizedCompletion,
    TokenUsage,
)
from router.storage.cache import ContentCache, compute_cache_key
from router.storage.records import JsonlStore, ResponseRecord
from router.storage.run_metadata import (
    create_run_dir,
    new_run_metadata,
    write_run_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

VERIFICATION_PROMPTS = [
    "Reply with exactly one word: pong",
    "What is 2 + 2? Reply with only the digit.",
]


def make_fake_completion(gen_config: GenerationConfig) -> NormalizedCompletion:
    """Used only in --dry-run mode so the rest of the pipeline can be
    exercised without spending money or needing an API key."""
    return NormalizedCompletion(
        provider=gen_config.provider,
        requested_model_id=gen_config.requested_model_id,
        served_model_id=gen_config.requested_model_id,
        text="[dry-run] fake response",
        request_id="dryrun-0000",
        usage=TokenUsage(input_tokens=12, output_tokens=4),
        latency_ms=1.0,
        timestamp_utc=datetime.now(UTC),
        generation_config=gen_config,
        status=CompletionStatus.OK,
        stop_reason="end_turn",
        truncated=False,
        retries=0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default="claude-haiku-4-5", help="Model id from configs/models.yaml")
    parser.add_argument("--num-prompts", type=int, default=2, help="How many of the tiny built-in prompts to use (max 2)")
    parser.add_argument("--live", action="store_true", help="Actually call the Anthropic API. Omit for a free dry run.")
    args = parser.parse_args()

    num_prompts = max(1, min(args.num_prompts, len(VERIFICATION_PROMPTS)))
    prompts = VERIFICATION_PROMPTS[:num_prompts]

    try:
        config = load_app_config()
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    model_entry = config.models.get_model(args.provider, args.model)

    git_sha = get_git_sha(cwd=str(REPO_ROOT))
    git_dirty = get_git_dirty(cwd=str(REPO_ROOT))
    metadata = new_run_metadata(
        config_hash=config.config_hash,
        pricing_config_version=config.pricing.pricing_config_version,
        git_sha=git_sha,
        git_dirty=git_dirty,
    )
    run_dir = create_run_dir(REPO_ROOT / "runs", metadata.run_id)
    configure_logging(level=config.secrets.router_log_level, run_dir=run_dir)
    logger = get_logger("verify_stage0")

    if args.live:
        api_key = config.require_api_key()
        client = AnthropicModelClient(api_key=api_key, base_url=config.secrets.anthropic_base_url)
        print(f"LIVE MODE: will make up to {num_prompts} real Anthropic API call(s) against {model_entry.id}.")
    else:
        client = None
        print(f"DRY RUN: no API calls will be made. Pass --live to call {model_entry.id} for real.")

    cache = ContentCache(REPO_ROOT / "data" / "cache")
    store = JsonlStore(run_dir / "records.jsonl")

    real_calls_made = 0
    cache_hits = 0
    total_cost = 0.0

    for prompt_index, prompt in enumerate(prompts):
        prompt_id = f"verify-{prompt_index}"
        prompt_hash = hash_text(prompt)

        # Call each prompt twice with identical conditions to prove the
        # second call is served from cache, not the network.
        for call_number in range(2):
            gen_config = build_generation_config(model_entry, system_prompt=None)
            cache_key = compute_cache_key(
                provider=args.provider,
                model_id=model_entry.id,
                prompt_hash=prompt_hash,
                generation_config=gen_config,
                sample_index=0,
            )

            def compute(prompt: str = prompt, gen_config: GenerationConfig = gen_config) -> NormalizedCompletion:
                nonlocal real_calls_made
                real_calls_made += 1
                if client is not None:
                    return client.complete(prompt, model_entry, sample_index=0)
                return make_fake_completion(gen_config)

            completion, cache_hit = cache.get_or_compute(cache_key, compute)
            if cache_hit:
                cache_hits += 1

            cost = calculate_cost(
                usage=completion.usage,
                model_id=model_entry.id,
                pricing=config.pricing,
                timestamp=completion.timestamp_utc,
            )
            total_cost += cost.total_cost

            record = ResponseRecord(
                record_id=cache_key,
                run_id=metadata.run_id,
                prompt_id=prompt_id,
                prompt_hash=prompt_hash,
                prompt_text=prompt,
                provider=args.provider,
                requested_model_id=model_entry.id,
                served_model_id=completion.served_model_id,
                sample_index=0,
                generation_config=completion.generation_config,
                status=completion.status.value,
                stop_reason=completion.stop_reason,
                truncated=completion.truncated,
                response_text=completion.text,
                usage=completion.usage,
                cost=cost,
                latency_ms=completion.latency_ms,
                timestamp_utc=completion.timestamp_utc,
                request_id=completion.request_id,
                pricing_config_version=config.pricing.pricing_config_version,
                git_sha=git_sha,
                cache_hit=cache_hit,
                retries=completion.retries,
                error_type=completion.error_type,
                error_message=completion.error_message,
            )
            store.append(record)

            logger.info(
                "recorded completion",
                extra={
                    "prompt_id": prompt_id,
                    "call_number": call_number,
                    "cache_hit": cache_hit,
                    "status": completion.status.value,
                    "cost_usd": cost.total_cost,
                },
            )

    metadata.utc_end = datetime.now(UTC)
    write_run_metadata(run_dir, metadata)

    print()
    print("=== Stage 0 verification summary ===")
    print(f"run_dir:          {run_dir}")
    print(f"mode:             {'LIVE' if args.live else 'DRY RUN'}")
    print(f"model:            {model_entry.id}")
    print(f"prompts:          {len(prompts)}")
    print(f"records written:  {len(prompts) * 2}")
    print(f"cache hits:       {cache_hits} (expected {len(prompts)})")
    print(f"compute() calls:  {real_calls_made} (expected {len(prompts)}; proves cache avoided duplicate calls)")
    print(f"total cost:       {total_cost:.6f} {config.pricing.currency}")
    print(f"pricing version:  {config.pricing.pricing_config_version}")

    if cache_hits != len(prompts):
        print("WARNING: cache hit count did not match expectations.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
