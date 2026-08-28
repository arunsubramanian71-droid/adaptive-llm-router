#!/usr/bin/env python
"""Model-pair pilot runner.

Runs a controlled, reproducible subset of a benchmark (default:
data/benchmarks/pilot_v1.jsonl) through two candidate models and persists
every response via Stage 0's existing ResponseRecord/JsonlStore/
ContentCache/cost-calculator machinery. This script only collects
responses — it never scores them. Evaluation (router.evaluation.*) is a
separate, later stage; in particular this script never imports or runs
UnsandboxedSubprocessCodeEvalEvaluator or any other evaluator.

SAFETY: without --live, this script makes zero network calls of any kind
and never touches an API key. It always prints the exact number of
requests a live run would make, and refuses to proceed (dry run or live)
if that number exceeds --max-requests (itself capped at a hard ceiling,
ABSOLUTE_MAX_REQUESTS, that no CLI flag can raise).

First pilot (see project instructions): 20 prompts, stratified evenly
across the benchmark's task categories, seed=20260825, k=1, both
candidate models -> 20 x 2 x 1 = 40 requests.

Example (dry run — always safe, no key needed):

    python scripts/run_model_pair_pilot.py

Example (live — spends money, requires ANTHROPIC_API_KEY):

    python scripts/run_model_pair_pilot.py --live
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from router.config import AppConfig, ConfigError, ModelEntry, load_app_config
from router.cost.calculator import calculate_cost
from router.dataset.loader import DatasetError, load_dataset
from router.dataset.schemas import DatasetItem
from router.git_utils import get_git_dirty, get_git_sha
from router.hashing import hash_text
from router.logging import configure_logging, get_logger
from router.models.anthropic_adapter import (
    build_generation_config as build_anthropic_generation_config,
)
from router.models.google_gemini_adapter import (
    build_generation_config as build_google_generation_config,
)
from router.models.openai_adapter import build_generation_config as build_openai_generation_config
from router.models.registry import build_model_client
from router.models.schemas import CompletionStatus, GenerationConfig, NormalizedCompletion
from router.storage.cache import ContentCache, compute_cache_key
from router.storage.records import JsonlStore, ResponseRecord
from router.storage.run_metadata import (
    create_run_dir,
    new_run_metadata,
    write_run_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_PATH = REPO_ROOT / "data" / "benchmarks" / "pilot_v1.jsonl"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"

DEFAULT_NUM_PROMPTS = 20
DEFAULT_K = 1
# Pilot prompt-selection seed. Deliberately distinct from the benchmark's own
# build seed (1337, see build_pilot_benchmark.py) -- this one controls which
# subset of the already-fixed 200-item pilot gets run, not the benchmark itself.
DEFAULT_SEED = 20260825
DEFAULT_MAX_REQUESTS = 100
# Hard ceiling: no --max-requests value, however large, can exceed this.
ABSOLUTE_MAX_REQUESTS = 500
NUM_MODELS_IN_PAIR = 2


class ModelClientProtocol(Protocol):
    def complete(
        self, prompt: str, model_entry: ModelEntry, sample_index: int, system_prompt: str | None = None
    ) -> NormalizedCompletion: ...


_GENERATION_CONFIG_BUILDERS = {
    "anthropic": build_anthropic_generation_config,
    "openai": build_openai_generation_config,
    "google": build_google_generation_config,
}


def build_generation_config_for_provider(
    provider: str, model_entry: ModelEntry, system_prompt: str | None
) -> GenerationConfig:
    """Each adapter's build_generation_config() hardcodes its own provider
    name into the result -- dispatch to the right one so a model's
    GenerationConfig.provider always matches which adapter actually served
    it, even when model_a and model_b are on different providers."""
    builder = _GENERATION_CONFIG_BUILDERS.get(provider)
    if builder is None:
        raise ConfigError(f"no generation-config builder registered for provider {provider!r}")
    return builder(model_entry, system_prompt)


class ModelTarget(NamedTuple):
    """One side of the pair: which provider, which configured model, and
    the client that actually knows how to call that provider. Grouping
    these together (rather than three parallel arguments) makes it
    impossible to accidentally pair model A's entry with model B's client
    -- a real risk once the two models can be on different providers."""

    provider: str
    entry: ModelEntry
    client: ModelClientProtocol


# ---------------------------------------------------------------------------
# Pure, independently-testable pieces
# ---------------------------------------------------------------------------


def select_stratified_prompts(items: list[DatasetItem], num_prompts: int, seed: int) -> list[DatasetItem]:
    """Deterministic stratified sample: as-even-as-possible per task_category,
    seeded shuffle within each category (sorted input first, for
    reproducibility regardless of the input list's original order), sorted
    by prompt_id in the result."""
    by_category: dict[str, list[DatasetItem]] = defaultdict(list)
    for item in items:
        by_category[item.metadata.get("task_category", "UNKNOWN")].append(item)

    categories = sorted(by_category)
    if not categories:
        return []

    base, remainder = divmod(num_prompts, len(categories))
    targets = {cat: base + (1 if i < remainder else 0) for i, cat in enumerate(categories)}

    selected: list[DatasetItem] = []
    for cat in categories:
        pool = sorted(by_category[cat], key=lambda it: it.prompt_id)
        random.Random(seed).shuffle(pool)
        selected.extend(pool[: targets[cat]])

    selected.sort(key=lambda it: it.prompt_id)
    return selected


def expected_request_count(num_prompts: int, k: int, num_models: int = NUM_MODELS_IN_PAIR) -> int:
    return num_prompts * num_models * k


def load_benchmark_version(benchmark_path: Path) -> str | None:
    manifest_path = benchmark_path.with_name(f"{benchmark_path.stem}.manifest.json")
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    version = data.get("benchmark_version")
    return version if isinstance(version, str) else None


def build_pilot_config(
    *,
    benchmark_path: Path,
    benchmark_version: str | None,
    provider: str,
    model_a_id: str,
    model_b_id: str,
    num_prompts_requested: int,
    selected: list[DatasetItem],
    k: int,
    seed: int,
    category_counts: dict[str, int],
    max_requests: int,
    expected_requests: int,
) -> dict[str, Any]:
    return {
        "benchmark_path": str(benchmark_path),
        "benchmark_version": benchmark_version,
        "provider": provider,
        "model_a": model_a_id,
        "model_b": model_b_id,
        "num_prompts_requested": num_prompts_requested,
        "num_prompts_selected": len(selected),
        "k": k,
        "seed": seed,
        "category_counts": dict(sorted(category_counts.items())),
        "selected_prompt_ids": [it.prompt_id for it in selected],
        "max_requests": max_requests,
        "expected_request_count": expected_requests,
    }


def validate_args(args: argparse.Namespace) -> list[str]:
    errors = []
    if args.num_prompts < 1:
        errors.append("--num-prompts must be >= 1")
    if args.k < 1:
        errors.append("--k must be >= 1")
    if args.max_requests < 1:
        errors.append("--max-requests must be >= 1")
    if args.max_requests > ABSOLUTE_MAX_REQUESTS:
        errors.append(f"--max-requests cannot exceed the hard safety ceiling of {ABSOLUTE_MAX_REQUESTS}")
    if not args.benchmark_path.exists():
        errors.append(f"benchmark file not found: {args.benchmark_path}")
    return errors


# ---------------------------------------------------------------------------
# Live execution (only ever called after the safety gate passes and --live
# was passed explicitly)
# ---------------------------------------------------------------------------


def run_pilot(
    *,
    config: AppConfig,
    items: list[DatasetItem],
    model_a: ModelTarget,
    model_b: ModelTarget,
    k: int,
    run_id: str,
    run_dir: Path,
    cache_dir: Path,
    git_sha: str | None,
    logger: Any = None,
) -> dict[str, int]:
    """Runs both models over every item. model_a and model_b may be on the
    same provider (the common case today) or different ones -- each
    ModelTarget carries its own client, so this loop never assumes a
    shared provider."""
    cache = ContentCache(cache_dir)
    store = JsonlStore(run_dir / "records.jsonl")

    n_calls_made = 0
    n_cache_hits = 0
    n_errors = 0

    for item in items:
        for target in (model_a, model_b):
            provider, model_entry, client = target.provider, target.entry, target.client
            for sample_index in range(k):
                gen_config = build_generation_config_for_provider(provider, model_entry, item.system_prompt)
                prompt_hash = hash_text(item.prompt)
                cache_key = compute_cache_key(
                    provider=provider,
                    model_id=model_entry.id,
                    prompt_hash=prompt_hash,
                    generation_config=gen_config,
                    sample_index=sample_index,
                )

                def compute(
                    item: DatasetItem = item,
                    model_entry: ModelEntry = model_entry,
                    sample_index: int = sample_index,
                    client: ModelClientProtocol = client,
                ) -> NormalizedCompletion:
                    nonlocal n_calls_made
                    n_calls_made += 1
                    return client.complete(item.prompt, model_entry, sample_index, system_prompt=item.system_prompt)

                completion, cache_hit = cache.get_or_compute(cache_key, compute)
                if cache_hit:
                    n_cache_hits += 1
                if completion.status != CompletionStatus.OK:
                    n_errors += 1

                cost = calculate_cost(
                    usage=completion.usage,
                    model_id=model_entry.id,
                    pricing=config.pricing,
                    timestamp=completion.timestamp_utc,
                )

                store.append(
                    ResponseRecord(
                        record_id=cache_key,
                        run_id=run_id,
                        prompt_id=item.prompt_id,
                        prompt_hash=prompt_hash,
                        prompt_text=item.prompt,
                        provider=provider,
                        requested_model_id=model_entry.id,
                        served_model_id=completion.served_model_id,
                        sample_index=sample_index,
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
                )

                if logger is not None:
                    logger.info(
                        "recorded completion",
                        extra={
                            "prompt_id": item.prompt_id,
                            "model_id": model_entry.id,
                            "sample_index": sample_index,
                            "cache_hit": cache_hit,
                            "status": completion.status.value,
                        },
                    )

    return {
        "n_records": len(items) * NUM_MODELS_IN_PAIR * k,
        "n_calls_made": n_calls_made,
        "n_cache_hits": n_cache_hits,
        "n_errors": n_errors,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model-a", default="claude-haiku-4-5", help="cheap candidate (default: primary pilot pair)")
    parser.add_argument("--model-b", default="claude-opus-5", help="strong candidate (default: primary pilot pair)")
    parser.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS)
    parser.add_argument("--live", action="store_true", help="Make real API calls. Omit for a zero-network dry run.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    errors = validate_args(args)
    if errors:
        for err in errors:
            print(f"error: {err}", file=sys.stderr)
        return 1

    try:
        items = load_dataset(args.benchmark_path)
    except DatasetError as exc:
        print(f"error loading benchmark: {exc}", file=sys.stderr)
        return 1

    selected = select_stratified_prompts(items, args.num_prompts, args.seed)
    category_counts = Counter(it.metadata.get("task_category", "UNKNOWN") for it in selected)
    n_requests = expected_request_count(len(selected), args.k)

    try:
        config = load_app_config()
        model_a_entry = config.models.get_model(args.provider, args.model_a)
        model_b_entry = config.models.get_model(args.provider, args.model_b)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    git_sha = get_git_sha(cwd=str(REPO_ROOT))
    git_dirty = get_git_dirty(cwd=str(REPO_ROOT))
    benchmark_version = load_benchmark_version(args.benchmark_path)

    metadata = new_run_metadata(
        config_hash=config.config_hash,
        pricing_config_version=config.pricing.pricing_config_version,
        git_sha=git_sha,
        git_dirty=git_dirty,
        seed=args.seed,
    )
    expected_run_dir = args.runs_root / metadata.run_id

    print("=== Model-pair pilot runner ===")
    print(f"mode:               {'LIVE' if args.live else 'DRY RUN (zero network calls)'}")
    print(f"benchmark:          {args.benchmark_path}")
    print(f"benchmark_version:  {benchmark_version or 'unknown'}")
    print(f"provider:           {args.provider}")
    print(f"model A (cheap):    {model_a_entry.id}")
    print(f"model B (strong):   {model_b_entry.id}")
    print(f"num_prompts:        {len(selected)} selected (requested {args.num_prompts})")
    print(f"k (samples/model):  {args.k}")
    print(f"seed:               {args.seed}")
    print(f"category counts:    {dict(sorted(category_counts.items()))}")
    print(f"prompt IDs:         {[it.prompt_id for it in selected]}")
    print(
        f"expected requests:  {len(selected)} prompts x {NUM_MODELS_IN_PAIR} models x k={args.k} = {n_requests}"
    )
    print(f"max_requests limit: {args.max_requests} (hard ceiling {ABSOLUTE_MAX_REQUESTS})")
    print(f"expected run dir:   {expected_run_dir}")
    print(f"git_sha:            {git_sha or 'no-git'}")
    print(f"config_hash:        {config.config_hash}")
    print(f"pricing_version:    {config.pricing.pricing_config_version}")

    if n_requests > args.max_requests:
        print(
            f"\nREFUSING TO RUN: expected request count {n_requests} exceeds --max-requests {args.max_requests}",
            file=sys.stderr,
        )
        return 2

    if not args.live:
        print("\nDry run complete. Zero network calls were made. Pass --live to execute for real.")
        return 0

    api_key = config.require_api_key(args.provider)  # never logged or printed below
    base_url = {"anthropic": config.secrets.anthropic_base_url, "openai": config.secrets.openai_base_url}.get(
        args.provider
    )
    # model_a and model_b share --provider today (no CLI flag exists yet for
    # per-model providers), so one client currently serves both -- but
    # run_pilot() itself (see ModelTarget) does not assume that; it accepts
    # an independent client per model, so a future per-model CLI flag would
    # only need to change this construction, not run_pilot().
    client = build_model_client(config, args.provider, api_key=api_key, base_url=base_url)

    run_dir = create_run_dir(args.runs_root, metadata.run_id)
    configure_logging(level=config.secrets.router_log_level, run_dir=run_dir)
    logger = get_logger("run_model_pair_pilot")

    pilot_config = build_pilot_config(
        benchmark_path=args.benchmark_path,
        benchmark_version=benchmark_version,
        provider=args.provider,
        model_a_id=model_a_entry.id,
        model_b_id=model_b_entry.id,
        num_prompts_requested=args.num_prompts,
        selected=selected,
        k=args.k,
        seed=args.seed,
        category_counts=category_counts,
        max_requests=args.max_requests,
        expected_requests=n_requests,
    )
    (run_dir / "pilot_config.json").write_text(json.dumps(pilot_config, indent=2), encoding="utf-8")

    summary = run_pilot(
        config=config,
        items=selected,
        model_a=ModelTarget(provider=args.provider, entry=model_a_entry, client=client),
        model_b=ModelTarget(provider=args.provider, entry=model_b_entry, client=client),
        k=args.k,
        run_id=metadata.run_id,
        run_dir=run_dir,
        cache_dir=args.cache_dir,
        git_sha=git_sha,
        logger=logger,
    )

    metadata.utc_end = datetime.now(UTC)
    write_run_metadata(run_dir, metadata)

    print("\n=== Pilot run summary ===")
    print(f"run_dir:          {run_dir}")
    print(f"records written:  {summary['n_records']}")
    print(f"actual API calls: {summary['n_calls_made']}")
    print(f"cache hits:       {summary['n_cache_hits']}")
    print(f"errors:           {summary['n_errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
