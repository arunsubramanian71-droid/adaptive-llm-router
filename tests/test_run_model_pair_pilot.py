from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_model_pair_pilot as pilot_mod

from router.dataset.schemas import DatasetItem, TaskType
from router.models.anthropic_adapter import build_generation_config
from router.models.schemas import CompletionStatus, NormalizedCompletion, TokenUsage
from router.storage.records import JsonlStore

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_model_pair_pilot.py"
REAL_BENCHMARK_PATH = REPO_ROOT / "data" / "benchmarks" / "pilot_v1.jsonl"


def _make_items(specs: list[tuple[str, str]]) -> list[DatasetItem]:
    """specs: list of (prompt_id, task_category)."""
    return [
        DatasetItem(
            prompt_id=pid,
            task_type=TaskType.EXACT_MATCH,
            prompt=f"prompt for {pid}",
            reference_answer="x",
            metadata={"task_category": category},
        )
        for pid, category in specs
    ]


# ---------------------------------------------------------------------------
# select_stratified_prompts
# ---------------------------------------------------------------------------


def test_select_stratified_prompts_even_split():
    items = _make_items(
        [(f"a{i}", "cat_a") for i in range(10)]
        + [(f"b{i}", "cat_b") for i in range(10)]
    )
    selected = pilot_mod.select_stratified_prompts(items, num_prompts=6, seed=1)
    counts: dict[str, int] = {}
    for it in selected:
        counts[it.metadata["task_category"]] = counts.get(it.metadata["task_category"], 0) + 1
    assert counts == {"cat_a": 3, "cat_b": 3}
    assert len(selected) == 6


def test_select_stratified_prompts_deterministic_with_seed():
    items = _make_items([(f"p{i}", "cat") for i in range(20)])
    a = [it.prompt_id for it in pilot_mod.select_stratified_prompts(items, 8, seed=42)]
    b = [it.prompt_id for it in pilot_mod.select_stratified_prompts(items, 8, seed=42)]
    assert a == b


def test_select_stratified_prompts_different_seeds_can_differ():
    items = _make_items([(f"p{i}", "cat") for i in range(50)])
    a = [it.prompt_id for it in pilot_mod.select_stratified_prompts(items, 10, seed=1)]
    b = [it.prompt_id for it in pilot_mod.select_stratified_prompts(items, 10, seed=2)]
    assert a != b


def test_select_stratified_prompts_result_sorted_by_id():
    items = _make_items([("z1", "cat_a"), ("a1", "cat_a"), ("m1", "cat_b")])
    selected = pilot_mod.select_stratified_prompts(items, num_prompts=3, seed=1)
    ids = [it.prompt_id for it in selected]
    assert ids == sorted(ids)


def test_select_stratified_prompts_caps_at_available_pool():
    items = _make_items([("only1", "cat_a")])
    selected = pilot_mod.select_stratified_prompts(items, num_prompts=10, seed=1)
    assert len(selected) == 1


def test_select_stratified_prompts_real_benchmark_matches_first_pilot_spec():
    items = pilot_mod.load_dataset(REAL_BENCHMARK_PATH)
    selected = pilot_mod.select_stratified_prompts(items, pilot_mod.DEFAULT_NUM_PROMPTS, pilot_mod.DEFAULT_SEED)
    assert len(selected) == 20
    counts: dict[str, int] = {}
    for it in selected:
        counts[it.metadata["task_category"]] = counts.get(it.metadata["task_category"], 0) + 1
    assert counts == {
        "coding": 4,
        "factual_knowledge": 4,
        "logical_reasoning": 4,
        "mathematical_reasoning": 4,
        "structured_extraction_constraint_following": 4,
    }
    # Pinned golden output — same seed + same committed benchmark must always
    # reproduce this exact prompt set. Guards against silent nondeterminism.
    assert [it.prompt_id for it in selected] == [
        "bbh-date_understanding-202",
        "bbh-date_understanding-94",
        "bbh-object_counting-222",
        "bbh-object_counting-248",
        "gsm8k-test-1191",
        "gsm8k-test-146",
        "gsm8k-test-663",
        "gsm8k-test-823",
        "humaneval-HumanEval-131",
        "humaneval-HumanEval-152",
        "humaneval-HumanEval-64",
        "humaneval-HumanEval-69",
        "ifeval-1531",
        "ifeval-1999",
        "ifeval-3323",
        "ifeval-3380",
        "truthfulqa-mc1-322",
        "truthfulqa-mc1-511",
        "truthfulqa-mc1-675",
        "truthfulqa-mc1-677",
    ]


# ---------------------------------------------------------------------------
# expected_request_count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k,expected", [(1, 40), (3, 120), (5, 200), (6, 240)])
def test_expected_request_count_first_pilot_spec(k, expected):
    assert pilot_mod.expected_request_count(20, k) == expected


def test_expected_request_count_generic():
    assert pilot_mod.expected_request_count(num_prompts=7, k=2, num_models=3) == 42


# ---------------------------------------------------------------------------
# load_benchmark_version
# ---------------------------------------------------------------------------


def test_load_benchmark_version_real_benchmark():
    assert pilot_mod.load_benchmark_version(REAL_BENCHMARK_PATH) == "pilot_v1"


def test_load_benchmark_version_missing_manifest(tmp_path: Path):
    fake = tmp_path / "nomanifest.jsonl"
    fake.write_text("", encoding="utf-8")
    assert pilot_mod.load_benchmark_version(fake) is None


# ---------------------------------------------------------------------------
# validate_args / safety limits
# ---------------------------------------------------------------------------


def _args(**overrides) -> argparse.Namespace:
    base = {
        "benchmark_path": REAL_BENCHMARK_PATH,
        "num_prompts": 20,
        "k": 1,
        "max_requests": pilot_mod.DEFAULT_MAX_REQUESTS,
        "seed": pilot_mod.DEFAULT_SEED,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_validate_args_accepts_defaults():
    assert pilot_mod.validate_args(_args()) == []


def test_validate_args_rejects_non_positive_num_prompts():
    assert any("--num-prompts" in e for e in pilot_mod.validate_args(_args(num_prompts=0)))


def test_validate_args_rejects_non_positive_k():
    assert any("--k" in e for e in pilot_mod.validate_args(_args(k=0)))


def test_validate_args_rejects_non_positive_max_requests():
    assert any("--max-requests" in e for e in pilot_mod.validate_args(_args(max_requests=0)))


def test_validate_args_rejects_max_requests_above_hard_ceiling():
    errors = pilot_mod.validate_args(_args(max_requests=pilot_mod.ABSOLUTE_MAX_REQUESTS + 1))
    assert any("hard safety ceiling" in e for e in errors)


def test_validate_args_accepts_max_requests_at_hard_ceiling():
    errors = pilot_mod.validate_args(_args(max_requests=pilot_mod.ABSOLUTE_MAX_REQUESTS))
    assert errors == []


def test_validate_args_rejects_missing_benchmark_file(tmp_path: Path):
    errors = pilot_mod.validate_args(_args(benchmark_path=tmp_path / "nope.jsonl"))
    assert any("benchmark file not found" in e for e in errors)


# ---------------------------------------------------------------------------
# run_pilot: mocked generation + cache behavior + persistence
# ---------------------------------------------------------------------------


class FakeModelClient:
    """Duck-typed stand-in for AnthropicModelClient — no network, records
    every call it receives so tests can assert on call counts per model."""

    def __init__(self, force_error_for_model_id: str | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self._force_error_for_model_id = force_error_for_model_id

    def complete(self, prompt, model_entry, sample_index, system_prompt=None) -> NormalizedCompletion:
        self.calls.append((model_entry.id, sample_index))
        gen_config = build_generation_config(model_entry, system_prompt)
        if model_entry.id == self._force_error_for_model_id:
            return NormalizedCompletion(
                provider="anthropic",
                requested_model_id=model_entry.id,
                usage=TokenUsage(),
                timestamp_utc=datetime.now(UTC),
                generation_config=gen_config,
                status=CompletionStatus.ERROR,
                error_type="RateLimitError",
                error_message="synthetic test error",
            )
        return NormalizedCompletion(
            provider="anthropic",
            requested_model_id=model_entry.id,
            served_model_id=model_entry.id,
            text=f"fake response from {model_entry.id} sample {sample_index}",
            request_id=f"req-{model_entry.id}-{sample_index}",
            usage=TokenUsage(input_tokens=20, output_tokens=10),
            latency_ms=12.0,
            timestamp_utc=datetime.now(UTC),
            generation_config=gen_config,
            status=CompletionStatus.OK,
            stop_reason="end_turn",
            truncated=False,
            retries=0,
        )


def _fake_config(pricing_config):
    return SimpleNamespace(pricing=pricing_config)


def test_run_pilot_calls_both_models_and_writes_records(tmp_path: Path, haiku_entry, opus_entry, pricing_config):
    items = _make_items([("p1", "cat"), ("p2", "cat")])
    client = FakeModelClient()
    run_dir = tmp_path / "run1"
    run_dir.mkdir()

    summary = pilot_mod.run_pilot(
        config=_fake_config(pricing_config),
        items=items,
        model_a=pilot_mod.ModelTarget(provider="anthropic", entry=haiku_entry, client=client),
        model_b=pilot_mod.ModelTarget(provider="anthropic", entry=opus_entry, client=client),
        k=1,
        run_id="test-run-1",
        run_dir=run_dir,
        cache_dir=tmp_path / "cache",
        git_sha="deadbeef",
    )

    assert summary["n_records"] == 4  # 2 items x 2 models x k=1
    assert summary["n_calls_made"] == 4
    assert summary["n_cache_hits"] == 0
    assert summary["n_errors"] == 0
    assert {c[0] for c in client.calls} == {"claude-haiku-4-5", "claude-opus-5"}

    records = JsonlStore(run_dir / "records.jsonl").read_all()
    assert len(records) == 4
    assert {r.requested_model_id for r in records} == {"claude-haiku-4-5", "claude-opus-5"}
    assert all(r.run_id == "test-run-1" for r in records)
    assert all(r.cost is not None and r.cost.total_cost > 0 for r in records)


def test_run_pilot_second_run_hits_cache_and_makes_no_calls(tmp_path: Path, haiku_entry, opus_entry, pricing_config):
    items = _make_items([("p1", "cat"), ("p2", "cat")])
    cache_dir = tmp_path / "cache"

    run_dir_1 = tmp_path / "run1"
    run_dir_1.mkdir()
    client_1 = FakeModelClient()
    summary_1 = pilot_mod.run_pilot(
        config=_fake_config(pricing_config),
        items=items,
        model_a=pilot_mod.ModelTarget(provider="anthropic", entry=haiku_entry, client=client_1),
        model_b=pilot_mod.ModelTarget(provider="anthropic", entry=opus_entry, client=client_1),
        k=1,
        run_id="run-1",
        run_dir=run_dir_1,
        cache_dir=cache_dir,
        git_sha=None,
    )
    assert summary_1["n_calls_made"] == 4

    run_dir_2 = tmp_path / "run2"
    run_dir_2.mkdir()
    client_2 = FakeModelClient()  # fresh client -- if called, calls list would be non-empty
    summary_2 = pilot_mod.run_pilot(
        config=_fake_config(pricing_config),
        items=items,
        model_a=pilot_mod.ModelTarget(provider="anthropic", entry=haiku_entry, client=client_2),
        model_b=pilot_mod.ModelTarget(provider="anthropic", entry=opus_entry, client=client_2),
        k=1,
        run_id="run-2",
        run_dir=run_dir_2,
        cache_dir=cache_dir,  # same cache dir as run 1
        git_sha=None,
    )
    assert summary_2["n_calls_made"] == 0
    assert summary_2["n_cache_hits"] == 4
    assert client_2.calls == []  # the fake client's complete() was never invoked


def test_run_pilot_records_errors_without_caching_them(tmp_path: Path, haiku_entry, opus_entry, pricing_config):
    items = _make_items([("p1", "cat")])
    client = FakeModelClient(force_error_for_model_id="claude-opus-5")
    run_dir = tmp_path / "run1"
    run_dir.mkdir()

    summary = pilot_mod.run_pilot(
        config=_fake_config(pricing_config),
        items=items,
        model_a=pilot_mod.ModelTarget(provider="anthropic", entry=haiku_entry, client=client),
        model_b=pilot_mod.ModelTarget(provider="anthropic", entry=opus_entry, client=client),
        k=1,
        run_id="run-err",
        run_dir=run_dir,
        cache_dir=tmp_path / "cache",
        git_sha=None,
    )
    assert summary["n_errors"] == 1

    records = {r.requested_model_id: r for r in JsonlStore(run_dir / "records.jsonl").read_all()}
    assert records["claude-opus-5"].status == "error"
    assert records["claude-opus-5"].error_type == "RateLimitError"
    assert records["claude-haiku-4-5"].status == "ok"

    # Errored calls must not be cached -- a second run must retry them.
    run_dir_2 = tmp_path / "run2"
    run_dir_2.mkdir()
    client_2 = FakeModelClient(force_error_for_model_id="claude-opus-5")
    summary_2 = pilot_mod.run_pilot(
        config=_fake_config(pricing_config),
        items=items,
        model_a=pilot_mod.ModelTarget(provider="anthropic", entry=haiku_entry, client=client_2),
        model_b=pilot_mod.ModelTarget(provider="anthropic", entry=opus_entry, client=client_2),
        k=1,
        run_id="run-err-2",
        run_dir=run_dir_2,
        cache_dir=tmp_path / "cache",
        git_sha=None,
    )
    assert summary_2["n_calls_made"] == 1  # haiku's success was cached; opus's error was retried
    assert summary_2["n_cache_hits"] == 1


def test_run_pilot_never_imports_unsandboxed_code_evaluator():
    # UnsandboxedSubprocessCodeEvalEvaluator must not appear anywhere in this
    # module's namespace -- this script only collects responses, never scores
    # or executes them.
    assert "UnsandboxedSubprocessCodeEvalEvaluator" not in dir(pilot_mod)
    assert not hasattr(pilot_mod, "evaluation")


# ---------------------------------------------------------------------------
# build_pilot_config
# ---------------------------------------------------------------------------


def test_build_pilot_config_contents():
    items = _make_items([("p1", "cat_a"), ("p2", "cat_b")])
    config = pilot_mod.build_pilot_config(
        benchmark_path=REAL_BENCHMARK_PATH,
        benchmark_version="pilot_v1",
        provider="anthropic",
        model_a_id="claude-haiku-4-5",
        model_b_id="claude-opus-5",
        num_prompts_requested=20,
        selected=items,
        k=1,
        seed=42,
        category_counts={"cat_a": 1, "cat_b": 1},
        max_requests=100,
        expected_requests=4,
    )
    assert config["model_a"] == "claude-haiku-4-5"
    assert config["model_b"] == "claude-opus-5"
    assert config["num_prompts_selected"] == 2
    assert config["selected_prompt_ids"] == ["p1", "p2"]
    assert config["expected_request_count"] == 4
    assert config["benchmark_version"] == "pilot_v1"


# ---------------------------------------------------------------------------
# CLI-level integration (subprocess, dry-run only -- no network, no key)
# ---------------------------------------------------------------------------


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_cli_default_dry_run_succeeds_and_reports_40_requests():
    result = _run_cli()
    assert result.returncode == 0
    assert "DRY RUN" in result.stdout
    assert "= 40" in result.stdout
    assert "Zero network calls were made" in result.stdout


def test_cli_refuses_when_exceeding_max_requests():
    result = _run_cli("--num-prompts", "100", "--max-requests", "50")
    assert result.returncode == 2
    assert "REFUSING TO RUN" in result.stderr


def test_cli_rejects_max_requests_above_hard_ceiling():
    result = _run_cli("--max-requests", str(pilot_mod.ABSOLUTE_MAX_REQUESTS + 1))
    assert result.returncode == 1
    assert "hard safety ceiling" in result.stderr


def test_cli_rejects_missing_benchmark_file():
    result = _run_cli("--benchmark-path", "data/benchmarks/does-not-exist.jsonl")
    assert result.returncode == 1
    assert "benchmark file not found" in result.stderr


def test_cli_never_creates_a_run_dir_without_live(tmp_path: Path):
    runs_root = tmp_path / "runs"
    result = _run_cli("--runs-root", str(runs_root))
    assert result.returncode == 0
    assert not runs_root.exists()


def test_cli_dry_run_does_not_require_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = _run_cli()
    assert result.returncode == 0
    assert "api" not in result.stderr.lower()
