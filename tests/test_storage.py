from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from router.cost.schemas import CostBreakdown
from router.models.anthropic_adapter import build_generation_config
from router.models.schemas import TokenUsage
from router.storage.records import JsonlStore, ResponseRecord


def _make_record(haiku_entry, sample_index: int = 0) -> ResponseRecord:
    gen_config = build_generation_config(haiku_entry, system_prompt=None)
    return ResponseRecord(
        record_id=f"key-{sample_index}",
        run_id="run-1",
        prompt_id="prompt-1",
        prompt_hash="hash-1",
        prompt_text="What is 2+2?",
        provider="anthropic",
        requested_model_id=haiku_entry.id,
        served_model_id=haiku_entry.id,
        sample_index=sample_index,
        generation_config=gen_config,
        status="ok",
        stop_reason="end_turn",
        truncated=False,
        response_text="4",
        usage=TokenUsage(input_tokens=10, output_tokens=1),
        cost=CostBreakdown(
            model_id=haiku_entry.id,
            pricing_config_version="test",
            currency="USD",
            input_cost=0.00001,
            output_cost=0.000005,
            cache_write_cost=0.0,
            cache_read_cost=0.0,
            reasoning_cost=0.0,
        ),
        latency_ms=123.4,
        timestamp_utc=datetime.now(UTC),
        request_id="req_test",
        pricing_config_version="test",
        git_sha="deadbeef",
        cache_hit=False,
        retries=0,
    )


def test_jsonl_round_trip(tmp_path: Path, haiku_entry):
    store = JsonlStore(tmp_path / "records.jsonl")
    record = _make_record(haiku_entry)
    store.append(record)

    loaded = store.read_all()
    assert len(loaded) == 1
    assert loaded[0].record_id == record.record_id
    assert loaded[0].usage.input_tokens == 10
    assert loaded[0].cost.total_cost == record.cost.total_cost


def test_jsonl_one_line_per_record_never_aggregated(tmp_path: Path, haiku_entry):
    store = JsonlStore(tmp_path / "records.jsonl")
    for i in range(6):  # up to k=6 samples per ADR-0001
        store.append(_make_record(haiku_entry, sample_index=i))

    loaded = store.read_all()
    assert len(loaded) == 6
    assert [r.sample_index for r in loaded] == list(range(6))


def test_read_all_on_missing_file_returns_empty(tmp_path: Path):
    store = JsonlStore(tmp_path / "does_not_exist" / "records.jsonl")
    assert store.read_all() == []


def test_scoring_placeholders_default_pending(tmp_path: Path, haiku_entry):
    store = JsonlStore(tmp_path / "records.jsonl")
    record = _make_record(haiku_entry)
    store.append(record)
    loaded = store.read_all()[0]
    assert loaded.score_status == "pending"
    assert loaded.q_hat is None
    assert loaded.parse_ok is None
