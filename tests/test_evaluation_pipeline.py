from __future__ import annotations

from router.dataset.schemas import DatasetItem, TaskType
from router.evaluation.evaluators import default_evaluator_registry
from router.evaluation.pipeline import run_evaluation_pipeline


def test_run_evaluation_pipeline_scores_matching_records(make_response_record):
    items = [
        DatasetItem(prompt_id="p1", task_type=TaskType.EXACT_MATCH, prompt="2+2?", reference_answer="4"),
        DatasetItem(prompt_id="p2", task_type=TaskType.JUDGE_SCORED, prompt="write a poem"),
    ]
    records = [
        make_response_record("r1", "p1", response_text="4"),
        make_response_record("r2", "p1", response_text="5"),
        make_response_record("r3", "p2", response_text="roses are red"),  # judge-scored, skipped here
    ]

    results = run_evaluation_pipeline(items, records, default_evaluator_registry())

    by_record = {r.record_id: r for r in results}
    assert by_record["r1"].score == 1.0
    assert by_record["r2"].score == 0.0
    assert "r3" not in by_record  # judge-scored items aren't handled by objective evaluators


def test_run_evaluation_pipeline_skips_unknown_prompt(make_response_record):
    items = [DatasetItem(prompt_id="p1", task_type=TaskType.EXACT_MATCH, prompt="x", reference_answer="y")]
    records = [make_response_record("r1", "unknown-prompt")]
    results = run_evaluation_pipeline(items, records, default_evaluator_registry())
    assert results == []
