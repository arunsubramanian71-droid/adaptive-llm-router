"""Response/evaluation pipeline.

Ties a dataset to a set of stored `ResponseRecord`s and a registry of
objective `Evaluator`s (one per `TaskType`), producing one `EvalResult` per
`(record_id, evaluator_name)`. Judge-scored items are handled separately by
`router.evaluation.judge.pipeline` (a judge is a model call, not a pure
function) and merged in by the caller if desired.
"""

from __future__ import annotations

from router.dataset.schemas import DatasetItem, TaskType
from router.evaluation.evaluators.base import Evaluator
from router.evaluation.schemas import EvalResult
from router.storage.records import ResponseRecord


def index_dataset(items: list[DatasetItem]) -> dict[str, DatasetItem]:
    return {item.prompt_id: item for item in items}


def evaluate_record(
    item: DatasetItem,
    record: ResponseRecord,
    evaluator_registry: dict[TaskType, Evaluator],
) -> EvalResult | None:
    """Returns None for JUDGE_SCORED items — those go through the judge
    pipeline instead, not an objective evaluator."""
    evaluator = evaluator_registry.get(item.task_type)
    if evaluator is None:
        return None
    return evaluator.evaluate(item, record.record_id, record.response_text)


def run_evaluation_pipeline(
    items: list[DatasetItem],
    records: list[ResponseRecord],
    evaluator_registry: dict[TaskType, Evaluator],
) -> list[EvalResult]:
    items_by_id = index_dataset(items)
    results: list[EvalResult] = []
    for record in records:
        item = items_by_id.get(record.prompt_id)
        if item is None:
            continue
        result = evaluate_record(item, record, evaluator_registry)
        if result is not None:
            results.append(result)
    return results
