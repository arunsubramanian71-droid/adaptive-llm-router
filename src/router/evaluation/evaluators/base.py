"""Objective-evaluator interface.

Each evaluator scores one response against a `DatasetItem`'s ground truth
for exactly one `TaskType`, deterministically — no model call, no
randomness. `evaluate()` must never raise for a malformed response (empty
text, unparsable JSON, ...); a malformed response is a score of 0.0 with
`error` set, not an exception, so a bad response never breaks the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from router.dataset.schemas import DatasetItem, TaskType
from router.evaluation.schemas import EvalResult


class Evaluator(ABC):
    name: str
    task_type: TaskType

    @abstractmethod
    def evaluate(self, item: DatasetItem, record_id: str, response_text: str | None) -> EvalResult:
        raise NotImplementedError

    def _empty_response_result(self, item: DatasetItem, record_id: str) -> EvalResult:
        return EvalResult(
            record_id=record_id,
            prompt_id=item.prompt_id,
            evaluator_name=self.name,
            score=0.0,
            passed=False,
            error="empty or missing response text",
        )
