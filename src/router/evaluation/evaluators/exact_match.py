from __future__ import annotations

import re
from typing import cast

from router.dataset.schemas import DatasetItem, TaskType
from router.evaluation.evaluators.base import Evaluator
from router.evaluation.schemas import EvalResult

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = text.strip(".,!?;:'\"")
    return _WHITESPACE_RE.sub(" ", text)


class ExactMatchEvaluator(Evaluator):
    name = "exact_match"
    task_type = TaskType.EXACT_MATCH

    def evaluate(self, item: DatasetItem, record_id: str, response_text: str | None) -> EvalResult:
        if not response_text:
            return self._empty_response_result(item, record_id)

        candidates = item.reference_answer
        if isinstance(candidates, str):
            candidates = [candidates]
        # dataset/loader.py validates reference_answer is str | list[str] for
        # this task_type before an ExactMatchEvaluator ever sees it.
        candidates = cast("list[str]", candidates)

        normalized_response = _normalize(response_text)
        normalized_candidates = [_normalize(c) for c in candidates]
        matched = normalized_response in normalized_candidates

        return EvalResult(
            record_id=record_id,
            prompt_id=item.prompt_id,
            evaluator_name=self.name,
            score=1.0 if matched else 0.0,
            passed=matched,
            details={
                "normalized_response": normalized_response,
                "normalized_candidates": normalized_candidates,
            },
        )
