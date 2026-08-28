"""Structured-extraction evaluator.

`item.reference_answer` is a flat `dict[str, Any]` of expected field ->
value. The response is expected to contain a JSON object (optionally
wrapped in prose or a markdown code fence); each expected field is compared
independently, so a partially-correct extraction gets partial credit
instead of an all-or-nothing score.
"""

from __future__ import annotations

import json
import re
from typing import cast

from router.dataset.schemas import DatasetItem, TaskType
from router.evaluation.evaluators.base import Evaluator
from router.evaluation.schemas import EvalResult

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_FIRST_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> dict | None:
    for candidate in (
        text.strip(),
        (m.group(1) if (m := _JSON_FENCE_RE.search(text)) else None),
        (m.group(0) if (m := _FIRST_OBJECT_RE.search(text)) else None),
    ):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _values_match(expected: object, actual: object) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    return expected == actual


class StructuredExtractionEvaluator(Evaluator):
    name = "structured_extraction"
    task_type = TaskType.STRUCTURED_EXTRACTION

    def evaluate(self, item: DatasetItem, record_id: str, response_text: str | None) -> EvalResult:
        if not response_text:
            return self._empty_response_result(item, record_id)

        # dataset/loader.py validates reference_answer is a dict for this task_type.
        expected = cast(dict, item.reference_answer)
        parsed = _extract_json_object(response_text)

        if parsed is None:
            return EvalResult(
                record_id=record_id,
                prompt_id=item.prompt_id,
                evaluator_name=self.name,
                score=0.0,
                passed=False,
                error="response did not contain a parsable JSON object",
                details={"parse_ok": False},
            )

        field_results = {key: _values_match(value, parsed.get(key)) for key, value in expected.items()}
        n_correct = sum(field_results.values())
        n_total = len(expected) or 1
        score = n_correct / n_total

        return EvalResult(
            record_id=record_id,
            prompt_id=item.prompt_id,
            evaluator_name=self.name,
            score=score,
            passed=score == 1.0,
            details={"parse_ok": True, "field_results": field_results, "parsed": parsed},
        )
