"""Constraint-checking evaluator.

`item.reference_answer` is a list of constraint specs, each a dict with a
`type` key:

    {"type": "contains",     "value": "refund"}
    {"type": "not_contains", "value": "sorry"}
    {"type": "regex",        "pattern": r"^\\d{3}-\\d{4}$"}
    {"type": "max_words",    "value": 50}
    {"type": "min_words",    "value": 5}
    {"type": "max_chars",    "value": 280}

Score is the fraction of constraints satisfied; `passed` requires all of
them. An unrecognized constraint type counts as failed (recorded in
`details`) rather than raising, so one bad spec doesn't crash the pipeline.
"""

from __future__ import annotations

import re
from typing import cast

from router.dataset.schemas import DatasetItem, TaskType
from router.evaluation.evaluators.base import Evaluator
from router.evaluation.schemas import EvalResult


def _check_contains(text: str, spec: dict) -> bool:
    return spec["value"].lower() in text.lower()


def _check_not_contains(text: str, spec: dict) -> bool:
    return spec["value"].lower() not in text.lower()


def _check_regex(text: str, spec: dict) -> bool:
    return re.search(spec["pattern"], text) is not None


def _check_max_words(text: str, spec: dict) -> bool:
    return len(text.split()) <= spec["value"]


def _check_min_words(text: str, spec: dict) -> bool:
    return len(text.split()) >= spec["value"]


def _check_max_chars(text: str, spec: dict) -> bool:
    return len(text) <= spec["value"]


_CHECKS = {
    "contains": _check_contains,
    "not_contains": _check_not_contains,
    "regex": _check_regex,
    "max_words": _check_max_words,
    "min_words": _check_min_words,
    "max_chars": _check_max_chars,
}


class ConstraintCheckingEvaluator(Evaluator):
    name = "constraint_checking"
    task_type = TaskType.CONSTRAINT_CHECKING

    def evaluate(self, item: DatasetItem, record_id: str, response_text: str | None) -> EvalResult:
        if not response_text:
            return self._empty_response_result(item, record_id)

        # dataset/loader.py validates reference_answer is a list for this task_type.
        constraints = cast("list[dict]", item.reference_answer)
        results: list[dict] = []
        for spec in constraints:
            check = _CHECKS.get(spec.get("type", ""))
            if check is None:
                results.append({"spec": spec, "satisfied": False, "error": "unknown constraint type"})
                continue
            try:
                satisfied = check(response_text, spec)
            except (re.error, KeyError, TypeError, AttributeError) as exc:  # malformed spec: bad regex, missing/wrong-typed field
                results.append({"spec": spec, "satisfied": False, "error": str(exc)})
                continue
            results.append({"spec": spec, "satisfied": satisfied})

        n_satisfied = sum(r["satisfied"] for r in results)
        n_total = len(results) or 1
        score = n_satisfied / n_total

        return EvalResult(
            record_id=record_id,
            prompt_id=item.prompt_id,
            evaluator_name=self.name,
            score=score,
            passed=score == 1.0,
            details={"constraint_results": results},
        )
