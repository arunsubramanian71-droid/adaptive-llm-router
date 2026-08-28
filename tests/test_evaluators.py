from __future__ import annotations

import pytest

from router.dataset.schemas import DatasetItem, TaskType
from router.evaluation.evaluators import (
    ConstraintCheckingEvaluator,
    ExactMatchEvaluator,
    FixtureCodeEvalEvaluator,
    HeuristicMockCodeEvalEvaluator,
    StructuredExtractionEvaluator,
    UnsandboxedSubprocessCodeEvalEvaluator,
    default_evaluator_registry,
)

# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------


def test_exact_match_correct():
    item = DatasetItem(prompt_id="p1", task_type=TaskType.EXACT_MATCH, prompt="2+2?", reference_answer="4")
    result = ExactMatchEvaluator().evaluate(item, "r1", "  4.  ")
    assert result.score == 1.0
    assert result.passed is True


def test_exact_match_accepts_any_candidate():
    item = DatasetItem(
        prompt_id="p1", task_type=TaskType.EXACT_MATCH, prompt="capital of france?", reference_answer=["Paris", "paris, france"]
    )
    result = ExactMatchEvaluator().evaluate(item, "r1", "Paris")
    assert result.score == 1.0


def test_exact_match_incorrect():
    item = DatasetItem(prompt_id="p1", task_type=TaskType.EXACT_MATCH, prompt="2+2?", reference_answer="4")
    result = ExactMatchEvaluator().evaluate(item, "r1", "5")
    assert result.score == 0.0
    assert result.passed is False


def test_exact_match_empty_response():
    item = DatasetItem(prompt_id="p1", task_type=TaskType.EXACT_MATCH, prompt="2+2?", reference_answer="4")
    result = ExactMatchEvaluator().evaluate(item, "r1", None)
    assert result.score == 0.0
    assert result.error is not None


# ---------------------------------------------------------------------------
# structured_extraction
# ---------------------------------------------------------------------------


def test_structured_extraction_full_match():
    item = DatasetItem(
        prompt_id="p1",
        task_type=TaskType.STRUCTURED_EXTRACTION,
        prompt="extract name and age",
        reference_answer={"name": "Ada", "age": 30},
    )
    result = StructuredExtractionEvaluator().evaluate(item, "r1", '{"name": "Ada", "age": 30}')
    assert result.score == 1.0
    assert result.passed is True


def test_structured_extraction_partial_match():
    item = DatasetItem(
        prompt_id="p1",
        task_type=TaskType.STRUCTURED_EXTRACTION,
        prompt="extract name and age",
        reference_answer={"name": "Ada", "age": 30},
    )
    result = StructuredExtractionEvaluator().evaluate(item, "r1", '{"name": "Ada", "age": 99}')
    assert result.score == 0.5
    assert result.passed is False


def test_structured_extraction_handles_fenced_json():
    item = DatasetItem(
        prompt_id="p1", task_type=TaskType.STRUCTURED_EXTRACTION, prompt="x", reference_answer={"a": 1}
    )
    result = StructuredExtractionEvaluator().evaluate(item, "r1", "Here you go:\n```json\n{\"a\": 1}\n```")
    assert result.score == 1.0


def test_structured_extraction_unparsable_response():
    item = DatasetItem(
        prompt_id="p1", task_type=TaskType.STRUCTURED_EXTRACTION, prompt="x", reference_answer={"a": 1}
    )
    result = StructuredExtractionEvaluator().evaluate(item, "r1", "not json")
    assert result.score == 0.0
    assert result.details["parse_ok"] is False


# ---------------------------------------------------------------------------
# constraint_checking
# ---------------------------------------------------------------------------


def test_constraint_checking_all_satisfied():
    item = DatasetItem(
        prompt_id="p1",
        task_type=TaskType.CONSTRAINT_CHECKING,
        prompt="reply politely",
        reference_answer=[
            {"type": "contains", "value": "thank"},
            {"type": "max_words", "value": 20},
        ],
    )
    result = ConstraintCheckingEvaluator().evaluate(item, "r1", "Thank you for your patience.")
    assert result.score == 1.0
    assert result.passed is True


def test_constraint_checking_partial():
    item = DatasetItem(
        prompt_id="p1",
        task_type=TaskType.CONSTRAINT_CHECKING,
        prompt="reply politely",
        reference_answer=[
            {"type": "contains", "value": "thank"},
            {"type": "max_words", "value": 3},
        ],
    )
    result = ConstraintCheckingEvaluator().evaluate(item, "r1", "Thank you for your patience today.")
    assert result.score == 0.5
    assert result.passed is False


def test_constraint_checking_unknown_type_counts_as_failed():
    item = DatasetItem(
        prompt_id="p1", task_type=TaskType.CONSTRAINT_CHECKING, prompt="x", reference_answer=[{"type": "made_up"}]
    )
    result = ConstraintCheckingEvaluator().evaluate(item, "r1", "anything")
    assert result.score == 0.0
    assert result.details["constraint_results"][0]["error"] == "unknown constraint type"


def test_constraint_checking_regex():
    item = DatasetItem(
        prompt_id="p1",
        task_type=TaskType.CONSTRAINT_CHECKING,
        prompt="give order id",
        reference_answer=[{"type": "regex", "pattern": r"^ORD-\d{4}$"}],
    )
    assert ConstraintCheckingEvaluator().evaluate(item, "r1", "ORD-1234").score == 1.0
    assert ConstraintCheckingEvaluator().evaluate(item, "r1", "order 1234").score == 0.0


# ---------------------------------------------------------------------------
# code_eval — mock/fixture only; the unsandboxed executor is never invoked
# ---------------------------------------------------------------------------


def test_default_registry_uses_non_executing_mock_for_code():
    registry = default_evaluator_registry()
    assert isinstance(registry[TaskType.CODE_GENERATION], HeuristicMockCodeEvalEvaluator)


def test_heuristic_mock_code_eval_rewards_def_and_return():
    item = DatasetItem(
        prompt_id="p1",
        task_type=TaskType.CODE_GENERATION,
        prompt="write add",
        reference_answer={"entry_point": "add", "test_cases": [{"call": "add(1,2)", "expected": 3}]},
    )
    ev = HeuristicMockCodeEvalEvaluator()

    full = ev.evaluate(item, "r1", "```python\ndef add(a, b):\n    return a + b\n```")
    no_return = ev.evaluate(item, "r2", "```python\ndef add(a, b):\n    pass\n```")
    no_def = ev.evaluate(item, "r3", "not code at all")

    assert full.score == 1.0
    assert no_return.score == pytest.approx(0.6)
    assert no_def.score == pytest.approx(0.0)


def test_fixture_code_eval_evaluator():
    item = DatasetItem(
        prompt_id="p1", task_type=TaskType.CODE_GENERATION, prompt="x", reference_answer={"test_cases": []}
    )
    ev = FixtureCodeEvalEvaluator(fixture={"r1": 0.75})
    assert ev.evaluate(item, "r1", "some code").score == 0.75
    assert ev.evaluate(item, "r2", "some code").score == 0.0  # default_score


def test_unsandboxed_code_eval_refuses_without_explicit_opt_in():
    with pytest.raises(ValueError, match="confirm_unsandboxed_execution"):
        UnsandboxedSubprocessCodeEvalEvaluator()
