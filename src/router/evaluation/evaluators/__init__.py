from router.dataset.schemas import TaskType
from router.evaluation.evaluators.base import Evaluator
from router.evaluation.evaluators.code_eval import (
    FixtureCodeEvalEvaluator,
    HeuristicMockCodeEvalEvaluator,
    UnsandboxedSubprocessCodeEvalEvaluator,
)
from router.evaluation.evaluators.constraint_checking import ConstraintCheckingEvaluator
from router.evaluation.evaluators.exact_match import ExactMatchEvaluator
from router.evaluation.evaluators.structured_extraction import StructuredExtractionEvaluator


def default_evaluator_registry() -> dict[TaskType, Evaluator]:
    """A fresh registry each call — evaluators are stateless but this avoids
    any accidental shared mutable state across callers.

    CODE_GENERATION maps to the non-executing heuristic mock, never to
    `UnsandboxedSubprocessCodeEvalEvaluator` — that class is opt-in only
    and must be constructed explicitly by a caller who has accepted its
    documented risk, so it can never be reached through this default path.
    """
    evaluators: list[Evaluator] = [
        ExactMatchEvaluator(),
        StructuredExtractionEvaluator(),
        ConstraintCheckingEvaluator(),
        HeuristicMockCodeEvalEvaluator(),
    ]
    return {e.task_type: e for e in evaluators}


__all__ = [
    "ConstraintCheckingEvaluator",
    "Evaluator",
    "ExactMatchEvaluator",
    "FixtureCodeEvalEvaluator",
    "HeuristicMockCodeEvalEvaluator",
    "StructuredExtractionEvaluator",
    "UnsandboxedSubprocessCodeEvalEvaluator",
    "default_evaluator_registry",
]
