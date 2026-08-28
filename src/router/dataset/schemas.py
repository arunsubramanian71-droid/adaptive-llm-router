"""Dataset schema.

A benchmark prompt plus enough ground truth to score a response
*objectively* (exact_match / structured_extraction / constraint_checking /
code_generation) or to hand to a judge (judge_scored). `reference_answer`'s
shape depends on `task_type` — see each evaluator module for the exact
contract it expects.

Building this schema is Stage 1 infrastructure; it does not itself contain
any prompts. The actual 200-prompt benchmark is a later, separate step.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    EXACT_MATCH = "exact_match"
    STRUCTURED_EXTRACTION = "structured_extraction"
    CONSTRAINT_CHECKING = "constraint_checking"
    CODE_GENERATION = "code_generation"
    JUDGE_SCORED = "judge_scored"


class DatasetItem(BaseModel):
    prompt_id: str
    task_type: TaskType
    prompt: str
    system_prompt: str | None = None

    # Shape depends on task_type:
    #   exact_match:           str | list[str]  (any accepted answer)
    #   structured_extraction: dict[str, Any]    (expected field -> value)
    #   constraint_checking:   list[dict]        (constraint specs, see
    #                                              evaluation/evaluators/constraint_checking.py)
    #   code_generation:       dict              (test_cases spec, see
    #                                              evaluation/evaluators/code_eval.py)
    #   judge_scored:          str | None         (optional rubric/reference text)
    reference_answer: Any | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
