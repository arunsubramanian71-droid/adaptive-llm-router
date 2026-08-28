"""Dataset loading and validation.

Reads a JSONL file of `DatasetItem` rows and validates it eagerly: unique
`prompt_id`s, every row parses against the schema, and (best-effort) that
`reference_answer` has the shape its `task_type` expects.
"""

from __future__ import annotations

from pathlib import Path

from router.dataset.schemas import DatasetItem, TaskType


class DatasetError(ValueError):
    pass


_EXPECTED_REFERENCE_SHAPE: dict[TaskType, tuple[type, ...]] = {
    TaskType.EXACT_MATCH: (str, list),
    TaskType.STRUCTURED_EXTRACTION: (dict,),
    TaskType.CONSTRAINT_CHECKING: (list,),
    TaskType.CODE_GENERATION: (dict,),
}


def _validate_reference_shape(item: DatasetItem) -> None:
    expected = _EXPECTED_REFERENCE_SHAPE.get(item.task_type)
    if expected is None:  # JUDGE_SCORED — reference_answer is optional free text
        return
    if item.reference_answer is None or not isinstance(item.reference_answer, expected):
        raise DatasetError(
            f"{item.prompt_id}: task_type={item.task_type.value} expects reference_answer "
            f"of type {expected}, got {type(item.reference_answer)}"
        )


def load_dataset(path: Path) -> list[DatasetItem]:
    if not path.exists():
        raise DatasetError(f"dataset file not found: {path}")

    items: list[DatasetItem] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = DatasetItem.model_validate_json(line)
            except Exception as exc:
                raise DatasetError(f"{path}:{line_number}: invalid dataset row: {exc}") from exc

            if item.prompt_id in seen_ids:
                raise DatasetError(f"{path}:{line_number}: duplicate prompt_id {item.prompt_id!r}")
            seen_ids.add(item.prompt_id)

            _validate_reference_shape(item)
            items.append(item)

    if not items:
        raise DatasetError(f"dataset file has no rows: {path}")
    return items
