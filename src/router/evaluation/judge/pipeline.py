from __future__ import annotations

from router.dataset.schemas import DatasetItem
from router.evaluation.judge.base import JudgeClient, JudgeVerdict
from router.storage.records import ResponseRecord


def run_judge_pipeline(
    items_by_prompt_id: dict[str, DatasetItem],
    records: list[ResponseRecord],
    judge_client: JudgeClient,
    rubric: str | None = None,
) -> list[JudgeVerdict]:
    verdicts: list[JudgeVerdict] = []
    for record in records:
        item = items_by_prompt_id.get(record.prompt_id)
        if item is None:
            continue
        reference = item.reference_answer if isinstance(item.reference_answer, str) else None
        verdicts.append(
            judge_client.judge(
                prompt_id=record.prompt_id,
                record_id=record.record_id,
                prompt=item.prompt,
                response_text=record.response_text,
                rubric=rubric,
                reference=reference,
            )
        )
    return verdicts
