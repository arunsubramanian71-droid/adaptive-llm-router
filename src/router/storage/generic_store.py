"""Generic append-only JSONL store for any pydantic model.

Added alongside (not in place of) `storage.records.JsonlStore` — Stage 0's
concrete store is left untouched. Every new record type introduced in later
phases (evaluation results, judge verdicts, prompt aggregates, routing
decisions, ...) reuses this instead of hand-rolling another read/append pair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class JsonlRecordStore(Generic[T]):
    def __init__(self, path: Path, model_cls: type[T]) -> None:
        self.path = path
        self.model_cls = model_cls
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: T) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json())
            f.write("\n")

    def append_all(self, records: list[T]) -> None:
        if not records:
            return
        with self.path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(record.model_dump_json())
                f.write("\n")

    def read_all(self) -> list[T]:
        if not self.path.exists():
            return []
        records: list[T] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(self.model_cls.model_validate_json(line))
        return records
