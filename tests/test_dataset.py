from __future__ import annotations

from pathlib import Path

import pytest

from router.dataset.loader import DatasetError, load_dataset
from router.dataset.schemas import DatasetItem, TaskType


def _write(path: Path, items: list[DatasetItem]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json())
            f.write("\n")


def test_load_dataset_round_trip(tmp_path: Path):
    items = [
        DatasetItem(prompt_id="p1", task_type=TaskType.EXACT_MATCH, prompt="2+2?", reference_answer="4"),
        DatasetItem(
            prompt_id="p2",
            task_type=TaskType.STRUCTURED_EXTRACTION,
            prompt="extract",
            reference_answer={"name": "Ada"},
        ),
    ]
    path = tmp_path / "ds.jsonl"
    _write(path, items)

    loaded = load_dataset(path)
    assert len(loaded) == 2
    assert loaded[0].prompt_id == "p1"
    assert loaded[1].reference_answer == {"name": "Ada"}


def test_load_dataset_missing_file_raises(tmp_path: Path):
    with pytest.raises(DatasetError):
        load_dataset(tmp_path / "nope.jsonl")


def test_load_dataset_empty_file_raises(tmp_path: Path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_dataset(path)


def test_load_dataset_duplicate_prompt_id_raises(tmp_path: Path):
    items = [
        DatasetItem(prompt_id="dup", task_type=TaskType.EXACT_MATCH, prompt="a", reference_answer="x"),
        DatasetItem(prompt_id="dup", task_type=TaskType.EXACT_MATCH, prompt="b", reference_answer="y"),
    ]
    path = tmp_path / "ds.jsonl"
    _write(path, items)
    with pytest.raises(DatasetError, match="duplicate"):
        load_dataset(path)


def test_load_dataset_wrong_reference_shape_raises(tmp_path: Path):
    # exact_match with a dict reference_answer instead of str/list is invalid.
    path = tmp_path / "ds.jsonl"
    path.write_text(
        '{"prompt_id":"p1","task_type":"exact_match","prompt":"x","reference_answer":{"a":1}}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError):
        load_dataset(path)


def test_load_dataset_malformed_json_raises(tmp_path: Path):
    path = tmp_path / "ds.jsonl"
    path.write_text("not json at all\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_dataset(path)
