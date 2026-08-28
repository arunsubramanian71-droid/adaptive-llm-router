from __future__ import annotations

from pathlib import Path

import pytest

from router.storage.run_metadata import (
    create_run_dir,
    new_run_metadata,
    read_run_metadata,
    write_run_metadata,
)


def test_run_dir_created_and_not_overwritten(tmp_path: Path):
    metadata = new_run_metadata(
        config_hash="abc123",
        pricing_config_version="v1",
        git_sha="deadbeef",
        git_dirty=False,
    )
    run_dir = create_run_dir(tmp_path, metadata.run_id)
    assert run_dir.exists()

    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, metadata.run_id)


def test_run_metadata_round_trip(tmp_path: Path):
    metadata = new_run_metadata(
        config_hash="abc123",
        pricing_config_version="v1",
        git_sha=None,
        git_dirty=None,
        seed=42,
    )
    run_dir = create_run_dir(tmp_path, metadata.run_id)
    write_run_metadata(run_dir, metadata)

    loaded = read_run_metadata(run_dir)
    assert loaded.run_id == metadata.run_id
    assert loaded.config_hash == "abc123"
    assert loaded.seed == 42
    assert loaded.git_sha is None
    assert "no-git" in metadata.run_id  # git-less repos still get a stable run id


def test_run_id_includes_git_sha_when_present():
    metadata = new_run_metadata(
        config_hash="cfg",
        pricing_config_version="v1",
        git_sha="0123456789abcdef",
        git_dirty=True,
    )
    assert "012345678" in metadata.run_id
    assert metadata.git_dirty is True
