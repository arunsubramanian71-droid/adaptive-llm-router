"""Immutable per-run metadata.

Every experiment run gets its own directory under `runs/` named from its
start time, git SHA, and config hash, so two runs can never silently
collide or overwrite each other.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

TRACKED_PACKAGES = ["anthropic", "pydantic", "pydantic-settings", "PyYAML"]


class RunMetadata(BaseModel):
    run_id: str
    utc_start: datetime
    utc_end: datetime | None = None
    git_sha: str | None = None
    git_dirty: bool | None = None
    config_hash: str
    pricing_config_version: str
    dependency_versions: dict[str, str]
    python_version: str
    platform: str
    seed: int | None = None


def dependency_versions(packages: list[str] = TRACKED_PACKAGES) -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = "unknown"
    return versions


def make_run_id(timestamp: datetime, git_sha: str | None, config_hash: str) -> str:
    ts_str = timestamp.strftime("%Y%m%dT%H%M%SZ")
    sha_str = (git_sha or "no-git")[:12]
    return f"{ts_str}__{sha_str}__{config_hash}"


def create_run_dir(runs_root: Path, run_id: str) -> Path:
    run_dir = runs_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists, refusing to overwrite: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_dir


def write_run_metadata(run_dir: Path, metadata: RunMetadata) -> None:
    path = run_dir / "run_metadata.json"
    path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")


def read_run_metadata(run_dir: Path) -> RunMetadata:
    path = run_dir / "run_metadata.json"
    return RunMetadata.model_validate_json(path.read_text(encoding="utf-8"))


def new_run_metadata(
    config_hash: str,
    pricing_config_version: str,
    git_sha: str | None,
    git_dirty: bool | None,
    seed: int | None = None,
    timestamp: datetime | None = None,
) -> RunMetadata:
    timestamp = timestamp or datetime.now(UTC)
    run_id = make_run_id(timestamp, git_sha, config_hash)
    return RunMetadata(
        run_id=run_id,
        utc_start=timestamp,
        git_sha=git_sha,
        git_dirty=git_dirty,
        config_hash=config_hash,
        pricing_config_version=pricing_config_version,
        dependency_versions=dependency_versions(),
        python_version=sys.version,
        platform=platform.platform(),
        seed=seed,
    )
