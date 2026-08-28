from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from router.experiment.config import (
    ExperimentConfig,
    ExperimentConfigError,
    RouterSpec,
    load_experiment_config,
    save_experiment_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _valid_config(**overrides) -> ExperimentConfig:
    base = {
        "experiment_name": "test_exp",
        "dataset_path": Path("data/example_dataset.jsonl"),
        "provider": "anthropic",
        "strong_model_id": "claude-sonnet-5",
        "cheap_model_id": "claude-haiku-4-5",
        "k": 3,
        "delta": 0.4,
        "router": RouterSpec(type="tfidf_logreg"),
        "output_dir": Path("runs/test_exp"),
    }
    base.update(overrides)
    return ExperimentConfig(**base)


def test_bundled_example_config_loads():
    config = load_experiment_config(REPO_ROOT / "configs" / "experiments" / "example_pilot.yaml")
    assert config.provider == "anthropic"
    assert config.router.type == "tfidf_logreg"
    assert config.k >= 1


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(ExperimentConfigError):
        load_experiment_config(tmp_path / "nope.yaml")


def test_k_must_be_positive():
    with pytest.raises(ValidationError):
        _valid_config(k=0)


def test_delta_must_be_in_unit_range():
    with pytest.raises(ValidationError):
        _valid_config(delta=1.5)


def test_bootstrap_n_must_be_positive():
    with pytest.raises(ValidationError):
        _valid_config(bootstrap_n=0)


def test_save_and_reload_round_trip(tmp_path: Path):
    config = _valid_config()
    path = tmp_path / "exp.yaml"
    save_experiment_config(config, path)

    reloaded = load_experiment_config(path)
    assert reloaded.experiment_name == config.experiment_name
    assert reloaded.k == config.k
    assert reloaded.router.type == config.router.type
    assert reloaded.dataset_path == config.dataset_path


def test_thresholds_default_to_none():
    config = _valid_config()
    assert config.thresholds is None


def test_router_hyperparams_default_empty_dict():
    spec = RouterSpec(type="gradient_boosting")
    assert spec.hyperparams == {}
