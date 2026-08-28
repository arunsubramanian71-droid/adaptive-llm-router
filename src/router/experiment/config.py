"""Experiment configuration system.

Captures one experiment's full parameterization — dataset, model-pair role
assignment, k, delta, router choice, calibration method, threshold sweep,
bootstrap settings, output location — as a single validated, YAML-backed
object, the same pattern `router.config` uses for Stage 0. This is what a
real experiment run instantiates; nothing here freezes k, delta, or the
model pair — those are just fields a caller fills in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

RouterType = Literal["tfidf_logreg", "handcrafted_logreg", "gradient_boosting"]
CalibrationMethodName = Literal["platt", "isotonic"]


class ExperimentConfigError(ValueError):
    pass


class RouterSpec(BaseModel):
    type: RouterType
    hyperparams: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    experiment_name: str
    dataset_path: Path

    provider: str
    strong_model_id: str
    cheap_model_id: str

    k: int
    delta: float

    router: RouterSpec
    calibration_method: CalibrationMethodName | None = None
    thresholds: list[float] | None = None  # None => analysis module's default sweep
    bootstrap_n: int = 2000
    bootstrap_seed: int | None = None

    output_dir: Path
    seed: int | None = None

    @field_validator("k")
    @classmethod
    def _k_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("k must be >= 1")
        return v

    @field_validator("delta")
    @classmethod
    def _delta_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("delta must be in [0, 1] (it thresholds a [0,1]-scored quality gap)")
        return v

    @field_validator("bootstrap_n")
    @classmethod
    def _bootstrap_n_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("bootstrap_n must be >= 1")
        return v


def load_experiment_config(path: Path) -> ExperimentConfig:
    if not path.exists():
        raise ExperimentConfigError(f"experiment config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ExperimentConfigError(f"experiment config did not parse to a mapping: {path}")
    try:
        return ExperimentConfig.model_validate(data)
    except Exception as exc:  # pydantic.ValidationError
        raise ExperimentConfigError(f"invalid experiment config {path}: {exc}") from exc


def save_experiment_config(config: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    as_dict = json.loads(config.model_dump_json())  # round-trip through JSON to get plain str/number types
    path.write_text(yaml.safe_dump(as_dict, sort_keys=False), encoding="utf-8")
