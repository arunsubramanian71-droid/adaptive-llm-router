"""Ablation framework.

Generic on purpose: it doesn't know what a "run" means (fit a router,
sweep thresholds, whatever) — the caller supplies `run_fn`, which takes a
merged config dict and returns a flat metrics dict. This module's only job
is bookkeeping: apply each variant's overrides on top of a base config,
call `run_fn`, and collect the results into one comparable table.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class AblationResult(BaseModel):
    variant_name: str
    overrides: dict[str, Any]
    metrics: dict[str, float]


def run_ablation(
    base_config: dict[str, Any],
    variants: dict[str, dict[str, Any]],
    run_fn: Callable[[dict[str, Any]], dict[str, float]],
) -> list[AblationResult]:
    results = []
    for variant_name, overrides in variants.items():
        merged_config = {**base_config, **overrides}
        metrics = run_fn(merged_config)
        results.append(AblationResult(variant_name=variant_name, overrides=overrides, metrics=metrics))
    return results
