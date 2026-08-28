"""Paired bootstrap confidence intervals.

For comparing two paired per-prompt sequences (e.g. router-A's cost vs
router-B's cost on the same prompts) without assuming a parametric
distribution. Resamples prompt indices (not the two sequences
independently) so pairing is preserved in every bootstrap draw.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from pydantic import BaseModel


class BootstrapResult(BaseModel):
    point_estimate: float
    ci_low: float
    ci_high: float
    n_bootstrap: int
    alpha: float


def paired_bootstrap_ci(
    values_a: list[float],
    values_b: list[float],
    statistic_fn: Callable[[list[float], list[float]], float],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> BootstrapResult:
    if len(values_a) != len(values_b):
        raise ValueError("paired arrays must have equal length")
    if not values_a:
        raise ValueError("cannot bootstrap an empty sample")

    n = len(values_a)
    rng = np.random.default_rng(seed)
    point_estimate = statistic_fn(values_a, values_b)

    draws = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample_a = [values_a[j] for j in idx]
        sample_b = [values_b[j] for j in idx]
        draws[i] = statistic_fn(sample_a, sample_b)

    ci_low = float(np.percentile(draws, 100 * alpha / 2))
    ci_high = float(np.percentile(draws, 100 * (1 - alpha / 2)))
    return BootstrapResult(
        point_estimate=point_estimate, ci_low=ci_low, ci_high=ci_high, n_bootstrap=n_bootstrap, alpha=alpha
    )


def mean_difference(values_a: list[float], values_b: list[float]) -> float:
    """The most common `statistic_fn`: mean(a) - mean(b)."""
    return sum(values_a) / len(values_a) - sum(values_b) / len(values_b)
