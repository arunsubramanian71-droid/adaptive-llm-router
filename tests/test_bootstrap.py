from __future__ import annotations

import pytest

from router.analysis.bootstrap import mean_difference, paired_bootstrap_ci


def test_paired_bootstrap_point_estimate_matches_statistic():
    values_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    values_b = [0.5, 1.5, 2.5, 3.5, 4.5]
    result = paired_bootstrap_ci(values_a, values_b, mean_difference, n_bootstrap=500, seed=0)
    assert result.point_estimate == pytest.approx(0.5)


def test_paired_bootstrap_ci_contains_point_estimate():
    values_a = [1.0, 2.0, 3.0, 2.5, 1.5, 4.0, 3.5]
    values_b = [1.1, 1.9, 3.2, 2.4, 1.6, 3.9, 3.4]
    result = paired_bootstrap_ci(values_a, values_b, mean_difference, n_bootstrap=1000, seed=1)
    assert result.ci_low <= result.point_estimate <= result.ci_high


def test_paired_bootstrap_deterministic_with_seed():
    values_a = [1.0, 2.0, 3.0, 4.0]
    values_b = [0.0, 1.0, 2.0, 3.0]
    r1 = paired_bootstrap_ci(values_a, values_b, mean_difference, n_bootstrap=200, seed=7)
    r2 = paired_bootstrap_ci(values_a, values_b, mean_difference, n_bootstrap=200, seed=7)
    assert r1.ci_low == r2.ci_low
    assert r1.ci_high == r2.ci_high


def test_paired_bootstrap_no_difference_ci_straddles_zero():
    values_a = [1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5]
    values_b = [1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5]  # identical -> difference is always exactly 0
    result = paired_bootstrap_ci(values_a, values_b, mean_difference, n_bootstrap=500, seed=2)
    assert result.point_estimate == pytest.approx(0.0)
    assert result.ci_low == pytest.approx(0.0)
    assert result.ci_high == pytest.approx(0.0)


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0, 2.0], [1.0], mean_difference)


def test_paired_bootstrap_rejects_empty():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], [], mean_difference)
