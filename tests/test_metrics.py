from __future__ import annotations

import pytest

from router.analysis.metrics import (
    accuracy,
    confusion_matrix_counts,
    cost_per_correct_answer,
    cost_per_request,
    cost_savings,
    pr_auc,
    precision_recall_f1,
    quality_degradation,
    quality_retention,
    total_cost,
)


def test_accuracy_basic():
    assert accuracy([1, 0, 1, 0], [1, 0, 0, 0]) == 0.75


def test_accuracy_empty():
    assert accuracy([], []) == 0.0


def test_precision_recall_f1():
    # tp=1 (idx0), fp=1 (idx2), fn=1 (idx3), tn=1 (idx1)
    y_true = [1, 0, 0, 1]
    y_pred = [1, 0, 1, 0]
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)


def test_confusion_matrix_counts():
    y_true = [1, 0, 0, 1]
    y_pred = [1, 0, 1, 0]
    counts = confusion_matrix_counts(y_true, y_pred)
    assert counts == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


def test_pr_auc_perfect_separation():
    y_true = [0, 0, 1, 1]
    probs = [0.1, 0.2, 0.8, 0.9]
    assert pr_auc(y_true, probs) == pytest.approx(1.0)


def test_pr_auc_requires_both_classes():
    with pytest.raises(ValueError):
        pr_auc([1, 1, 1], [0.5, 0.6, 0.7])


def test_cost_per_request_and_total_cost():
    costs = [0.01, 0.02, 0.03]
    assert cost_per_request(costs) == pytest.approx(0.02)
    assert total_cost(costs) == pytest.approx(0.06)


def test_cost_per_request_empty():
    assert cost_per_request([]) == 0.0


def test_cost_savings():
    savings = cost_savings(baseline_total_cost=10.0, routed_total_cost=6.0)
    assert savings["absolute"] == pytest.approx(4.0)
    assert savings["relative"] == pytest.approx(0.4)


def test_cost_savings_zero_baseline():
    savings = cost_savings(baseline_total_cost=0.0, routed_total_cost=0.0)
    assert savings["relative"] == 0.0


def test_quality_retention_and_degradation():
    assert quality_retention(routed_quality=0.8, reference_quality=1.0) == pytest.approx(0.8)
    assert quality_retention(routed_quality=0.8, reference_quality=0.0) == 0.0
    assert quality_degradation(reference_quality=1.0, routed_quality=0.8) == pytest.approx(0.2)


def test_cost_per_correct_answer():
    assert cost_per_correct_answer(10.0, 5) == pytest.approx(2.0)
    assert cost_per_correct_answer(10.0, 0) == float("inf")
