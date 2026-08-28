"""Routing/cost/quality metrics.

Classification metrics (accuracy/precision/recall/F1/PR-AUC/confusion
matrix) score the routing *decision* against a delta-threshold label.
Cost/quality metrics score what routing actually bought — always computed
from measured per-prompt cost/quality inputs the caller supplies, never
estimated here.
"""

from __future__ import annotations

from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    return correct / len(y_true)


def precision_recall_f1(y_true: list[int], y_pred: list[int]) -> tuple[float, float, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return float(precision), float(recall), float(f1)


def pr_auc(y_true: list[int], probs: list[float]) -> float:
    if len(set(y_true)) < 2:
        raise ValueError("PR-AUC is undefined with only one class present in y_true")
    return float(average_precision_score(y_true, probs))


def confusion_matrix_counts(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred, strict=True):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 0 and p == 0:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def cost_per_request(costs: list[float]) -> float:
    return sum(costs) / len(costs) if costs else 0.0


def total_cost(costs: list[float]) -> float:
    return sum(costs)


def cost_savings(baseline_total_cost: float, routed_total_cost: float) -> dict[str, float]:
    absolute = baseline_total_cost - routed_total_cost
    relative = (absolute / baseline_total_cost) if baseline_total_cost > 0 else 0.0
    return {"absolute": absolute, "relative": relative}


def quality_retention(routed_quality: float, reference_quality: float) -> float:
    """routed_quality / reference_quality — 1.0 means routing matched the
    reference (e.g. always-strong) quality exactly."""
    if reference_quality == 0:
        return 0.0
    return routed_quality / reference_quality


def quality_degradation(reference_quality: float, routed_quality: float) -> float:
    return reference_quality - routed_quality


def cost_per_correct_answer(total_cost_value: float, n_correct: int) -> float:
    if n_correct == 0:
        return float("inf")
    return total_cost_value / n_correct
