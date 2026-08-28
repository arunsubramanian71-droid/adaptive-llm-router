"""Operating-threshold (tau) sweep.

Sweeps a router's probability output over candidate tau values and scores
each as a routing decision — classification metrics against a
delta-threshold label, plus (when the caller supplies per-prompt realized
cost/quality for both roles) the average cost and quality that operating
point would have produced. All inputs are measured/stored values; nothing
here estimates cost or quality on its own.
"""

from __future__ import annotations

from pydantic import BaseModel

from router.analysis.metrics import accuracy, precision_recall_f1

DEFAULT_THRESHOLDS = [round(i / 20, 2) for i in range(21)]  # 0.00, 0.05, ..., 1.00


class ThresholdResult(BaseModel):
    tau: float
    n_total: int
    n_routed_strong: int
    strong_rate: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    avg_cost: float | None = None
    avg_quality: float | None = None


def sweep_thresholds(
    y_true: list[int],
    probs: list[float],
    thresholds: list[float] | None = None,
    cost_if_strong: list[float] | None = None,
    cost_if_cheap: list[float] | None = None,
    quality_if_strong: list[float] | None = None,
    quality_if_cheap: list[float] | None = None,
) -> list[ThresholdResult]:
    if len(y_true) != len(probs):
        raise ValueError("y_true and probs must be the same length")
    thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    n_total = len(probs)

    results = []
    for tau in thresholds:
        y_pred = [1 if p >= tau else 0 for p in probs]
        n_strong = sum(y_pred)
        precision, recall, f1 = precision_recall_f1(y_true, y_pred)

        avg_cost = None
        if cost_if_strong is not None and cost_if_cheap is not None:
            costs = [
                cost_if_strong[i] if y_pred[i] == 1 else cost_if_cheap[i] for i in range(n_total)
            ]
            avg_cost = sum(costs) / n_total if n_total else None

        avg_quality = None
        if quality_if_strong is not None and quality_if_cheap is not None:
            qualities = [
                quality_if_strong[i] if y_pred[i] == 1 else quality_if_cheap[i] for i in range(n_total)
            ]
            avg_quality = sum(qualities) / n_total if n_total else None

        results.append(
            ThresholdResult(
                tau=tau,
                n_total=n_total,
                n_routed_strong=n_strong,
                strong_rate=n_strong / n_total if n_total else 0.0,
                accuracy=accuracy(y_true, y_pred),
                precision=precision,
                recall=recall,
                f1=f1,
                avg_cost=avg_cost,
                avg_quality=avg_quality,
            )
        )
    return results
