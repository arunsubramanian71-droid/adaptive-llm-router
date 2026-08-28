from __future__ import annotations

from pydantic import BaseModel

from router.analysis.thresholds import ThresholdResult


class FrontierPoint(BaseModel):
    tau: float
    avg_cost: float
    avg_quality: float


def cost_quality_frontier(threshold_results: list[ThresholdResult]) -> list[FrontierPoint]:
    return [
        FrontierPoint(tau=r.tau, avg_cost=r.avg_cost, avg_quality=r.avg_quality)
        for r in threshold_results
        if r.avg_cost is not None and r.avg_quality is not None
    ]


def pareto_frontier(points: list[FrontierPoint]) -> list[FrontierPoint]:
    """Non-dominated points only (lower cost, higher quality is better).
    A point is dominated if some other point is at least as good on both
    axes and strictly better on at least one."""
    non_dominated = []
    for p in points:
        dominated = any(
            q is not p
            and q.avg_cost <= p.avg_cost
            and q.avg_quality >= p.avg_quality
            and (q.avg_cost < p.avg_cost or q.avg_quality > p.avg_quality)
            for q in points
        )
        if not dominated:
            non_dominated.append(p)
    return sorted(non_dominated, key=lambda pt: pt.avg_cost)
