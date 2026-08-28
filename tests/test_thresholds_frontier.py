from __future__ import annotations

from router.analysis.frontier import FrontierPoint, cost_quality_frontier, pareto_frontier
from router.analysis.thresholds import sweep_thresholds


def test_sweep_thresholds_basic_classification_metrics():
    y_true = [1, 1, 0, 0]
    probs = [0.9, 0.8, 0.3, 0.1]
    results = sweep_thresholds(y_true, probs, thresholds=[0.0, 0.5, 1.0])
    by_tau = {r.tau: r for r in results}

    assert by_tau[0.0].n_routed_strong == 4  # everyone >= 0
    assert by_tau[1.0].n_routed_strong == 0  # nobody >= 1.0 except exactly 1.0 probs (none here)
    assert by_tau[0.5].accuracy == 1.0  # perfectly separates at 0.5


def test_sweep_thresholds_computes_cost_and_quality_when_provided():
    y_true = [1, 0]
    probs = [0.9, 0.1]
    results = sweep_thresholds(
        y_true,
        probs,
        thresholds=[0.5],
        cost_if_strong=[0.10, 0.10],
        cost_if_cheap=[0.01, 0.01],
        quality_if_strong=[1.0, 1.0],
        quality_if_cheap=[0.5, 0.5],
    )
    result = results[0]
    # prompt 0 routed strong (0.9 >= 0.5): cost 0.10, quality 1.0
    # prompt 1 routed cheap (0.1 < 0.5): cost 0.01, quality 0.5
    assert result.avg_cost == (0.10 + 0.01) / 2
    assert result.avg_quality == (1.0 + 0.5) / 2


def test_sweep_thresholds_without_cost_quality_leaves_them_none():
    results = sweep_thresholds([1, 0], [0.9, 0.1], thresholds=[0.5])
    assert results[0].avg_cost is None
    assert results[0].avg_quality is None


def test_sweep_thresholds_rejects_mismatched_lengths():
    import pytest

    with pytest.raises(ValueError):
        sweep_thresholds([1, 0, 1], [0.5, 0.5])


def test_cost_quality_frontier_filters_incomplete_points():
    from router.analysis.thresholds import ThresholdResult

    results = [
        ThresholdResult(
            tau=0.1, n_total=2, n_routed_strong=2, strong_rate=1.0, accuracy=1.0, precision=1.0,
            recall=1.0, f1=1.0, avg_cost=0.1, avg_quality=0.9,
        ),
        ThresholdResult(
            tau=0.9, n_total=2, n_routed_strong=0, strong_rate=0.0, accuracy=0.5, precision=0.0,
            recall=0.0, f1=0.0, avg_cost=None, avg_quality=None,
        ),
    ]
    frontier = cost_quality_frontier(results)
    assert len(frontier) == 1
    assert frontier[0].tau == 0.1


def test_pareto_frontier_removes_dominated_points():
    points = [
        FrontierPoint(tau=0.0, avg_cost=0.10, avg_quality=1.0),   # best quality, most expensive
        FrontierPoint(tau=0.5, avg_cost=0.05, avg_quality=0.9),   # good tradeoff
        FrontierPoint(tau=0.9, avg_cost=0.01, avg_quality=0.5),   # cheapest, lowest quality
        FrontierPoint(tau=1.0, avg_cost=0.06, avg_quality=0.85),  # dominated by tau=0.5 (more cost, less quality)
    ]
    frontier = pareto_frontier(points)
    taus_on_frontier = {p.tau for p in frontier}
    assert taus_on_frontier == {0.0, 0.5, 0.9}
    assert 1.0 not in taus_on_frontier
    # sorted ascending by cost
    assert [p.tau for p in frontier] == [0.9, 0.5, 0.0]
