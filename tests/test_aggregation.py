from __future__ import annotations

from router.aggregation.aggregator import (
    aggregate_q_hat,
    build_prompt_aggregates,
    build_score_lookup,
    compute_delta,
    compute_label,
)
from router.aggregation.kdelta_analysis import sweep_k_delta


def _synthetic_records_and_scores(make_response_record):
    """6 strong + 6 cheap samples for p1 (k up to 6, per ADR-0001), fewer
    for p2 to exercise the "not enough samples" path. Scores are made up
    for aggregation-math testing only."""
    records = []
    scores: dict[str, float] = {}

    strong_scores_p1 = [1.0, 1.0, 0.8, 1.0, 0.6, 1.0]
    cheap_scores_p1 = [0.4, 0.6, 0.2, 0.4, 0.0, 0.4]
    for i, s in enumerate(strong_scores_p1):
        rec = make_response_record(f"p1-strong-{i}", "p1", model_id="claude-sonnet-5", sample_index=i)
        records.append(rec)
        scores[rec.record_id] = s
    for i, s in enumerate(cheap_scores_p1):
        rec = make_response_record(f"p1-cheap-{i}", "p1", model_id="claude-haiku-4-5", sample_index=i)
        records.append(rec)
        scores[rec.record_id] = s

    # p2 has only cheap samples — strong q_hat should come back None.
    rec = make_response_record("p2-cheap-0", "p2", model_id="claude-haiku-4-5", sample_index=0)
    records.append(rec)
    scores[rec.record_id] = 0.5

    return records, scores


def test_aggregate_q_hat_mean_over_k_samples(make_response_record):
    records, scores = _synthetic_records_and_scores(make_response_record)
    q_hat, n = aggregate_q_hat(records, scores, "p1", "claude-sonnet-5", k=3)
    assert n == 3
    assert q_hat == (1.0 + 1.0 + 0.8) / 3


def test_aggregate_q_hat_respects_k_boundary(make_response_record):
    records, scores = _synthetic_records_and_scores(make_response_record)
    q_hat_k1, n1 = aggregate_q_hat(records, scores, "p1", "claude-sonnet-5", k=1)
    _q_hat_k6, n6 = aggregate_q_hat(records, scores, "p1", "claude-sonnet-5", k=6)
    assert n1 == 1
    assert n6 == 6
    assert q_hat_k1 == 1.0


def test_aggregate_q_hat_missing_model_returns_none(make_response_record):
    records, scores = _synthetic_records_and_scores(make_response_record)
    q_hat, n = aggregate_q_hat(records, scores, "p2", "claude-sonnet-5", k=6)
    assert q_hat is None
    assert n == 0


def test_compute_delta_and_label(make_response_record):
    records, scores = _synthetic_records_and_scores(make_response_record)
    q_strong, _ = aggregate_q_hat(records, scores, "p1", "claude-sonnet-5", k=6)
    q_cheap, _ = aggregate_q_hat(records, scores, "p1", "claude-haiku-4-5", k=6)
    delta = compute_delta(q_strong, q_cheap)
    assert delta is not None and delta > 0

    assert compute_label(delta, delta_threshold=0.2) == 1
    assert compute_label(delta, delta_threshold=0.9) == 0
    assert compute_label(None, delta_threshold=0.2) is None


def test_build_prompt_aggregates(make_response_record):
    records, scores = _synthetic_records_and_scores(make_response_record)
    aggregates = build_prompt_aggregates(
        records, scores, ["p1", "p2"], "anthropic", "claude-sonnet-5", "claude-haiku-4-5", k=6
    )
    by_id = {a.prompt_id: a for a in aggregates}

    assert by_id["p1"].delta_hat is not None
    assert by_id["p1"].n_samples_strong == 6
    assert by_id["p1"].n_samples_cheap == 6

    assert by_id["p2"].q_hat_strong is None
    assert by_id["p2"].delta_hat is None  # can't compute without both q_hats


def test_build_score_lookup_from_eval_results():
    from router.evaluation.schemas import EvalResult

    results = [
        EvalResult(record_id="r1", prompt_id="p1", evaluator_name="exact_match", score=1.0),
        EvalResult(record_id="r2", prompt_id="p1", evaluator_name="exact_match", score=0.0),
    ]
    lookup = build_score_lookup(results)
    assert lookup == {"r1": 1.0, "r2": 0.0}


def test_sweep_k_delta_offline_no_new_calls(make_response_record):
    records, scores = _synthetic_records_and_scores(make_response_record)
    points = sweep_k_delta(
        records,
        scores,
        prompt_ids=["p1", "p2"],
        provider="anthropic",
        strong_model_id="claude-sonnet-5",
        cheap_model_id="claude-haiku-4-5",
        k_values=[1, 3, 6],
        delta_values=[0.2, 0.4, 0.6],
    )
    assert len(points) == 3 * 3  # every (k, delta) combination

    # p1 always has a computable delta; p2 never does (no strong samples) —
    # so exactly one of two prompts is ever labeled, regardless of k/delta.
    for point in points:
        assert point.n_prompts_total == 2
        assert point.n_prompts_labeled == 1

    # Larger delta threshold should never increase the positive rate.
    by_delta_at_k6 = {p.delta: p for p in points if p.k == 6}
    assert by_delta_at_k6[0.6].label_positive_rate <= by_delta_at_k6[0.2].label_positive_rate
