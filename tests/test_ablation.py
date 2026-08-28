from __future__ import annotations

from router.analysis.ablation import run_ablation


def test_run_ablation_merges_overrides_and_collects_metrics():
    base_config = {"k": 6, "delta": 0.4, "router": "tfidf_logreg"}
    variants = {
        "baseline": {},
        "lower_delta": {"delta": 0.2},
        "gb_router": {"router": "gradient_boosting"},
    }

    def fake_run(config: dict) -> dict[str, float]:
        # deterministic synthetic "metric" derived from the config, purely
        # to prove wiring — not a real experiment result.
        return {"delta_used": config["delta"], "router_name_length": float(len(config["router"]))}

    results = run_ablation(base_config, variants, fake_run)
    by_name = {r.variant_name: r for r in results}

    assert by_name["baseline"].overrides == {}
    assert by_name["baseline"].metrics["delta_used"] == 0.4

    assert by_name["lower_delta"].overrides == {"delta": 0.2}
    assert by_name["lower_delta"].metrics["delta_used"] == 0.2

    assert by_name["gb_router"].metrics["router_name_length"] == len("gradient_boosting")

    # base_config itself must be untouched by merging.
    assert base_config["delta"] == 0.4
