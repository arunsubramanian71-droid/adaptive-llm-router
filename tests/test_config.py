from __future__ import annotations

from datetime import UTC
from pathlib import Path

import pytest
import yaml

from router.config import (
    ConfigError,
    load_app_config,
    load_models_config,
)


def test_default_configs_load_and_validate():
    config = load_app_config()
    assert config.models.all_model_ids() <= set(config.pricing.models.keys())
    assert config.config_hash
    assert len(config.config_hash) == 16


def test_missing_config_file_raises(tmp_path: Path):
    with pytest.raises(ConfigError):
        load_models_config(tmp_path / "does-not-exist.yaml")


def test_malformed_models_config_raises(tmp_path: Path):
    bad = tmp_path / "models.yaml"
    bad.write_text("version: not-an-int\nproviders: {}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_models_config(bad)


def test_pricing_missing_for_configured_model_raises(tmp_path: Path):
    models_path = tmp_path / "models.yaml"
    pricing_path = tmp_path / "pricing.yaml"
    models_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "providers": {
                    "anthropic": {
                        "adapter": "router.models.anthropic_adapter.AnthropicModelClient",
                        "models": [
                            {
                                "id": "claude-unpriced-9",
                                "display_name": "Unpriced",
                                "max_output_tokens": 100,
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    pricing_path.write_text(
        yaml.safe_dump(
            {
                "version": "1",
                "currency": "USD",
                "pricing_config_version": "test",
                "models": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_app_config(models_path=models_path, pricing_path=pricing_path)


def test_get_model_unknown_raises():
    config = load_app_config()
    with pytest.raises(ConfigError):
        config.models.get_model("anthropic", "does-not-exist")
    with pytest.raises(ConfigError):
        config.models.get_model("unknown-provider", "claude-sonnet-5")


def test_rate_for_no_matching_period_raises(pricing_config):
    from datetime import datetime

    model_pricing = pricing_config.models["claude-sonnet-5"]
    with pytest.raises(ConfigError):
        model_pricing.rate_for(datetime(2020, 1, 1, tzinfo=UTC))
