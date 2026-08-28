from __future__ import annotations

import pytest

from router.config import ConfigError, load_app_config
from router.models.anthropic_adapter import AnthropicModelClient
from router.models.google_gemini_adapter import GoogleGeminiModelClient
from router.models.openai_adapter import OpenAIModelClient
from router.models.registry import build_model_client, resolve_adapter_class

# ---------------------------------------------------------------------------
# resolve_adapter_class
# ---------------------------------------------------------------------------


def test_resolve_adapter_class_anthropic():
    cls = resolve_adapter_class("router.models.anthropic_adapter.AnthropicModelClient")
    assert cls is AnthropicModelClient


def test_resolve_adapter_class_openai():
    cls = resolve_adapter_class("router.models.openai_adapter.OpenAIModelClient")
    assert cls is OpenAIModelClient


def test_resolve_adapter_class_google():
    cls = resolve_adapter_class("router.models.google_gemini_adapter.GoogleGeminiModelClient")
    assert cls is GoogleGeminiModelClient


def test_resolve_adapter_class_rejects_malformed_path():
    with pytest.raises(ConfigError, match="invalid adapter path"):
        resolve_adapter_class("not_a_dotted_path")


def test_resolve_adapter_class_rejects_missing_module():
    with pytest.raises(ConfigError, match="could not import"):
        resolve_adapter_class("router.models.does_not_exist.SomeClient")


def test_resolve_adapter_class_rejects_missing_class():
    with pytest.raises(ConfigError, match="has no class"):
        resolve_adapter_class("router.models.anthropic_adapter.NotARealClass")


def test_resolve_adapter_class_rejects_non_modelclient_target():
    with pytest.raises(ConfigError, match="not a ModelClient subclass"):
        resolve_adapter_class("router.config.ModelEntry")


# ---------------------------------------------------------------------------
# build_model_client -- no network calls, api_key is a dummy string
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config():
    return load_app_config()


@pytest.mark.parametrize(
    "provider,expected_cls",
    [
        ("anthropic", AnthropicModelClient),
        ("openai", OpenAIModelClient),
        ("google", GoogleGeminiModelClient),
    ],
)
def test_build_model_client_instantiates_correct_adapter(config, provider, expected_cls):
    client = build_model_client(config, provider, api_key="dummy-no-network")
    assert isinstance(client, expected_cls)
    assert client.provider == provider


def test_build_model_client_unknown_provider_raises(config):
    with pytest.raises(ConfigError, match="unknown provider"):
        build_model_client(config, "not-a-real-provider", api_key="dummy")


# ---------------------------------------------------------------------------
# Requirement 11: models from different providers coexist in one experiment
# configuration (the real, committed configs/models.yaml + pricing.yaml).
# ---------------------------------------------------------------------------


def test_models_from_three_providers_coexist_in_one_config(config):
    providers_present = set(config.models.providers.keys())
    assert {"anthropic", "openai", "google"} <= providers_present

    anthropic_ids = {m.id for m in config.models.providers["anthropic"].models}
    openai_ids = {m.id for m in config.models.providers["openai"].models}
    google_ids = {m.id for m in config.models.providers["google"].models}

    assert "claude-haiku-4-5" in anthropic_ids
    assert "claude-opus-5" in anthropic_ids
    assert "gpt-5.6-luna" in openai_ids
    assert "gpt-5.6-sol" in openai_ids
    assert "gemini-2.5-flash-lite" in google_ids
    assert "gemini-3.1-pro-preview" in google_ids

    # No accidental model-id collisions across providers.
    assert not (anthropic_ids & openai_ids)
    assert not (anthropic_ids & google_ids)
    assert not (openai_ids & google_ids)

    # get_model() resolves each one under its own provider, and every
    # configured model has a matching pricing entry (AppConfig's own
    # cross-validation already enforces this at load time -- re-asserted
    # here as the explicit "registry" proof this requirement asks for).
    for provider, model_id in [
        ("anthropic", "claude-haiku-4-5"),
        ("openai", "gpt-5.6-luna"),
        ("google", "gemini-2.5-flash-lite"),
    ]:
        entry = config.models.get_model(provider, model_id)
        assert entry.id == model_id
        pricing = config.pricing.pricing_for(model_id)
        assert pricing.provider == provider


def test_get_model_wrong_provider_does_not_leak_across_providers(config):
    with pytest.raises(ConfigError):
        config.models.get_model("openai", "claude-haiku-4-5")
    with pytest.raises(ConfigError):
        config.models.get_model("anthropic", "gpt-5.6-luna")
    with pytest.raises(ConfigError):
        config.models.get_model("google", "gpt-5.6-luna")


# ---------------------------------------------------------------------------
# Requirement 7: no provider key is required just to load configuration.
# ---------------------------------------------------------------------------


def test_load_app_config_requires_no_api_keys(monkeypatch, tmp_path):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # Point at an empty dir so no stray .env file on disk leaks a real key in.
    monkeypatch.chdir(tmp_path)
    config = load_app_config()  # must not raise
    assert config.secrets.anthropic_api_key is None
    assert config.secrets.openai_api_key is None
    assert config.secrets.resolved_google_api_key is None


def test_require_api_key_per_provider_error_messages(config, monkeypatch):
    monkeypatch.setattr(config.secrets, "anthropic_api_key", None)
    monkeypatch.setattr(config.secrets, "openai_api_key", None)
    monkeypatch.setattr(config.secrets, "gemini_api_key", None)
    monkeypatch.setattr(config.secrets, "google_api_key", None)

    for provider in ("anthropic", "openai", "google"):
        with pytest.raises(ConfigError, match=provider):
            config.require_api_key(provider)


def test_require_api_key_default_is_still_anthropic(config, monkeypatch):
    # Backward compatibility: existing zero-arg call sites (verify_stage0.py,
    # run_model_pair_pilot.py) must keep working unchanged.
    monkeypatch.setattr(config.secrets, "anthropic_api_key", "sk-ant-fake-value")
    assert config.require_api_key() == "sk-ant-fake-value"


def test_resolved_google_api_key_precedence(config, monkeypatch):
    monkeypatch.setattr(config.secrets, "gemini_api_key", "gemini-value")
    monkeypatch.setattr(config.secrets, "google_api_key", None)
    assert config.secrets.resolved_google_api_key == "gemini-value"

    monkeypatch.setattr(config.secrets, "google_api_key", "google-value")
    assert config.secrets.resolved_google_api_key == "google-value"  # GOOGLE_API_KEY wins when both set
