"""Configuration loading and validation.

Secrets (API keys) come only from the environment / `.env` file. Experiment
parameters and pricing come only from YAML. Nothing here hardcodes a key or
a price — see configs/models.yaml and configs/pricing.yaml.

Everything is validated eagerly at load time (`load_app_config`) so a bad
config fails fast with a clear error instead of surfacing mid-experiment.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from router.hashing import hash_text

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_PATH = REPO_ROOT / "configs" / "models.yaml"
DEFAULT_PRICING_PATH = REPO_ROOT / "configs" / "pricing.yaml"


class ConfigError(ValueError):
    """Raised when configuration is missing or fails validation."""


# ---------------------------------------------------------------------------
# Secrets (environment only)
# ---------------------------------------------------------------------------


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_base_url: str | None = Field(default=None, alias="ANTHROPIC_BASE_URL")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")

    # Google's own SDK/docs support both names; if both are set, GOOGLE_API_KEY
    # wins (confirmed in the google-genai README and ai.google.dev docs).
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")

    router_log_level: str = Field(default="INFO", alias="ROUTER_LOG_LEVEL")

    @property
    def resolved_google_api_key(self) -> str | None:
        return self.google_api_key or self.gemini_api_key


# ---------------------------------------------------------------------------
# Model / generation configuration (configs/models.yaml)
# ---------------------------------------------------------------------------

EffortLevel = Literal["low", "medium", "high", "xhigh", "max"]


class ThinkingConfig(BaseModel):
    type: Literal["adaptive", "enabled", "disabled"] | None = None
    display: Literal["summarized", "omitted"] | None = None
    budget_tokens: int | None = None


class ModelEntry(BaseModel):
    id: str
    display_name: str
    role_hint: str | None = None
    max_output_tokens: int
    thinking: ThinkingConfig | None = None
    effort: EffortLevel | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] = Field(default_factory=list)
    inference_geo: str | None = None


class ProviderModelsConfig(BaseModel):
    adapter: str
    models: list[ModelEntry]

    @field_validator("models")
    @classmethod
    def _non_empty(cls, v: list[ModelEntry]) -> list[ModelEntry]:
        if not v:
            raise ValueError("provider must declare at least one model")
        return v


class GenerationDefaults(BaseModel):
    max_output_tokens: int = 1024
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] = Field(default_factory=list)


class ModelsConfig(BaseModel):
    version: int
    providers: dict[str, ProviderModelsConfig]
    generation_defaults: GenerationDefaults = Field(default_factory=GenerationDefaults)

    def get_model(self, provider: str, model_id: str) -> ModelEntry:
        provider_cfg = self.providers.get(provider)
        if provider_cfg is None:
            raise ConfigError(f"unknown provider {provider!r} in models.yaml")
        for entry in provider_cfg.models:
            if entry.id == model_id:
                return entry
        raise ConfigError(f"model {model_id!r} not configured under provider {provider!r}")

    def all_model_ids(self) -> set[str]:
        return {m.id for p in self.providers.values() for m in p.models}


# ---------------------------------------------------------------------------
# Pricing configuration (configs/pricing.yaml)
# ---------------------------------------------------------------------------


class RatePeriod(BaseModel):
    effective_from: datetime
    effective_until: datetime | None = None
    input_per_mtok: float
    output_per_mtok: float
    cache_write_5m_per_mtok: float | None = None
    cache_write_1h_per_mtok: float | None = None
    cache_read_per_mtok: float | None = None
    reasoning_per_mtok: float | None = None

    def covers(self, ts: datetime) -> bool:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return ts >= self.effective_from and (
            self.effective_until is None or ts < self.effective_until
        )


class ModelPricing(BaseModel):
    provider: str
    rates: list[RatePeriod]

    @field_validator("rates")
    @classmethod
    def _non_empty(cls, v: list[RatePeriod]) -> list[RatePeriod]:
        if not v:
            raise ValueError("model pricing must declare at least one rate period")
        return v

    def rate_for(self, ts: datetime) -> RatePeriod:
        matches = [r for r in self.rates if r.covers(ts)]
        if not matches:
            raise ConfigError(f"no pricing rate covers timestamp {ts.isoformat()}")
        if len(matches) > 1:
            raise ConfigError(f"overlapping pricing rate periods cover timestamp {ts.isoformat()}")
        return matches[0]


class PricingConfig(BaseModel):
    version: str
    currency: str = "USD"
    pricing_config_version: str
    models: dict[str, ModelPricing]

    def pricing_for(self, model_id: str) -> ModelPricing:
        m = self.models.get(model_id)
        if m is None:
            raise ConfigError(f"no pricing configured for model {model_id!r}")
        return m


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> tuple[dict, str]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ConfigError(f"config file did not parse to a mapping: {path}")
    return data, raw


def load_models_config(path: Path = DEFAULT_MODELS_PATH) -> tuple[ModelsConfig, str]:
    data, raw = _load_yaml(path)
    try:
        return ModelsConfig.model_validate(data), raw
    except Exception as exc:  # pydantic.ValidationError
        raise ConfigError(f"invalid models config {path}: {exc}") from exc


def load_pricing_config(path: Path = DEFAULT_PRICING_PATH) -> tuple[PricingConfig, str]:
    data, raw = _load_yaml(path)
    try:
        return PricingConfig.model_validate(data), raw
    except Exception as exc:
        raise ConfigError(f"invalid pricing config {path}: {exc}") from exc


class AppConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    secrets: Secrets
    models: ModelsConfig
    pricing: PricingConfig
    models_config_path: Path
    pricing_config_path: Path
    config_hash: str

    @model_validator(mode="after")
    def _cross_validate(self) -> AppConfig:
        missing = self.models.all_model_ids() - set(self.pricing.models.keys())
        if missing:
            raise ValueError(
                f"models configured without pricing entries: {sorted(missing)}"
            )
        return self

    def require_api_key(self, provider: str = "anthropic") -> str:
        """Look up the API key for `provider` (default "anthropic", so
        existing zero-arg call sites keep working unchanged). Raises
        ConfigError rather than returning None/empty — callers can rely on
        getting either a usable key or a clear failure."""
        key = self._resolve_api_key(provider)
        if not key:
            raise ConfigError(
                f"no API key configured for provider {provider!r} (env var or .env — "
                f"see .env.example). Refusing to make a real API call without it."
            )
        return key

    def _resolve_api_key(self, provider: str) -> str | None:
        if provider == "anthropic":
            return self.secrets.anthropic_api_key
        if provider == "openai":
            return self.secrets.openai_api_key
        if provider == "google":
            return self.secrets.resolved_google_api_key
        raise ConfigError(f"unknown provider {provider!r}")


def load_app_config(
    models_path: Path = DEFAULT_MODELS_PATH,
    pricing_path: Path = DEFAULT_PRICING_PATH,
) -> AppConfig:
    """Load and validate all configuration eagerly. Raises ConfigError on any
    problem so failures surface at startup, not mid-run."""
    models_cfg, models_raw = load_models_config(models_path)
    pricing_cfg, pricing_raw = load_pricing_config(pricing_path)
    secrets = Secrets()

    config_hash = hash_text(models_raw + "\n---\n" + pricing_raw)[:16]

    try:
        return AppConfig(
            secrets=secrets,
            models=models_cfg,
            pricing=pricing_cfg,
            models_config_path=models_path,
            pricing_config_path=pricing_path,
            config_hash=config_hash,
        )
    except Exception as exc:  # pydantic.ValidationError from cross-validation
        raise ConfigError(str(exc)) from exc
