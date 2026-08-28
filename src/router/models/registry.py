"""Provider/adapter registry.

`ProviderModelsConfig.adapter` (a dotted class path, e.g.
`"router.models.anthropic_adapter.AnthropicModelClient"`) has been part of
`configs/models.yaml`'s schema since Stage 0, but nothing resolved it — every
caller imported a concrete adapter class by name. That's what made this
"an Anthropic router with a provider abstraction around it" rather than
genuinely provider-agnostic: adding a provider meant writing an adapter
*and* hunting down every hardcoded import.

`build_model_client` closes that gap: given an `AppConfig` and a provider
name, it resolves the configured adapter class dynamically and constructs
it. Callers (the pilot runner, experiment scripts, ...) no longer need to
know which adapter class backs which provider string.
"""

from __future__ import annotations

import importlib
import logging

from router.config import AppConfig, ConfigError
from router.models.base import ModelClient


def resolve_adapter_class(dotted_path: str) -> type[ModelClient]:
    module_path, sep, class_name = dotted_path.rpartition(".")
    if not sep:
        raise ConfigError(f"invalid adapter path {dotted_path!r} — expected 'module.path.ClassName'")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ConfigError(f"could not import adapter module {module_path!r}: {exc}") from exc
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ConfigError(f"module {module_path!r} has no class {class_name!r}") from exc
    if not (isinstance(cls, type) and issubclass(cls, ModelClient)):
        raise ConfigError(f"{dotted_path} is not a ModelClient subclass")
    return cls


def build_model_client(
    config: AppConfig,
    provider: str,
    api_key: str,
    base_url: str | None = None,
    max_retries: int | None = None,
    logger: logging.Logger | None = None,
) -> ModelClient:
    """Construct the configured adapter for `provider`. Every adapter shares
    the same constructor signature (api_key, base_url, max_retries, logger)
    by convention — see router.models.base.ModelClient."""
    provider_cfg = config.models.providers.get(provider)
    if provider_cfg is None:
        raise ConfigError(f"unknown provider {provider!r} in models.yaml")
    cls = resolve_adapter_class(provider_cfg.adapter)
    kwargs: dict = {"api_key": api_key, "base_url": base_url, "logger": logger}
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return cls(**kwargs)
