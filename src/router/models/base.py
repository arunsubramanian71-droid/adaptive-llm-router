"""Provider-agnostic model client interface.

Every provider adapter (Anthropic today; others later) implements this one
method and returns a `NormalizedCompletion`. Nothing outside `router.models`
should import a provider SDK directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from router.config import ModelEntry
from router.models.schemas import NormalizedCompletion


class ModelClient(ABC):
    provider: str

    @abstractmethod
    def complete(
        self,
        prompt: str,
        model_entry: ModelEntry,
        sample_index: int,
        system_prompt: str | None = None,
    ) -> NormalizedCompletion:
        """Make one completion call and return a normalized result.

        Must never raise for ordinary provider errors (bad request, rate
        limit, timeout, malformed response) — those are captured in the
        returned `NormalizedCompletion.status`/`error_*` fields instead, so
        callers can persist a record for every attempted call.
        """
        raise NotImplementedError
