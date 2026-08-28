"""Content-addressed cache for model completions.

The cache key is derived from everything that determines what a call would
produce: provider, model id, prompt content, the full generation
configuration, and sample index. Changing any of those changes the key, so
distinct experimental conditions can never collide, and identical conditions
always hit the same cache entry.

Only successful completions are cached — a transient error should not
permanently poison a cache slot.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from router.hashing import hash_object
from router.models.schemas import CompletionStatus, GenerationConfig, NormalizedCompletion


def compute_cache_key(
    provider: str,
    model_id: str,
    prompt_hash: str,
    generation_config: GenerationConfig,
    sample_index: int,
) -> str:
    material = {
        "provider": provider,
        "model_id": model_id,
        "prompt_hash": prompt_hash,
        "sample_index": sample_index,
        "generation_config": generation_config.model_dump(mode="json"),
    }
    return hash_object(material)


class ContentCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        # Two-level fan-out keeps any one directory from holding every entry.
        return self.cache_dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> NormalizedCompletion | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        return NormalizedCompletion.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, key: str, completion: NormalizedCompletion) -> None:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(completion.model_dump_json(), encoding="utf-8")

    def get_or_compute(
        self, key: str, compute_fn: Callable[[], NormalizedCompletion]
    ) -> tuple[NormalizedCompletion, bool]:
        """Returns (completion, cache_hit)."""
        cached = self.get(key)
        if cached is not None:
            return cached, True
        result = compute_fn()
        if result.status == CompletionStatus.OK:
            self.put(key, result)
        return result, False
