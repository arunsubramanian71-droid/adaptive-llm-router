"""Response-level persistence.

One record per (prompt, provider, model, generation configuration,
sample_index) — never aggregated away. The schema carries placeholder
fields (`score_status`, `q_hat`, `label`) so a later offline scoring stage
can fill them in without changing the record shape or re-calling any API.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from router.cost.schemas import CostBreakdown
from router.models.schemas import GenerationConfig, TokenUsage


class ResponseRecord(BaseModel):
    record_id: str  # content-addressed cache key for this exact call
    run_id: str

    prompt_id: str
    prompt_hash: str
    prompt_text: str | None = None

    provider: str
    requested_model_id: str
    served_model_id: str | None = None
    sample_index: int

    generation_config: GenerationConfig

    status: str
    stop_reason: str | None = None
    truncated: bool = False
    parse_ok: bool | None = None  # filled by a later scoring stage

    response_text: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost: CostBreakdown | None = None

    latency_ms: float | None = None
    timestamp_utc: datetime
    request_id: str | None = None

    pricing_config_version: str | None = None
    git_sha: str | None = None

    cache_hit: bool = False
    retries: int = 0
    error_type: str | None = None
    error_message: str | None = None

    # Evaluation placeholders for the (not-yet-built) scoring stage.
    score_status: str = "pending"
    q_hat: float | None = None
    label: str | None = None

    raw_response: dict[str, Any] | None = None


class JsonlStore:
    """Append-only JSON-Lines store — one line per ResponseRecord."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ResponseRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json())
            f.write("\n")

    def read_all(self) -> list[ResponseRecord]:
        if not self.path.exists():
            return []
        records: list[ResponseRecord] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(ResponseRecord.model_validate_json(line))
        return records
