from __future__ import annotations

from pydantic import BaseModel


class CostBreakdown(BaseModel):
    """Transparent per-call cost breakdown — never collapse to one number
    without keeping the parts, since later analysis needs to attribute cost
    to cache usage vs. fresh generation."""

    model_id: str
    pricing_config_version: str
    currency: str

    input_cost: float
    output_cost: float
    cache_write_cost: float
    cache_read_cost: float
    reasoning_cost: float

    @property
    def total_cost(self) -> float:
        return (
            self.input_cost
            + self.output_cost
            + self.cache_write_cost
            + self.cache_read_cost
            + self.reasoning_cost
        )
