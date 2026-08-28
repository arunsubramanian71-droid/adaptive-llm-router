from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class RouteRequest(BaseModel):
    prompt: str


class RouteResponse(BaseModel):
    prompt: str
    selected_role: Literal["cheap", "strong"]
    probability: float
    policy_name: str
    selected_model_id: str
    estimated_cost_usd: float
    tau: float
    disclaimer: str


class InfoResponse(BaseModel):
    disclaimer: str
    strong_model_id: str
    cheap_model_id: str
    tau: float
    router_name: str
