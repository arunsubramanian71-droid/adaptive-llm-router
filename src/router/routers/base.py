"""Learned-router interface.

A `Router` predicts P(this prompt needs the strong model) from the prompt
text alone — no access to `q_hat`/`delta_hat` (that's the oracle's
privilege, see `router.policies.oracle`). `predict_proba` returns raw,
uncalibrated model output; `router.routers.calibration` turns that into a
calibrated probability, and `router.policies.router_policy.RouterPolicy`
turns a calibrated probability plus a threshold into a deployable `Policy`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Router(ABC):
    name: str

    @abstractmethod
    def fit(self, prompts: list[str], labels: list[int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, prompts: list[str]) -> list[float]:
        raise NotImplementedError


def validate_binary_labels(labels: list[int]) -> None:
    unique = set(labels)
    if unique - {0, 1}:
        raise ValueError(f"labels must be 0/1, got values {sorted(unique)}")
    if len(unique) < 2:
        raise ValueError(
            "need both classes (0 and 1) present in training labels to fit a router "
            f"(got only {unique})"
        )
