"""Probability calibration for router output.

A router's raw `predict_proba` is not necessarily a calibrated probability
(e.g. "of the prompts scored 0.7, 70% actually needed strong"). This module
fits a 1-D calibration map from raw score -> calibrated probability (Platt
scaling = logistic regression on the score; isotonic = a monotonic
step function), and provides the two standard ways to check whether it
worked: Brier score and Expected Calibration Error (ECE).
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


class CalibrationMethod(str, Enum):
    PLATT = "platt"
    ISOTONIC = "isotonic"


class ProbabilityCalibrator:
    def __init__(self, method: CalibrationMethod = CalibrationMethod.PLATT) -> None:
        self.method = method
        self._model: LogisticRegression | IsotonicRegression | None = None

    def fit(self, y_true: list[int], raw_probs: list[float]) -> ProbabilityCalibrator:
        y = np.asarray(y_true, dtype=int)
        p = np.asarray(raw_probs, dtype=float)
        if self.method == CalibrationMethod.PLATT:
            model = LogisticRegression()
            model.fit(p.reshape(-1, 1), y)
        else:
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(p, y)
        self._model = model
        return self

    def transform(self, raw_probs: list[float]) -> list[float]:
        if self._model is None:
            raise RuntimeError("calibrator has not been fit yet")
        p = np.asarray(raw_probs, dtype=float)
        if self.method == CalibrationMethod.PLATT:
            idx = list(self._model.classes_).index(1)
            return self._model.predict_proba(p.reshape(-1, 1))[:, idx].tolist()
        return self._model.transform(p).tolist()


def brier_score(y_true: list[int], probs: list[float]) -> float:
    return float(brier_score_loss(y_true, probs))


def expected_calibration_error(y_true: list[int], probs: list[float], n_bins: int = 10) -> float:
    """Standard ECE: partition [0, 1] into `n_bins` equal-width bins by
    predicted probability, and take the sample-weighted average gap
    between each bin's mean predicted probability and its actual positive
    rate."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    n = len(p)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        if not mask.any():
            continue
        bin_confidence = p[mask].mean()
        bin_accuracy = y[mask].mean()
        ece += (mask.sum() / n) * abs(bin_accuracy - bin_confidence)
    return float(ece)
