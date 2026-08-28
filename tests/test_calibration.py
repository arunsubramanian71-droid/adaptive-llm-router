from __future__ import annotations

import pytest

from router.routers.calibration import (
    CalibrationMethod,
    ProbabilityCalibrator,
    brier_score,
    expected_calibration_error,
)


def test_brier_score_perfect_predictions_is_zero():
    assert brier_score([1, 0, 1, 0], [1.0, 0.0, 1.0, 0.0]) == pytest.approx(0.0)


def test_brier_score_worst_case_predictions_is_one():
    assert brier_score([1, 0], [0.0, 1.0]) == pytest.approx(1.0)


def test_brier_score_uninformative_midpoint():
    assert brier_score([1, 0], [0.5, 0.5]) == pytest.approx(0.25)


def test_ece_perfectly_calibrated_is_near_zero():
    # Half the "0.9" predictions are correct... use groups matching stated
    # confidence exactly so each bin's accuracy equals its confidence.
    y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # 40% positive overall
    probs = [0.4] * 10
    assert expected_calibration_error(y_true, probs, n_bins=10) == pytest.approx(0.0, abs=1e-6)


def test_ece_overconfident_predictions_is_high():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]  # only 50% actually positive
    probs = [0.95] * 8  # but predicted very confident
    ece = expected_calibration_error(y_true, probs, n_bins=10)
    assert ece > 0.4


def test_ece_empty_input_is_zero():
    assert expected_calibration_error([], [], n_bins=10) == 0.0


@pytest.mark.parametrize("method", [CalibrationMethod.PLATT, CalibrationMethod.ISOTONIC])
def test_calibrator_output_in_range_and_monotonic_ish(method):
    y_true = [0] * 20 + [1] * 20
    raw_probs = [i / 40 for i in range(40)]  # monotonically increasing raw score
    calibrator = ProbabilityCalibrator(method=method).fit(y_true, raw_probs)

    calibrated = calibrator.transform(raw_probs)
    assert len(calibrated) == 40
    assert all(0.0 <= p <= 1.0 for p in calibrated)
    # Higher raw score should not map to a *lower* calibrated probability
    # for this monotonic training signal.
    assert calibrated[-1] >= calibrated[0]


def test_calibrator_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        ProbabilityCalibrator().transform([0.5])


def test_calibration_improves_or_maintains_brier_on_miscalibrated_scores():
    # Raw scores are systematically overconfident (always near 0 or 1) even
    # though the true positive rate in each extreme group is only 70%.
    y_true = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0] * 2 + [0, 0, 0, 0, 0, 0, 0, 1, 1, 1] * 2
    raw_probs = [0.95] * 20 + [0.05] * 20

    raw_brier = brier_score(y_true, raw_probs)
    calibrator = ProbabilityCalibrator(method=CalibrationMethod.PLATT).fit(y_true, raw_probs)
    calibrated_brier = brier_score(y_true, calibrator.transform(raw_probs))

    assert calibrated_brier <= raw_brier + 1e-9
