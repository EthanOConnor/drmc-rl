from __future__ import annotations

import pytest

from drmc_rl.search.strong_league import DavidsonCalibration


def test_davidson_calibration_is_normalized_and_monotone() -> None:
    calibration = DavidsonCalibration(
        slope=1.7,
        bias=-0.1,
        draw_logit=-2.5,
        artifact_sha256="test",
    )
    low = calibration.wdl(-0.8)
    middle = calibration.wdl(0.0)
    high = calibration.wdl(0.8)
    assert low.win < middle.win < high.win
    assert low.loss > middle.loss > high.loss
    for value in (low, middle, high):
        assert value.win + value.draw + value.loss == pytest.approx(1.0)
