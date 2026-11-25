"""Tests for src/validation.py and for frame_timestamps in src/video_io.py."""

import math

import numpy as np
import pytest

from src import validation
from src.synthetic import write_synthetic_video
from src.video_io import frame_timestamps


def test_error_measures_of_one_value():
    assert validation.error(10.0, 8.0) == 2.0
    assert validation.absolute_error(8.0, 10.0) == 2.0
    assert validation.percent_error(102.0, 100.0) == pytest.approx(2.0)
    assert validation.percent_error(98.0, 100.0) == pytest.approx(-2.0)
    assert validation.percent_error(-102.0, -100.0) == pytest.approx(-2.0)  # the sign says too high or too low, also for a negative reference


def test_a_percent_error_needs_a_reference_that_is_not_zero():
    with pytest.raises(ValueError, match="not zero"):
        validation.percent_error(1.0, 0.0)


def test_error_measures_of_a_series():
    measured, reference = [1.0, 2.0, 3.0], [1.0, 1.0, 1.0]
    assert validation.mean_absolute_error(measured, reference) == pytest.approx(1.0)
    assert validation.root_mean_square_error(measured, reference) == pytest.approx(math.sqrt(5.0 / 3.0))
    assert validation.bias(measured, reference) == pytest.approx(1.0)
    assert validation.bias([1.0, 3.0], [2.0, 2.0]) == pytest.approx(0.0)  # errors of both signs cancel in the bias
    assert validation.mean_absolute_error([1.0, 3.0], [2.0, 2.0]) == pytest.approx(1.0)  # and not in the MAE


def test_values_that_are_not_numbers_are_skipped_in_pairs():
    assert validation.mean_absolute_error([1.0, np.nan, 3.0], [2.0, 2.0, np.nan]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no pair"):
        validation.mean_absolute_error([np.nan], [1.0])
    with pytest.raises(ValueError, match="must match"):
        validation.mean_absolute_error([1.0, 2.0], [1.0])


def test_compare_value_with_a_percent_or_an_absolute_tolerance():
    inside = validation.compare_value("Speed", "120 rpm", "Mean speed", "rpm", 120.3, 120.0, tolerance_percent=0.5)
    assert inside["passed"] is True and inside["error"] == pytest.approx(0.3) and inside["percent_error"] == pytest.approx(0.25)
    assert inside["tolerance"] == "0.5 %"
    outside = validation.compare_value("Speed", "120 rpm", "Mean speed", "rpm", 121.0, 120.0, tolerance_percent=0.5)
    assert outside["passed"] is False
    points = validation.compare_value("Belt", "x", "Slip", "percentage points", 1.2, 1.0, tolerance_abs=0.3)
    assert points["passed"] is True and points["tolerance"] == "0.3 percentage points"
    assert validation.compare_value("Belt", "x", "Slip", "percentage points", 1.4, 1.0, tolerance_abs=0.3)["passed"] is False


def test_compare_value_without_a_tolerance_only_reports_the_error():
    row = validation.compare_value("Frame rate", "wrong on purpose", "Mean speed", "rpm", 60.0, 120.0)
    assert row["passed"] is None and row["tolerance"] == "none" and row["percent_error"] == pytest.approx(-50.0)


def test_a_zero_reference_works_with_an_absolute_tolerance_only():
    row = validation.compare_value("Belt", "x", "Slip", "percentage points", 0.05, 0.0, tolerance_abs=0.3)
    assert row["passed"] is True and row["percent_error"] is None
    with pytest.raises(ValueError, match="absolute tolerance"):
        validation.compare_value("Belt", "x", "Slip", "percentage points", 0.05, 0.0, tolerance_percent=1.0)


def test_compare_series_gives_the_error_measures():
    row = validation.compare_series("noise", "Speed in every frame", "rpm", [1.0, 2.0, np.nan, 4.0], [1.0, 1.0, 1.0, 1.0])
    assert row["n"] == 3 and row["mae"] == pytest.approx(4.0 / 3.0) and row["bias"] == pytest.approx(4.0 / 3.0)
    assert row["max_abs_error"] == pytest.approx(3.0) and row["rmse"] == pytest.approx(math.sqrt(10.0 / 3.0))


# ------------------------------------------------------------------ the time axis


def test_evenly_spaced_frames_that_match_the_frame_rate_pass():
    result = validation.check_time_axis(np.arange(120) / 60.0, 60.0)
    assert result["available"] and result["ok"] and not result["problems"]
    assert result["fps_from_times"] == pytest.approx(60.0) and result["max_irregularity_percent"] == pytest.approx(0.0, abs=1e-6)


def test_a_wrong_frame_rate_is_found_from_the_frame_times():
    result = validation.check_time_axis(np.arange(120) / 60.0, 30.0)
    assert not result["ok"] and result["step_error_percent"] == pytest.approx(-50.0)
    assert "the frame times say 60.00 fps but the analysis uses 30 fps" in result["problems"][0]


def test_dropped_frames_and_a_variable_frame_rate_are_found():
    times = np.arange(120) / 60.0
    times[60:] += 1.0 / 60.0  # one frame was dropped: one step is twice as long
    result = validation.check_time_axis(times, 60.0)
    assert not result["ok"] and result["max_irregularity_percent"] == pytest.approx(100.0)
    assert "variable frame rate or dropped frames" in result["problems"][0]
    assert result["longest_step_s"] == pytest.approx(2.0 / 60.0)


def test_frame_times_that_cannot_be_used_are_reported_and_not_judged():
    zeros = validation.check_time_axis(np.zeros(100), 60.0)
    assert zeros["available"] is False and "do not" in zeros["reason"].lower() or "does not" in zeros["reason"]
    assert validation.check_time_axis([0.0, 0.1], 10.0)["available"] is False
    assert validation.check_time_axis([0.0, np.nan, 0.2], 10.0)["available"] is False


def test_frame_timestamps_of_a_video(tmp_path):
    path = tmp_path / "disc.mp4"
    write_synthetic_video(path, rpm=60.0, fps=30.0, seconds=2.0)
    times = frame_timestamps(path)
    assert len(times) == 60 and np.allclose(np.diff(times), 1.0 / 30.0, atol=1e-4)
    assert len(frame_timestamps(path, max_frames=10)) == 10
    later = frame_timestamps(path, start_frame=20)
    assert len(later) == 40 and later[0] == pytest.approx(20.0 / 30.0, abs=0.04)
    assert validation.check_time_axis(times, 30.0)["ok"]
