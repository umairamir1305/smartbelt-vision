"""Tests for src/belt_drive.py. Two simulated pulleys with known speeds are used."""

import math

import numpy as np
import pandas as pd
import pytest

from src.belt_drive import (
    add_belt_columns,
    common_rotation,
    expected_ratio_from,
    merge_pulley_tables,
    summarise_belt_drive,
)
from src.kinematics import AnalysisError, add_angle_columns, add_speed_columns
from src.signal_analysis import analyse_speed_signal
from src.synthetic import make_synthetic_table
from src.tracking import add_tracking_quality

DRIVER_CENTRE = (200.0, 150.0)
DRIVEN_CENTRE = (500.0, 150.0)


def pulley(rpm, centre, radius_px, lost=(), noise_px=0.05, seed=1, signal=True, **kwargs):
    """A simulated pulley taken through angle, speed and (optionally) the signal analysis."""
    table = make_synthetic_table(rpm=rpm, centre=centre, radius_px=radius_px, lost_frames=lost, noise_px=noise_px, seed=seed, **kwargs)
    table = add_speed_columns(add_angle_columns(add_tracking_quality(table, centre), centre))
    if signal:
        table, _ = analyse_speed_signal(table)
    return table


def drive(driver_rpm=120.0, ratio=2.0, slip_percent=0.0, crossed=False, driver_lost=(), driven_lost=(), **kwargs):
    """Merged table of a driver and a driven pulley. The driven pulley has the ideal speed minus the slip."""
    driven_rpm = driver_rpm / ratio * (1 - slip_percent / 100.0) * (-1 if crossed else 1)
    driver = pulley(driver_rpm, DRIVER_CENTRE, 80.0, lost=driver_lost, seed=1, **kwargs)
    driven = pulley(driven_rpm, DRIVEN_CENTRE, 120.0, lost=driven_lost, seed=2, **kwargs)
    return merge_pulley_tables(driver, driven)


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def test_merge_prefixes_every_column_but_frame_and_time():
    driver = pulley(120.0, DRIVER_CENTRE, 80.0, signal=False)
    driven = pulley(60.0, DRIVEN_CENTRE, 120.0, signal=False)
    merged = merge_pulley_tables(driver, driven)

    assert merged.columns[:2].tolist() == ["frame", "time_s"]
    assert len(merged) == len(driver)
    for column in driver.columns:
        if column not in ("frame", "time_s"):
            assert f"driver_{column}" in merged.columns and f"driven_{column}" in merged.columns
    assert merged["driver_rpm"].mean() == pytest.approx(120.0, abs=0.5)
    assert merged["driven_rpm"].mean() == pytest.approx(60.0, abs=0.5)


def test_merge_rejects_tables_of_different_frames():
    one = pulley(120.0, DRIVER_CENTRE, 80.0, signal=False, seconds=5.0)
    other = pulley(60.0, DRIVEN_CENTRE, 120.0, signal=False, seconds=4.0)
    with pytest.raises(AnalysisError, match="same frames"):
        merge_pulley_tables(one, other)

    shifted = other.copy()
    shifted = pulley(60.0, DRIVEN_CENTRE, 120.0, signal=False, seconds=5.0, fps=30.0)
    with pytest.raises(AnalysisError):
        merge_pulley_tables(one, shifted)


# ---------------------------------------------------------------------------
# Expected ratio
# ---------------------------------------------------------------------------


def test_expected_ratio_from_radii_or_from_a_reference():
    assert expected_ratio_from(25.0, 50.0) == (2.0, "pulley radii")
    assert expected_ratio_from(25.0, 50.0, reference_ratio=2.03) == (2.03, "reference ratio")  # the reference wins
    assert expected_ratio_from(reference_ratio=1.5) == (1.5, "reference ratio")
    assert expected_ratio_from() == (None, None)
    assert expected_ratio_from(25.0, None) == (None, None)  # one radius is not enough


def test_expected_ratio_rejects_non_positive_values():
    with pytest.raises(AnalysisError):
        expected_ratio_from(0.0, 50.0)
    with pytest.raises(AnalysisError):
        expected_ratio_from(reference_ratio=-1.0)


# ---------------------------------------------------------------------------
# Rotation in common and the ratio
# ---------------------------------------------------------------------------


def test_common_rotation_counts_the_turns_of_both_pulleys():
    merged = drive(120.0, 2.0)
    rotation = common_rotation(merged)
    assert rotation["runs"] == 1 and rotation["common_frames"] == 300
    assert rotation["duration_s"] == pytest.approx(299 / 60.0)
    assert rotation["driver_rotation_rad"] / (2 * math.pi) == pytest.approx(2.0 * 299 / 60.0, rel=1e-3)  # 2 turns per second
    assert rotation["driven_rotation_rad"] / (2 * math.pi) == pytest.approx(1.0 * 299 / 60.0, rel=1e-3)


def test_a_short_gap_in_one_pulley_changes_nothing():
    merged = drive(120.0, 2.0, driven_lost=[100, 101])
    rotation = common_rotation(merged)
    assert rotation["runs"] == 1
    summary = summarise_belt_drive(merged)
    assert summary["measured_ratio"] == pytest.approx(2.0, rel=1e-3)


def test_a_long_gap_splits_the_runs_and_the_ratio_stays_right():
    # 12 degrees per frame for the driver and 6 for the driven: a gap of 12 frames is 156 degrees for the driver.
    merged = drive(120.0, 2.0, driver_lost=range(100, 112))
    assert merged["driver_segment"].dropna().nunique() == 2
    rotation = common_rotation(merged)
    assert rotation["runs"] == 2

    summary = summarise_belt_drive(merged)
    assert summary["measured_ratio"] == pytest.approx(2.0, rel=2e-3)
    # The time of the gap is left out of the duration.
    assert rotation["duration_s"] < 299 / 60.0 - 11 / 60.0


def test_pulleys_with_no_common_frames_cannot_be_compared():
    merged = drive(120.0, 2.0, seconds=2.0)
    merged["driven_angle_rad"] = np.nan
    with pytest.raises(AnalysisError, match="fewer than 2 frames in common"):
        common_rotation(merged)


# ---------------------------------------------------------------------------
# Per-frame columns
# ---------------------------------------------------------------------------


def test_ratio_column_and_surface_speeds_for_an_ideal_belt():
    merged = add_belt_columns(drive(120.0, 2.0), expected_ratio=2.0, driver_radius_mm=25.0, driven_radius_mm=50.0)
    ratio = merged["speed_ratio"]
    assert ratio.dropna().median() == pytest.approx(2.0, rel=0.01)
    assert merged["speed_loss_percent"].dropna().abs().max() < 2.0  # noise only

    # 120 rpm on 25 mm is omega x R = 12.566 x 0.025 = 0.314 m/s on both pulleys.
    assert merged["driver_surface_speed_m_s"].dropna().median() == pytest.approx(0.314, rel=0.01)
    assert merged["driven_surface_speed_m_s"].dropna().median() == pytest.approx(0.314, rel=0.01)


def test_a_slower_driven_pulley_gives_the_slip_as_speed_loss():
    merged = add_belt_columns(drive(120.0, 2.0, slip_percent=2.0), expected_ratio=2.0)
    assert merged["speed_ratio"].dropna().median() == pytest.approx(2.0 / 0.98, rel=0.01)
    assert merged["speed_loss_percent"].dropna().median() == pytest.approx(2.0, abs=0.3)


def test_a_crossed_belt_gives_a_negative_ratio():
    merged = add_belt_columns(drive(120.0, 2.0, crossed=True))
    assert merged["speed_ratio"].dropna().median() == pytest.approx(-2.0, rel=0.01)


def test_the_single_frame_speed_is_used_when_there_is_no_smoothing():
    merged = drive(120.0, 2.0, signal=False)
    assert "driver_rpm_smooth" not in merged.columns
    assert add_belt_columns(merged)["speed_ratio"].dropna().median() == pytest.approx(2.0, rel=0.01)


def test_frames_where_the_driven_pulley_is_almost_stopped_have_no_ratio():
    merged = drive(120.0, 2.0)
    merged.loc[50:60, "driven_rpm_smooth"] = 0.5  # nearly stopped (the typical speed is 60 rpm)
    ratio = add_belt_columns(merged)["speed_ratio"]
    assert ratio[50:61].isna().all()
    assert ratio.notna().sum() > 250


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_of_an_ideal_belt_with_radii():
    merged = add_belt_columns(drive(120.0, 2.0), 2.0, 25.0, 50.0)
    summary = summarise_belt_drive(merged, 2.0, "pulley radii", 25.0, 50.0)

    assert summary["measured_ratio"] == pytest.approx(2.0, rel=1e-3)
    assert summary["direction"] == "same direction (open belt)"
    assert summary["expected_source"] == "pulley radii"
    assert summary["ratio_deviation_percent"] == pytest.approx(0.0, abs=0.1)
    assert summary["speed_loss_percent"] == pytest.approx(0.0, abs=0.1)
    assert summary["driver_mean_rpm"] == pytest.approx(120.0, rel=2e-3)
    assert summary["driven_mean_rpm"] == pytest.approx(60.0, rel=2e-3)
    assert summary["driver_surface_speed_m_s"] == pytest.approx(0.314, rel=0.01)
    assert summary["driven_surface_speed_m_s"] == pytest.approx(0.314, rel=0.01)
    # A 1 mm error in a 25 mm radius is 4 percent, in a 50 mm radius 2 percent.
    assert summary["radius_sensitivity_driver_percent_per_mm"] == pytest.approx(4.0)
    assert summary["radius_sensitivity_driven_percent_per_mm"] == pytest.approx(2.0)
    assert summary["ratio_range"][0] < 2.0 < summary["ratio_range"][1]


def test_summary_finds_the_slip_and_the_deviation_of_the_ratio():
    merged = add_belt_columns(drive(120.0, 2.0, slip_percent=3.0), 2.0)
    summary = summarise_belt_drive(merged, 2.0, "pulley radii")

    assert summary["measured_ratio"] == pytest.approx(2.0 / 0.97, rel=1e-3)
    assert summary["speed_loss_percent"] == pytest.approx(3.0, abs=0.1)
    # The ratio is 3.09 percent above the expected one, the speed loss is 3 percent of the ideal driven speed.
    assert summary["ratio_deviation_percent"] == pytest.approx(3.09, abs=0.15)
    assert summary["driver_surface_speed_m_s"] is None  # no radii were given


def test_a_wrong_radius_looks_like_slip_and_a_reference_ratio_removes_it():
    merged = add_belt_columns(drive(120.0, 2.0), 2.0)

    # The driven radius is typed in 1 mm too small (49 instead of 50 mm): the expected ratio is 1.96, so the
    # driven pulley should turn at 120 / 1.96 = 61.2 rpm. It turns at 60 rpm, so an ideal belt seems to
    # have a speed loss of 2 percent. A radius error of 1 mm on 50 mm is a fake slip of 2 percent.
    wrong_expected, _ = expected_ratio_from(25.0, 49.0)
    assert summarise_belt_drive(merged, wrong_expected, "pulley radii")["speed_loss_percent"] == pytest.approx(2.0, abs=0.15)

    # A reference ratio measured with no load (here the same ideal drive) gives 0.
    reference = summarise_belt_drive(merged)["measured_ratio"]
    assert summarise_belt_drive(merged, reference, "reference ratio")["speed_loss_percent"] == pytest.approx(0.0, abs=1e-9)


def test_summary_without_an_expected_ratio_has_no_deviation():
    summary = summarise_belt_drive(drive(120.0, 2.0))
    assert summary["measured_ratio"] == pytest.approx(2.0, rel=1e-3)
    assert summary["expected_ratio"] is None
    assert summary["ratio_deviation_percent"] is None and summary["speed_loss_percent"] is None


def test_summary_of_a_crossed_belt():
    summary = summarise_belt_drive(drive(120.0, 2.0, crossed=True), 2.0, "pulley radii")
    assert summary["direction"] == "opposite directions (crossed belt)"
    assert summary["measured_ratio"] == pytest.approx(2.0, rel=1e-3)
    assert summary["speed_loss_percent"] == pytest.approx(0.0, abs=0.1)


def test_a_pulley_that_does_not_turn_gives_an_error():
    merged = drive(120.0, 2.0)
    still = merged.copy()
    still["driven_angle_rad"] = 1.0 + still["driven_angle_rad"].to_numpy()[0] * 0  # a constant angle
    with pytest.raises(AnalysisError, match="driven pulley turned less than 5 degrees"):
        summarise_belt_drive(still)

    still = merged.copy()
    still["driver_angle_rad"] = 1.0
    with pytest.raises(AnalysisError, match="driver pulley turned less than 5 degrees"):
        summarise_belt_drive(still)


def test_the_inputs_are_not_changed():
    merged = drive(120.0, 2.0, seconds=2.0)
    columns = list(merged.columns)
    add_belt_columns(merged, 2.0, 25.0, 50.0)
    assert list(merged.columns) == columns
    assert isinstance(merged, pd.DataFrame)
