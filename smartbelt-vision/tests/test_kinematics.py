"""Tests for src/kinematics.py: angle convention, unwrapping, gaps, rotation summary and the circle fit."""

import math

import numpy as np
import pandas as pd
import pytest

from src.kinematics import (
    AnalysisError,
    MIN_VALID_FRAMES,
    add_angle_columns,
    add_speed_columns,
    angle_from_position,
    fit_centre_from_track,
    fit_circle,
    rad_s_to_rpm,
    rpm_to_rad_s,
    summarise_rotation,
    summarise_speed,
)
from src.synthetic import disc_geometry, true_marker_angle, true_marker_speed_rpm, write_synthetic_video
from src.tracking import settings_for_colour, track_marker

CENTRE = (200.0, 150.0)
RADIUS = 100.0


def table_from_angles(angles_rad, valid=None, centre=CENTRE, radius=RADIUS, fps=60.0):
    """A tracking table for a marker that really is at the given angles (counter-clockwise positive).

    valid is an optional list of True/False per frame. Frames that are not valid have no position.
    """
    angles_rad = np.asarray(angles_rad, dtype=float)
    count = len(angles_rad)
    table = pd.DataFrame(
        {
            "frame": np.arange(count),
            "time_s": np.arange(count) / fps,
            "marker_x": centre[0] + radius * np.cos(angles_rad),
            "marker_y": centre[1] - radius * np.sin(angles_rad),
        }
    )
    valid = np.ones(count, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    table.loc[~valid, ["marker_x", "marker_y"]] = np.nan
    table["tracking_valid"] = valid
    return table


def steady_angles(count, step_deg):
    return np.radians(step_deg) * np.arange(count)


# ---------------------------------------------------------------------------
# Angle convention
# ---------------------------------------------------------------------------


def test_angle_from_position_in_the_main_directions():
    centre = (100, 100)
    assert angle_from_position(200, 100, centre) == pytest.approx(0.0)  # right of the centre
    assert angle_from_position(100, 0, centre) == pytest.approx(math.pi / 2)  # above (smaller y)
    assert abs(angle_from_position(0, 100, centre)) == pytest.approx(math.pi)  # left
    assert angle_from_position(100, 200, centre) == pytest.approx(-math.pi / 2)  # below (larger y)
    assert angle_from_position(200, 0, centre) == pytest.approx(math.pi / 4)  # up and right
    assert angle_from_position(0, 200, centre) == pytest.approx(-3 * math.pi / 4)  # down and left


def test_angle_from_position_gives_nan_for_missing_positions():
    angles = angle_from_position([np.nan, 200.0], [np.nan, 100.0], (100, 100))
    assert np.isnan(angles[0])
    assert angles[1] == pytest.approx(0.0)


def test_angle_convention_matches_the_simulated_video_generator():
    # The simulator puts the marker at the true angle. Calculating the angle back must give the same value.
    width, height = 320, 240
    centre, _, orbit_radius, _ = disc_geometry(width, height)
    for degrees in (0, 30, 100, 179, -45, -170):
        angle = math.radians(degrees)
        x = centre[0] + orbit_radius * math.cos(angle)
        y = centre[1] - orbit_radius * math.sin(angle)
        assert angle_from_position(x, y, centre) == pytest.approx(angle)


# ---------------------------------------------------------------------------
# Unwrapping
# ---------------------------------------------------------------------------


def test_unwrapping_removes_the_jump_at_pi():
    truth = steady_angles(80, 17.0)  # 80 steps of 17 degrees: about 3.8 turns
    result = add_angle_columns(table_from_angles(truth), CENTRE)

    wrapped = result["angle_wrapped_rad"].to_numpy()
    assert wrapped.max() <= math.pi and wrapped.min() > -math.pi
    assert np.abs(np.diff(wrapped)).max() > 5.0  # the wrapped angle does jump by almost 2 pi
    assert np.allclose(result["angle_rad"].to_numpy(), truth)  # the unwrapped angle does not
    assert (result["segment"] == 0).all()


def test_unwrapping_clockwise_rotation():
    truth = -steady_angles(80, 17.0)
    result = add_angle_columns(table_from_angles(truth), CENTRE)
    assert np.allclose(result["angle_rad"].to_numpy(), truth)


def test_unwrapped_angle_starts_at_the_first_wrapped_angle():
    # A true start angle of 4 rad is outside -pi to +pi. The unwrapped series starts at the wrapped
    # value (4 - 2 pi) and is shifted by a whole number of turns, but the differences are right.
    truth = 4.0 + steady_angles(40, 15.0)
    result = add_angle_columns(table_from_angles(truth), CENTRE)
    angle = result["angle_rad"].to_numpy()
    assert angle[0] == pytest.approx(4.0 - 2 * math.pi)
    assert np.allclose(angle - angle[0], truth - truth[0])


def test_frames_without_a_trusted_position_have_no_angle():
    truth = steady_angles(40, 10.0)
    valid = np.ones(40, dtype=bool)
    valid[[5, 6, 20]] = False
    result = add_angle_columns(table_from_angles(truth, valid), CENTRE)

    assert result.loc[[5, 6, 20], "angle_rad"].isna().all()
    assert result.loc[[5, 6, 20], "angle_wrapped_rad"].isna().all()
    assert result.loc[[5, 6, 20], "segment"].isna().all()
    assert np.allclose(result.loc[valid, "angle_rad"].to_numpy(), truth[valid])  # the rest is still right
    assert (result.loc[valid, "segment"] == 0).all()


def test_the_input_table_is_not_changed():
    table = table_from_angles(steady_angles(30, 10.0))
    add_angle_columns(table, CENTRE)
    assert "angle_rad" not in table.columns


# ---------------------------------------------------------------------------
# Gaps and segments
# ---------------------------------------------------------------------------


def test_a_short_gap_does_not_split_the_angle():
    # 20 degrees per frame, 3 frames lost: the marker turns 4 x 20 = 80 degrees across the gap (under 90).
    truth = steady_angles(40, 20.0)
    valid = np.ones(40, dtype=bool)
    valid[15:18] = False
    result = add_angle_columns(table_from_angles(truth, valid), CENTRE)

    assert result["segment"].dropna().nunique() == 1
    assert np.allclose(result.loc[valid, "angle_rad"].to_numpy(), truth[valid])


def test_a_long_gap_splits_the_angle_into_segments():
    # 20 degrees per frame, 5 frames lost: 6 x 20 = 120 degrees across the gap (over 90).
    truth = steady_angles(40, 20.0)
    valid = np.ones(40, dtype=bool)
    valid[15:20] = False
    result = add_angle_columns(table_from_angles(truth, valid), CENTRE)

    segment = result["segment"]
    assert (segment[:15] == 0).all()
    assert segment[15:20].isna().all()
    assert (segment[20:] == 1).all()
    # 120 degrees is less than half a turn, so the unwrapped angle after the gap is still right.
    assert np.allclose(result.loc[valid, "angle_rad"].to_numpy(), truth[valid])


def test_a_gap_longer_than_half_a_turn_gives_a_wrong_angle_and_is_flagged():
    # 50 degrees per frame, 4 frames lost: 5 x 50 = 250 degrees across the gap. np.unwrap cannot know that
    # and reads it as 250 - 360 = -110 degrees. Every angle after the gap is one turn too low.
    # This is the reason for the segment rule: the gap must be flagged.
    truth = steady_angles(30, 50.0)
    valid = np.ones(30, dtype=bool)
    valid[10:14] = False
    result = add_angle_columns(table_from_angles(truth, valid), CENTRE)

    assert (result["segment"][14:] == 1).all()
    assert np.allclose(result.loc[valid, "angle_rad"].to_numpy()[:10], truth[:10])
    after_gap = result["angle_rad"].to_numpy()[14:]
    assert np.allclose(after_gap, truth[14:] - 2 * math.pi)


def test_gaps_at_the_start_and_the_end_do_not_create_segments():
    truth = steady_angles(40, 20.0)
    valid = np.ones(40, dtype=bool)
    valid[:8] = False
    valid[-6:] = False
    result = add_angle_columns(table_from_angles(truth, valid), CENTRE)
    assert result["segment"].dropna().nunique() == 1


def test_rate_after_the_gap_is_used_when_there_is_nothing_before_it():
    # Only the very first frame is valid before the gap, so the rate has to come from the frames after it.
    truth = steady_angles(30, 20.0)
    valid = np.ones(30, dtype=bool)
    valid[1:5] = False  # 4 frames lost: 5 x 20 = 100 degrees across the gap
    result = add_angle_columns(table_from_angles(truth, valid), CENTRE)
    assert result["segment"].dropna().nunique() == 2


def test_too_few_valid_frames_raise_an_error():
    truth = steady_angles(40, 10.0)
    valid = np.zeros(40, dtype=bool)
    valid[: MIN_VALID_FRAMES - 1] = True
    with pytest.raises(AnalysisError, match="trusted marker position"):
        add_angle_columns(table_from_angles(truth, valid), CENTRE)


# ---------------------------------------------------------------------------
# Rotation summary
# ---------------------------------------------------------------------------


def test_summary_of_a_counter_clockwise_rotation():
    truth = np.linspace(0.0, 6 * math.pi, 181)  # exactly 3 turns in 180 steps
    summary = summarise_rotation(add_angle_columns(table_from_angles(truth), CENTRE))

    assert summary["direction"] == "counter-clockwise"
    assert summary["turns"] == pytest.approx(3.0)
    assert summary["total_rotation_rad"] == pytest.approx(6 * math.pi)
    assert summary["angle_frames"] == 181
    assert (summary["first_frame"], summary["last_frame"]) == (0, 180)
    assert summary["duration_s"] == pytest.approx(180 / 60.0)
    assert summary["segments"] == 1


def test_summary_of_a_clockwise_rotation():
    truth = -np.linspace(0.0, 4 * math.pi, 101)
    summary = summarise_rotation(add_angle_columns(table_from_angles(truth), CENTRE))
    assert summary["direction"] == "clockwise"
    assert summary["turns"] == pytest.approx(-2.0)


def test_summary_of_a_marker_that_hardly_moves():
    truth = np.full(30, 1.0) + np.linspace(0.0, 0.02, 30)
    summary = summarise_rotation(add_angle_columns(table_from_angles(truth), CENTRE))
    assert summary["direction"] == "no clear rotation"


def test_summary_with_a_split_gap_separates_the_two_totals():
    truth = steady_angles(40, 20.0)
    valid = np.ones(40, dtype=bool)
    valid[15:20] = False  # 120 degrees across the gap, so the angle is split
    summary = summarise_rotation(add_angle_columns(table_from_angles(truth, valid), CENTRE))

    assert summary["segments"] == 2
    assert math.degrees(summary["total_rotation_rad"]) == pytest.approx(39 * 20.0)  # first to last frame
    # Inside the segments: frames 0 to 14 (14 steps) and frames 20 to 39 (19 steps).
    assert math.degrees(summary["rotation_in_segments_rad"]) == pytest.approx((14 + 19) * 20.0)
    assert summary["longest_segment_frames"] == 20


def test_summary_needs_angle_values():
    table = add_angle_columns(table_from_angles(steady_angles(30, 10.0)), CENTRE)
    table["angle_rad"] = np.nan
    with pytest.raises(AnalysisError):
        summarise_rotation(table)


# ---------------------------------------------------------------------------
# Effect of a wrong centre
# ---------------------------------------------------------------------------


def test_a_wrong_centre_gives_an_angle_error_once_per_turn():
    truth = np.radians(np.arange(0, 720, 1.0))  # two turns in 1 degree steps
    table = table_from_angles(truth, centre=(200.0, 150.0), radius=100.0)

    offset = 5.0  # centre given 5 pixels too far to the right
    wrong = add_angle_columns(table, (200.0 + offset, 150.0))
    error = wrong["angle_rad"].to_numpy() - truth

    # The largest angle error is asin(offset / radius), about 2.87 degrees here.
    assert np.abs(error).max() == pytest.approx(math.asin(offset / 100.0), rel=0.02)
    # It repeats every turn: the error one turn later is the same.
    assert np.allclose(error[:360], error[360:], atol=1e-9)
    # And it averages out over whole turns, so the mean speed is not affected by it.
    assert error[360] - error[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Circle fit
# ---------------------------------------------------------------------------


def test_fit_circle_recovers_a_perfect_circle():
    angles = np.radians(np.arange(0, 360, 7.0))
    x = 1500.0 + 300.0 * np.cos(angles)  # large coordinates, as in a 1080p frame
    y = 900.0 - 300.0 * np.sin(angles)
    centre_x, centre_y, radius, rms = fit_circle(x, y)
    assert (centre_x, centre_y, radius) == pytest.approx((1500.0, 900.0, 300.0), abs=1e-6)
    assert rms == pytest.approx(0.0, abs=1e-6)


def test_fit_circle_also_works_on_a_short_arc_without_noise():
    angles = np.radians(np.arange(0, 90, 3.0))
    x = 200.0 + 100.0 * np.cos(angles)
    y = 150.0 - 100.0 * np.sin(angles)
    centre_x, centre_y, radius, _ = fit_circle(x, y)
    assert (centre_x, centre_y, radius) == pytest.approx((200.0, 150.0, 100.0), abs=1e-6)


def test_fit_circle_with_noisy_points():
    rng = np.random.default_rng(7)
    angles = np.radians(np.arange(0, 720, 2.0))
    x = 640.0 + 250.0 * np.cos(angles) + rng.normal(0.0, 0.3, len(angles))
    y = 360.0 - 250.0 * np.sin(angles) + rng.normal(0.0, 0.3, len(angles))
    centre_x, centre_y, radius, rms = fit_circle(x, y)
    assert (centre_x, centre_y) == pytest.approx((640.0, 360.0), abs=0.2)
    assert radius == pytest.approx(250.0, abs=0.2)
    assert rms == pytest.approx(0.3, abs=0.1)


def test_fit_circle_needs_three_points():
    with pytest.raises(AnalysisError):
        fit_circle([1.0, 2.0], [1.0, 2.0])


def test_fit_centre_from_track_on_a_full_circle():
    table = table_from_angles(steady_angles(120, 10.0))  # 1200 degrees, whole circle covered
    fit = fit_centre_from_track(table)
    assert fit["usable"]
    assert fit["centre"] == pytest.approx(CENTRE, abs=1e-6)
    assert fit["radius_px"] == pytest.approx(RADIUS, abs=1e-6)
    assert fit["coverage_deg"] >= 340


def test_fit_centre_from_track_is_not_trusted_for_a_short_arc():
    table = table_from_angles(steady_angles(30, 3.0))  # only 87 degrees of the circle
    fit = fit_centre_from_track(table)
    assert not fit["usable"]
    assert "covered only" in fit["reason"]
    assert fit["centre"] is not None  # it is still reported, but flagged


def test_fit_centre_from_track_with_too_few_points():
    table = table_from_angles(steady_angles(15, 20.0))
    fit = fit_centre_from_track(table)
    assert not fit["usable"]
    assert fit["centre"] is None
    assert "only 15" in fit["reason"]


def test_fit_ignores_frames_that_are_not_trusted():
    truth = steady_angles(120, 10.0)
    valid = np.ones(120, dtype=bool)
    table = table_from_angles(truth, valid)
    table.loc[[10, 50], ["marker_x", "marker_y"]] = [10.0, 10.0]  # two wild positions
    table.loc[[10, 50], "tracking_valid"] = False
    fit = fit_centre_from_track(table)
    assert fit["centre"] == pytest.approx(CENTRE, abs=1e-6)


# ---------------------------------------------------------------------------
# Against the simulated video, where the true angle is known
# ---------------------------------------------------------------------------


def track_and_calculate(path, centre):
    table = track_marker(path, settings_for_colour("green"), centre=centre)
    return add_angle_columns(table, centre)


@pytest.fixture(scope="module")
def slowing_video_with_gap(tmp_path_factory):
    """Speed falls off from 150 rpm, 3 Hz ripple, 3 frames hidden. 30 fps, so 30 degrees per frame at the start."""
    path = tmp_path_factory.mktemp("videos") / "slowing.mp4"
    write_synthetic_video(
        path, rpm=150.0, fps=30.0, seconds=4.0, width=320, height=240,
        ripple_deg=3.0, ripple_hz=3.0, decay_s=2.0, hidden_frames=range(50, 53),
    )
    return path


def test_angle_from_a_simulated_video_matches_the_true_angle(slowing_video_with_gap):
    result = track_and_calculate(slowing_video_with_gap, (160, 120))
    truth = true_marker_angle(
        result["time_s"].to_numpy(), 150.0,
        ripple_amplitude_rad=math.radians(3.0), ripple_hz=3.0, decay_s=2.0,
    )
    has_angle = result["angle_rad"].notna().to_numpy()
    error_deg = np.degrees(result["angle_rad"].to_numpy()[has_angle] - truth[has_angle])

    assert has_angle.sum() == 117  # 120 frames, 3 hidden
    assert np.abs(error_deg).max() < 0.5
    # By frame 50 the speed has fallen to about 65 rpm (13 degrees per frame), so the 4 steps across
    # the gap are about 52 degrees. That is under 90 degrees, so the angle is not split.
    assert result["segment"].dropna().nunique() == 1


def test_the_circle_fit_finds_the_true_centre_in_a_simulated_video(slowing_video_with_gap):
    table = track_marker(slowing_video_with_gap, settings_for_colour("green"), centre=(160, 120))
    fit = fit_centre_from_track(table)
    assert fit["usable"]
    assert fit["centre"] == pytest.approx((160, 120), abs=0.2)


# ---------------------------------------------------------------------------
# rpm conversion
# ---------------------------------------------------------------------------


def test_rpm_conversion_with_known_values():
    assert rad_s_to_rpm(2 * math.pi) == pytest.approx(60.0)  # one turn per second
    assert rad_s_to_rpm(10 * math.pi) == pytest.approx(300.0)
    assert rad_s_to_rpm(0.0) == 0.0
    assert rad_s_to_rpm(-2 * math.pi) == pytest.approx(-60.0)  # clockwise stays negative
    assert rpm_to_rad_s(60.0) == pytest.approx(2 * math.pi)
    assert rpm_to_rad_s(300.0) == pytest.approx(10 * math.pi)


def test_rpm_conversion_round_trip_and_arrays():
    rpm = np.array([0.0, 12.5, 120.0, 1800.0, -45.0])
    assert np.allclose(rad_s_to_rpm(rpm_to_rad_s(rpm)), rpm)
    assert rad_s_to_rpm(np.array([2 * math.pi, 4 * math.pi])).tolist() == pytest.approx([60.0, 120.0])


# ---------------------------------------------------------------------------
# Angular velocity
# ---------------------------------------------------------------------------


def with_speed(truth, valid=None, fps=60.0):
    table = table_from_angles(truth, valid, fps=fps)
    return add_speed_columns(add_angle_columns(table, CENTRE))


def test_constant_speed_is_exact_everywhere_including_the_ends():
    # 12 degrees per frame at 60 fps is 720 degrees per second, which is 2 turns per second, 120 rpm.
    result = with_speed(steady_angles(60, 12.0))
    assert np.allclose(result["rpm"].to_numpy(), 120.0)
    assert np.allclose(result["angular_velocity_rad_s"].to_numpy(), 4 * math.pi)


def test_clockwise_rotation_gives_negative_speed():
    result = with_speed(-steady_angles(60, 12.0))
    assert np.allclose(result["rpm"].to_numpy(), -120.0)


def test_constant_acceleration_is_found_by_the_central_and_the_end_differences():
    # theta = 0.5 * alpha * t^2 with alpha = 8 rad/s^2, so omega = alpha * t. np.gradient is exact for this.
    fps = 100.0
    time_s = np.arange(50) / fps
    result = with_speed(0.5 * 8.0 * time_s**2, fps=fps)
    assert np.allclose(result["angular_velocity_rad_s"].to_numpy(), 8.0 * time_s)


def test_lost_frames_inside_a_segment_do_not_disturb_the_speed():
    # The real time of every frame is used, so the speed stays 120 rpm right through the gap.
    valid = np.ones(60, dtype=bool)
    valid[[10, 11, 30, 31, 32]] = False
    result = with_speed(steady_angles(60, 12.0), valid)

    assert result.loc[~pd.Series(valid), "rpm"].isna().all()  # no speed where there is no angle
    assert np.allclose(result.loc[valid, "rpm"].to_numpy(), 120.0)


def test_no_speed_is_calculated_across_a_split_gap():
    # 50 degrees per frame, 4 frames lost. The gap is flagged and the angle after it is one turn too low,
    # but the speed on both sides must still be the true 500 rpm (50 degrees x 60 fps / 360 x 60).
    valid = np.ones(30, dtype=bool)
    valid[10:14] = False
    result = with_speed(steady_angles(30, 50.0), valid)

    assert result["segment"].dropna().nunique() == 2
    assert np.allclose(result.loc[valid, "rpm"].to_numpy(), 500.0)

    # A plain np.gradient over all the valid frames would not give that, because of the missing turn.
    wrong = np.gradient(result.loc[valid, "angle_rad"].to_numpy(), result.loc[valid, "time_s"].to_numpy())
    assert not np.allclose(wrong, 500.0, atol=1.0)


def test_a_segment_of_one_frame_has_no_speed_and_two_frames_give_one_difference():
    # Frames 0 to 19 valid, a long gap, one lone frame, another long gap, two more frames.
    truth = steady_angles(60, 20.0)
    valid = np.zeros(60, dtype=bool)
    valid[0:20] = True
    valid[30] = True
    valid[45:47] = True
    result = with_speed(truth, valid)

    assert result["segment"].dropna().nunique() == 3
    assert np.isnan(result.loc[30, "rpm"])  # a lone frame: no speed
    assert result.loc[45, "rpm"] == pytest.approx(200.0)  # 20 degrees per frame x 60 fps / 360 x 60 = 200 rpm
    assert result.loc[46, "rpm"] == pytest.approx(200.0)


def test_speed_columns_need_the_angle_columns_first():
    with pytest.raises(AnalysisError, match="add_angle_columns"):
        add_speed_columns(table_from_angles(steady_angles(30, 10.0)))


def test_speed_calculation_does_not_change_the_input_table():
    table = add_angle_columns(table_from_angles(steady_angles(30, 10.0)), CENTRE)
    add_speed_columns(table)
    assert "rpm" not in table.columns


def test_a_fast_ripple_is_reduced_by_the_central_difference_as_the_formula_says():
    # np.gradient averages over two frame intervals. A ripple at frequency f is reduced by sin(x) / x
    # with x = 2 pi f / fps. At fps / 10 that is 0.9355. This is why very fast fluctuations read too small.
    fps = 60.0
    frequency = fps / 10.0
    time_s = np.arange(0, 20, 1 / fps)
    amplitude = 0.05  # rad
    theta = 2 * math.pi * 2.0 * time_s + amplitude * np.sin(2 * math.pi * frequency * time_s)
    omega = np.gradient(theta, time_s, edge_order=2) - 2 * math.pi * 2.0

    measured = (omega[30:-30].max() - omega[30:-30].min()) / 2
    x = 2 * math.pi * frequency / fps
    expected = amplitude * 2 * math.pi * frequency * math.sin(x) / x
    assert measured == pytest.approx(expected, rel=0.01)


# ---------------------------------------------------------------------------
# Speed summary
# ---------------------------------------------------------------------------


def test_summary_of_a_steady_speed():
    summary = summarise_speed(with_speed(steady_angles(60, 12.0)))
    assert summary["mean_rpm"] == pytest.approx(120.0)
    assert (summary["min_rpm"], summary["max_rpm"]) == pytest.approx((120.0, 120.0))
    assert summary["std_rpm"] == pytest.approx(0.0, abs=1e-9)
    assert summary["variation_percent"] == pytest.approx(0.0, abs=1e-9)
    assert summary["trend_rpm"] == pytest.approx(0.0, abs=1e-6)
    assert summary["direction"] == "counter-clockwise"
    assert summary["speed_samples"] == 60


def test_summary_of_clockwise_rotation_reports_positive_speeds():
    summary = summarise_speed(with_speed(-steady_angles(60, 12.0)))
    assert summary["direction"] == "clockwise"
    assert summary["mean_rpm"] == pytest.approx(120.0)
    assert summary["min_rpm"] == pytest.approx(120.0)
    assert summary["max_rpm"] == pytest.approx(120.0)


def test_summary_mean_is_rotation_divided_by_time_not_the_mean_of_noisy_values():
    # Speed alternates between 100 and 140 rpm every frame (noise), but the average speed is exactly 120.
    rng = np.random.default_rng(3)
    time_s = np.arange(121) / 60.0
    angle = 4 * math.pi * time_s + 0.0  # 120 rpm
    noisy = angle + rng.normal(0.0, 0.003, len(angle))
    summary = summarise_speed(with_speed(noisy))
    assert summary["mean_rpm"] == pytest.approx(120.0, abs=0.3)
    assert summary["std_rpm"] > 1.0  # the single frame values scatter
    assert summary["max_rpm"] > summary["mean_rpm"] > summary["min_rpm"]


def test_summary_mean_leaves_out_the_split_gap():
    valid = np.ones(40, dtype=bool)
    valid[15:21] = False  # long gap: 7 steps of 20 degrees = 140 degrees, so the angle is split
    result = with_speed(steady_angles(40, 20.0), valid)
    assert result["segment"].dropna().nunique() == 2
    assert summarise_speed(result)["mean_rpm"] == pytest.approx(200.0)


def test_summary_reports_a_large_trend_for_a_slowing_disc():
    fps = 60.0
    time_s = np.arange(300) / fps
    omega0, tau = 2 * math.pi * 300 / 60, 4.0
    angle = omega0 * tau * (1 - np.exp(-time_s / tau))
    summary = summarise_speed(with_speed(angle, fps=fps))

    assert summary["max_rpm"] == pytest.approx(300.0, rel=0.01)
    assert summary["trend_rpm"] < -150.0  # the speed falls by well over 100 rpm during the clip
    assert summary["variation_percent"] > 30.0  # the spread is mostly the slowing down


def test_summary_of_a_marker_that_does_not_move():
    truth = np.full(40, 1.0)
    summary = summarise_speed(with_speed(truth))
    assert summary["direction"] == "no clear rotation"
    assert summary["mean_rpm"] == pytest.approx(0.0, abs=1e-9)
    assert math.isnan(summary["variation_percent"])


def test_summary_needs_speed_values():
    table = add_angle_columns(table_from_angles(steady_angles(30, 10.0)), CENTRE)
    with pytest.raises(AnalysisError, match="add_speed_columns"):
        summarise_speed(table)
    table = add_speed_columns(table)
    table["rpm"] = np.nan
    with pytest.raises(AnalysisError, match="no speed values"):
        summarise_speed(table)


# ---------------------------------------------------------------------------
# Against the simulated video, where the true speed is known
# ---------------------------------------------------------------------------


def test_speed_from_a_simulated_video_matches_the_true_speed(slowing_video_with_gap):
    result = add_speed_columns(track_and_calculate(slowing_video_with_gap, (160, 120)))

    truth = true_marker_speed_rpm(
        result["time_s"].to_numpy(), 150.0,
        ripple_amplitude_rad=math.radians(3.0), ripple_hz=3.0, decay_s=2.0,
    )
    has_speed = result["rpm"].notna().to_numpy()
    error = result["rpm"].to_numpy()[has_speed] - truth[has_speed]

    # At 30 fps a 3 Hz ripple is reduced a little by the central difference, and the one-sided end
    # differences are noisier. A few rpm is the right size for this small 320 x 240 test video.
    assert np.abs(error).max() < 4.0
    assert np.sqrt(np.mean(error**2)) < 1.5
