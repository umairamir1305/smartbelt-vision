"""Tests for src/slip_analysis.py: scale, belt speed from straight line fits, slip, wrap angle and the calculators."""

import math

import numpy as np
import pandas as pd
import pytest

from src.slip_analysis import (
    MIN_PASS_FRAMES,
    add_belt_speed_columns,
    belt_friction_coefficient,
    combine_scales,
    find_belt_passes,
    mark_on_straight_span,
    power_difference,
    power_from_torque,
    scale_from_marker_radius,
    summarise_slip,
    tension_ratio,
    wrap_angle_open_belt,
)
from src.synthetic import belt_loop

FPS = 60.0


def belt_mark_table(speed_px_s=1150.0, seconds=5.0, noise_px=0.3, seed=1, lost=(), ratio=2.0):
    """The tracked belt mark of a simulated belt drive (960 x 540), without making a video.

    Returns (table, on_span, loop, centres, radii). The mark runs round the belt at speed_px_s.
    """
    loop = belt_loop(960, 540, ratio)
    count = int(seconds * FPS)
    time_s = np.arange(count) / FPS
    rng = np.random.default_rng(seed)
    points = np.array([loop.point_at(0.15 * loop.length + speed_px_s * t) for t in time_s])
    x = points[:, 0] + rng.normal(0.0, noise_px, count)
    y = points[:, 1] + rng.normal(0.0, noise_px, count)
    valid = np.ones(count, dtype=bool)
    valid[list(lost)] = False
    x[~valid], y[~valid] = np.nan, np.nan
    table = pd.DataFrame(
        {
            "frame": np.arange(count), "time_s": time_s, "marker_x": x, "marker_y": y,
            "area_px": 600.0, "tracking_valid": valid, "tracking_quality": np.where(valid, "Good", "Lost"),
        }
    )
    centres = [loop.c1, loop.c2]
    radii = [loop.a1 / 0.95, loop.a2 / 0.95]  # the disc radii, as the region radii would be
    return table, mark_on_straight_span(x, y, centres, radii), loop, centres, radii


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------


def test_scale_from_a_marker_radius():
    assert scale_from_marker_radius(18.225, 72.9) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        scale_from_marker_radius(0.0, 72.9)
    with pytest.raises(ValueError):
        scale_from_marker_radius(18.0, -1.0)


def test_combining_scales_reports_how_much_they_disagree():
    scale, disagreement = combine_scales({"driver": 0.25, "driven": 0.26})
    assert scale == pytest.approx(0.255)
    assert disagreement == pytest.approx(100 * 0.01 / 0.255)
    assert combine_scales({"driver": 0.25, "driven": None}) == (0.25, 0.0)
    assert combine_scales({"driver": None}) == (None, 0.0)


# ---------------------------------------------------------------------------
# Straight spans and passes
# ---------------------------------------------------------------------------


def test_points_inside_a_pulley_circle_are_not_on_a_straight_span():
    on_span = mark_on_straight_span([100, 100, 300, np.nan], [100, 400, 100, 100], [(100, 150)], [60])
    assert on_span.tolist() == [False, True, True, False]  # (100, 100) is 50 px from the centre, nan has no position


def test_the_belt_speed_of_a_pass_is_found_from_a_straight_line_fit():
    table, on_span, loop, _, _ = belt_mark_table(speed_px_s=1150.0, noise_px=0.42)
    passes = find_belt_passes(table, on_span)

    assert len(passes) >= 2 and all(p["accepted"] for p in passes)
    # Each pass is only about 11 frames long here (the straight spans are short), and still gives the speed to
    # about a quarter of a percent, although every single position has an error of 0.4 px. That error would
    # be 3 percent in a speed from two neighbouring frames.
    for belt_pass in passes:
        assert belt_pass["frames"] >= MIN_PASS_FRAMES
        assert belt_pass["speed_px_s"] == pytest.approx(1150.0, rel=0.005)
        assert belt_pass["residual_px"] < 0.7
    # The lower span runs to the right (about 0 degrees), the upper span to the left (about 180 degrees).
    for belt_pass in passes:
        direction = abs(belt_pass["direction_deg"])
        assert min(direction, abs(direction - 180)) < 25


def test_a_lost_frame_ends_a_pass_and_short_runs_are_ignored():
    # A slower belt (400 px/s), so that every pass is long: about 32 frames.
    table, on_span, loop, _, _ = belt_mark_table(speed_px_s=400.0, noise_px=0.2)
    whole = find_belt_passes(table, on_span)
    assert all(p["frames"] > 25 for p in whole)
    first = whole[0]
    middle = (first["start_index"] + first["end_index"]) // 2

    table_with_gap, on_span_gap, *_ = belt_mark_table(speed_px_s=400.0, noise_px=0.2, lost=[middle])
    split = find_belt_passes(table_with_gap, on_span_gap)
    assert len(split) == len(whole) + 1  # the pass was cut in two, and both parts are long enough
    assert all(p["speed_px_s"] == pytest.approx(400.0, rel=0.01) for p in split)

    # The passes are 32 and 33 frames long: they pass a limit of 30 but not a limit of 40.
    assert len(find_belt_passes(table, on_span, min_frames=30)) == len(whole)
    assert find_belt_passes(table, on_span, min_frames=40) == []


def test_passes_that_are_too_short_are_left_out():
    # At 1150 px/s the passes are about 11 frames. Asking for 20 frames in a row finds none.
    table, on_span, *_ = belt_mark_table(speed_px_s=1150.0)
    assert len(find_belt_passes(table, on_span, min_frames=8)) >= 2
    assert find_belt_passes(table, on_span, min_frames=20) == []


def test_a_mark_that_is_not_on_a_straight_line_is_rejected():
    # Every point is allowed here, so the pass takes in the curved parts of the belt too.
    table, _, loop, _, _ = belt_mark_table(noise_px=0.1)
    everywhere = np.ones(len(table), dtype=bool)
    passes = find_belt_passes(table, everywhere)
    assert len(passes) == 1 and passes[0]["frames"] == len(table)  # the whole loop is one long run
    assert not passes[0]["accepted"] and passes[0]["residual_px"] > 20


def test_frames_that_are_not_consecutive_do_not_make_a_pass():
    table, on_span, *_ = belt_mark_table()
    table["frame"] = table["frame"] * 2  # every second frame is missing from the video
    assert find_belt_passes(table, on_span) == []


# ---------------------------------------------------------------------------
# Table columns
# ---------------------------------------------------------------------------


def merged_table(count, with_surface=True):
    merged = pd.DataFrame({"frame": np.arange(count), "time_s": np.arange(count) / FPS})
    if with_surface:
        merged["driver_surface_speed_m_s"] = 0.3
        merged["driven_surface_speed_m_s"] = 0.29
    return merged


def test_belt_columns_hold_the_speed_of_each_accepted_pass_and_the_slip_of_each_frame():
    table, on_span, *_ = belt_mark_table(speed_px_s=1150.0, noise_px=0.2)
    passes = find_belt_passes(table, on_span)
    merged = add_belt_speed_columns(merged_table(len(table)), table, on_span, passes, scale_mm_per_px=0.25)

    for column in ("belt_marker_x", "belt_marker_y", "belt_area_px", "belt_tracking_valid", "belt_tracking_quality",
                   "belt_on_span", "belt_speed_m_s", "driver_slip_percent", "driven_slip_percent"):
        assert column in merged.columns
    speed = merged["belt_speed_m_s"]
    assert speed.notna().sum() == sum(p["frames"] for p in passes)
    assert speed.dropna().median() == pytest.approx(1150.0 * 0.25 / 1000.0, rel=0.003)  # 0.2875 m/s

    # driver slip = (0.3 - 0.2875) / 0.3 = 4.2 percent, driven slip = (0.2875 - 0.29) / 0.2875 = -0.9 percent
    assert merged["driver_slip_percent"].dropna().median() == pytest.approx(4.17, abs=0.15)
    assert merged["driven_slip_percent"].dropna().median() == pytest.approx(-0.87, abs=0.15)
    assert merged["driver_slip_percent"][speed.isna()].isna().all()  # no belt speed, no slip


def test_the_calibration_factor_scales_the_belt_speed():
    table, on_span, *_ = belt_mark_table(noise_px=0.1)
    passes = find_belt_passes(table, on_span)
    plain = add_belt_speed_columns(merged_table(len(table)), table, on_span, passes, 0.25)
    scaled = add_belt_speed_columns(merged_table(len(table)), table, on_span, passes, 0.25, belt_speed_factor=1.04)
    assert scaled["belt_speed_m_s"].dropna().median() == pytest.approx(1.04 * plain["belt_speed_m_s"].dropna().median())


def test_without_surface_speeds_there_are_no_slip_columns():
    table, on_span, *_ = belt_mark_table()
    merged = add_belt_speed_columns(merged_table(len(table), with_surface=False), table, on_span, find_belt_passes(table, on_span), 0.25)
    assert "belt_speed_m_s" in merged.columns and "driver_slip_percent" not in merged.columns


def test_without_a_scale_only_the_tracking_columns_are_added():
    table, on_span, *_ = belt_mark_table()
    merged = add_belt_speed_columns(merged_table(len(table)), table, on_span, find_belt_passes(table, on_span), None)
    assert "belt_marker_x" in merged.columns and "belt_on_span" in merged.columns
    assert "belt_speed_m_s" not in merged.columns and "driver_slip_percent" not in merged.columns


def test_tables_of_different_length_are_rejected():
    table, on_span, *_ = belt_mark_table()
    with pytest.raises(ValueError, match="same frames"):
        add_belt_speed_columns(merged_table(10), table, on_span, [], 0.25)


# ---------------------------------------------------------------------------
# Slip summary
# ---------------------------------------------------------------------------


def make_passes(speeds, frames=40, accepted=True):
    return [{"speed_px_s": speed, "frames": frames, "accepted": accepted} for speed in speeds]


def test_slip_definitions_with_known_speeds():
    # Belt speed 1150 px/s at 0.25 mm per pixel is 0.2875 m/s. Driver surface 0.3 m/s, driven surface 0.28 m/s.
    summary = summarise_slip(make_passes([1150.0, 1150.0]), 0.25, 0.3, 0.28)
    assert summary["available"] and summary["reason"] == ""
    assert summary["belt_speed_m_s"] == pytest.approx(0.2875)
    assert summary["driver_slip_percent"] == pytest.approx((0.3 - 0.2875) / 0.3 * 100)  # 4.17 percent
    assert summary["driven_slip_percent"] == pytest.approx((0.2875 - 0.28) / 0.2875 * 100)  # 2.61 percent
    assert summary["speed_loss_percent"] == pytest.approx((0.3 - 0.28) / 0.3 * 100)  # 6.67 percent
    # The two slips together are the speed loss: (1 - s1)(1 - s2) = 1 - loss.
    assert (1 - summary["driver_slip_percent"] / 100) * (1 - summary["driven_slip_percent"] / 100) == pytest.approx(
        1 - summary["speed_loss_percent"] / 100
    )
    assert summary["calibration_factor_if_no_load"] == pytest.approx(0.3 / 0.2875)
    assert summary["passes"] == 2 and summary["span_frames"] == 80


def test_the_calibration_factor_makes_the_driver_slip_zero_when_it_is_used():
    first = summarise_slip(make_passes([1150.0]), 0.25, 0.3, 0.3)
    second = summarise_slip(make_passes([1150.0]), 0.25, 0.3, 0.3, belt_speed_factor=first["calibration_factor_if_no_load"])
    assert second["driver_slip_percent"] == pytest.approx(0.0, abs=1e-9)


def test_the_belt_speed_is_the_weighted_mean_of_the_passes_and_the_spread_is_reported():
    passes = [
        {"speed_px_s": 1100.0, "frames": 10, "accepted": True},
        {"speed_px_s": 1200.0, "frames": 30, "accepted": True},
        {"speed_px_s": 5000.0, "frames": 50, "accepted": False},  # rejected: not used
    ]
    summary = summarise_slip(passes, 0.25, 0.3, 0.28)
    assert summary["belt_speed_m_s"] == pytest.approx((1100 * 10 + 1200 * 30) / 40 * 0.25 / 1000)
    assert summary["rejected_passes"] == 1 and summary["passes"] == 2
    assert summary["belt_speed_spread_percent"] == pytest.approx(100 * np.std([1100, 1200], ddof=1) / 1175)
    assert summarise_slip(make_passes([1150.0]), 0.25, 0.3, 0.28)["belt_speed_spread_percent"] is None  # one pass only


def test_slip_is_unavailable_when_the_belt_cannot_be_tracked():
    for passes in ([], make_passes([1150.0], frames=12), make_passes([1150.0], accepted=False)):
        summary = summarise_slip(passes, 0.25, 0.3, 0.28)
        assert not summary["available"]
        assert summary["reason"].startswith("Slip estimation unavailable because independent belt motion could not be tracked reliably.")
        assert summary["driver_slip_percent"] is None and summary["belt_speed_m_s"] is None
    assert "12 frames" in summarise_slip(make_passes([1150.0], frames=12), 0.25, 0.3, 0.28)["reason"]
    assert "1 rejected" in summarise_slip(make_passes([1150.0], accepted=False), 0.25, 0.3, 0.28)["reason"]


def test_the_scale_disagreement_is_carried_through():
    summary = summarise_slip(make_passes([1150.0]), 0.25, 0.3, 0.28, scale_disagreement_percent=1.7)
    assert summary["scale_disagreement_percent"] == 1.7


def test_slip_from_a_simulated_belt_mark_matches_the_true_belt_speed():
    # True belt speed 1148.8 px/s at 0.25 mm/px = 0.2872 m/s. Surface speed of the driver is 0.2901 m/s (1 percent faster).
    table, on_span, *_ = belt_mark_table(speed_px_s=1148.77, noise_px=0.42)
    summary = summarise_slip(find_belt_passes(table, on_span), 0.25, 0.2901, 0.2814)
    assert summary["belt_speed_m_s"] == pytest.approx(0.2872, rel=0.002)
    assert summary["driver_slip_percent"] == pytest.approx(1.0, abs=0.25)
    assert summary["driven_slip_percent"] == pytest.approx(2.0, abs=0.25)


# ---------------------------------------------------------------------------
# Wrap angle
# ---------------------------------------------------------------------------


def test_wrap_angles_of_an_open_belt():
    small, large = wrap_angle_open_belt(50.0, 50.0, 300.0)  # equal pulleys: half a turn each
    assert small == pytest.approx(math.pi) and large == pytest.approx(math.pi)

    small, large = wrap_angle_open_belt(25.0, 50.0, 200.0)  # asin(25 / 200) = 7.18 degrees
    assert math.degrees(small) == pytest.approx(180 - 2 * 7.1808, abs=0.001)
    assert math.degrees(small) + math.degrees(large) == pytest.approx(360.0)


def test_wrap_angle_agrees_with_the_geometry_of_the_simulated_belt():
    loop = belt_loop(960, 540, 2.0)
    centre_distance = loop.c2[0] - loop.c1[0]
    small, large = wrap_angle_open_belt(loop.a1, loop.a2, centre_distance)
    assert small == pytest.approx(loop.wrap_driver_rad)
    assert large == pytest.approx(loop.wrap_driven_rad)


@pytest.mark.parametrize("arguments", [(0.0, 50.0, 200.0), (60.0, 50.0, 200.0), (25.0, 50.0, 20.0)])
def test_wrap_angle_rejects_impossible_geometry(arguments):
    with pytest.raises(ValueError):
        wrap_angle_open_belt(*arguments)


# ---------------------------------------------------------------------------
# Calculators
# ---------------------------------------------------------------------------


def test_belt_friction_coefficient_from_the_tensions():
    # T1 / T2 = 2 and a wrap angle of pi: mu = ln 2 / pi = 0.2206.
    assert belt_friction_coefficient(120.0, 60.0, math.pi) == pytest.approx(math.log(2) / math.pi)
    assert belt_friction_coefficient(60.0, 60.0, math.pi) == 0.0  # no difference of tensions, no friction needed

    mu = belt_friction_coefficient(130.0, 40.0, math.radians(150))
    assert tension_ratio(mu, math.radians(150)) == pytest.approx(130.0 / 40.0)  # the two relations undo each other


@pytest.mark.parametrize("arguments", [(0.0, 10.0, 3.0), (10.0, -1.0, 3.0), (50.0, 100.0, 3.0), (100.0, 50.0, 0.0)])
def test_friction_calculator_rejects_impossible_values(arguments):
    with pytest.raises(ValueError):
        belt_friction_coefficient(*arguments)


def test_tension_ratio_rejects_impossible_values():
    with pytest.raises(ValueError):
        tension_ratio(-0.1, 3.0)
    with pytest.raises(ValueError):
        tension_ratio(0.3, 0.0)


def test_power_from_torque_and_the_power_difference():
    assert power_from_torque(2.0, 120.0) == pytest.approx(2.0 * 4 * math.pi)  # 120 rpm is 4 pi rad/s: 25.13 W
    assert power_from_torque(0.0, 120.0) == 0.0
    assert power_from_torque(2.0, -120.0) == pytest.approx(power_from_torque(2.0, 120.0))  # the direction does not matter

    result = power_difference(2.0, 120.0, 3.7, 60.0)
    assert result["power_in_w"] == pytest.approx(25.133, abs=0.001)
    assert result["power_out_w"] == pytest.approx(3.7 * 2 * math.pi, abs=0.001)  # 23.249 W
    assert result["power_difference_w"] == pytest.approx(25.133 - 23.249, abs=0.002)
    assert result["efficiency_percent"] == pytest.approx(92.5, abs=0.1)


def test_power_calculator_rejects_impossible_values():
    with pytest.raises(ValueError):
        power_from_torque(-1.0, 100.0)
    with pytest.raises(ValueError, match="input power"):
        power_difference(0.0, 100.0, 1.0, 50.0)
