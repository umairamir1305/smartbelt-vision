"""Tests for the simulated belt drive video (src/synthetic.py): two discs with one marker each."""

import math

import numpy as np
import pytest

from src.synthetic import (
    belt_drive_geometry,
    belt_loop,
    make_belt_drive_frame,
    true_marker_angle,
    write_synthetic_belt_video,
)
from src.tracking import find_marker, settings_for_colour, track_marker
from src.video_io import get_video_info


def test_geometry_fits_in_the_frame_and_follows_the_ratio():
    driver_centre, driver_radius, driven_centre, driven_radius, marker_radius = belt_drive_geometry(960, 540, 2.0)
    assert driven_radius == pytest.approx(2.0 * driver_radius)
    assert driver_centre[1] == driven_centre[1] == 270
    # The discs do not overlap and stay inside the frame.
    assert driven_centre[0] - driven_radius > driver_centre[0] + driver_radius
    assert driver_centre[0] - driver_radius > 0
    assert driven_centre[0] + driven_radius < 960
    assert driven_centre[1] - driven_radius > 0 and driven_centre[1] + driven_radius < 540
    assert marker_radius >= 5.0


@pytest.mark.parametrize(
    "width, height, ratio",
    [(960, 540, 0), (960, 540, 3.0), (400, 540, 2.0)],
)
def test_geometry_rejects_impossible_sizes(width, height, ratio):
    with pytest.raises(ValueError):
        belt_drive_geometry(width, height, ratio)


def test_each_marker_is_where_its_angle_says():
    width, height, ratio = 960, 540, 1.5
    driver_centre, driver_radius, driven_centre, driven_radius, _ = belt_drive_geometry(width, height, ratio)
    frame = make_belt_drive_frame(width, height, ratio, math.radians(90), math.radians(200))

    driver = find_marker(frame, settings_for_colour("green"))
    driven = find_marker(frame, settings_for_colour("orange"))
    # The driver marker is straight above its centre. Image y points down, so y is smaller.
    assert driver.x == pytest.approx(driver_centre[0], abs=1.0)
    assert driver.y == pytest.approx(driver_centre[1] - 0.75 * driver_radius, abs=1.0)
    assert driven.x == pytest.approx(driven_centre[0] + 0.75 * driven_radius * math.cos(math.radians(200)), abs=1.0)
    assert driven.y == pytest.approx(driven_centre[1] - 0.75 * driven_radius * math.sin(math.radians(200)), abs=1.0)


def test_unknown_colours_are_rejected():
    with pytest.raises(ValueError):
        make_belt_drive_frame(960, 540, 2.0, 0.0, 0.0, driver_colour="purple")


def test_true_speeds_follow_ratio_slip_and_direction(tmp_path):
    info = write_synthetic_belt_video(tmp_path / "a.mp4", driver_rpm=120.0, ratio=2.0, fps=30.0, seconds=1.0)
    assert (info["ideal_driven_rpm"], info["driven_rpm"]) == (60.0, 60.0)

    info = write_synthetic_belt_video(tmp_path / "b.mp4", driver_rpm=120.0, ratio=2.0, slip_percent=2.0, fps=30.0, seconds=1.0)
    assert info["driven_rpm"] == pytest.approx(58.8)  # 2 percent slower than the ideal 60 rpm

    info = write_synthetic_belt_video(tmp_path / "c.mp4", driver_rpm=120.0, ratio=2.0, crossed=True, fps=30.0, seconds=1.0)
    assert info["driven_rpm"] == -60.0  # opposite direction

    # The angle arrays are the true angles of the two discs. At 120 rpm (2 turns per second) and 30 fps
    # the driver turns 2 x 59 / 30 turns between the first and the last of 60 frames.
    info = write_synthetic_belt_video(tmp_path / "d.mp4", driver_rpm=120.0, ratio=2.0, fps=30.0, seconds=2.0)
    assert info["driver_angle_rad"][-1] - info["driver_angle_rad"][0] == pytest.approx(2 * math.pi * 2.0 * 59 / 30)
    assert np.allclose(info["driven_angle_rad"], true_marker_angle(info["times_s"], 60.0, start_angle_rad=math.radians(150.0)))


def test_the_video_has_the_right_size_and_both_markers_can_be_tracked(tmp_path):
    path = tmp_path / "belt.mp4"
    info = write_synthetic_belt_video(path, driver_rpm=90.0, ratio=1.5, fps=30.0, seconds=2.0)
    video = get_video_info(path)
    assert (video.width, video.height, video.frame_count) == (960, 540, 60)

    for colour, centre, radius, angles in (
        ("green", info["driver_centre"], info["driver_radius_px"], info["driver_angle_rad"]),
        ("orange", info["driven_centre"], info["driven_radius_px"], info["driven_angle_rad"]),
    ):
        table = track_marker(path, settings_for_colour(colour), centre=centre, region_radius_px=1.05 * radius)
        assert table["marker_x"].notna().all()
        expected_x = centre[0] + 0.75 * radius * np.cos(angles)
        expected_y = centre[1] - 0.75 * radius * np.sin(angles)
        error = np.hypot(table["marker_x"] - expected_x, table["marker_y"] - expected_y)
        assert error.max() < 0.7


def test_each_marker_can_be_hidden_on_its_own(tmp_path):
    path = tmp_path / "hidden.mp4"
    info = write_synthetic_belt_video(
        path, driver_rpm=90.0, ratio=1.5, fps=30.0, seconds=2.0, hidden_driver_frames=range(10, 14), hidden_driven_frames=[30]
    )
    driver = track_marker(path, settings_for_colour("green"), centre=info["driver_centre"], region_radius_px=1.05 * info["driver_radius_px"])
    driven = track_marker(path, settings_for_colour("orange"), centre=info["driven_centre"], region_radius_px=1.05 * info["driven_radius_px"])
    assert driver["marker_x"].isna().sum() == 4 and driver["marker_x"].isna()[10:14].all()
    assert driven["marker_x"].isna().sum() == 1 and driven["marker_x"].isna()[30]


def test_bad_settings_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_synthetic_belt_video(tmp_path / "x.mp4", slip_percent=100.0)
    with pytest.raises(ValueError):
        write_synthetic_belt_video(tmp_path / "x.mp4", fps=0)
    with pytest.raises(ValueError):
        write_synthetic_belt_video(tmp_path / "x.mp4", ratio=5.0)


# ---------------------------------------------------------------------------
# The belt and the mark on it
# ---------------------------------------------------------------------------


def test_the_belt_loop_is_closed_continuous_and_has_the_right_wrap_angles():
    loop = belt_loop(960, 540, 2.0)
    positions = np.array([loop.point_at(u) for u in np.arange(0.0, loop.length + 1.0, 1.0)])
    steps = np.hypot(*np.diff(positions, axis=0).T)
    assert steps.min() == pytest.approx(1.0, abs=0.02) and steps.max() == pytest.approx(1.0, abs=0.02)  # no jumps
    assert np.allclose(positions[0], loop.point_at(loop.length), atol=1e-6)  # closed

    expected_driver_wrap = math.pi - 2 * math.asin((loop.a2 - loop.a1) / (loop.c2[0] - loop.c1[0]))
    assert loop.wrap_driver_rad == pytest.approx(expected_driver_wrap)
    assert loop.wrap_driver_rad + loop.wrap_driven_rad == pytest.approx(2 * math.pi)
    assert loop.length == pytest.approx(2 * loop.straight + loop.wrap_driven_rad * loop.a2 + loop.wrap_driver_rad * loop.a1)


def test_the_belt_runs_to_the_right_on_the_lower_span_and_to_the_left_on_the_upper_span():
    loop = belt_loop(960, 540, 2.0)
    lower_start, lower_end = loop.point_at(5.0), loop.point_at(loop.straight - 5.0)
    assert lower_end[0] > lower_start[0] and lower_end[1] > loop.c1[1]  # to the right, below the centres (image y down)
    upper_start = loop.point_at(loop.straight + loop.arc_driven + 5.0)
    upper_end = loop.point_at(2 * loop.straight + loop.arc_driven - 5.0)
    assert upper_end[0] < upper_start[0] and upper_end[1] < loop.c1[1]  # to the left, above the centres


def test_the_belt_video_has_a_mark_that_moves_at_the_true_belt_speed(tmp_path):
    from src.tracking import settings_for_colour, track_marker

    path = tmp_path / "belt.mp4"
    info = write_synthetic_belt_video(
        path, driver_rpm=120.0, ratio=2.0, driver_slip_percent=1.0, slip_percent=2.0, draw_belt=True,
        fps=30.0, seconds=3.0, scale_mm_per_px=0.25,
    )
    # Belt speed = driver surface speed minus 1 percent. Driven speed = ideal speed minus 1 and 2 percent.
    assert info["belt_speed_px_s"] == pytest.approx(2 * math.pi * 2.0 * info["belt_loop"].a1 * 0.99)
    assert info["belt_speed_m_s"] == pytest.approx(info["belt_speed_px_s"] * 0.25 / 1000)
    assert info["driven_rpm"] == pytest.approx(60.0 * 0.99 * 0.98)
    assert info["driver_surface_speed_m_s"] / info["belt_speed_m_s"] == pytest.approx(1 / 0.99)
    assert info["driven_surface_speed_m_s"] / info["belt_speed_m_s"] == pytest.approx(0.98, abs=1e-6)

    table = track_marker(path, settings_for_colour("blue"))
    assert table["marker_x"].notna().all()
    error = np.hypot(table["marker_x"] - info["belt_positions"][:, 0], table["marker_y"] - info["belt_positions"][:, 1])
    assert error.max() < 1.0


def test_the_belt_mark_can_be_hidden_and_a_crossed_belt_is_not_drawn(tmp_path):
    from src.tracking import settings_for_colour, track_marker

    path = tmp_path / "hidden_mark.mp4"
    write_synthetic_belt_video(path, draw_belt=True, fps=30.0, seconds=1.0, hidden_belt_frames=range(5, 10))
    table = track_marker(path, settings_for_colour("blue"))
    assert table["marker_x"].isna().sum() == 5 and table["marker_x"].isna()[5:10].all()

    with pytest.raises(ValueError, match="open belt"):
        write_synthetic_belt_video(tmp_path / "x.mp4", draw_belt=True, crossed=True)
    with pytest.raises(ValueError, match="slip percentages"):
        write_synthetic_belt_video(tmp_path / "x.mp4", driver_slip_percent=100.0)
    with pytest.raises(ValueError, match="scale"):
        write_synthetic_belt_video(tmp_path / "x.mp4", scale_mm_per_px=0.0)


def test_without_a_belt_the_returned_dictionary_has_no_belt_entries(tmp_path):
    info = write_synthetic_belt_video(tmp_path / "plain.mp4", fps=30.0, seconds=1.0)
    assert "belt_speed_px_s" not in info
    assert info["driver_marker_radius_mm"] == pytest.approx(0.75 * info["driver_radius_px"] * 0.25)
