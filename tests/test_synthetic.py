"""Tests for src/synthetic.py (simulated data). They also pin down the angle convention."""

import math

import numpy as np
import pytest

from src.synthetic import (
    MARKER_COLOURS_BGR,
    disc_geometry,
    make_disc_frame,
    make_synthetic_table,
    true_marker_angle,
    true_marker_speed_rpm,
    write_synthetic_video,
)
from src.video_io import get_video_info, read_frame


def green_marker_centroid(frame):
    """Find the green marker as the centroid of strongly green pixels. Returns (x, y)."""
    blue, green, red = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    mask = (green > 150) & (blue < 100) & (red < 100)
    ys, xs = np.nonzero(mask)
    assert len(xs) > 0, "the marker was not found in the frame"
    return xs.mean(), ys.mean()


def test_true_marker_angle_for_constant_speed():
    # 60 rpm is one turn per second, so a quarter of a turn takes 0.25 s.
    angles = true_marker_angle([0.0, 0.25, 0.5], rpm=60.0, start_angle_rad=0.0)
    assert np.allclose(angles, [0.0, math.pi / 2, math.pi])


def test_negative_rpm_is_clockwise():
    angles = true_marker_angle([0.0, 0.25], rpm=-60.0, start_angle_rad=0.0)
    assert angles[1] == pytest.approx(-math.pi / 2)


def test_ripple_adds_a_sine_term():
    # Ripple of 0.1 rad at 1 Hz: a quarter of a period (0.25 s) adds exactly 0.1 rad.
    angles = true_marker_angle(
        [0.25], rpm=0.0, start_angle_rad=0.0, ripple_amplitude_rad=0.1, ripple_hz=1.0
    )
    assert angles[0] == pytest.approx(0.1)


def test_marker_is_to_the_right_of_centre_at_angle_zero():
    width, height = 320, 240
    centre, _, orbit_radius, _ = disc_geometry(width, height)
    x, y = green_marker_centroid(make_disc_frame(width, height, 0.0))
    assert x == pytest.approx(centre[0] + orbit_radius, abs=1.5)
    assert y == pytest.approx(centre[1], abs=1.5)


def test_marker_is_above_centre_at_90_degrees():
    # Counter-clockwise on the screen: from the 3 o'clock position to the 12 o'clock position.
    # The image y axis points down, so "above" means a SMALLER y value.
    width, height = 320, 240
    centre, _, orbit_radius, _ = disc_geometry(width, height)
    x, y = green_marker_centroid(make_disc_frame(width, height, math.pi / 2))
    assert x == pytest.approx(centre[0], abs=1.5)
    assert y == pytest.approx(centre[1] - orbit_radius, abs=1.5)


def test_unknown_colour_is_rejected():
    with pytest.raises(ValueError):
        make_disc_frame(320, 240, 0.0, colour="purple")


def test_every_listed_colour_can_be_drawn():
    for colour in MARKER_COLOURS_BGR:
        frame = make_disc_frame(320, 240, 0.0, colour=colour)
        assert frame.shape == (240, 320, 3)


def test_noise_changes_the_frame_but_keeps_the_shape():
    clean = make_disc_frame(320, 240, 0.0)
    noisy = make_disc_frame(320, 240, 0.0, noise_sigma=8.0)
    assert noisy.shape == clean.shape
    assert not np.array_equal(noisy, clean)


def test_write_synthetic_video_matches_the_settings(tmp_path):
    path = tmp_path / "sim.mp4"
    true_angle = write_synthetic_video(path, rpm=60.0, fps=20.0, seconds=1.0, width=160, height=120)
    info = get_video_info(path)
    assert len(true_angle) == 20
    assert info.frame_count == 20
    assert info.fps == pytest.approx(20.0)
    assert (info.width, info.height) == (160, 120)
    # 60 rpm at 20 fps is 18 degrees of rotation per frame.
    assert np.allclose(np.degrees(np.diff(true_angle)), 18.0)


def test_bad_settings_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_synthetic_video(tmp_path / "a.mp4", fps=0)
    with pytest.raises(ValueError):
        write_synthetic_video(tmp_path / "a.mp4", width=50, height=50)


def test_marker_can_be_left_out_of_a_frame():
    frame = make_disc_frame(320, 240, 0.0, show_marker=False)
    blue, green, red = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    assert not ((green > 150) & (blue < 100) & (red < 100)).any()


def test_hidden_frames_have_no_marker_in_the_video(tmp_path):
    path = tmp_path / "hidden.mp4"
    write_synthetic_video(
        path, rpm=60.0, fps=20.0, seconds=1.0, width=320, height=240, hidden_frames=range(5, 8)
    )
    for frame_number, should_have_marker in [(4, True), (5, False), (7, False), (8, True)]:
        frame = read_frame(path, frame_number)
        blue, green, red = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        has_marker = bool(((green > 150) & (blue < 100) & (red < 100)).any())
        assert has_marker == should_have_marker, f"frame {frame_number}"


def test_constant_speed_and_decay_agree_at_the_start():
    times = np.linspace(0.0, 0.05, 6)
    steady = true_marker_angle(times, rpm=120.0, start_angle_rad=0.0)
    slowing = true_marker_angle(times, rpm=120.0, start_angle_rad=0.0, decay_s=5.0)
    # Over the first 0.05 s the speed has hardly fallen (1 percent), so the angles are very close.
    assert np.allclose(slowing, steady, atol=0.006)
    assert slowing[-1] < steady[-1]  # but it is already a little behind


def test_decay_starts_at_the_given_speed_and_approaches_a_total_angle():
    rpm, decay_s = 60.0, 2.0
    omega = 2 * math.pi * rpm / 60.0

    # The speed at t = 0 is the starting speed: a tiny step gives omega.
    step = 1e-6
    start_speed = (true_marker_angle([step], rpm, 0.0, decay_s=decay_s)[0] - 0.0) / step
    assert start_speed == pytest.approx(omega, rel=1e-5)

    # After a very long time the disc has turned omega * decay_s in total.
    final = true_marker_angle([1000.0], rpm, 0.0, decay_s=decay_s)[0]
    assert final == pytest.approx(omega * decay_s)

    # After one time constant the speed is 1 / e of the start: the angle change over a short step shows it.
    speed_at_tau = (true_marker_angle([decay_s + step], rpm, 0.0, decay_s=decay_s)[0]
                    - true_marker_angle([decay_s], rpm, 0.0, decay_s=decay_s)[0]) / step
    assert speed_at_tau == pytest.approx(omega / math.e, rel=1e-4)


def test_negative_decay_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_synthetic_video(tmp_path / "a.mp4", decay_s=-1.0)


def test_true_speed_values():
    assert true_marker_speed_rpm([0.0, 3.0], rpm=120.0) == pytest.approx([120.0, 120.0])

    # Slowing down: after one time constant the speed is 1 / e of the start.
    assert true_marker_speed_rpm([2.0], rpm=100.0, decay_s=2.0)[0] == pytest.approx(100.0 / math.e)

    # Ripple of 0.1 rad at 2 Hz adds 60 * A * f = 12 rpm at t = 0, and nothing a quarter period later.
    ripple_start = true_marker_speed_rpm([0.0], rpm=0.0, ripple_amplitude_rad=0.1, ripple_hz=2.0)[0]
    ripple_quarter = true_marker_speed_rpm([0.125], rpm=0.0, ripple_amplitude_rad=0.1, ripple_hz=2.0)[0]
    assert ripple_start == pytest.approx(12.0)
    assert ripple_quarter == pytest.approx(0.0, abs=1e-9)


def test_true_speed_is_the_derivative_of_the_true_angle():
    settings = dict(rpm=200.0, ripple_amplitude_rad=0.05, ripple_hz=3.0, decay_s=2.5)
    times = np.linspace(0.0, 4.0, 17)
    step = 1e-6
    numerical = (
        true_marker_angle(times + step, **settings) - true_marker_angle(times - step, **settings)
    ) / (2 * step)
    assert np.allclose(numerical * 60.0 / (2 * math.pi), true_marker_speed_rpm(times, **settings), rtol=1e-6, atol=1e-6)


def test_synthetic_table_columns_and_time_axis():
    table = make_synthetic_table(rpm=120.0, fps=60.0, seconds=2.0)
    assert list(table.columns) == ["frame", "time_s", "marker_x", "marker_y", "area_px"]
    assert len(table) == 120
    assert table["frame"].tolist() == list(range(120))
    assert table["time_s"].iloc[60] == pytest.approx(1.0)


def test_synthetic_table_marker_moves_on_a_circle_at_the_true_angle():
    centre = (200.0, 150.0)
    table = make_synthetic_table(rpm=60.0, fps=30.0, seconds=2.0, centre=centre, radius_px=80.0, noise_px=0.0)

    distance = np.hypot(table["marker_x"] - centre[0], table["marker_y"] - centre[1])
    assert np.allclose(distance, 80.0)

    # The angle seen from the centre (counter-clockwise positive, image y points down) is the true angle.
    angle = np.arctan2(centre[1] - table["marker_y"], table["marker_x"] - centre[0])
    expected = np.angle(np.exp(1j * true_marker_angle(table["time_s"], 60.0)))  # wrapped to -pi..pi
    assert np.allclose(angle, expected)


def test_synthetic_table_noise_is_repeatable_and_has_the_right_size():
    first = make_synthetic_table(noise_px=0.2, seed=3)
    second = make_synthetic_table(noise_px=0.2, seed=3)
    different = make_synthetic_table(noise_px=0.2, seed=4)
    clean = make_synthetic_table(noise_px=0.0)

    assert first.equals(second)
    assert not first.equals(different)
    assert (first["marker_x"] - clean["marker_x"]).std() == pytest.approx(0.2, rel=0.15)


def test_synthetic_table_lost_frames_are_empty():
    table = make_synthetic_table(lost_frames=[5, 6, 9999])  # a frame number past the end is ignored
    assert table.loc[[5, 6], ["marker_x", "marker_y", "area_px"]].isna().all().all()
    assert table["marker_x"].isna().sum() == 2


def test_synthetic_table_uses_ripple_and_decay():
    table = make_synthetic_table(rpm=100.0, fps=60.0, seconds=3.0, ripple_deg=3.0, ripple_hz=2.0, decay_s=2.0, noise_px=0.0)
    angle = np.unwrap(np.arctan2(240.0 - table["marker_y"], table["marker_x"] - 320.0))
    truth = true_marker_angle(table["time_s"], 100.0, ripple_amplitude_rad=math.radians(3.0), ripple_hz=2.0, decay_s=2.0)
    assert np.allclose(angle - angle[0], truth - truth[0])


def test_synthetic_table_rejects_bad_settings():
    with pytest.raises(ValueError):
        make_synthetic_table(fps=0)
    with pytest.raises(ValueError):
        make_synthetic_table(seconds=-1)
