"""Tests for pick_points.py.

The real window cannot be opened in a test, so OpenCV's window functions are replaced
by fake ones. The fake waitKey plays a script of clicks and key presses.
"""

import math
import re

import cv2
import pytest

import pick_points
from src.synthetic import disc_geometry, true_marker_angle, write_synthetic_video
from src.tracking import find_marker
from src.video_io import read_frame


def marker_position(width, height):
    """Where the marker is in frame 0 of a simulated video."""
    centre, _, orbit_radius, _ = disc_geometry(width, height)
    angle = float(true_marker_angle([0.0], rpm=60.0)[0])
    return centre[0] + orbit_radius * math.cos(angle), centre[1] - orbit_radius * math.sin(angle)


class FakeWindow:
    """Stands in for the OpenCV window. Collects what was shown and plays a script of actions."""

    def __init__(self, monkeypatch, script, close_after=None):
        self.script = list(script)
        self.close_after = close_after  # after this many waitKey calls the window counts as closed
        self.wait_calls = 0
        self.callback = None
        self.last_view = None
        monkeypatch.setattr(cv2, "namedWindow", lambda *args, **kwargs: None)
        monkeypatch.setattr(cv2, "setMouseCallback", self.set_mouse_callback)
        monkeypatch.setattr(cv2, "imshow", self.show)
        monkeypatch.setattr(cv2, "waitKey", self.wait_key)
        monkeypatch.setattr(cv2, "getWindowProperty", self.window_property)
        monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    def window_property(self, name, property_id):
        if self.close_after is not None and self.wait_calls >= self.close_after:
            return 0.0
        return 1.0

    def set_mouse_callback(self, name, callback, param=None):
        self.callback = callback

    def show(self, name, picture):
        self.last_view = picture

    def wait_key(self, delay=0):
        self.wait_calls += 1
        if not self.script:
            if self.close_after is not None:
                return -1  # no key pressed: the loop must end because the window was closed
            return ord("q")  # never loop forever
        action = self.script.pop(0)
        if action[0] == "click":
            self.callback(cv2.EVENT_LBUTTONDOWN, action[1], action[2], 0, None)
            return -1
        return action[1]


def numbers_after(flag, text, count):
    """Read the numbers that follow a flag in the printed command, for example --centre 320 240."""
    match = re.search(flag + r"((?: -?\d+(?:\.\d+)?){" + str(count) + "})", text)
    assert match, f"{flag} not found in the printed command"
    return [float(value) for value in match.group(1).split()]


# ---------------------------------------------------------------------------
# Clicks and keys
# ---------------------------------------------------------------------------


def test_clicks_are_stored_in_order():
    state = pick_points.new_state()
    assert pick_points.next_step(state) == "marker"
    pick_points.handle_click(state, 10, 20)
    pick_points.handle_click(state, 30, 40)
    assert state == {"marker": (10, 20), "centre": (30, 40), "edge": None}
    assert pick_points.next_step(state) == "edge"
    pick_points.handle_click(state, 50, 60)
    assert pick_points.next_step(state) is None
    pick_points.handle_click(state, 70, 80)  # a fourth click changes nothing
    assert state["edge"] == (50, 60)


def test_undo_removes_the_last_click_and_reset_removes_all():
    state = pick_points.new_state()
    for point in [(1, 1), (2, 2), (3, 3)]:
        pick_points.handle_click(state, *point)
    pick_points.undo_last_click(state)
    assert state == {"marker": (1, 1), "centre": (2, 2), "edge": None}
    pick_points.reset_state(state)
    assert all(value is None for value in state.values())
    pick_points.undo_last_click(state)  # undo with nothing to undo is harmless


# ---------------------------------------------------------------------------
# Numbers worked out from the clicks
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("videos") / "small.mp4"
    write_synthetic_video(path, rpm=60.0, fps=20.0, seconds=0.5, width=640, height=480)
    return path


def test_results_from_three_good_clicks(small_video):
    frame = read_frame(small_video, 0)
    marker_x, marker_y = marker_position(640, 480)
    state = {"marker": (round(marker_x), round(marker_y)), "centre": (320, 240), "edge": (320 + 182, 240)}

    results = pick_points.compute_results(frame, state)
    assert results["message"] == ""
    assert results["centre"] == (320, 240)
    assert results["pulley_radius_px"] == pytest.approx(182)
    assert results["region_radius_px"] == pytest.approx(182 * 1.05)

    # The colour settings that came from the click must find the marker again.
    detection = find_marker(frame, results["settings"])
    assert detection is not None
    assert math.hypot(detection.x - marker_x, detection.y - marker_y) < 0.6

    # The area limits are 0.3 x and 3 x the measured marker area, so they surround it.
    assert results["settings"].min_area_px < results["blob"].area_px < results["settings"].max_area_px


def test_clicking_on_grey_gives_a_message_and_no_settings(small_video):
    frame = read_frame(small_video, 0)
    results = pick_points.compute_results(frame, {"marker": (250, 240), "centre": None, "edge": None})
    assert results["settings"] is None
    assert "grey" in results["message"]


def test_an_edge_click_next_to_the_centre_is_flagged(small_video):
    frame = read_frame(small_video, 0)
    results = pick_points.compute_results(frame, {"marker": None, "centre": (320, 240), "edge": (325, 240)})
    assert results["region_radius_px"] is None
    assert "very close" in results["message"]


def test_printed_command_contains_all_values(small_video):
    frame = read_frame(small_video, 0)
    marker_x, marker_y = marker_position(640, 480)
    state = {"marker": (round(marker_x), round(marker_y)), "centre": (320, 240), "edge": (502, 240)}
    results = pick_points.compute_results(frame, state)

    text = "\n".join(pick_points.format_results("sample_data/my video.mp4", results))
    assert 'python run_analysis.py "sample_data/my video.mp4"' in text  # a path with spaces is quoted
    for flag in ("--hue", "--sat-min", "--val-min", "--min-area", "--max-area", "--centre", "--region-radius"):
        assert flag in text


# ---------------------------------------------------------------------------
# The whole picker, with a fake window
# ---------------------------------------------------------------------------


def test_full_run_with_a_fake_window(small_video, tmp_path, monkeypatch, capsys):
    marker_x, marker_y = marker_position(640, 480)
    script = [
        ("click", round(marker_x), round(marker_y)),
        ("click", 320, 240),
        ("click", 502, 240),
        ("key", ord("q")),
    ]
    window = FakeWindow(monkeypatch, script)

    code = pick_points.run_picker(small_video, 0, output_dir=tmp_path)
    printed = capsys.readouterr().out

    assert code == 0
    assert numbers_after("--centre", printed, 2) == [320, 240]
    assert numbers_after("--region-radius", printed, 1) == [round(182 * 1.05)]
    assert "--hue" in printed
    assert (tmp_path / "small_picked.png").exists()
    assert window.last_view.shape == (480, 640, 3)  # a small frame is shown at full size


def test_clicks_are_converted_back_when_the_picture_is_shown_smaller(tmp_path, monkeypatch, capsys):
    path = tmp_path / "big.mp4"
    write_synthetic_video(path, rpm=60.0, fps=10.0, seconds=0.5, width=1920, height=1080)
    scale = min(1.0, pick_points.MAX_DISPLAY_WIDTH / 1920, pick_points.MAX_DISPLAY_HEIGHT / 1080)
    assert scale < 1.0

    marker_x, marker_y = marker_position(1920, 1080)
    # The clicks are given in WINDOW pixels, as the real window would report them.
    script = [
        ("click", marker_x * scale, marker_y * scale),
        ("click", 960 * scale, 540 * scale),
        ("click", (960 + 410) * scale, 540 * scale),
        ("key", 13),  # Enter finishes
    ]
    window = FakeWindow(monkeypatch, script)

    code = pick_points.run_picker(path, 0, output_dir=tmp_path)
    printed = capsys.readouterr().out

    assert code == 0
    centre_x, centre_y = numbers_after("--centre", printed, 2)
    assert abs(centre_x - 960) <= 1 and abs(centre_y - 540) <= 1
    (region_radius,) = numbers_after("--region-radius", printed, 1)
    assert abs(region_radius - 410 * 1.05) <= 2
    assert window.last_view.shape[0] <= pick_points.MAX_DISPLAY_HEIGHT
    assert window.last_view.shape[1] <= pick_points.MAX_DISPLAY_WIDTH


def test_undo_and_reset_keys(small_video, tmp_path, monkeypatch, capsys):
    marker_x, marker_y = marker_position(640, 480)
    script = [
        ("click", 10, 10),
        ("click", 20, 20),
        ("key", ord("u")),  # undo the second click
        ("key", ord("r")),  # start again
        ("click", round(marker_x), round(marker_y)),
        ("click", 320, 240),
        ("key", ord("q")),
    ]
    FakeWindow(monkeypatch, script)

    code = pick_points.run_picker(small_video, 0, output_dir=tmp_path)
    printed = capsys.readouterr().out
    assert code == 0
    assert numbers_after("--centre", printed, 2) == [320, 240]
    assert "--region-radius" not in printed  # the third click was never made


def test_finishing_without_any_click_returns_1(small_video, tmp_path, monkeypatch, capsys):
    FakeWindow(monkeypatch, [("key", ord("q"))])
    assert pick_points.run_picker(small_video, 0, output_dir=tmp_path) == 1
    assert "Nothing was picked" in capsys.readouterr().out
    assert not list(tmp_path.glob("*_picked.png"))


def test_a_build_of_opencv_without_windows_gives_a_helpful_error(small_video, tmp_path, monkeypatch):
    def no_window_support(*args, **kwargs):
        raise cv2.error("The function is not implemented. Rebuild the library with Windows support.")

    monkeypatch.setattr(cv2, "namedWindow", no_window_support)
    monkeypatch.setattr(cv2, "destroyAllWindows", no_window_support)
    with pytest.raises(OSError, match="opencv-python-headless"):
        pick_points.run_picker(small_video, 0, output_dir=tmp_path)


def test_main_reports_a_missing_video(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["pick_points.py", "no_such_video.mp4"])
    assert pick_points.main() == 1
    assert "not found" in capsys.readouterr().out


def test_closing_the_window_with_the_x_button_ends_the_picker(small_video, tmp_path, monkeypatch, capsys):
    marker_x, marker_y = marker_position(640, 480)
    window = FakeWindow(monkeypatch, [("click", round(marker_x), round(marker_y))], close_after=40)

    code = pick_points.run_picker(small_video, 0, output_dir=tmp_path)
    printed = capsys.readouterr().out

    assert code == 0  # the marker click was made before the window was closed
    assert "--hue" in printed
    assert window.wait_calls >= 40


def test_a_window_that_looks_closed_at_the_start_is_not_treated_as_closed(small_video, tmp_path, monkeypatch, capsys):
    # The window reports "not visible" from the very first call. The picker must keep going
    # for the first few loops, so the clicks that come early are still handled.
    marker_x, marker_y = marker_position(640, 480)
    script = [("click", round(marker_x), round(marker_y)), ("click", 320, 240)]
    FakeWindow(monkeypatch, script, close_after=0)

    code = pick_points.run_picker(small_video, 0, output_dir=tmp_path)
    printed = capsys.readouterr().out
    assert code == 0
    assert numbers_after("--centre", printed, 2) == [320, 240]


# ---------------------------------------------------------------------------
# One pulley of a belt drive (--pulley driver / driven)
# ---------------------------------------------------------------------------


def test_options_for_a_pulley_of_a_belt_drive_have_the_pulley_name(small_video):
    frame = read_frame(small_video, 0)
    marker_x, marker_y = marker_position(640, 480)
    state = {"marker": (round(marker_x), round(marker_y)), "centre": (320, 240), "edge": (502, 240)}
    results = pick_points.compute_results(frame, state)

    lines = pick_points.format_results("sample_data/belt.mp4", results, pulley="driven")
    text = "\n".join(lines)
    for flag in ("--driven-hue", "--driven-sat-min", "--driven-val-min", "--driven-min-area", "--driven-max-area",
                 "--driven-centre 320 240", "--driven-region-radius"):
        assert flag in text
    assert "Options for the driven pulley" in text
    assert "python run_analysis.py" not in text
    assert "python run_belt_drive.py" not in text  # only the options are printed, to be added to one command

    plain = "\n".join(pick_points.format_results("sample_data/belt.mp4", results))
    assert "python run_analysis.py sample_data/belt.mp4 --hue" in plain  # without a pulley nothing changes


def test_full_run_for_one_pulley_saves_a_picture_with_the_pulley_name(small_video, tmp_path, monkeypatch, capsys):
    marker_x, marker_y = marker_position(640, 480)
    script = [("click", round(marker_x), round(marker_y)), ("click", 320, 240), ("click", 502, 240), ("key", ord("q"))]
    FakeWindow(monkeypatch, script)

    code = pick_points.run_picker(small_video, 0, output_dir=tmp_path, pulley="driver")
    printed = capsys.readouterr().out
    assert code == 0
    assert numbers_after("--driver-centre", printed, 2) == [320, 240]
    assert (tmp_path / "small_picked_driver.png").exists()
    assert not (tmp_path / "small_picked.png").exists()


def test_the_pulley_option_is_checked_on_the_command_line(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["pick_points.py", "no_such_video.mp4", "--pulley", "wheel"])
    with pytest.raises(SystemExit):
        pick_points.main()
    assert "invalid choice" in capsys.readouterr().err


def test_options_for_the_mark_on_the_belt_have_no_centre(small_video):
    frame = read_frame(small_video, 0)
    marker_x, marker_y = marker_position(640, 480)
    state = {"marker": (round(marker_x), round(marker_y)), "centre": (320, 240), "edge": (502, 240)}
    results = pick_points.compute_results(frame, state)

    text = "\n".join(pick_points.format_results("sample_data/belt.mp4", results, pulley="belt"))
    for flag in ("--belt-hue", "--belt-sat-min", "--belt-val-min", "--belt-min-area", "--belt-max-area"):
        assert flag in text
    assert "--belt-centre" not in text and "--belt-region-radius" not in text and "Rotation centre" not in text
    assert "Options for the mark on the belt" in text


def test_picking_the_belt_mark_needs_one_click_and_q(small_video, tmp_path, monkeypatch, capsys):
    marker_x, marker_y = marker_position(640, 480)
    window = FakeWindow(monkeypatch, [("click", round(marker_x), round(marker_y)), ("key", ord("q"))])

    code = pick_points.run_picker(small_video, 0, output_dir=tmp_path, pulley="belt")
    printed = capsys.readouterr().out
    assert code == 0
    assert "--belt-hue" in printed and "--belt-centre" not in printed
    assert (tmp_path / "small_picked_belt.png").exists()
    assert window.last_view.shape == (480, 640, 3)
