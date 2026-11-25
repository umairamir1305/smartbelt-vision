"""Tests for src/tracking.py. Simulated frames from src/synthetic.py give a known right answer."""

import math
from dataclasses import replace

import cv2
import numpy as np
import pandas as pd
import pytest

from src.synthetic import MARKER_COLOURS_BGR, disc_geometry, make_disc_frame, true_marker_angle, write_synthetic_video
from src.tracking import (
    TrackingError,
    TrackingSettings,
    add_tracking_quality,
    annotate_frame,
    colour_range_from_patch,
    describe_settings,
    find_marker,
    make_colour_mask,
    make_mask_preview,
    measure_blob_at,
    require_enough_detections,
    settings_for_colour,
    suggest_area_limits,
    summarise_tracking,
    track_marker,
    validate_settings,
    write_annotated_video,
)
from src.video_io import get_video_info

WIDTH, HEIGHT = 320, 240
CENTRE = (WIDTH // 2, HEIGHT // 2)
MARKER_AREA = math.pi * (0.03 * HEIGHT) ** 2  # area of the simulated marker in square pixels


def true_position(angle_rad, width=WIDTH, height=HEIGHT):
    """Where the simulated marker really is for a given angle."""
    centre, _, orbit_radius, _ = disc_geometry(width, height)
    return centre[0] + orbit_radius * math.cos(angle_rad), centre[1] - orbit_radius * math.sin(angle_rad)


def distance_between(detection, true_x, true_y):
    return math.hypot(detection.x - true_x, detection.y - true_y)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_settings_for_colour_and_overrides():
    green = settings_for_colour("green")
    assert (green.hue_low, green.hue_high) == (35, 85)
    changed = settings_for_colour("green", sat_min=150, min_area_px=200)
    assert changed.sat_min == 150
    assert changed.min_area_px == 200
    assert changed.val_min == green.val_min  # untouched values stay the same


def test_unknown_colour_is_rejected():
    with pytest.raises(TrackingError, match="Unknown colour"):
        settings_for_colour("purple")


@pytest.mark.parametrize(
    "bad_values",
    [
        {"hue_low": -1},
        {"hue_high": 180},
        {"sat_min": 300},
        {"val_min": -5},
        {"min_area_px": 0},
        {"min_area_px": 500, "max_area_px": 400},
        {"blur_size": 4},
        {"blur_size": 1},
        {"morph_size": -1},
    ],
)
def test_bad_settings_are_rejected(bad_values):
    with pytest.raises(TrackingError):
        validate_settings(replace(TrackingSettings(), **bad_values))


def test_default_settings_are_valid():
    validate_settings(TrackingSettings())


def test_describe_settings_mentions_wrap_around_for_red():
    assert "wraps around" in describe_settings(settings_for_colour("red"))
    assert "wraps around" not in describe_settings(settings_for_colour("green"))


def test_suggest_area_limits():
    assert suggest_area_limits(1000.0) == (300, 3000)
    assert suggest_area_limits(5.0)[0] == 10  # never below 10 pixels


# ---------------------------------------------------------------------------
# Colour mask and marker detection
# ---------------------------------------------------------------------------


def test_hue_range_that_wraps_around_zero_is_handled():
    # Hue 0 and hue 175 are both red. Hue 90 (cyan) and hue 20 (orange) are not.
    hsv_row = np.array([[[0, 255, 255], [175, 255, 255], [90, 255, 255], [20, 255, 255]]], dtype=np.uint8)
    bgr_row = cv2.cvtColor(hsv_row, cv2.COLOR_HSV2BGR)
    settings = settings_for_colour("red", blur_size=0, morph_size=0)
    assert make_colour_mask(bgr_row, settings)[0].tolist() == [255, 255, 0, 0]


def test_grey_and_dark_pixels_are_not_picked_up():
    # White disc, dark background and a black dot have no strong colour.
    frame = make_disc_frame(WIDTH, HEIGHT, 0.0, show_marker=False)
    for colour in MARKER_COLOURS_BGR:
        assert make_colour_mask(frame, settings_for_colour(colour)).sum() == 0


@pytest.mark.parametrize("colour", sorted(MARKER_COLOURS_BGR))
def test_the_preset_for_each_colour_finds_the_marker(colour):
    angle = math.radians(40)
    frame = make_disc_frame(WIDTH, HEIGHT, angle, colour)
    detection = find_marker(frame, settings_for_colour(colour))
    true_x, true_y = true_position(angle)
    assert detection is not None
    assert distance_between(detection, true_x, true_y) < 0.6
    assert detection.area_px == pytest.approx(MARKER_AREA, rel=0.25)


def test_position_is_accurate_over_a_full_turn():
    settings = settings_for_colour("green")
    worst = 0.0
    for degrees in range(0, 360, 7):
        angle = math.radians(degrees)
        detection = find_marker(make_disc_frame(WIDTH, HEIGHT, angle), settings)
        assert detection is not None
        worst = max(worst, distance_between(detection, *true_position(angle)))
    assert worst < 0.6


def test_marker_is_still_found_in_a_noisy_frame():
    angle = math.radians(100)
    frame = make_disc_frame(WIDTH, HEIGHT, angle, noise_sigma=10.0, rng=np.random.default_rng(3))
    detection = find_marker(frame, settings_for_colour("green"))
    assert detection is not None
    assert distance_between(detection, *true_position(angle)) < 1.5


def test_hidden_marker_gives_none():
    frame = make_disc_frame(WIDTH, HEIGHT, 0.5, show_marker=False)
    assert find_marker(frame, settings_for_colour("green")) is None


def test_area_limits_reject_regions_that_are_too_small_or_too_big():
    frame = make_disc_frame(WIDTH, HEIGHT, 0.5)
    assert find_marker(frame, settings_for_colour("green", min_area_px=1000)) is None
    assert find_marker(frame, settings_for_colour("green", max_area_px=100)) is None


def test_the_biggest_region_is_chosen_and_the_pulley_region_excludes_distractors():
    angle = math.radians(40)
    frame = make_disc_frame(WIDTH, HEIGHT, angle)
    cv2.circle(frame, (30, 30), 20, MARKER_COLOURS_BGR["green"], -1)  # a bigger green blob far from the pulley
    settings = settings_for_colour("green")

    without_region = find_marker(frame, settings)
    assert without_region is not None
    assert distance_between(without_region, 30, 30) < 1.0  # the bigger blob wins

    with_region = find_marker(frame, settings, centre=CENTRE, region_radius_px=100)
    assert with_region is not None
    assert distance_between(with_region, *true_position(angle)) < 0.6  # the region leaves the distractor out


def test_pulley_region_outside_the_frame_gives_none():
    frame = make_disc_frame(WIDTH, HEIGHT, 0.5)
    assert find_marker(frame, settings_for_colour("green"), centre=(5000, 5000), region_radius_px=50) is None


# ---------------------------------------------------------------------------
# Picking helpers (used by pick_points.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("colour", sorted(MARKER_COLOURS_BGR))
def test_colour_settings_worked_out_from_a_click_find_the_marker(colour):
    angle = math.radians(40)
    frame = make_disc_frame(WIDTH, HEIGHT, angle, colour)
    true_x, true_y = true_position(angle)
    settings = colour_range_from_patch(frame, true_x, true_y)
    detection = find_marker(frame, settings)
    assert detection is not None
    assert distance_between(detection, true_x, true_y) < 0.6


def test_red_click_gives_a_range_that_wraps_around_zero():
    angle = math.radians(40)
    frame = make_disc_frame(WIDTH, HEIGHT, angle, "red")
    settings = colour_range_from_patch(frame, *true_position(angle))
    assert settings.hue_low > settings.hue_high


def test_clicking_on_a_grey_area_is_rejected():
    frame = make_disc_frame(WIDTH, HEIGHT, 0.5)
    with pytest.raises(TrackingError, match="grey"):
        colour_range_from_patch(frame, 110, 120)  # on the white disc


def test_clicking_outside_the_picture_is_rejected():
    frame = make_disc_frame(WIDTH, HEIGHT, 0.5)
    with pytest.raises(TrackingError, match="outside"):
        colour_range_from_patch(frame, 5000, 5000)


def test_measure_blob_at_returns_the_clicked_region():
    angle = math.radians(40)
    frame = make_disc_frame(WIDTH, HEIGHT, angle)
    true_x, true_y = true_position(angle)
    settings = settings_for_colour("green")

    blob = measure_blob_at(frame, settings, true_x, true_y)
    assert blob is not None
    assert blob.area_px == pytest.approx(MARKER_AREA, rel=0.25)
    assert measure_blob_at(frame, settings, 20, 20) is None  # nothing green there


# ---------------------------------------------------------------------------
# Tracking quality rules, on a table made by hand
# ---------------------------------------------------------------------------


def make_circle_table(count=10, centre=(200, 150), radius=100.0, step_deg=30.0, area=500.0):
    """A table of marker positions on a perfect circle. All frames are fine until changed by a test."""
    rows = []
    for i in range(count):
        angle = math.radians(step_deg * i)
        rows.append((i, centre[0] + radius * math.cos(angle), centre[1] - radius * math.sin(angle), area))
    table = pd.DataFrame(rows, columns=["frame", "marker_x", "marker_y", "area_px"])
    table.insert(1, "time_s", table["frame"] / 30.0)
    return table


def make_damaged_table():
    """Frame 3 lost, frame 5 wrong area, frame 7 far away from the circle."""
    table = make_circle_table()
    table.loc[3, ["marker_x", "marker_y", "area_px"]] = np.nan
    table.loc[5, "area_px"] = 100.0
    table.loc[7, ["marker_x", "marker_y"]] = [50.0, 40.0]
    return table


def test_quality_rules_on_a_table_made_by_hand():
    rated = add_tracking_quality(make_damaged_table(), centre=(200, 150))

    assert rated["tracking_quality"].tolist() == [
        "Good", "Good", "Good", "Lost", "Good", "Moderate", "Good", "Poor", "Moderate", "Good",
    ]
    assert rated["tracking_valid"].tolist() == [True, True, True, False, True, True, True, False, True, True]
    assert rated.loc[3, "quality_notes"] == "marker not found"
    assert rated.loc[5, "quality_notes"] == "area"
    assert rated.loc[7, "quality_notes"] == "distance from centre, jump"
    assert rated.loc[8, "quality_notes"] == "jump"  # the jump back from the bad frame
    assert rated.loc[4, "tracking_quality"] == "Good"  # no jump check right after a lost frame
    assert rated.loc[0, "radius_px"] == pytest.approx(100.0)


def test_without_a_centre_only_the_area_check_is_used():
    table = make_circle_table()
    table.loc[5, "area_px"] = 100.0
    table.loc[7, ["marker_x", "marker_y"]] = [50.0, 40.0]
    rated = add_tracking_quality(table, centre=None)

    assert rated.loc[5, "tracking_quality"] == "Moderate"
    assert rated.loc[7, "tracking_quality"] == "Good"
    assert rated["radius_px"].isna().all()


def test_a_table_with_no_marker_at_all_is_all_lost():
    table = make_circle_table(count=4)
    table[["marker_x", "marker_y", "area_px"]] = np.nan
    rated = add_tracking_quality(table, centre=(200, 150))
    assert rated["tracking_quality"].tolist() == ["Lost"] * 4
    assert not rated["tracking_valid"].any()


def test_summary_counts_and_overall_quality():
    centre = (200, 150)
    summary = summarise_tracking(add_tracking_quality(make_damaged_table(), centre), centre)

    assert summary["frames"] == 10
    assert summary["found"] == 9
    assert summary["detection_rate"] == pytest.approx(0.9)
    assert (summary["good"], summary["moderate"], summary["poor"], summary["lost"]) == (6, 2, 1, 1)
    assert summary["longest_gap_frames"] == 1
    assert (summary["area_problems"], summary["distance_problems"], summary["jump_problems"]) == (1, 1, 2)
    assert summary["median_radius_px"] == pytest.approx(100.0)
    assert summary["median_area_px"] == pytest.approx(500.0)
    # 6 of 10 frames are Good (below 90 percent) but 8 of 10 are Good or Moderate (80 percent).
    assert summary["overall_quality"] == "Moderate"


def test_overall_quality_good_and_poor():
    centre = (200, 150)
    good = summarise_tracking(add_tracking_quality(make_circle_table(), centre), centre)
    assert good["overall_quality"] == "Good"
    assert good["radius_spread_percent"] == pytest.approx(0.0, abs=1e-9)

    table = make_circle_table()
    table.loc[0:5, ["marker_x", "marker_y", "area_px"]] = np.nan  # 6 of 10 frames lost
    poor = summarise_tracking(add_tracking_quality(table, centre), centre)
    assert poor["overall_quality"] == "Poor"
    assert poor["longest_gap_frames"] == 6


def test_the_longest_gap_is_the_longest_run_of_lost_frames():
    table = make_circle_table(count=12)
    table.loc[[1, 2], ["marker_x", "marker_y", "area_px"]] = np.nan
    table.loc[[6, 7, 8, 9], ["marker_x", "marker_y", "area_px"]] = np.nan
    summary = summarise_tracking(add_tracking_quality(table, (200, 150)), (200, 150))
    assert summary["longest_gap_frames"] == 4


def test_too_few_detections_raise_an_error_with_advice():
    with pytest.raises(TrackingError, match="enough frames"):
        require_enough_detections({"found": 3, "frames": 10, "detection_rate": 0.3})
    require_enough_detections({"found": 9, "frames": 10, "detection_rate": 0.9})  # no error


# ---------------------------------------------------------------------------
# Tracking a whole video
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hidden_marker_video(tmp_path_factory):
    """60 frames at 20 fps, 60 rpm (18 degrees per frame), marker hidden in frames 20 to 24."""
    path = tmp_path_factory.mktemp("videos") / "hidden.mp4"
    write_synthetic_video(
        path, rpm=60.0, fps=20.0, seconds=3.0, width=WIDTH, height=HEIGHT, hidden_frames=range(20, 25)
    )
    return path


def test_tracking_a_video_with_a_hidden_marker(hidden_marker_video):
    table = track_marker(hidden_marker_video, settings_for_colour("green"), centre=CENTRE)

    assert len(table) == 60
    assert list(table.columns) == [
        "frame", "time_s", "marker_x", "marker_y", "area_px", "radius_px",
        "tracking_valid", "tracking_quality", "quality_notes",
    ]
    assert table["frame"].tolist() == list(range(60))
    assert table["time_s"].iloc[1] == pytest.approx(0.05)

    lost_frames = table.index[table["tracking_quality"] == "Lost"].tolist()
    assert lost_frames == [20, 21, 22, 23, 24]

    # The positions that were found agree with the true marker positions.
    found = table[table["tracking_quality"] != "Lost"]
    true_angles = true_marker_angle(found["time_s"].to_numpy(), 60.0)
    centre, _, orbit_radius, _ = disc_geometry(WIDTH, HEIGHT)
    expected_x = centre[0] + orbit_radius * np.cos(true_angles)
    expected_y = centre[1] - orbit_radius * np.sin(true_angles)
    errors = np.hypot(found["marker_x"].to_numpy() - expected_x, found["marker_y"].to_numpy() - expected_y)
    assert errors.max() < 0.6

    summary = summarise_tracking(table, CENTRE)
    assert summary["lost"] == 5
    assert summary["longest_gap_frames"] == 5
    assert summary["overall_quality"] == "Good"  # 55 of 60 frames are Good (91.7 percent)


def test_tracking_works_without_a_centre(hidden_marker_video):
    table = track_marker(hidden_marker_video, settings_for_colour("green"))
    assert table["radius_px"].isna().all()
    assert (table["tracking_quality"] == "Lost").sum() == 5


def test_start_frame_max_frames_and_progress(hidden_marker_video):
    calls = []
    table = track_marker(
        hidden_marker_video,
        settings_for_colour("green"),
        centre=CENTRE,
        start_frame=10,
        max_frames=8,
        progress=lambda done, total: calls.append((done, total)),
    )
    assert table["frame"].tolist() == list(range(10, 18))
    assert table["time_s"].iloc[0] == pytest.approx(0.5)  # time is frame / fps, so it starts at 10 / 20
    assert calls[0] == (1, 8)
    assert calls[-1] == (8, 8)


def test_region_radius_gives_the_same_positions(hidden_marker_video):
    plain = track_marker(hidden_marker_video, settings_for_colour("green"), centre=CENTRE)
    limited = track_marker(hidden_marker_video, settings_for_colour("green"), centre=CENTRE, region_radius_px=100)
    assert np.allclose(plain["marker_x"].to_numpy(), limited["marker_x"].to_numpy(), equal_nan=True)


def test_bad_geometry_and_start_frame_are_rejected(hidden_marker_video):
    settings = settings_for_colour("green")
    with pytest.raises(TrackingError, match="outside the frame"):
        track_marker(hidden_marker_video, settings, centre=(5000, 100))
    with pytest.raises(TrackingError, match="needs a rotation centre"):
        track_marker(hidden_marker_video, settings, region_radius_px=100)
    with pytest.raises(TrackingError, match="beyond the end"):
        track_marker(hidden_marker_video, settings, start_frame=500)
    with pytest.raises(TrackingError):
        track_marker(hidden_marker_video, replace(settings, hue_low=300))


def test_a_wrong_colour_finds_nothing_in_the_video(hidden_marker_video):
    table = track_marker(hidden_marker_video, settings_for_colour("blue"), centre=CENTRE)
    assert (table["tracking_quality"] == "Lost").all()
    summary = summarise_tracking(table, CENTRE)
    assert summary["overall_quality"] == "Poor"
    with pytest.raises(TrackingError, match="enough frames"):
        require_enough_detections(summary)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def test_annotated_video_has_the_same_frames_and_size(hidden_marker_video, tmp_path):
    table = track_marker(hidden_marker_video, settings_for_colour("green"), centre=CENTRE)
    out_path = write_annotated_video(
        hidden_marker_video, table, tmp_path / "sub" / "tracked.mp4", fps=20.0, centre=CENTRE, region_radius_px=100
    )
    info = get_video_info(out_path)
    assert out_path.exists()
    assert info.frame_count == 60
    assert (info.width, info.height) == (WIDTH, HEIGHT)


def test_annotated_video_accepts_extra_text_for_every_frame(hidden_marker_video, tmp_path):
    table = track_marker(hidden_marker_video, settings_for_colour("green"), centre=CENTRE)
    texts = [f"angle {number}" for number in range(len(table))]
    out_path = write_annotated_video(hidden_marker_video, table, tmp_path / "text.mp4", fps=20.0, centre=CENTRE, extra_text=texts)
    assert get_video_info(out_path).frame_count == 60

    with pytest.raises(TrackingError, match="one text for every row"):
        write_annotated_video(hidden_marker_video, table, tmp_path / "bad.mp4", fps=20.0, extra_text=["too short"])


def test_extra_text_is_drawn_on_the_frame():
    frame = make_disc_frame(WIDTH, HEIGHT, 0.5)
    plain = annotate_frame(frame, 0, 0.0, None, "Lost", None, None, [])
    with_text = annotate_frame(frame, 0, 0.0, None, "Lost", None, None, [], extra_text="angle 12.3 deg")
    assert not np.array_equal(plain, with_text)
    # The text sits in the second line (rows 30 to 60), so the area below it is unchanged.
    assert np.array_equal(plain[70:], with_text[70:])


def test_annotated_video_needs_a_non_empty_table(hidden_marker_video, tmp_path):
    table = track_marker(hidden_marker_video, settings_for_colour("green"), centre=CENTRE).iloc[0:0]
    with pytest.raises(TrackingError):
        write_annotated_video(hidden_marker_video, table, tmp_path / "x.mp4", fps=20.0)


def test_mask_preview_has_two_panels_and_reports_the_detection():
    frame = make_disc_frame(WIDTH, HEIGHT, math.radians(40))
    picture, detection = make_mask_preview(frame, settings_for_colour("green"), centre=CENTRE, region_radius_px=100)
    assert picture.shape == (HEIGHT, 2 * WIDTH, 3)
    assert detection is not None

    _, nothing = make_mask_preview(frame, settings_for_colour("blue"))
    assert nothing is None


def test_mask_preview_is_scaled_down_when_it_is_too_wide():
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    picture, _ = make_mask_preview(frame, settings_for_colour("green"), max_width=1000)
    assert picture.shape[1] <= 1000
