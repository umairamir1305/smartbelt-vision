"""Tests for src/pipeline.py (the analysis of one pulley) and for drawing several markers on a video."""

import pandas as pd
import pytest

from src.pipeline import analyse_pulley, save_plots, save_spectrum_files, video_labels
from src.signal_analysis import SignalSettings
from src.synthetic import write_synthetic_belt_video, write_synthetic_video
from src.tracking import (
    TrackingError,
    annotate_layers,
    settings_for_colour,
    track_marker,
    write_annotated_video_layers,
)
from src.video_io import get_video_info


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("videos") / "synthetic_pipeline.mp4"
    write_synthetic_video(path, rpm=60.0, fps=30.0, seconds=4.0, width=320, height=240)
    return path


def test_analyse_pulley_returns_everything(video):
    result = analyse_pulley(video, settings_for_colour("green"), centre=(160, 120))

    assert result.info.frame_count == 120
    assert result.centre == (160, 120) and result.used_centre == (160, 120)
    assert result.tracking["overall_quality"] == "Good"
    assert result.rotation["turns"] == pytest.approx(3.97, abs=0.02)
    assert result.speed["mean_rpm"] == pytest.approx(60.0, abs=0.1)
    assert result.signal is not None and result.signal["frames"] == 120
    assert result.fit["usable"]
    assert result.notes == [] and result.signal_problem is None and result.analysis_problem is None
    assert result.tracking_seconds > 0
    for column in ("angle_rad", "rpm", "rpm_smooth", "rpm_fluctuation"):
        assert column in result.table.columns


def test_without_a_centre_only_the_tracking_is_done(video):
    result = analyse_pulley(video, settings_for_colour("green"))
    assert result.rotation is None and result.speed is None and result.signal is None and result.fit is None
    assert "angle_rad" not in result.table.columns
    assert result.used_centre is None


def test_problems_are_stored_and_not_raised(video):
    # 6 frames: too few for an angle.
    short = analyse_pulley(video, settings_for_colour("green"), centre=(160, 120), max_frames=6)
    assert short.rotation is None and "trusted marker position" in short.analysis_problem

    # 15 frames: an angle and a speed, but too few for the signal analysis.
    medium = analyse_pulley(video, settings_for_colour("green"), centre=(160, 120), max_frames=15)
    assert medium.speed is not None and medium.signal is None
    assert "frames in a row" in medium.signal_problem


def test_fitted_centre_is_used_when_asked_and_trustworthy(video):
    result = analyse_pulley(video, settings_for_colour("green"), centre=(166, 120), use_fitted_centre=True)
    assert result.used_centre != (166, 120)
    assert result.used_centre == pytest.approx((160, 120), abs=0.3)
    assert result.tracking["radius_spread_percent"] < 1.0  # the frames were rated again with the better centre

    short = analyse_pulley(video, settings_for_colour("green"), centre=(166, 120), use_fitted_centre=True, max_frames=12)
    assert short.used_centre == (166, 120)
    assert any("fitted centre was not used" in note for note in short.notes)


def test_bad_inputs_raise_errors(video):
    with pytest.raises(TrackingError, match="needs a rotation centre"):
        analyse_pulley(video, settings_for_colour("green"), use_fitted_centre=True)
    with pytest.raises(TrackingError, match="outside the frame"):
        analyse_pulley(video, settings_for_colour("green"), centre=(5000, 5))
    with pytest.raises(Exception, match="smoothing window"):
        analyse_pulley(video, settings_for_colour("green"), centre=(160, 120), signal_settings=SignalSettings(smooth_window_s=0))


def test_progress_is_reported(video):
    calls = []
    analyse_pulley(video, settings_for_colour("green"), centre=(160, 120), max_frames=30, progress=lambda a, b: calls.append((a, b)))
    assert calls[-1] == (30, 30)


def test_save_helpers_use_the_given_folder(video, tmp_path):
    result = analyse_pulley(video, settings_for_colour("green"), centre=(160, 120))
    out = tmp_path / "somewhere"
    plots = save_plots(result.table, result.used_centre, "run", False, result.speed["mean_rpm"], result.signal, out)
    spectrum = save_spectrum_files(result.signal, "run", out)

    assert sorted(path.name for path in plots) == [
        "run_angle.png", "run_fluctuation.png", "run_rpm.png", "run_spectrum.png", "run_tracking_quality.png", "run_trajectory.png",
    ]
    assert [path.name for path in spectrum] == ["run_spectrum.csv", "run_peaks.csv"]
    assert all(path.parent == out and path.exists() for path in plots + spectrum)


def test_video_labels_are_exported_from_the_pipeline():
    table = pd.DataFrame({"angle_wrapped_rad": [0.5], "rpm": [100.4]})
    assert video_labels(table) == ["angle 28.6 deg   speed 100 rpm"]


# ---------------------------------------------------------------------------
# Several markers on one video
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def belt_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("videos") / "belt.mp4"
    info = write_synthetic_belt_video(path, driver_rpm=90.0, ratio=1.5, fps=30.0, seconds=2.0)
    driver = track_marker(path, settings_for_colour("green"), centre=info["driver_centre"], region_radius_px=1.05 * info["driver_radius_px"])
    driven = track_marker(path, settings_for_colour("orange"), centre=info["driven_centre"], region_radius_px=1.05 * info["driven_radius_px"])
    return path, info, driver, driven


def test_two_markers_are_drawn_on_one_video(belt_video, tmp_path):
    path, info, driver, driven = belt_video
    layers = [
        {"name": "driver", "table": driver, "centre": info["driver_centre"], "region_radius_px": 100, "extra_text": None},
        {"name": "driven", "table": driven, "centre": info["driven_centre"], "region_radius_px": 150,
         "extra_text": [f"angle {number}" for number in range(len(driven))]},
    ]
    out = write_annotated_video_layers(path, layers, tmp_path / "sub" / "both.mp4", fps=30.0, header_texts=["ratio 1.5"] * len(driver))
    video = get_video_info(out)
    assert out.exists() and video.frame_count == 60 and (video.width, video.height) == (960, 540)


def test_layer_video_checks_its_inputs(belt_video, tmp_path):
    path, info, driver, driven = belt_video
    layer = {"name": "a", "table": driver, "centre": None, "region_radius_px": None, "extra_text": None}
    with pytest.raises(TrackingError, match="no tracked frames"):
        write_annotated_video_layers(path, [], tmp_path / "x.mp4", fps=30.0)
    with pytest.raises(TrackingError, match="same number of frames"):
        write_annotated_video_layers(path, [layer, dict(layer, table=driven.iloc[:10])], tmp_path / "x.mp4", fps=30.0)
    with pytest.raises(TrackingError, match="one text for every row"):
        write_annotated_video_layers(path, [dict(layer, extra_text=["too short"])], tmp_path / "x.mp4", fps=30.0)
    with pytest.raises(TrackingError, match="header_texts"):
        write_annotated_video_layers(path, [layer], tmp_path / "x.mp4", fps=30.0, header_texts=["one"])


def test_annotate_layers_draws_names_and_text():
    import numpy as np

    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    layer = {"name": "driver", "marker": (150.0, 100.0, 300.0), "quality": "Good", "centre": (100, 100),
             "region_radius_px": 50, "trail": [(140.0, 100.0, "Good")], "text": "angle 10.0 deg"}
    drawn = annotate_layers(frame, 5, 0.5, [layer], "ratio 2")
    assert drawn.max() > 0 and frame.max() == 0  # the original is not changed
    lost = annotate_layers(frame, 5, 0.5, [dict(layer, marker=None, quality="Lost")])
    assert not np.array_equal(drawn, lost)
