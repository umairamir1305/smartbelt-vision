"""Tests for src/app_support.py: the parts of the app that need no browser."""

import io
import zipfile
from dataclasses import replace

import pytest

from src import app_support as support
from src.synthetic import write_synthetic_belt_video
from src.tracking import TrackingError, find_marker, settings_for_colour
from src.video_io import get_video_info, read_frame


@pytest.fixture(scope="module")
def marked(tmp_path_factory):
    """A simulated belt drive with a mark on the belt: driver slip 1 percent, driven slip 2 percent."""
    path = tmp_path_factory.mktemp("app_videos") / "synthetic_marked.mp4"
    info = write_synthetic_belt_video(
        path, driver_rpm=120.0, ratio=2.0, driver_slip_percent=1.0, slip_percent=2.0, draw_belt=True,
        width=640, height=360, fps=60.0, seconds=4.0, scale_mm_per_px=0.25,
    )
    return path, info


def pulley_inputs(info, with_mm=True):
    items = {}
    for name, colour in (("driver", "green"), ("driven", "orange")):
        items[name] = support.PulleyInput(
            settings_for_colour(colour), tuple(info[f"{name}_centre"]), round(info[f"{name}_radius_px"] * 1.05),
            info[f"{name}_effective_radius_mm"] if with_mm else None,
            info[f"{name}_marker_radius_mm"] if with_mm else None,
        )
    return items


@pytest.fixture(scope="module")
def belt_result(marked):
    path, info = marked
    config = support.RunConfig(str(path), "belt", pulley_inputs(info), belt_settings=settings_for_colour("blue"))
    return support.run_analysis(config)


@pytest.fixture(scope="module")
def single_result(marked):
    path, info = marked
    config = support.RunConfig(str(path), "single", {"pulley": pulley_inputs(info)["driver"]})
    return support.run_analysis(config)


# ------------------------------------------------------------------ the video


def test_sample_videos_are_listed_by_name_and_other_files_are_left_out(tmp_path):
    for name in ("b.mp4", "a.MOV", "notes.txt", "c.avi"):
        (tmp_path / name).write_bytes(b"x")
    assert [path.name for path in support.list_sample_videos(tmp_path)] == ["a.MOV", "b.mp4", "c.avi"]
    assert support.list_sample_videos(tmp_path / "missing") == []


def test_an_uploaded_video_is_written_to_a_folder_on_this_computer(tmp_path):
    path = support.save_uploaded_video("clip.mp4", b"video bytes", tmp_path)
    assert path == tmp_path / "clip.mp4" and path.read_bytes() == b"video bytes"
    default_folder = support.save_uploaded_video("other.MOV", b"data")
    assert default_folder.exists() and default_folder.parent.name.startswith("smartbelt_")


def test_uploaded_names_cannot_leave_the_folder_and_bad_files_are_refused(tmp_path):
    folder = tmp_path / "uploads"
    assert support.save_uploaded_video("../evil.mp4", b"x", folder).parent == folder
    with pytest.raises(ValueError, match="does not look like a video"):
        support.save_uploaded_video("report.pdf", b"x", folder)
    with pytest.raises(ValueError, match="is empty"):
        support.save_uploaded_video("empty.mp4", b"", folder)


def test_video_rows_and_warnings(marked):
    path, _ = marked
    info = support.video_info(path)
    rows = dict(support.video_rows(info, path.name))
    assert rows["Picture size"] == "640 x 360 pixels" and rows["Frame rate"] == "60 fps (read from file)"
    assert support.video_info(path, 30.0).fps == 30.0
    assert not support.video_warnings(info)


# ------------------------------------------------------------------ pictures


def test_the_frame_preview_shows_the_grid_and_the_centres(marked):
    path, info = marked
    plain = support.frame_preview(path, 0)
    assert plain.shape == (360, 640, 3)
    with_grid = support.frame_preview(path, 0, grid_step=100)
    with_centre = support.frame_preview(path, 0, [(tuple(info["driver_centre"]), 60)])
    assert (with_grid != plain).any() and (with_centre != plain).any()
    assert (support.frame_preview(path, 0, [(None, None)]) == plain).all()  # a centre that is None is skipped


def test_the_mask_preview_finds_the_marker(marked):
    path, _ = marked
    picture, detection = support.mask_preview(path, 0, settings_for_colour("green"))
    assert detection is not None and picture.shape[2] == 3
    _, nothing = support.mask_preview(path, 0, settings_for_colour("magenta"))
    assert nothing is None


def test_colour_limits_from_a_pixel_on_the_marker(marked):
    path, _ = marked
    marker = find_marker(read_frame(path, 0), settings_for_colour("green"))
    settings, detection = support.settings_from_pixel(path, 0, marker.x, marker.y)
    assert detection is not None and 35 <= settings.hue_low <= 60 and settings.hue_high > settings.hue_low
    assert settings.min_area_px < detection.area_px < settings.max_area_px
    with pytest.raises(TrackingError, match="almost grey"):
        support.settings_from_pixel(path, 0, 3, 3)
    with pytest.raises(TrackingError, match="outside the picture"):
        support.settings_from_pixel(path, 0, 5000, 5)


# ------------------------------------------------------------------ the analysis


def test_a_single_pulley_run_gives_the_speed_and_the_plots(single_result):
    result = single_result
    assert result.simulated and result.video_name == "synthetic_marked.mp4"
    assert result.pulleys["pulley"].speed["mean_rpm"] == pytest.approx(120.0, abs=0.5)
    assert [suffix for suffix, _, _ in result.figures["pulley"]] == [
        "trajectory", "angle", "rpm", "fluctuation", "spectrum", "tracking_quality",
    ]
    assert all(png.startswith(b"\x89PNG") for _, _, png in result.figures["pulley"])
    assert result.outcome is None and "rpm" in result.table.columns
    assert support.csv_name(result) == "synthetic_marked_analysis.csv"


def test_a_belt_drive_run_gives_the_ratio_and_the_slip(belt_result):
    outcome = belt_result.outcome
    assert outcome.belt["measured_ratio"] == pytest.approx(2.06, abs=0.02)
    assert outcome.belt["speed_loss_percent"] == pytest.approx(2.98, abs=0.3)
    assert outcome.slip["driver_slip_percent"] == pytest.approx(1.0, abs=0.3)
    assert outcome.slip["driven_slip_percent"] == pytest.approx(2.0, abs=0.3)
    assert [suffix for suffix, _, _ in belt_result.figures["belt"]] == ["belt_speeds", "belt_surface_speed", "belt_mark_speed", "belt_slip"]
    assert support.csv_name(belt_result) == "synthetic_marked_belt_analysis.csv"
    assert "driver_slip_percent" in belt_result.table.columns


def test_without_a_belt_mark_there_is_no_slip(marked):
    path, info = marked
    config = support.RunConfig(str(path), "belt", pulley_inputs(info))
    result = support.run_analysis(config)
    assert result.outcome.slip is None and result.outcome.belt_table is None
    assert [suffix for suffix, _, _ in result.figures["belt"]] == ["belt_speeds", "belt_surface_speed"]


def test_a_belt_mark_with_no_scale_gives_the_reason_and_no_number(marked):
    path, info = marked
    config = support.RunConfig(str(path), "belt", pulley_inputs(info, with_mm=False), belt_settings=settings_for_colour("blue"))
    result = support.run_analysis(config)
    assert result.outcome.slip is None
    notes = support.result_notes(result)
    assert any("Slip estimation unavailable" in note and "a scale is needed" in note for note in notes)


def test_a_wrong_centre_is_a_plain_error(marked):
    path, info = marked
    items = pulley_inputs(info)
    items["driver"] = replace(items["driver"], centre=(5000.0, 5000.0))
    with pytest.raises(TrackingError):
        support.run_analysis(support.RunConfig(str(path), "belt", items))


def test_progress_is_reported_for_every_marker(marked):
    path, info = marked
    labels = []

    def factory(label):
        labels.append(label)
        return lambda done, total: None

    config = support.RunConfig(str(path), "belt", pulley_inputs(info), belt_settings=settings_for_colour("blue"), max_frames=60)
    support.run_analysis(config, factory)
    assert labels == ["Tracking driver", "Tracking driven", "Tracking belt mark"]


# ------------------------------------------------------------------ what is shown and downloaded


def test_the_summary_sentences_repeat_numbers_and_never_name_a_cause(belt_result):
    text = " ".join(support.engineering_summary(belt_result))
    assert "120.00 rpm" in text and "speed ratio is 2.06" in text and "slip is +1." in text
    for word in ("fault", "defect", "worn", "bearing", "failure", "misaligned"):
        assert word not in text.lower()


def test_the_settings_rows_list_what_was_used(belt_result):
    rows = dict(support.settings_rows(belt_result))
    assert rows["Analysis"] == "Two-pulley belt drive"
    assert "Driver pulley: marker colour" in rows and "Belt mark: colour" in rows
    assert rows["Smoothing"].startswith("Savitzky-Golay, window 0.2 s")
    assert rows["Trend removed"] == "polynomial of order 1"


def test_the_plots_come_as_a_zip_with_clear_names(belt_result, single_result):
    names = zipfile.ZipFile(io.BytesIO(support.plots_zip(belt_result))).namelist()
    assert "synthetic_marked_driver_rpm.png" in names and "synthetic_marked_belt_slip.png" in names
    assert len(names) == 6 + 6 + 4
    single_names = zipfile.ZipFile(io.BytesIO(support.plots_zip(single_result))).namelist()
    assert "synthetic_marked_rpm.png" in single_names and len(single_names) == 6


def test_the_csv_is_the_table(belt_result):
    text = support.csv_bytes(belt_result).decode("utf-8")
    assert text.splitlines()[0].startswith("driver_frame") or "driver_rpm" in text.splitlines()[0]
    assert len(text.splitlines()) == len(belt_result.table) + 1


def test_the_report_has_the_results_the_graphs_and_the_limitations(belt_result):
    html = support.report_html(belt_result, "Run with a slack belt.")
    for part in ("Results: Driver pulley", "Results: Driven pulley", "Belt drive", "Comparison of the two pulleys", "Belt slip",
                 "Slip on driver", "Limitations", "Run with a slack belt.", "SIMULATED data", "Analysis settings"):
        assert part in html
    assert html.count("data:image/png;base64,") == 16
    assert "http://" not in html and "https://" not in html


def test_the_report_of_a_single_pulley(single_result):
    html = support.report_html(single_result)
    assert "<h2>Results</h2>" in html and "Belt drive" not in html and html.count("data:image/png;base64,") == 6


def test_an_annotated_frame_is_the_right_frame(belt_result):
    picture, frame_number, time_s = support.annotated_frame(belt_result, 30)
    assert picture.shape == (360, 640, 3) and frame_number == 30 and time_s == pytest.approx(0.5, abs=0.01)
    last, last_number, _ = support.annotated_frame(belt_result, 10_000)  # clamped to the last frame
    assert last_number == len(belt_result.table) - 1
    first, first_number, _ = support.annotated_frame(belt_result, -5)
    assert first_number == 0
    assert (picture != read_frame(belt_result.info.path, 30)).any()  # markers and labels are drawn


def test_the_tracked_video_has_one_frame_per_row(belt_result, tmp_path):
    path = support.write_tracked_video(belt_result, tmp_path / "tracked.mp4")
    assert get_video_info(path).frame_count == len(belt_result.table)
