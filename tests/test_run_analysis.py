"""Tests for run_analysis.py. The script is run through its main() function on small simulated videos."""

import re

import pandas as pd
import pytest

import run_analysis
from src.synthetic import write_synthetic_video
from src.video_io import get_video_info

TRACKING_COLUMNS = [
    "frame", "time_s", "marker_x", "marker_y", "area_px", "radius_px",
    "tracking_valid", "tracking_quality", "quality_notes",
]
ANGLE_COLUMNS = ["angle_wrapped_rad", "angle_rad", "segment"]
SPEED_COLUMNS = ["angular_velocity_rad_s", "rpm"]
SIGNAL_COLUMNS = [
    "speed_outlier", "speed_interpolated", "rpm_clean", "rpm_smooth", "rpm_trend", "rpm_fluctuation",
    "angular_acceleration_rad_s2",
]

# The test videos are 320 x 240, so the rotation centre is (160, 120) and the marker is 72 pixels from it.
CENTRE = (160, 120)


@pytest.fixture(scope="module")
def green_video(tmp_path_factory):
    """40 frames, 20 fps, 60 rpm (18 degrees per frame), green marker, hidden in frames 10 to 12."""
    path = tmp_path_factory.mktemp("videos") / "green.mp4"
    write_synthetic_video(path, rpm=60.0, fps=20.0, seconds=2.0, width=320, height=240, hidden_frames=range(10, 13))
    return path


@pytest.fixture(scope="module")
def red_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("videos") / "red.mp4"
    write_synthetic_video(path, rpm=-60.0, fps=20.0, seconds=2.0, width=320, height=240, colour="red")
    return path


@pytest.fixture(scope="module")
def slowing_video(tmp_path_factory):
    """120 rpm at the start, speed falling off with a time constant of 1.5 s. 30 fps, 3 s."""
    path = tmp_path_factory.mktemp("videos") / "slowing.mp4"
    write_synthetic_video(path, rpm=120.0, fps=30.0, seconds=3.0, width=320, height=240, decay_s=1.5)
    return path


@pytest.fixture(scope="module")
def ripple_video(tmp_path_factory):
    """150 rpm (30 degrees per frame at 30 fps) with a 4 degree, 3 Hz ripple in the angle. 6 s, 320 x 240."""
    path = tmp_path_factory.mktemp("videos") / "ripple.mp4"
    write_synthetic_video(path, rpm=150.0, fps=30.0, seconds=6.0, width=320, height=240, ripple_deg=4.0, ripple_hz=3.0)
    return path


@pytest.fixture(scope="module")
def simulated_named_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("videos") / "synthetic_test.mp4"
    write_synthetic_video(path, rpm=60.0, fps=20.0, seconds=2.0, width=320, height=240)
    return path


@pytest.fixture
def run_cli(monkeypatch, tmp_path):
    """Run the script with the given command line arguments. Results go into a temporary folder."""
    out_dir = tmp_path / "out"
    monkeypatch.setattr(run_analysis, "OUTPUT_DIR", out_dir)

    def run(*arguments):
        monkeypatch.setattr("sys.argv", ["run_analysis.py", *[str(argument) for argument in arguments]])
        return run_analysis.main(), out_dir

    return run


def test_normal_run_saves_table_plots_and_tracked_video(run_cli, green_video, capsys):
    code, out_dir = run_cli(green_video, "--colour", "green", "--centre", *CENTRE)
    printed = capsys.readouterr().out

    assert code == 0
    table = pd.read_csv(out_dir / "green_analysis.csv")
    assert list(table.columns) == TRACKING_COLUMNS + ANGLE_COLUMNS + SPEED_COLUMNS + SIGNAL_COLUMNS
    assert len(table) == 40
    assert table["marker_x"].isna().sum() == 3  # the three hidden frames are empty, not zero
    assert table["angle_rad"].isna().sum() == 3
    assert table["rpm"].isna().sum() == 3
    assert table["segment"].dropna().nunique() == 1
    assert table["rpm"].dropna().tolist() == pytest.approx([60.0] * 37, abs=1.5)  # 60 rpm, a little noise
    assert table["angular_velocity_rad_s"].dropna().mean() == pytest.approx(2 * 3.14159265, rel=0.01)

    assert table["speed_interpolated"].sum() == 3  # the three hidden frames were filled in for the analysis
    assert table["rpm_clean"].notna().all()

    for name in ("trajectory", "angle", "rpm", "fluctuation", "spectrum", "tracking_quality"):
        assert (out_dir / f"green_{name}.png").stat().st_size > 1000
    spectrum = pd.read_csv(out_dir / "green_spectrum.csv")
    assert list(spectrum.columns) == ["frequency_hz", "amplitude_rpm"]
    assert spectrum["frequency_hz"].iloc[0] == 0.0
    assert list(pd.read_csv(out_dir / "green_peaks.csv").columns) == [
        "rank", "frequency_hz", "period_s", "amplitude_rpm", "percent_of_mean", "order", "note",
    ]
    assert get_video_info(out_dir / "green_tracked.mp4").frame_count == 40

    assert "37 of 40" in printed
    assert "Tracking quality     : Good" in printed
    assert "Longest gap          : 3 frames" in printed
    assert "Angle frames         : 37 of 40" in printed
    # 39 steps of 18 degrees from frame 0 to frame 39.
    assert "counter-clockwise, 702.0 degrees (1.950 turns) from frame 0 to frame 39" in printed
    assert "Angle segments       : 1 (no long gap)" in printed
    assert "Speed frames         : 37 frames have a speed" in printed
    mean_speed = float(re.search(r"Mean speed +: ([\d.]+) rpm, counter-clockwise", printed).group(1))
    assert mean_speed == pytest.approx(60.0, abs=0.1)
    assert re.search(r"Speed range +: [\d.]+ to [\d.]+ rpm", printed)
    assert re.search(r"Speed variation +: [\d.]+ rpm standard deviation", printed)
    assert "the speed changes by about" not in printed  # a steady speed gets no slow-down note
    assert "Analysis stretch     : frames 0 to 39 (40 frames, 2.000 s), 3 lost frames filled in" in printed
    assert "Smoothing            : Savitzky-Golay, 5 frames (0.250 s), polynomial order 2" in printed
    assert "Speed noise" in printed and "Speed fluctuation" in printed and "Acceleration" in printed
    assert "Spectrum             : resolution 0.50 Hz, up to 10.0 Hz" in printed
    assert "SIMULATED" not in printed  # the file name does not start with "synthetic"


def test_a_clockwise_red_marker(run_cli, red_video, capsys):
    code, out_dir = run_cli(red_video, "--hue", 168, 12, "--centre", *CENTRE, "--no-video", "--no-plots")
    printed = capsys.readouterr().out
    assert code == 0
    assert "wraps around 0" in printed
    assert "clockwise, 702.0 degrees" in printed
    assert re.search(r"Mean speed +: 60\.\d\d rpm, clockwise", printed)  # reported as a positive speed
    table = pd.read_csv(out_dir / "red_analysis.csv")
    assert table["tracking_valid"].all()
    assert table["angle_rad"].iloc[-1] < table["angle_rad"].iloc[0]  # clockwise: the angle goes down
    assert table["rpm"].mean() == pytest.approx(-60.0, abs=0.5)  # the CSV keeps the sign


def test_a_ripple_is_listed_as_a_dominant_frequency(run_cli, ripple_video, capsys):
    code, out_dir = run_cli(ripple_video, "--centre", *CENTRE, "--no-video")
    printed = capsys.readouterr().out
    assert code == 0

    assert "Dominant frequencies : 1 found" in printed
    assert "Frequency  Period" in printed and "Amplitude" in printed
    assert re.search(r"1\s+3\.0\d Hz\s+0\.33\d s\s+1\d\.\d\d rpm", printed)  # about 11 to 12 rpm at 30 fps
    assert "source not identified" in printed
    assert "does not show the cause" in printed  # the caution is always printed with peaks

    peaks = pd.read_csv(out_dir / "ripple_peaks.csv")
    assert len(peaks) == 1
    assert peaks["rank"].iloc[0] == 1
    assert peaks["frequency_hz"].iloc[0] == pytest.approx(3.0, abs=0.1)
    assert peaks["order"].iloc[0] == pytest.approx(1.2, abs=0.05)  # 3 Hz against a 2.5 Hz rotation
    spectrum = pd.read_csv(out_dir / "ripple_spectrum.csv")
    assert spectrum["amplitude_rpm"].iloc[(spectrum["frequency_hz"] - 3.0).abs().argmin()] > 5.0


def test_min_peak_percent_decides_whether_the_ripple_is_listed(run_cli, ripple_video, capsys):
    # The ripple is about 11 rpm on 150 rpm, which is 7 or 8 percent of the speed.
    code, _ = run_cli(ripple_video, "--centre", *CENTRE, "--min-peak-percent", 5, "--no-video", "--no-plots")
    assert code == 0 and "Dominant frequencies : 1 found" in capsys.readouterr().out

    code, _ = run_cli(ripple_video, "--centre", *CENTRE, "--min-peak-percent", 20, "--no-video", "--no-plots")
    printed = capsys.readouterr().out
    assert code == 0
    assert "Dominant frequencies : none above 30" in printed  # 20 percent of 150 rpm


def test_a_steady_disc_has_no_dominant_frequency(run_cli, green_video, capsys):
    code, out_dir = run_cli(green_video, "--centre", *CENTRE, "--no-video", "--no-plots")
    printed = capsys.readouterr().out
    assert code == 0
    assert "Dominant frequencies : none above" in printed
    assert "does not show the cause" not in printed
    assert len(pd.read_csv(out_dir / "green_peaks.csv")) == 0  # the file exists, with only the header


def test_signal_options_change_the_analysis(run_cli, green_video, capsys):
    code, out_dir = run_cli(
        green_video, "--centre", *CENTRE, "--smooth-window", 0.4, "--trend-order", 0, "--max-gap-frames", 0,
        "--keep-outliers", "--no-video", "--no-plots",
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert "Analysis stretch     : frames 13 to 39 (27 frames" in printed  # a gap of 3 now ends the stretch
    assert "lost frames filled in" not in printed
    assert "Outliers             : not checked (--keep-outliers)" in printed
    assert "Savitzky-Golay, 9 frames (0.450 s)" in printed
    assert "trend of order 0 removed" in printed
    table = pd.read_csv(out_dir / "green_analysis.csv")
    assert table.loc[:12, "rpm_clean"].isna().all()
    assert table.loc[13:, "rpm_clean"].notna().all()


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--smooth-window", 0], "smoothing window"),
        (["--trend-order", 9], "trend order"),
        (["--max-gap-frames", -2], "cannot be negative"),
        (["--min-peak-percent", -1], "negative percentage"),
    ],
)
def test_bad_signal_options_give_a_plain_error_message(run_cli, green_video, capsys, arguments, message):
    code, out_dir = run_cli(green_video, "--centre", *CENTRE, *arguments)
    assert code == 1
    assert message in capsys.readouterr().out
    assert not out_dir.exists()  # nothing was done before the problem was found


def test_a_clip_that_is_too_short_for_the_signal_analysis_still_gives_angle_and_speed(run_cli, green_video, capsys):
    code, out_dir = run_cli(green_video, "--centre", *CENTRE, "--max-frames", 15, "--no-video", "--no-plots")
    printed = capsys.readouterr().out
    assert code == 0  # not an error: the angle and the speed are there
    assert "Mean speed" in printed
    assert "Signal analysis      : not done: Only 15 frames in a row have a speed" in printed
    table = pd.read_csv(out_dir / "green_analysis.csv")
    assert "rpm" in table.columns and "rpm_clean" not in table.columns
    assert not (out_dir / "green_spectrum.csv").exists()


def test_a_slowing_disc_gets_a_note_and_the_speed_falls(run_cli, slowing_video, capsys):
    code, out_dir = run_cli(slowing_video, "--centre", *CENTRE, "--no-video", "--no-plots")
    printed = capsys.readouterr().out
    assert code == 0
    assert "the speed changes by about" in printed
    assert "the speed is not steady" in printed  # the fluctuation spectrum is not reliable
    rpm = pd.read_csv(out_dir / "slowing_analysis.csv")["rpm"]
    assert rpm.iloc[:5].mean() == pytest.approx(120.0, rel=0.05)
    assert rpm.iloc[-5:].mean() < 0.3 * rpm.iloc[:5].mean()  # exp(-3 / 1.5) is 0.135 of the start


def test_video_labels_prefer_the_smoothed_speed_and_fall_back_to_the_single_frame_speed():
    table = pd.DataFrame(
        {
            "angle_wrapped_rad": [0.0, 0.0, 0.0],
            "rpm": [100.4, 100.4, 100.4],
            "rpm_smooth": [float("nan"), 120.2, float("nan")],
        }
    )
    assert run_analysis.video_labels(table) == [
        "angle 0.0 deg   speed 100 rpm", "angle 0.0 deg   speed 120 rpm", "angle 0.0 deg   speed 100 rpm",
    ]


def test_video_labels_show_angle_and_speed():
    table = pd.DataFrame(
        {
            "angle_wrapped_rad": [0.5, float("nan"), -1.5707963267948966],
            "rpm": [119.6, float("nan"), -60.2],
        }
    )
    assert run_analysis.video_labels(table) == ["angle 28.6 deg   speed 120 rpm", "", "angle 270.0 deg   speed -60 rpm"]

    without_speed = table.drop(columns="rpm")
    assert run_analysis.video_labels(without_speed) == ["angle 28.6 deg", "", "angle 270.0 deg"]


def test_simulated_videos_are_marked_as_simulated(run_cli, simulated_named_video, capsys):
    code, _ = run_cli(simulated_named_video, "--centre", *CENTRE, "--no-video", "--no-plots")
    assert code == 0
    assert "Data                 : SIMULATED video (not a real measurement)" in capsys.readouterr().out


def test_no_video_and_no_plots_options(run_cli, green_video):
    code, out_dir = run_cli(green_video, "--centre", *CENTRE, "--no-video", "--no-plots")
    assert code == 0
    assert (out_dir / "green_analysis.csv").exists()
    assert not (out_dir / "green_tracked.mp4").exists()
    assert not list(out_dir.glob("*.png"))


def test_without_a_centre_there_is_no_angle(run_cli, green_video, capsys):
    code, out_dir = run_cli(green_video, "--no-video")
    printed = capsys.readouterr().out
    assert code == 0
    table = pd.read_csv(out_dir / "green_analysis.csv")
    assert list(table.columns) == TRACKING_COLUMNS
    assert table["radius_px"].isna().all()
    assert "Rotation centre      : not given" in printed
    assert "Angle frames" not in printed
    assert "Mean speed" not in printed
    # The plots that do not need an angle are still made.
    assert (out_dir / "green_trajectory.png").exists()
    assert (out_dir / "green_tracking_quality.png").exists()
    assert not (out_dir / "green_angle.png").exists()
    assert not (out_dir / "green_rpm.png").exists()


def test_circle_fit_is_reported_and_agrees_with_the_true_centre(run_cli, green_video, capsys):
    code, _ = run_cli(green_video, "--centre", *CENTRE, "--no-video", "--no-plots")
    printed = capsys.readouterr().out
    assert code == 0
    match = re.search(r"Circle fit +: centre \(([\d.]+), ([\d.]+)\), radius ([\d.]+) px", printed)
    assert match, "the circle fit line was not printed"
    fit_x, fit_y, fit_radius = (float(value) for value in match.groups())
    assert (fit_x, fit_y) == pytest.approx(CENTRE, abs=0.3)
    assert fit_radius == pytest.approx(72.0, abs=0.3)
    assert "try --use-fitted-centre" not in printed  # the centre is right, so no hint


def test_a_wrong_centre_gets_a_hint_and_the_fitted_centre_can_be_used(run_cli, green_video, capsys):
    wrong_centre = (166, 120)  # 6 pixels too far to the right
    code, _ = run_cli(green_video, "--centre", *wrong_centre, "--no-video", "--no-plots")
    printed = capsys.readouterr().out
    assert code == 0
    assert "try --use-fitted-centre" in printed
    wrong_spread = float(printed.split("spread ")[1].split(" percent")[0])
    assert wrong_spread > 5.0

    code, out_dir = run_cli(green_video, "--centre", *wrong_centre, "--use-fitted-centre", "--no-video", "--no-plots")
    printed = capsys.readouterr().out
    assert code == 0
    assert "from the circle fit, not the one you gave" in printed
    assert "try --use-fitted-centre" not in printed  # it is already in use
    fitted_spread = float(printed.split("spread ")[1].split(" percent")[0])
    assert fitted_spread < 1.0  # the frames were rated again with the better centre

    table = pd.read_csv(out_dir / "green_analysis.csv")
    assert table["radius_px"].median() == pytest.approx(72.0, abs=0.5)


def test_fitted_centre_option_needs_a_centre(run_cli, green_video, capsys):
    code, _ = run_cli(green_video, "--use-fitted-centre")
    assert code == 1
    assert "--use-fitted-centre needs --centre" in capsys.readouterr().out


def test_a_fit_that_cannot_be_trusted_is_not_used(run_cli, green_video, capsys):
    # 12 frames are only 216 degrees of the circle, and fewer than 20 trusted points are left.
    code, out_dir = run_cli(
        green_video, "--centre", 166, 120, "--use-fitted-centre", "--max-frames", 12, "--no-video", "--no-plots"
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert "Circle fit           : not done: only" in printed
    assert "The fitted centre was not used" in printed
    assert "(166, 120) pixels" in printed  # the centre that was given is still used


def test_too_few_valid_frames_save_the_files_and_explain(run_cli, green_video, capsys):
    code, out_dir = run_cli(green_video, "--centre", *CENTRE, "--max-frames", 6, "--no-video", "--no-plots")
    printed = capsys.readouterr().out
    assert code == 1
    assert "Angle                : not calculated" in printed
    assert "Speed frames" not in printed
    assert "Only 6 frames have a trusted marker position" in printed
    table = pd.read_csv(out_dir / "green_analysis.csv")
    assert len(table) == 6 and "angle_rad" not in table.columns  # the tracking part is saved for debugging


def test_region_radius_and_area_options_are_used(run_cli, green_video, capsys):
    code, _ = run_cli(
        green_video, "--centre", *CENTRE, "--region-radius", 100, "--min-area", 50, "--max-area", 400,
        "--sat-min", 120, "--val-min", 90, "--no-video", "--no-plots",
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert "saturation from 120, brightness from 90, area 50 to 400" in printed
    assert "search region 100 pixels" in printed


def test_fps_option_changes_the_time_axis(run_cli, green_video):
    code, out_dir = run_cli(green_video, "--fps", 40, "--no-video", "--no-plots")
    assert code == 0
    table = pd.read_csv(out_dir / "green_analysis.csv")
    assert table["time_s"].iloc[1] == pytest.approx(0.025)


def test_start_frame_and_max_frames(run_cli, green_video):
    code, out_dir = run_cli(green_video, "--start-frame", 5, "--max-frames", 30, "--centre", *CENTRE, "--no-video", "--no-plots")
    assert code == 0
    table = pd.read_csv(out_dir / "green_analysis.csv")
    assert table["frame"].tolist() == list(range(5, 35))


def test_preview_saves_a_picture_and_nothing_else(run_cli, green_video, capsys):
    code, out_dir = run_cli(green_video, "--preview", "--centre", *CENTRE)
    printed = capsys.readouterr().out
    assert code == 0
    assert (out_dir / "green_mask_preview.png").exists()
    assert not (out_dir / "green_analysis.csv").exists()
    assert "Marker found         : yes" in printed


def test_preview_with_the_wrong_colour_says_so(run_cli, green_video, capsys):
    code, out_dir = run_cli(green_video, "--preview", "--colour", "blue")
    printed = capsys.readouterr().out
    assert code == 1
    assert "Marker found         : no" in printed
    assert (out_dir / "green_mask_preview.png").exists()  # the picture is still saved, it helps to see why


def test_wrong_colour_gives_advice_but_still_saves_the_files(run_cli, green_video, capsys):
    code, out_dir = run_cli(green_video, "--colour", "blue", "--centre", *CENTRE, "--no-video")
    printed = capsys.readouterr().out
    assert code == 1
    assert "Marker could not be detected in enough frames" in printed
    assert "Tracking quality     : Poor" in printed
    assert (out_dir / "green_analysis.csv").exists()
    assert (out_dir / "green_tracking_quality.png").exists()  # this plot helps to see what went wrong


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--centre", 9000, 100], "outside the frame"),
        (["--region-radius", 100], "needs --centre"),
        (["--hue", 200, 10], "hue limits"),
        (["--sat-min", 400], "between 0 and 255"),
        (["--min-area", 500, "--max-area", 100], "area limits"),
        (["--start-frame", 9999], "beyond the end"),
        (["--fps", 0], "positive number"),
    ],
)
def test_bad_options_give_a_plain_error_message(run_cli, green_video, capsys, arguments, message):
    code, _ = run_cli(green_video, *arguments)
    assert code == 1
    assert message in capsys.readouterr().out


def test_missing_video_gives_an_error(run_cli, capsys):
    code, _ = run_cli("no_such_video.mp4")
    assert code == 1
    assert "not found" in capsys.readouterr().out
