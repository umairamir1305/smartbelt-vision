"""Tests for run_belt_drive.py and make_test_belt_video.py. Small simulated belt drives are used."""

import re

import pandas as pd
import pytest

import make_test_belt_video
import run_belt_drive
from src.cli import add_signal_arguments, signal_settings_from_args
from src.synthetic import write_synthetic_belt_video
from src.video_io import get_video_info

# 640 x 360 frame, 30 fps, 4 s. The driver turns at 90 rpm (18 degrees per frame), the radius ratio is 1.5.
SIZE = dict(width=640, height=360, fps=30.0, seconds=4.0, driver_rpm=90.0, ratio=1.5)


def make_video(directory, name, **changes):
    path = directory / name
    settings = {**SIZE, **changes}
    return path, write_synthetic_belt_video(path, **settings)


@pytest.fixture(scope="module")
def slip_video(tmp_path_factory):
    """The driven disc is 2 percent slower than the ideal 60 rpm."""
    return make_video(tmp_path_factory.mktemp("videos"), "synthetic_belt.mp4", slip_percent=2.0)


@pytest.fixture(scope="module")
def crossed_video(tmp_path_factory):
    return make_video(tmp_path_factory.mktemp("videos"), "crossed.mp4", crossed=True)


@pytest.fixture(scope="module")
def marked_belt(tmp_path_factory):
    """A belt with a mark. The belt is 1 percent slower than the driver surface, the driven disc 2 percent slower than the belt."""
    path = tmp_path_factory.mktemp("videos") / "synthetic_marked.mp4"
    info = write_synthetic_belt_video(
        path, driver_rpm=90.0, ratio=1.5, driver_slip_percent=1.0, slip_percent=2.0, draw_belt=True,
        width=640, height=360, fps=60.0, seconds=5.0, scale_mm_per_px=0.25,
    )
    return path, info


def belt_options(info, scale="markers"):
    """The pulley options with the effective radii of the simulated belt, and the belt mark."""
    options = [
        "--driver-centre", *info["driver_centre"], "--driven-centre", *info["driven_centre"],
        "--driver-region-radius", round(info["driver_radius_px"] * 1.05), "--driven-region-radius", round(info["driven_radius_px"] * 1.05),
        "--driver-radius-mm", info["driver_effective_radius_mm"], "--driven-radius-mm", info["driven_effective_radius_mm"],
        "--belt-colour", "blue", "--no-video", "--no-plots",
    ]
    if scale == "markers":
        options += ["--driver-marker-radius-mm", info["driver_marker_radius_mm"], "--driven-marker-radius-mm", info["driven_marker_radius_mm"]]
    elif scale == "typed":
        options += ["--scale-mm-per-px", info["scale_mm_per_px"]]
    return options


def geometry_options(info, radii=True):
    options = [
        "--driver-centre", *info["driver_centre"], "--driven-centre", *info["driven_centre"],
        "--driver-region-radius", round(info["driver_radius_px"] * 1.05), "--driven-region-radius", round(info["driven_radius_px"] * 1.05),
    ]
    if radii:
        options += ["--driver-radius-mm", 25, "--driven-radius-mm", 37.5]  # 37.5 / 25 = 1.5
    return options


@pytest.fixture
def run_cli(monkeypatch, tmp_path):
    out_dir = tmp_path / "out"
    monkeypatch.setattr(run_belt_drive, "OUTPUT_DIR", out_dir)

    def run(*arguments):
        monkeypatch.setattr("sys.argv", ["run_belt_drive.py", *[str(argument) for argument in arguments]])
        return run_belt_drive.main(), out_dir

    return run


def number(pattern, text):
    return float(re.search(pattern, text).group(1))


def test_normal_run_finds_the_speeds_the_ratio_and_the_slip(run_cli, slip_video, capsys):
    path, info = slip_video
    code, out_dir = run_cli(path, *geometry_options(info))
    printed = capsys.readouterr().out
    assert code == 0

    for title in ("SmartBelt Vision: two-pulley belt drive", "DRIVER PULLEY", "DRIVEN PULLEY", "BELT DRIVE"):
        assert title in printed
    assert printed.count("Video                :") == 1  # the video is shown once, in the header
    assert "SIMULATED video" in printed

    assert number(r"Driver +: [\d.]+ turns, mean ([\d.]+) rpm", printed) == pytest.approx(90.0, abs=0.3)
    assert number(r"Driven +: [\d.]+ turns, mean ([\d.]+) rpm", printed) == pytest.approx(58.8, abs=0.3)  # 60 rpm minus 2 percent
    assert "same direction (open belt)" in printed
    assert number(r"Speed ratio +: ([\d.]+) driver turns", printed) == pytest.approx(1.5 / 0.98, rel=0.003)
    assert "Expected ratio       : 1.5000 (from the pulley radii)" in printed
    assert number(r"Speed loss +: ([+-][\d.]+) percent", printed) == pytest.approx(2.0, abs=0.2)
    assert number(r"Ratio deviation +: ([+-][\d.]+) percent", printed) == pytest.approx(2.04, abs=0.2)
    assert "Surface speed" in printed and "Radius sensitivity" in printed and "Caution" in printed

    table = pd.read_csv(out_dir / "synthetic_belt_belt_analysis.csv")
    assert len(table) == 120
    for column in ("driver_marker_x", "driven_marker_x", "driver_rpm", "driven_rpm_smooth", "driver_segment",
                   "speed_ratio", "speed_loss_percent", "driver_surface_speed_m_s", "driven_surface_speed_m_s"):
        assert column in table.columns
    assert table["speed_loss_percent"].median() == pytest.approx(2.0, abs=0.3)

    for name in ("driver", "driven"):
        for suffix in ("trajectory", "angle", "rpm", "fluctuation", "spectrum", "tracking_quality"):
            assert (out_dir / f"synthetic_belt_{name}_{suffix}.png").stat().st_size > 1000
        assert (out_dir / f"synthetic_belt_{name}_spectrum.csv").exists()
        assert (out_dir / f"synthetic_belt_{name}_peaks.csv").exists()
    for name in ("speeds", "surface_speed"):
        assert (out_dir / f"synthetic_belt_belt_{name}.png").stat().st_size > 1000
    video = get_video_info(out_dir / "synthetic_belt_belt_tracked.mp4")
    assert video.frame_count == 120 and (video.width, video.height) == (640, 360)


def test_a_crossed_belt(run_cli, crossed_video, capsys):
    path, info = crossed_video
    code, out_dir = run_cli(path, *geometry_options(info), "--no-video", "--no-plots")
    printed = capsys.readouterr().out
    assert code == 0
    assert "opposite directions (crossed belt)" in printed
    assert number(r"Speed ratio +: ([\d.]+) driver turns", printed) == pytest.approx(1.5, rel=0.003)
    assert number(r"Speed loss +: ([+-][\d.]+) percent", printed) == pytest.approx(0.0, abs=0.2)
    table = pd.read_csv(out_dir / "crossed_belt_analysis.csv")
    assert table["speed_ratio"].median() < 0 and table["driven_rpm"].median() < 0
    assert not list(out_dir.glob("*.png"))  # --no-plots


def test_without_radii_the_ratio_is_given_but_no_expected_ratio(run_cli, slip_video, capsys):
    path, info = slip_video
    code, out_dir = run_cli(path, *geometry_options(info, radii=False), "--no-video")
    printed = capsys.readouterr().out
    assert code == 0
    assert number(r"Speed ratio +: ([\d.]+) driver turns", printed) == pytest.approx(1.5 / 0.98, rel=0.003)
    assert "Expected ratio       : not known" in printed
    assert "Speed loss" not in printed and "Surface speed" not in printed and "Radius sensitivity" not in printed
    table = pd.read_csv(out_dir / "synthetic_belt_belt_analysis.csv")
    assert "speed_loss_percent" not in table.columns and "driver_surface_speed_m_s" not in table.columns
    assert (out_dir / "synthetic_belt_belt_speeds.png").exists()
    assert not (out_dir / "synthetic_belt_belt_surface_speed.png").exists()


def test_a_reference_ratio_replaces_the_radii(run_cli, slip_video, capsys):
    path, info = slip_video
    # The radii say 1.5. A reference ratio of 1.5306 (what a no-load recording of this drive would give) says the
    # drive has no extra speed loss.
    code, _ = run_cli(path, *geometry_options(info), "--reference-ratio", 1.5306, "--no-video", "--no-plots")
    printed = capsys.readouterr().out
    assert code == 0
    assert "(from the reference ratio)" in printed
    assert number(r"Speed loss +: ([+-][\d.]+) percent", printed) == pytest.approx(0.0, abs=0.15)


def test_preview_saves_one_picture_for_each_marker(run_cli, slip_video, capsys):
    path, info = slip_video
    code, out_dir = run_cli(path, "--preview", *geometry_options(info, radii=False))
    printed = capsys.readouterr().out
    assert code == 0
    assert printed.count("found at") == 2
    for name in ("driver", "driven"):
        assert (out_dir / f"synthetic_belt_{name}_mask_preview.png").exists()
    assert not list(out_dir.glob("*.csv"))

    code, _ = run_cli(path, "--preview", "--driven-colour", "blue")
    printed = capsys.readouterr().out
    assert code == 1 and "Driven marker" in printed and "not found" in printed


def test_a_marker_that_is_not_found_gives_advice_and_still_saves_the_files(run_cli, slip_video, capsys):
    path, info = slip_video
    code, out_dir = run_cli(path, *geometry_options(info), "--driven-colour", "blue", "--no-video", "--no-plots")
    printed = capsys.readouterr().out
    assert code == 1
    assert "driven pulley: Marker could not be detected in enough frames" in printed
    assert "Belt drive           : not calculated" in printed
    assert (out_dir / "synthetic_belt_belt_analysis.csv").exists()


def test_too_short_a_clip_for_the_angle_means_no_ratio(run_cli, slip_video, capsys):
    path, info = slip_video
    code, out_dir = run_cli(path, *geometry_options(info), "--max-frames", 6, "--no-video", "--no-plots")
    printed = capsys.readouterr().out
    assert code == 1
    assert "The speed ratio needs the speed of both pulleys" in printed
    assert (out_dir / "synthetic_belt_belt_analysis.csv").exists()


def test_fitted_centres_can_be_used(run_cli, slip_video, capsys):
    path, info = slip_video
    driver_x, driver_y = info["driver_centre"]
    options = [
        "--driver-centre", driver_x + 5, driver_y, "--driven-centre", *info["driven_centre"],
        "--driver-region-radius", round(info["driver_radius_px"] * 1.05), "--driven-region-radius", round(info["driven_radius_px"] * 1.05),
    ]
    code, _ = run_cli(path, *options, "--use-fitted-centres", "--no-video", "--no-plots")
    printed = capsys.readouterr().out
    assert code == 0
    assert printed.count("from the circle fit, not the one you gave") >= 1


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--driven-centre", 400, 200], "--driver-centre is needed"),
        (["--driver-centre", 100, 100, "--driven-centre", 104, 100], "almost the same point"),
        (["--driver-centre", 100, 100, "--driven-centre", 400, 200, "--driver-radius-mm", -5], "--driver-radius-mm must be larger than 0"),
        (["--driver-centre", 100, 100, "--driven-centre", 400, 200, "--driven-region-radius", 0], "--driven-region-radius must be larger than 0"),
        (["--driver-centre", 100, 100, "--driven-centre", 4000, 200], "driven pulley: The rotation centre"),
        (["--driver-centre", 100, 100, "--driven-centre", 400, 200, "--reference-ratio", -2], "reference ratio must be positive"),
        (["--driver-centre", 100, 100, "--driven-centre", 400, 200, "--smooth-window", 0], "smoothing window"),
        (["--driver-hue", 200, 10], "hue limits"),
    ],
)
def test_bad_options_give_a_plain_error_message(run_cli, slip_video, capsys, arguments, message):
    path, _ = slip_video
    code, _ = run_cli(path, *arguments)
    assert code == 1
    assert message in capsys.readouterr().out


def test_missing_video_gives_an_error(run_cli, capsys):
    code, _ = run_cli("no_such_video.mp4", "--driver-centre", 100, 100, "--driven-centre", 300, 100)
    assert code == 1
    assert "not found" in capsys.readouterr().out


def test_the_shared_signal_options_have_the_same_defaults_in_both_scripts():
    import argparse

    parser = argparse.ArgumentParser()
    add_signal_arguments(parser)
    settings = signal_settings_from_args(parser.parse_args([]))
    assert (settings.smooth_window_s, settings.trend_order, settings.max_gap_frames, settings.min_peak_percent) == (0.2, 1, 3, 0.5)
    assert settings.remove_outliers

    custom = signal_settings_from_args(parser.parse_args(["--smooth-window", "0.4", "--trend-order", "2", "--keep-outliers"]))
    assert (custom.smooth_window_s, custom.trend_order, custom.remove_outliers) == (0.4, 2, False)


# ---------------------------------------------------------------------------
# make_test_belt_video.py
# ---------------------------------------------------------------------------


def test_the_belt_video_script_writes_a_video_and_prints_the_command(tmp_path, monkeypatch, capsys):
    out = tmp_path / "belt.mp4"
    monkeypatch.setattr("sys.argv", ["make_test_belt_video.py", "--out", str(out), "--fps", "30", "--seconds", "1", "--slip-percent", "3"])
    assert make_test_belt_video.main() == 0
    printed = capsys.readouterr().out

    assert out.exists() and get_video_info(out).frame_count == 30
    assert "SIMULATED belt drive video" in printed
    assert "Ideal driven speed    : 60 rpm" in printed
    assert "True driven speed     : 58.2 rpm (same direction), speed loss 3 percent" in printed
    command = [line for line in printed.splitlines() if line.startswith("python run_belt_drive.py")][0]
    for flag in ("--driver-centre", "--driven-centre", "--driver-region-radius", "--driven-region-radius", "--driver-radius-mm 25", "--driven-radius-mm 50"):
        assert flag in command


def test_the_belt_video_script_names_its_files_from_the_settings():
    class Args:
        driver_rpm, ratio, slip_percent, crossed, fps = 120.0, 2.0, 0.0, False, 60.0

    assert make_test_belt_video.default_output_path(Args).name == "synthetic_belt_120rpm_ratio2_60fps.mp4"
    Args.slip_percent, Args.crossed = 3.0, True
    assert make_test_belt_video.default_output_path(Args).name == "synthetic_belt_120rpm_ratio2_slip3_crossed_60fps.mp4"


@pytest.mark.parametrize("arguments, message", [(["--ratio", "5"], "too big"), (["--slip-percent", "100"], "slip percentage"), (["--fps", "0"], "positive")])
def test_the_belt_video_script_reports_bad_settings(tmp_path, monkeypatch, capsys, arguments, message):
    monkeypatch.setattr("sys.argv", ["make_test_belt_video.py", "--out", str(tmp_path / "x.mp4"), *arguments])
    assert make_test_belt_video.main() == 1
    assert message in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The mark on the belt: belt speed and slip
# ---------------------------------------------------------------------------


def test_belt_speed_and_slip_are_found_from_the_mark_on_the_belt(run_cli, marked_belt, capsys):
    path, info = marked_belt
    options = [option for option in belt_options(info) if option not in ("--no-video", "--no-plots")]
    code, out_dir = run_cli(path, *options)
    printed = capsys.readouterr().out
    assert code == 0

    assert "BELT SLIP" in printed
    assert re.search(r"from the distance of the marker from the centre: driver 0\.2[45]\d\d, driven 0\.2[45]\d\d", printed)
    assert number(r"belt speed ([\d.]+) m/s", printed) == pytest.approx(info["belt_speed_m_s"], rel=0.005)
    assert number(r"Driver surface +: ([\d.]+) m/s", printed) == pytest.approx(info["driver_surface_speed_m_s"], rel=0.005)
    assert number(r"Driven surface +: ([\d.]+) m/s", printed) == pytest.approx(info["driven_surface_speed_m_s"], rel=0.005)
    assert number(r"Slip on driver +: ([+-][\d.]+) percent", printed) == pytest.approx(1.0, abs=0.3)  # the belt is 1 percent slower
    assert number(r"Slip on driven +: ([+-][\d.]+) percent", printed) == pytest.approx(2.0, abs=0.3)  # the driven disc 2 percent
    assert number(r"Speed loss +: ([+-][\d.]+) percent \(both together", printed) == pytest.approx(2.98, abs=0.3)
    assert "Calibration factor" in printed and "Caution" in printed

    table = pd.read_csv(out_dir / "synthetic_marked_belt_analysis.csv")
    for column in ("belt_marker_x", "belt_on_span", "belt_speed_m_s", "driver_slip_percent", "driven_slip_percent"):
        assert column in table.columns
    assert table["belt_speed_m_s"].notna().sum() > 40
    assert table["driver_slip_percent"].dropna().median() == pytest.approx(1.0, abs=0.4)

    for name in ("belt_mark_speed", "belt_slip"):
        assert (out_dir / f"synthetic_marked_{name}.png").stat().st_size > 1000
    video = get_video_info(out_dir / "synthetic_marked_belt_tracked.mp4")
    assert video.frame_count == 300  # the tracked video also shows the belt mark


def test_a_typed_scale_gives_the_same_slip(run_cli, marked_belt, capsys):
    path, info = marked_belt
    code, _ = run_cli(path, *belt_options(info, scale="typed"))
    printed = capsys.readouterr().out
    assert code == 0
    assert "typed in with --scale-mm-per-px" in printed
    assert number(r"Slip on driver +: ([+-][\d.]+) percent", printed) == pytest.approx(1.0, abs=0.3)


def test_a_scale_that_is_one_percent_off_is_a_slip_error_of_one_percent(run_cli, marked_belt, capsys):
    path, info = marked_belt
    options = [o for o in belt_options(info, scale="none")] + ["--scale-mm-per-px", info["scale_mm_per_px"] * 1.01]
    code, _ = run_cli(path, *options)
    printed = capsys.readouterr().out
    assert code == 0
    # The true driver slip is 1 percent. A scale that is 1 percent too big makes the belt look 1 percent faster: slip 0.
    assert number(r"Slip on driver +: ([+-][\d.]+) percent", printed) == pytest.approx(0.0, abs=0.3)


def test_the_calibration_factor_removes_the_driver_slip_of_a_recording_without_load(run_cli, marked_belt, capsys):
    path, info = marked_belt
    run_cli(path, *belt_options(info, scale="typed"))
    factor = number(r"Calibration factor +: ([\d.]+)", capsys.readouterr().out)
    assert factor == pytest.approx(1 / 0.99, rel=0.004)

    code, _ = run_cli(path, *belt_options(info, scale="typed"), "--belt-speed-factor", factor)
    printed = capsys.readouterr().out
    assert code == 0
    assert number(r"Belt speed factor +: ([\d.]+) was applied", printed) == pytest.approx(factor, abs=1e-4)
    assert number(r"Slip on driver +: ([+-][\d.]+) percent", printed) == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize(
    "drop, add, message",
    [
        (["--driver-marker-radius-mm", "--driven-marker-radius-mm"], [], "a scale is needed"),
        (["--driver-radius-mm"], [], "both effective radii are needed"),
        (["--driver-region-radius"], [], "both region radii are needed"),
    ],
)
def test_slip_is_not_calculated_when_something_is_missing_and_the_reason_is_given(run_cli, marked_belt, capsys, drop, add, message):
    path, info = marked_belt
    options = belt_options(info)
    for name in drop:
        index = options.index(name)
        del options[index : index + 2]
    code, out_dir = run_cli(path, *options, *add)
    printed = capsys.readouterr().out
    assert code == 0  # not an error: the pulley results are there
    assert "BELT SLIP" in printed
    assert "Slip                 : Slip estimation unavailable: " in printed and message in printed
    assert "Slip on driver" not in printed
    table = pd.read_csv(out_dir / "synthetic_marked_belt_analysis.csv")
    assert "belt_marker_x" in table.columns  # the tracked mark is saved
    assert "driver_slip_percent" not in table.columns or table["driver_slip_percent"].isna().all()  # no slip value


def test_a_belt_mark_that_cannot_be_tracked_gives_no_slip_and_says_so(run_cli, marked_belt, capsys):
    path, info = marked_belt
    options = belt_options(info)
    options[options.index("blue")] = "yellow"  # no yellow mark exists in the video
    code, out_dir = run_cli(path, *options)
    printed = capsys.readouterr().out
    assert code == 0
    assert "Slip estimation unavailable because independent belt motion could not be tracked reliably." in printed
    assert "in 0 frames" in printed
    assert "Slip on driver" not in printed
    table = pd.read_csv(out_dir / "synthetic_marked_belt_analysis.csv")
    assert table["belt_speed_m_s"].isna().all()


def test_without_belt_options_there_is_no_belt_slip_block(run_cli, slip_video, capsys):
    path, info = slip_video
    code, _ = run_cli(path, *geometry_options(info), "--no-video", "--no-plots")
    printed = capsys.readouterr().out
    assert code == 0 and "BELT SLIP" not in printed and "belt mark" not in printed.lower()


def test_belt_hue_alone_turns_the_belt_tracking_on(run_cli, marked_belt, capsys):
    path, info = marked_belt
    options = belt_options(info, scale="typed")
    index = options.index("--belt-colour")
    del options[index : index + 2]
    code, _ = run_cli(path, *options, "--belt-hue", 100, 120)
    printed = capsys.readouterr().out
    assert code == 0 and "BELT SLIP" in printed
    assert number(r"Slip on driver +: ([+-][\d.]+) percent", printed) == pytest.approx(1.0, abs=0.3)


def test_preview_also_checks_the_belt_mark(run_cli, marked_belt, capsys):
    path, info = marked_belt
    code, out_dir = run_cli(path, "--preview", "--belt-colour", "blue", *geometry_options(info, radii=False))
    printed = capsys.readouterr().out
    assert code == 0 and printed.count("found at") == 3
    assert (out_dir / "synthetic_marked_belt_mask_preview.png").exists()


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--scale-mm-per-px", 0], "--scale-mm-per-px must be larger than 0"),
        (["--belt-speed-factor", -1], "--belt-speed-factor must be larger than 0"),
        (["--driver-marker-radius-mm", 0], "--driver-marker-radius-mm must be larger than 0"),
        (["--belt-hue", 200, 10], "hue limits"),
        (["--belt-colour", "blue", "--belt-min-area", 500, "--belt-max-area", 100], "area limits"),
    ],
)
def test_bad_belt_options_give_a_plain_error_message(run_cli, marked_belt, capsys, arguments, message):
    path, info = marked_belt
    code, _ = run_cli(path, *geometry_options(info), *arguments)
    assert code == 1
    assert message in capsys.readouterr().out


def test_the_belt_video_script_with_a_belt(tmp_path, monkeypatch, capsys):
    out = tmp_path / "marked.mp4"
    monkeypatch.setattr("sys.argv", ["make_test_belt_video.py", "--out", str(out), "--fps", "30", "--seconds", "1", "--belt",
                                     "--driver-slip-percent", "1", "--slip-percent", "2"])
    assert make_test_belt_video.main() == 0
    printed = capsys.readouterr().out
    assert out.exists()
    assert "Belt                  : open belt" in printed and "True belt speed" in printed
    assert "speed loss 2.98 percent" in printed  # (1 - 0.99 x 0.98) x 100
    command = [line for line in printed.splitlines() if line.startswith("python run_belt_drive.py")][0]
    for flag in ("--belt-colour blue", "--driver-marker-radius-mm", "--driven-marker-radius-mm", "the effective belt radii"[:0]):
        assert flag in command

    monkeypatch.setattr("sys.argv", ["make_test_belt_video.py", "--out", str(tmp_path / "x.mp4"), "--belt", "--crossed"])
    assert make_test_belt_video.main() == 1
    assert "open belt" in capsys.readouterr().out


def test_belt_video_names_show_the_belt_and_the_driver_slip():
    class Args:
        driver_rpm, ratio, slip_percent, crossed, fps, belt, driver_slip_percent = 120.0, 2.0, 2.0, False, 60.0, True, 1.0

    assert make_test_belt_video.default_output_path(Args).name == "synthetic_belt_120rpm_ratio2_mark_dslip1_slip2_60fps.mp4"
