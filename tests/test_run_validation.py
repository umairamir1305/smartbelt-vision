"""Tests for run_validation.py: the simulated checks, and the comparison of a real recording with a reference."""

import numpy as np
import pandas as pd
import pytest

import run_validation
from src.pipeline import analyse_pulley
from src.synthetic import write_synthetic_video
from src.tracking import settings_for_colour


def run(monkeypatch, capsys, *arguments):
    monkeypatch.setattr("sys.argv", ["run_validation.py", *[str(argument) for argument in arguments]])
    code = run_validation.main()
    return code, capsys.readouterr().out


@pytest.fixture(scope="module")
def disc(tmp_path_factory):
    """A 120 rpm disc video and the table that the analysis makes of it, saved as CSV like run_analysis.py does."""
    folder = tmp_path_factory.mktemp("validation")
    video = folder / "synthetic_disc.mp4"
    write_synthetic_video(video, rpm=120.0, fps=60.0, seconds=4.0)
    result = analyse_pulley(video, settings_for_colour("green"), centre=(320.0, 240.0))
    table = folder / "synthetic_disc_analysis.csv"
    result.table.to_csv(table, index=False)
    return video, table, result


def test_the_quick_simulated_suite_is_inside_its_tolerances(monkeypatch, capsys, tmp_path):
    code, out = run(monkeypatch, capsys, "--quick", "--out-dir", tmp_path)
    assert code == 0
    assert "Validation on simulated videos" in out and "OUTSIDE" not in out
    assert "12 of 12 checks are within their tolerance." in out
    results = pd.read_csv(tmp_path / "validation_results.csv")
    assert set(results["Group"]) == {"Speed", "Time axis", "Wrong frame rate", "Noise", "Frequency", "Belt drive"}
    assert (tmp_path / "validation_series.csv").exists() and (tmp_path / "simulated_validation_summary.png").stat().st_size > 5000
    wrong = results[results["Group"] == "Wrong frame rate"].iloc[0]
    assert wrong["Error (%)"] == pytest.approx(-50.0, abs=0.1) and wrong["Result"] == "no limit"  # shows what a wrong frame rate does


def test_no_plots_option_skips_the_picture(monkeypatch, capsys, tmp_path):
    code, _ = run(monkeypatch, capsys, "--quick", "--no-plots", "--out-dir", tmp_path)
    assert code == 0 and not (tmp_path / "simulated_validation_summary.png").exists()


def test_a_lost_marker_is_checked_inside_the_stretches_and_the_gap_is_a_known_limit(tmp_path, monkeypatch):
    case = ("Lost marker", "120 rpm, hidden 15 frames", {"rpm": 120, "fps": 60.0, "hidden_frames": range(100, 115)})
    monkeypatch.setattr(run_validation, "disc_cases", lambda quick: [case])
    rows, _ = run_validation.run_disc_checks(tmp_path, quick=False)
    by_quantity = {row["quantity"]: row for row in rows if row["group"] == "Lost marker"}
    assert by_quantity["Mean speed"]["passed"] is True
    assert by_quantity["Turns inside the stretches with a marker"]["passed"] is True
    gap = by_quantity["Turns including the gap (known limit)"]
    assert gap["passed"] is None and gap["error"] == pytest.approx(-1.0, abs=0.01)  # one whole turn: the angle across a long gap is ambiguous


def test_the_true_speed_series_of_a_constant_speed_is_constant():
    angles = np.arange(100) * np.radians(12.0)  # 12 degrees per frame at 60 fps is 120 rpm
    assert np.allclose(run_validation.true_rpm_series(angles, 60.0), 120.0)
    assert np.allclose(run_validation.true_rpm_series(-angles, 60.0), 120.0)  # the direction does not matter


def test_a_real_recording_inside_the_tolerance(monkeypatch, capsys, disc):
    _, table, result = disc
    code, out = run(monkeypatch, capsys, "--table", table, "--reference-rpm", 121.0, "--reference-turns", f"{abs(result.rotation['turns']):.2f}")
    assert code == 0
    assert "Measured speed       : 120.00 rpm" in out
    assert "OUTSIDE" not in out and "2 of 2 checks are within their tolerance." in out


def test_a_real_recording_outside_the_tolerance_gives_exit_code_1(monkeypatch, capsys, disc):
    _, table, _ = disc
    code, out = run(monkeypatch, capsys, "--table", table, "--reference-rpm", 100.0)
    assert code == 1 and "OUTSIDE" in out and "0 of 1 checks are within their tolerance." in out
    code, out = run(monkeypatch, capsys, "--table", table, "--reference-rpm", 100.0, "--tolerance-percent", 25)
    assert code == 0 and "within" in out


def test_a_table_needs_a_reference(monkeypatch, capsys, disc):
    _, table, _ = disc
    code, out = run(monkeypatch, capsys, "--table", table)
    assert code == 1 and "--reference-rpm or --reference-turns" in out


def test_the_frame_timing_of_a_video_is_checked(monkeypatch, capsys, disc):
    video, _, _ = disc
    code, out = run(monkeypatch, capsys, "--video", video)
    assert code == 0 and "the frames are evenly spaced and match 60 fps" in out
    code, out = run(monkeypatch, capsys, "--video", video, "--fps", 30)
    assert code == 0 and "the frame times say 60.00 fps but the analysis uses 30 fps" in out


def test_a_missing_file_gives_a_plain_message(monkeypatch, capsys):
    code, out = run(monkeypatch, capsys, "--video", "no_such_video.mp4")
    assert code == 1 and out.startswith("Error:")
