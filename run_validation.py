"""Validation: compare the measured numbers with a reference.

1. Simulated videos, where the true values are known:

    python run_validation.py

   Makes simulated videos (different speeds, frame rates, noise, a lost marker, a speed wobble, belt slip),
   runs the analysis on them and prints how far the measured numbers are from the true ones. It writes
   outputs/validation_results.csv, outputs/validation_series.csv and outputs/simulated_validation_summary.png.
   The exit code is 1 if a check is outside its tolerance.

2. A real recording, with a reference that you measured another way:

    python run_validation.py --table outputs/clip_analysis.csv --reference-rpm 148 --reference-turns 37
    python run_validation.py --video sample_data/clip.mp4

   --table is the CSV that run_analysis.py or the app saved (one pulley). --reference-rpm is for example a
   tachometer reading. --reference-turns is the number of turns that you counted by hand in the video, over the
   same frames. --video checks that the frames are evenly spaced in time (a phone video can have a variable
   frame rate, which makes the speeds wrong).
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import validation
from src.console import line
from src.kinematics import AnalysisError, summarise_rotation, summarise_speed
from src.pipeline import OUTPUT_DIR, PULLEYS, analyse_belt_drive, analyse_pulley
from src.reporting import plot_validation, save_figure
from src.synthetic import write_synthetic_belt_video, write_synthetic_video
from src.tracking import TrackingError, settings_for_colour
from src.video_io import VideoError, frame_timestamps, get_video_info

CENTRE = (320.0, 240.0)  # the rotation centre of the simulated disc videos (640 x 480)
SPEED_TOLERANCE_PERCENT = 0.5
TURNS_TOLERANCE = 0.05
POINTS_TOLERANCE = 0.3  # percentage points, for the speed loss and the slip
SECONDS = 4.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare measured numbers with a reference: simulated videos with known values, or your own recording."
    )
    parser.add_argument("--table", help="CSV table saved by run_analysis.py or the app (one pulley), to compare with a reference")
    parser.add_argument("--reference-rpm", type=float, help="reference speed in rpm, for example from a tachometer")
    parser.add_argument("--reference-turns", type=float, help="number of turns counted by hand over the same frames")
    parser.add_argument("--video", help="check the spacing of the frame times of this video")
    parser.add_argument("--fps", type=float, default=None, help="frame rate to compare the frame times with (default: the one in the file)")
    parser.add_argument("--tolerance-percent", type=float, default=2.0, help="tolerance of the speed in percent for --table (default 2)")
    parser.add_argument("--tolerance-turns", type=float, default=0.25, help="tolerance of the turns for --table (default 0.25)")
    parser.add_argument("--quick", action="store_true", help="fewer simulated cases (about 10 seconds)")
    parser.add_argument("--no-plots", action="store_true", help="do not save the plot")
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR), help="folder for the result files (default outputs)")
    return parser


# ---------------------------------------------------------------------------
# Simulated videos
# ---------------------------------------------------------------------------


def disc_cases(quick: bool) -> list:
    """(group, test, settings of the simulated disc video) for the speed checks."""
    speeds = (120,) if quick else (30, 60, 120, 240, 480)
    cases = [("Speed", f"{rpm:g} rpm, 60 fps", {"rpm": rpm, "fps": 60.0}) for rpm in speeds]
    if not quick:
        cases += [("Frame rate", f"120 rpm, {fps:g} fps", {"rpm": 120, "fps": fps}) for fps in (30.0, 120.0)]
    noise_levels = (6,) if quick else (6, 12)
    cases += [("Noise", f"120 rpm, noise {sigma:g}", {"rpm": 120, "fps": 60.0, "noise_sigma": float(sigma)}) for sigma in noise_levels]
    if not quick:
        cases.append(("Lost marker", "120 rpm, hidden 15 frames", {"rpm": 120, "fps": 60.0, "hidden_frames": range(100, 115)}))
    return cases


def true_rpm_series(true_angles, fps: float) -> np.ndarray:
    """The true speed in every frame, from the true angles, the same way the analysis finds it from the measured angles."""
    times = np.arange(len(true_angles)) / fps
    return np.abs(np.gradient(np.unwrap(np.asarray(true_angles, dtype=float)), times)) * 60.0 / (2.0 * np.pi)


def run_disc_checks(folder: Path, quick: bool):
    rows, series = [], []
    for number, (group, test, settings) in enumerate(disc_cases(quick)):
        path = folder / f"disc_{number}.mp4"
        angles = write_synthetic_video(path, seconds=SECONDS, **settings)
        result = analyse_pulley(path, settings_for_colour("green"), centre=CENTRE)
        rpm = settings["rpm"]
        rows.append(validation.compare_value(group, test, "Mean speed", "rpm", abs(result.speed["mean_rpm"]), rpm,
                                             tolerance_percent=SPEED_TOLERANCE_PERCENT))
        expected_turns = rpm / 60.0 * result.rotation["duration_s"]
        if group == "Lost marker":
            # A gap where the marker could have turned more than 90 degrees splits the angle in two parts. The turns across such a gap can
            # be off by whole turns (the gap here is 16 frames of 12 degrees = 192 degrees, and the angle across it is taken as -168 degrees).
            # So the turns are checked inside the stretches that have a marker, and the turns with the gap are shown without a limit.
            tracked = result.table.dropna(subset=["angle_rad"])
            seconds_with_marker = float(tracked.groupby("segment")["time_s"].agg(lambda s: s.max() - s.min()).sum())
            rows.append(validation.compare_value(group, test, "Turns inside the stretches with a marker", "turns",
                                                 abs(result.rotation["rotation_in_segments_rad"]) / (2.0 * np.pi), rpm / 60.0 * seconds_with_marker,
                                                 tolerance_abs=TURNS_TOLERANCE))
            rows.append(validation.compare_value(group, test, "Turns including the gap (known limit)", "turns", abs(result.rotation["turns"]),
                                                 expected_turns))
        else:
            rows.append(validation.compare_value(group, test, "Turns", "turns", abs(result.rotation["turns"]), expected_turns,
                                                 tolerance_abs=TURNS_TOLERANCE))
        if group == "Noise" or (group == "Speed" and rpm == 120):
            true_rpm = true_rpm_series(angles, settings["fps"])
            table = result.table
            series.append(validation.compare_series(test, "Speed in every frame", "rpm", table["rpm"].abs().to_numpy(), true_rpm))
            if group == "Noise":
                series.append(validation.compare_series(test, "Smoothed speed", "rpm", table["rpm_smooth"].abs().to_numpy(), true_rpm))
        if group == "Speed" and rpm == 120:
            stamps = frame_timestamps(path)
            check = validation.check_time_axis(stamps, settings["fps"])
            if check["available"]:
                rows.append(validation.compare_value("Time axis", test, "Largest irregularity of the frame spacing", "percent",
                                                     check["max_irregularity_percent"], 0.0, tolerance_abs=1.0))
                rows.append(validation.compare_value("Time axis", test, "Frame rate from the frame times", "fps",
                                                     check["fps_from_times"], settings["fps"], tolerance_percent=1.0))
            wrong = analyse_pulley(path, settings_for_colour("green"), centre=CENTRE, fps_override=settings["fps"] / 2.0)
            rows.append(validation.compare_value("Wrong frame rate", "120 rpm video read as 30 fps", "Mean speed", "rpm",
                                                 abs(wrong.speed["mean_rpm"]), rpm))  # no tolerance: it shows the size of the error
    return rows, series


def run_frequency_check(folder: Path):
    path = folder / "ripple.mp4"
    ripple_deg, ripple_hz, rpm = 4.0, 3.0, 150.0
    write_synthetic_video(path, rpm=rpm, fps=60.0, seconds=5.0, ripple_deg=ripple_deg, ripple_hz=ripple_hz, noise_sigma=6.0)
    result = analyse_pulley(path, settings_for_colour("green"), centre=CENTRE)
    peaks = result.signal["peaks"] if result.signal else []
    if not peaks:
        return []
    biggest = max(peaks, key=lambda peak: peak["amplitude_rpm"])
    expected_rpm = 60.0 * ripple_hz * np.radians(ripple_deg)  # amplitude of the speed wobble: 2 pi f A in rad/s, in rpm
    test = f"{rpm:g} rpm, wobble {ripple_deg:g} deg at {ripple_hz:g} Hz"
    return [
        validation.compare_value("Frequency", test, "Dominant frequency", "Hz", biggest["frequency_hz"], ripple_hz, tolerance_abs=0.1),
        validation.compare_value("Frequency", test, "Fluctuation amplitude", "rpm", biggest["amplitude_rpm"], expected_rpm, tolerance_percent=5.0),
    ]


def belt_cases(quick: bool) -> list:
    """(slip on the driver, slip on the driven pulley) in percent."""
    return [(1.0, 2.0)] if quick else [(0.0, 0.0), (1.0, 2.0), (3.0, 0.0), (0.0, 3.0)]


def run_belt_checks(folder: Path, quick: bool) -> list:
    rows = []
    ratio = 2.0
    for number, (driver_slip, driven_slip) in enumerate(belt_cases(quick)):
        path = folder / f"belt_{number}.mp4"
        info = write_synthetic_belt_video(path, driver_rpm=120.0, ratio=ratio, driver_slip_percent=driver_slip, slip_percent=driven_slip,
                                          draw_belt=True, seconds=5.0, scale_mm_per_px=0.25)
        colours = {"driver": "green", "driven": "orange"}
        outcome = analyse_belt_drive(
            path,
            {name: settings_for_colour(colours[name]) for name in PULLEYS},
            {name: tuple(info[f"{name}_centre"]) for name in PULLEYS},
            {name: round(info[f"{name}_radius_px"] * 1.05) for name in PULLEYS},
            {name: info[f"{name}_effective_radius_mm"] for name in PULLEYS},
            {name: info[f"{name}_marker_radius_mm"] for name in PULLEYS},
            belt_settings=settings_for_colour("blue"),
        )
        test = f"ratio {ratio:g}, driver slip {driver_slip:g} %, driven slip {driven_slip:g} %"
        loss = 1.0 - (1.0 - driver_slip / 100.0) * (1.0 - driven_slip / 100.0)
        belt, slip = outcome.belt, outcome.slip
        rows.append(validation.compare_value("Belt drive", test, "Speed ratio", "driver turns per driven turn", belt["measured_ratio"],
                                             ratio / (1.0 - loss), tolerance_percent=SPEED_TOLERANCE_PERCENT))
        rows.append(validation.compare_value("Belt drive", test, "Speed loss", "percentage points", belt["speed_loss_percent"], loss * 100.0,
                                             tolerance_abs=POINTS_TOLERANCE))
        if slip is not None and slip["available"]:
            rows.append(validation.compare_value("Belt drive", test, "Slip on the driver", "percentage points", slip["driver_slip_percent"],
                                                 driver_slip, tolerance_abs=POINTS_TOLERANCE))
            rows.append(validation.compare_value("Belt drive", test, "Slip on the driven pulley", "percentage points",
                                                 slip["driven_slip_percent"], driven_slip, tolerance_abs=POINTS_TOLERANCE))
    return rows


def run_simulated_suite(quick: bool = False, folder=None):
    """Run all simulated checks. Returns (rows of single values, rows of series)."""
    own_folder = None
    if folder is None:
        own_folder = tempfile.TemporaryDirectory(prefix="smartbelt_validation_")
        folder = own_folder.name
    folder = Path(folder)
    try:
        rows, series = run_disc_checks(folder, quick)
        rows += run_frequency_check(folder)
        rows += run_belt_checks(folder, quick)
    finally:
        if own_folder is not None:
            own_folder.cleanup()
    return rows, series


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def format_number(value, digits: int = 3):
    return "" if value is None else f"{value:.{digits}f}"


def rows_to_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Group": row["group"], "Test": row["test"], "Quantity": row["quantity"], "Unit": row["unit"],
                "Measured": row["measured"], "Reference": row["reference"], "Error": row["error"],
                "Error (%)": row["percent_error"], "Tolerance": row["tolerance"],
                "Result": {True: "within", False: "OUTSIDE", None: "no limit"}[row["passed"]],
            }
            for row in rows
        ]
    )


def print_rows(rows) -> None:
    """Print the checks as a table. The percent error of a quantity that is already in percent is left out."""
    header = f"{'Group':<16} {'Test':<52} {'Quantity':<40} {'Unit':<28} {'Measured':>9} {'Reference':>9} {'Error':>8} {'Error %':>8}  {'Tolerance':<22} Result"
    print(header)
    print("-" * len(header))
    for row in rows:
        show_percent = row["percent_error"] is not None and row["unit"] not in ("percentage points", "percent")
        percent = f"{row['percent_error']:+.3f}" if show_percent else ""
        result = {True: "within", False: "OUTSIDE", None: "no limit"}[row["passed"]]
        print(
            f"{row['group']:<16} {row['test']:<52} {row['quantity']:<40} {row['unit']:<28} {row['measured']:>9.3f} {row['reference']:>9.3f} "
            f"{row['error']:>+8.3f} {percent:>8}  {row['tolerance']:<22} {result}"
        )


def print_series(series) -> None:
    if not series:
        return
    frame = pd.DataFrame(series).rename(columns={"test": "Test", "quantity": "Quantity", "unit": "Unit", "n": "Frames", "mae": "MAE",
                                                 "rmse": "RMSE", "bias": "Bias", "max_abs_error": "Largest error"})
    print()
    print("Speed in every frame against the true speed (MAE: mean absolute error, RMSE: root mean square error)")
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def summary_line(rows) -> str:
    judged = [row for row in rows if row["passed"] is not None]
    inside = sum(1 for row in judged if row["passed"])
    return f"{inside} of {len(judged)} checks are within their tolerance."


# ---------------------------------------------------------------------------
# A real recording
# ---------------------------------------------------------------------------


def run_real(args) -> int:
    rows = []
    if args.table:
        table = pd.read_csv(args.table)
        speed, rotation = summarise_speed(table), summarise_rotation(table)
        name = Path(args.table).name
        line("Table", f"{name} ({len(table)} frames)")
        line("Measured speed", f"{abs(speed['mean_rpm']):.2f} rpm ({speed['direction']})")
        line("Measured rotation", f"{abs(rotation['turns']):.3f} turns in {rotation['duration_s']:.3f} s")
        if args.reference_rpm is not None:
            rows.append(validation.compare_value("Your recording", name, "Mean speed", "rpm", abs(speed["mean_rpm"]), args.reference_rpm,
                                                 tolerance_percent=args.tolerance_percent))
        if args.reference_turns is not None:
            rows.append(validation.compare_value("Your recording", name, "Turns", "turns", abs(rotation["turns"]), args.reference_turns,
                                                 tolerance_abs=args.tolerance_turns))
    if args.video:
        info = get_video_info(args.video, fps_override=args.fps)
        check = validation.check_time_axis(frame_timestamps(args.video), info.fps)
        print()
        if not check["available"]:
            line("Frame timing", check["reason"])
        else:
            line("Frame timing", f"{check['frames']} frames, {check['fps_from_times']:.2f} fps from the frame times, spacing varies by up to "
                                 f"{check['max_irregularity_percent']:.2f} percent")
            if check["ok"]:
                line("Result", f"the frames are evenly spaced and match {info.fps:g} fps")
            for problem in check["problems"]:
                line("Problem", problem)
    if rows:
        print()
        print_rows(rows)
        print()
        print(summary_line(rows))
    return 1 if any(row["passed"] is False for row in rows) else 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.table or args.video:
            if args.table and args.reference_rpm is None and args.reference_turns is None:
                print("Error: give --reference-rpm or --reference-turns to compare the table with.")
                return 1
            return run_real(args)

        print("Validation on simulated videos (the true values are known)")
        print("-" * 60)
        rows, series = run_simulated_suite(args.quick)
        print_rows(rows)
        print_series(series)
        print()
        print(summary_line(rows))

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows_to_frame(rows).to_csv(out_dir / "validation_results.csv", index=False)
        pd.DataFrame(series).to_csv(out_dir / "validation_series.csv", index=False)
        line("Saved", out_dir / "validation_results.csv")
        if not args.no_plots:
            line("Saved", save_figure(plot_validation(rows), out_dir / "simulated_validation_summary.png"))
        return 1 if any(row["passed"] is False for row in rows) else 0
    except (VideoError, TrackingError, AnalysisError, ValueError, OSError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
