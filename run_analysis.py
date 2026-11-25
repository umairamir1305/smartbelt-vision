"""Track the coloured marker in a video, calculate its angle and rotation speed, analyse the
speed signal and save the results.

Phase 5: tracking, angle, rotation speed (rpm), smoothing, speed fluctuation and frequency
spectrum. The belt (two pulleys and slip) comes in the next phases.

Run from the project folder, for example:
    python run_analysis.py sample_data/synthetic_disc_120rpm_60fps.mp4 --colour green --centre 320 240
    python run_analysis.py sample_data/disc_60fps_handspin_01.mp4 --colour orange --preview
    python run_analysis.py sample_data/disc_60fps_handspin_01.mp4 --hue 52 76 --sat-min 118 --val-min 95 --centre 962 541 --region-radius 380

Use pick_points.py to get the colour numbers and the centre by clicking on a frame.
Use --preview first. It saves one picture that shows what the colour settings pick up.
The angle, the speed and the signal analysis are only done when a rotation centre is given with --centre.
Change --smooth-window, --trend-order or --max-gap-frames to see how the analysis settings change the result.

Files made in the outputs folder (the video name is written as <video>):
    <video>_analysis.csv           one row per frame: marker position, tracking quality, angle, speed, smoothing
    <video>_tracked.mp4            the video with the tracking, the angle and the speed drawn on it
    <video>_trajectory.png         where the marker was in the frame
    <video>_angle.png              marker angle against time, wrapped and unwrapped
    <video>_rpm.png                rotation speed (single frames and smoothed) and angular acceleration
    <video>_fluctuation.png        speed fluctuation about the trend
    <video>_spectrum.png           frequency spectrum of the speed fluctuation
    <video>_spectrum.csv           the same spectrum as numbers
    <video>_peaks.csv              the dominant peaks of the spectrum
    <video>_tracking_quality.png   tracking quality of every frame
    <video>_mask_preview.png       only with --preview
"""

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

from src.console import (
    line,
    make_progress_printer,
    print_angle_summary,
    print_fit_summary,
    print_signal_summary,
    print_speed_summary,
    print_tracking_summary,
)
from src.kinematics import AnalysisError
from src.pipeline import analyse_pulley, save_plots, save_spectrum_files, video_labels
from src.cli import add_signal_arguments, signal_settings_from_args
from src.signal_analysis import validate_signal_settings
from src.tracking import (
    COLOUR_PRESETS,
    TrackingError,
    check_centre,
    describe_settings,
    make_mask_preview,
    require_enough_detections,
    settings_for_colour,
    validate_settings,
    write_annotated_video,
)
from src.utils import OUTPUT_DIR, looks_simulated, save_image
from src.video_io import VideoError, read_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track the coloured marker in a video and calculate its angle.")
    parser.add_argument("video", help="path to the video file")

    colour = parser.add_argument_group("marker colour and size")
    colour.add_argument(
        "--colour", default="green", choices=sorted(COLOUR_PRESETS), help="named marker colour (default green)"
    )
    colour.add_argument(
        "--hue",
        type=int,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="hue limits from 0 to 179, used instead of --colour. LOW larger than HIGH wraps around 0 (red)",
    )
    colour.add_argument("--sat-min", type=int, default=None, help="minimum saturation, 0 to 255 (default 100)")
    colour.add_argument("--val-min", type=int, default=None, help="minimum brightness, 0 to 255 (default 70)")
    colour.add_argument("--min-area", type=float, default=None, help="smallest marker area in square pixels (default 30)")
    colour.add_argument("--max-area", type=float, default=None, help="largest marker area in square pixels (default 20000)")

    pulley = parser.add_argument_group("pulley")
    pulley.add_argument(
        "--centre", type=float, nargs=2, metavar=("X", "Y"), help="rotation centre in pixels, for example 320 240"
    )
    pulley.add_argument(
        "--region-radius",
        type=float,
        default=None,
        help="search for the marker only within this many pixels of the centre (needs --centre)",
    )
    pulley.add_argument(
        "--use-fitted-centre",
        action="store_true",
        help="calculate the angle about the centre found by fitting a circle to the marker path, "
        "instead of the centre given with --centre (needs --centre)",
    )

    add_signal_arguments(parser)

    other = parser.add_argument_group("other")
    other.add_argument("--fps", type=float, default=None, help="frame rate to use instead of the one stored in the file")
    other.add_argument("--start-frame", type=int, default=0, help="first frame to use (default 0)")
    other.add_argument("--max-frames", type=int, default=None, help="stop after this many frames (default: whole video)")
    other.add_argument("--trail", type=int, default=40, help="past marker positions drawn as dots (default 40)")
    other.add_argument("--no-video", action="store_true", help="do not write the tracked video (faster)")
    other.add_argument("--no-plots", action="store_true", help="do not save the plots")
    other.add_argument(
        "--preview",
        action="store_true",
        help="only check the colour settings on one frame (the one given by --start-frame) and save a picture",
    )
    return parser


def settings_from_args(args):
    """Build the tracking settings from the colour name and any values typed on the command line."""
    changes = {}
    if args.sat_min is not None:
        changes["sat_min"] = args.sat_min
    if args.val_min is not None:
        changes["val_min"] = args.val_min
    if args.min_area is not None:
        changes["min_area_px"] = args.min_area
    if args.max_area is not None:
        changes["max_area_px"] = args.max_area

    settings = settings_for_colour(args.colour, **changes)
    if args.hue is not None:
        settings = replace(settings, hue_low=args.hue[0], hue_high=args.hue[1])
    return settings


def run_preview(args, settings, centre) -> int:
    frame = read_frame(args.video, args.start_frame)
    if centre is not None:
        check_centre(centre, frame.shape[1], frame.shape[0])

    picture, detection = make_mask_preview(frame, settings, centre, args.region_radius)
    path = save_image(OUTPUT_DIR / f"{Path(args.video).stem}_mask_preview.png", picture)

    line("Colour limits", describe_settings(settings))
    if detection is None:
        line("Marker found", "no")
        line("Saved", path)
        print(
            "The marker was not found with these settings. In the picture, the right panel should show the "
            "marker as a white blob. If it does not, change --hue, --sat-min and --val-min, or use pick_points.py."
        )
        return 1
    line("Marker found", f"yes, at ({detection.x:.1f}, {detection.y:.1f}) pixels, area {detection.area_px:.0f} square pixels")
    line("Saved", path)
    return 0


def run(args) -> int:
    settings = settings_from_args(args)
    validate_settings(settings)
    centre = tuple(args.centre) if args.centre is not None else None
    if args.region_radius is not None and centre is None:
        raise TrackingError("--region-radius needs --centre.")
    if args.use_fitted_centre and centre is None:
        raise TrackingError("--use-fitted-centre needs --centre.")
    signal_settings = signal_settings_from_args(args)
    validate_signal_settings(signal_settings)

    if args.preview:
        return run_preview(args, settings, centre)

    stem = Path(args.video).stem
    simulated = looks_simulated(args.video)

    result = analyse_pulley(
        args.video,
        settings,
        centre=centre,
        region_radius_px=args.region_radius,
        signal_settings=signal_settings,
        fps_override=args.fps,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        use_fitted_centre=args.use_fitted_centre,
        progress=make_progress_printer("Tracking"),
    )
    print()
    table, info, used_centre = result.table, result.info, result.used_centre
    speed, signal_result = result.speed, result.signal

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{stem}_analysis.csv"
    table.to_csv(csv_path, index=False, float_format="%.6f")

    spectrum_paths = save_spectrum_files(signal_result, stem, OUTPUT_DIR) if signal_result is not None else []

    plot_paths = []
    if not args.no_plots:
        plot_paths = save_plots(
            table, used_centre, stem, simulated, speed["mean_rpm"] if speed else None, signal_result, OUTPUT_DIR
        )

    video_path = None
    video_seconds = 0.0
    if not args.no_video:
        started = time.time()
        video_path = write_annotated_video(
            args.video,
            table,
            OUTPUT_DIR / f"{stem}_tracked.mp4",
            fps=info.fps,
            centre=used_centre,
            region_radius_px=args.region_radius,
            trail_length=args.trail,
            start_frame=args.start_frame,
            extra_text=video_labels(table) if result.rotation is not None else None,
        )
        video_seconds = time.time() - started

    print_tracking_summary(info, settings, centre, used_centre, args.region_radius, result.tracking, simulated)
    if result.fit is not None:
        print_fit_summary(result.fit, centre, fit_in_use=used_centre != centre)
    if result.rotation is not None:
        print_angle_summary(result.rotation)
    if speed is not None:
        print_speed_summary(speed)
    if signal_result is not None:
        print_signal_summary(signal_result)
    elif result.signal_problem is not None:
        line("Signal analysis", f"not done: {result.signal_problem}")
    if result.analysis_problem is not None:
        line("Angle" if result.rotation is None else "Speed", "not calculated, see the message at the end")
    for note in result.notes:
        line("Note", note)

    frames = result.tracking["frames"]
    line("Tracking time", f"{result.tracking_seconds:.1f} s ({frames / max(result.tracking_seconds, 1e-9):.0f} frames per second)")
    if video_path is not None:
        line("Video time", f"{video_seconds:.1f} s")
    line("Saved", csv_path)
    for path in spectrum_paths + plot_paths:
        line("Saved", path)
    if video_path is not None:
        line("Saved", video_path)

    # Checked last, so the files above exist even when something failed. They help to find out why.
    require_enough_detections(result.tracking)
    if result.analysis_problem is not None:
        raise AnalysisError(result.analysis_problem)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except (VideoError, TrackingError, AnalysisError, OSError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
