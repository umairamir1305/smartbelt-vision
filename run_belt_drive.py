"""Analyse a belt drive with two pulleys: track a marker on each pulley and compare their speeds.

Phase 7: two pulleys and the belt. Each pulley is analysed like the single pulley in run_analysis.py
(tracking, angle, speed, signal analysis). Then the speed ratio of the two pulleys is compared with the ratio
that the effective radii give. If a small mark of a third colour is stuck on the belt (--belt-colour), the
belt speed is measured where the belt runs straight, and the slip on each pulley is calculated.

Put a marker of a different colour on each pulley and keep both pulleys fully in the picture.
Use pick_points.py once for each pulley to get its colour numbers and centre. Add --pulley driver (or driven)
and it prints the options with the right names.

Run from the project folder, for example:
    python run_belt_drive.py sample_data/belt_60fps_01.mp4 --driver-colour green --driven-colour orange ^
        --driver-centre 140 270 --driven-centre 464 270 --driver-radius-mm 25 --driven-radius-mm 50

    python run_belt_drive.py sample_data/belt_60fps_01.mp4 --preview --driver-colour green --driven-colour orange

    python run_belt_drive.py sample_data/belt_60fps_01.mp4 --driver-centre 140 270 --driven-centre 464 270 ^
        --driver-region-radius 102 --driven-region-radius 204 --driver-radius-mm 23 --driven-radius-mm 46 ^
        --belt-colour blue --driver-marker-radius-mm 18 --driven-marker-radius-mm 36

For the slip, the scale (mm per pixel) is needed. Give it with --scale-mm-per-px, or give the distance in mm from the
centre of a pulley to its marker (--driver-marker-radius-mm, --driven-marker-radius-mm). Slip is a small
difference of speeds with errors of its own: look at the change between a recording with no load and one with
load, and use the calibration factor that is printed (--belt-speed-factor).

The effective radius is from the centre of the pulley to the middle of the belt, not to the outside of the pulley.
A 1 mm error in a radius of 25 mm is 4 percent, so for the speed loss a reference ratio is more reliable:
record the drive with no load, and give the ratio you measured with --reference-ratio.

Files made in the outputs folder (the video name is written as <video>):
    <video>_belt_analysis.csv           one row per frame, columns of both pulleys with driver_ and driven_ in front
    <video>_belt_tracked.mp4            the video with both markers, their angle and speed and the ratio drawn on it
    <video>_belt_speeds.png             speed of both pulleys and their ratio against time
    <video>_belt_surface_speed.png      surface speed of both pulleys (only with both radii)
    <video>_belt_mark_speed.png         belt speed from the mark and the surface speeds (only with a belt mark and slip)
    <video>_belt_slip.png               slip on the driver and on the driven pulley against time
    <video>_driver_*.png/.csv, <video>_driven_*.png/.csv   the plots and spectrum files of each pulley
    <video>_driver_mask_preview.png ... only with --preview
"""

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

from src.cli import add_signal_arguments, signal_settings_from_args
from src.console import (
    line,
    make_progress_printer,
    print_angle_summary,
    print_belt_summary,
    print_fit_summary,
    print_signal_summary,
    print_slip_summary,
    print_speed_summary,
    print_tracking_summary,
)
from src.kinematics import AnalysisError
from src.pipeline import (
    MIN_CENTRE_DISTANCE_PX,
    PULLEYS,
    analyse_belt_drive,
    make_belt_plots,
    save_plots,
    save_spectrum_files,
    video_labels,
)
from src.reporting import save_figure
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
    write_annotated_video_layers,
)
from src.utils import OUTPUT_DIR, looks_simulated, save_image
from src.video_io import VideoError, read_frame

DEFAULT_COLOURS = {"driver": "green", "driven": "orange"}

def add_pulley_arguments(parser: argparse.ArgumentParser, name: str) -> None:
    """The options of one pulley. Their names start with --driver- or --driven-."""
    group = parser.add_argument_group(f"{name} pulley")
    group.add_argument(
        f"--{name}-colour", default=DEFAULT_COLOURS[name], choices=sorted(COLOUR_PRESETS),
        help=f"named colour of the marker on the {name} pulley (default {DEFAULT_COLOURS[name]})",
    )
    group.add_argument(
        f"--{name}-hue", type=int, nargs=2, metavar=("LOW", "HIGH"),
        help="hue limits from 0 to 179, used instead of the colour name. LOW larger than HIGH wraps around 0 (red)",
    )
    group.add_argument(f"--{name}-sat-min", type=int, default=None, help="minimum saturation, 0 to 255 (default 100)")
    group.add_argument(f"--{name}-val-min", type=int, default=None, help="minimum brightness, 0 to 255 (default 70)")
    group.add_argument(f"--{name}-min-area", type=float, default=None, help="smallest marker area in square pixels (default 30)")
    group.add_argument(f"--{name}-max-area", type=float, default=None, help="largest marker area in square pixels (default 20000)")
    group.add_argument(f"--{name}-centre", type=float, nargs=2, metavar=("X", "Y"), help="rotation centre in pixels")
    group.add_argument(
        f"--{name}-region-radius", type=float, default=None,
        help="search for this marker only within this many pixels of its centre",
    )
    group.add_argument(
        f"--{name}-radius-mm", type=float, default=None,
        help="effective radius of the pulley in mm, from the centre to the middle of the belt",
    )
    group.add_argument(
        f"--{name}-marker-radius-mm", type=float, default=None,
        help="distance in mm from the centre of the pulley to the middle of its marker (gives the scale for the belt speed)",
    )


def add_belt_mark_arguments(parser: argparse.ArgumentParser) -> None:
    """The options of the mark on the belt and of the scale. Giving --belt-colour or --belt-hue turns the belt tracking on."""
    group = parser.add_argument_group("belt mark and scale")
    group.add_argument(
        "--belt-colour", default=None, choices=sorted(COLOUR_PRESETS),
        help="named colour of a small mark on the belt. Giving it (or --belt-hue) turns on the belt speed and the slip",
    )
    group.add_argument(
        "--belt-hue", type=int, nargs=2, metavar=("LOW", "HIGH"),
        help="hue limits of the belt mark from 0 to 179, used instead of the colour name",
    )
    group.add_argument("--belt-sat-min", type=int, default=None, help="minimum saturation of the belt mark (default 100)")
    group.add_argument("--belt-val-min", type=int, default=None, help="minimum brightness of the belt mark (default 70)")
    group.add_argument("--belt-min-area", type=float, default=None, help="smallest belt mark area in square pixels (default 30)")
    group.add_argument("--belt-max-area", type=float, default=None, help="largest belt mark area in square pixels (default 20000)")
    group.add_argument(
        "--scale-mm-per-px", type=float, default=None,
        help="millimetres per pixel in the plane of the pulleys, from a known distance. Used instead of the marker distances",
    )
    group.add_argument(
        "--belt-speed-factor", type=float, default=1.0,
        help="multiply the measured belt speed by this number (the calibration factor from a recording with no load)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse a belt drive with two pulleys.")
    parser.add_argument("video", help="path to the video file")
    for name in PULLEYS:
        add_pulley_arguments(parser, name)

    drive = parser.add_argument_group("belt drive")
    drive.add_argument(
        "--reference-ratio", type=float, default=None,
        help="speed ratio (driver / driven) measured with no load. Used instead of the ratio of the radii",
    )
    drive.add_argument(
        "--use-fitted-centres", action="store_true",
        help="calculate the angles about the centres found by fitting a circle to each marker path",
    )
    add_belt_mark_arguments(parser)
    add_signal_arguments(parser)

    other = parser.add_argument_group("other")
    other.add_argument("--fps", type=float, default=None, help="frame rate to use instead of the one stored in the file")
    other.add_argument("--start-frame", type=int, default=0, help="first frame to use (default 0)")
    other.add_argument("--max-frames", type=int, default=None, help="stop after this many frames (default: whole video)")
    other.add_argument("--trail", type=int, default=40, help="past marker positions drawn as dots (default 40)")
    other.add_argument("--no-video", action="store_true", help="do not write the tracked video (faster)")
    other.add_argument("--no-plots", action="store_true", help="do not save the plots")
    other.add_argument(
        "--preview", action="store_true",
        help="only check the colour settings of both markers on one frame (--start-frame) and save pictures",
    )
    return parser


def pulley_settings(args, name: str, default_colour=None):
    """The tracking settings of one marker from the colour name and any values typed on the command line."""
    changes = {}
    for option, field in (("sat_min", "sat_min"), ("val_min", "val_min"), ("min_area", "min_area_px"), ("max_area", "max_area_px")):
        value = getattr(args, f"{name}_{option}")
        if value is not None:
            changes[field] = value
    settings = settings_for_colour(getattr(args, f"{name}_colour") or default_colour, **changes)
    hue = getattr(args, f"{name}_hue")
    if hue is not None:
        settings = replace(settings, hue_low=hue[0], hue_high=hue[1])
    return settings


def belt_mark_settings(args):
    """The tracking settings of the belt mark, or None if no belt colour was given."""
    if args.belt_colour is None and args.belt_hue is None:
        return None
    return pulley_settings(args, "belt", default_colour="blue")


def pulley_centre(args, name: str):
    centre = getattr(args, f"{name}_centre")
    return tuple(centre) if centre is not None else None


def run_preview(args, settings, belt_settings) -> int:
    frame = read_frame(args.video, args.start_frame)
    items = [(name, settings[name], pulley_centre(args, name), getattr(args, f"{name}_region_radius")) for name in PULLEYS]
    if belt_settings is not None:
        items.append(("belt", belt_settings, None, None))

    found_all = True
    for name, marker_settings, centre, region in items:
        if centre is not None:
            check_centre(centre, frame.shape[1], frame.shape[0])
        picture, detection = make_mask_preview(frame, marker_settings, centre, region)
        path = save_image(OUTPUT_DIR / f"{Path(args.video).stem}_{name}_mask_preview.png", picture)

        line(f"{name.capitalize()} colour", describe_settings(marker_settings))
        if detection is None:
            found_all = False
            line(f"{name.capitalize()} marker", "not found. The right panel of the picture should show the marker as a white blob")
        else:
            line(f"{name.capitalize()} marker", f"found at ({detection.x:.1f}, {detection.y:.1f}) pixels, area {detection.area_px:.0f} square pixels")
        line("Saved", path)
    return 0 if found_all else 1


def ratio_text(merged) -> list:
    """The speed ratio of each frame as text for the top line of the tracked video."""
    texts = []
    for value in merged["speed_ratio"].to_numpy(dtype=float):
        texts.append("" if math.isnan(value) else f"speed ratio {abs(value):.3f}")
    return texts


def run(args) -> int:
    settings = {name: pulley_settings(args, name) for name in PULLEYS}
    for name in PULLEYS:
        validate_settings(settings[name])
    belt_settings = belt_mark_settings(args)
    if belt_settings is not None:
        validate_settings(belt_settings)
    if args.scale_mm_per_px is not None and args.scale_mm_per_px <= 0:
        raise TrackingError("--scale-mm-per-px must be larger than 0.")
    if args.belt_speed_factor <= 0:
        raise TrackingError("--belt-speed-factor must be larger than 0.")
    for name in PULLEYS:
        marker_radius = getattr(args, f"{name}_marker_radius_mm")
        if marker_radius is not None and marker_radius <= 0:
            raise TrackingError(f"--{name}-marker-radius-mm must be larger than 0.")
    signal_settings = signal_settings_from_args(args)
    validate_signal_settings(signal_settings)

    if args.preview:
        return run_preview(args, settings, belt_settings)

    centres = {name: pulley_centre(args, name) for name in PULLEYS}
    for name in PULLEYS:
        if centres[name] is None:
            raise TrackingError(f"--{name}-centre is needed. Use pick_points.py with --pulley {name} to find it.")
        region = getattr(args, f"{name}_region_radius")
        if region is not None and region <= 0:
            raise TrackingError(f"--{name}-region-radius must be larger than 0.")
    if math.dist(centres["driver"], centres["driven"]) < MIN_CENTRE_DISTANCE_PX:
        raise TrackingError("The two rotation centres are almost the same point. Each pulley needs its own centre.")

    radii = {name: getattr(args, f"{name}_radius_mm") for name in PULLEYS}
    for name, radius in radii.items():
        if radius is not None and radius <= 0:
            raise TrackingError(f"--{name}-radius-mm must be larger than 0.")
    stem = Path(args.video).stem
    simulated = looks_simulated(args.video)
    regions = {name: getattr(args, f"{name}_region_radius") for name in PULLEYS}
    marker_radii = {name: getattr(args, f"{name}_marker_radius_mm") for name in PULLEYS}

    def progress_factory(label):
        show = make_progress_printer(label)

        def callback(done, total):
            show(done, total)
            if done == total:
                print()  # end the progress line

        return callback

    outcome = analyse_belt_drive(
        args.video, settings, centres, regions, radii, marker_radii,
        belt_settings=belt_settings, scale_mm_per_px=args.scale_mm_per_px, belt_speed_factor=args.belt_speed_factor,
        reference_ratio=args.reference_ratio, signal_settings=signal_settings, fps_override=args.fps,
        start_frame=args.start_frame, max_frames=args.max_frames, use_fitted_centres=args.use_fitted_centres,
        progress_factory=progress_factory,
    )
    info, results, merged = outcome.info, outcome.pulleys, outcome.merged
    expected_ratio = outcome.expected_ratio
    belt, belt_problem = outcome.belt, outcome.belt_problem
    belt_table, slip, slip_message = outcome.belt_table, outcome.slip, outcome.slip_message
    if slip is not None and slip.get("scale_source") == "typed in":
        slip["scale_source"] = "typed in with --scale-mm-per-px"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{stem}_belt_analysis.csv"
    merged.to_csv(csv_path, index=False, float_format="%.6f")

    saved = []
    for name in PULLEYS:
        result = results[name]
        if result.signal is not None:
            saved += save_spectrum_files(result.signal, f"{stem}_{name}", OUTPUT_DIR)
        if not args.no_plots:
            mean_rpm = result.speed["mean_rpm"] if result.speed else None
            saved += save_plots(result.table, result.used_centre, f"{stem}_{name}", simulated, mean_rpm, result.signal, OUTPUT_DIR)
    if not args.no_plots:
        for suffix, _, figure in make_belt_plots(merged, belt, slip, expected_ratio, simulated):
            saved.append(save_figure(figure, OUTPUT_DIR / f"{stem}_{suffix}.png"))

    video_path = None
    video_seconds = 0.0
    if not args.no_video:
        started = time.time()
        layers = []
        for name in PULLEYS:
            result = results[name]
            layers.append(
                {
                    "name": name,
                    "table": result.table,
                    "centre": result.used_centre,
                    "region_radius_px": getattr(args, f"{name}_region_radius"),
                    "extra_text": video_labels(result.table) if result.rotation is not None else None,
                }
            )
        if belt_table is not None:
            layers.append({"name": "belt", "table": belt_table, "centre": None, "region_radius_px": None, "extra_text": None})
        video_path = write_annotated_video_layers(
            args.video,
            layers,
            OUTPUT_DIR / f"{stem}_belt_tracked.mp4",
            fps=info.fps,
            trail_length=args.trail,
            start_frame=args.start_frame,
            header_texts=ratio_text(merged) if "speed_ratio" in merged.columns else None,
        )
        video_seconds = time.time() - started

    # Print everything.
    print()
    print("SmartBelt Vision: two-pulley belt drive")
    print("=======================================")
    if simulated:
        line("Data", "SIMULATED video (not a real measurement)")
    line("Video", f"{info.path} ({info.width} x {info.height}, {info.fps:.2f} fps)")
    for name in PULLEYS:
        result = results[name]
        print_tracking_summary(
            result.info, settings[name], centres[name], result.used_centre, getattr(args, f"{name}_region_radius"),
            result.tracking, simulated, title=f"{name.upper()} PULLEY", show_video=False,
        )
        if result.fit is not None:
            print_fit_summary(result.fit, centres[name], fit_in_use=result.used_centre != centres[name])
        if result.rotation is not None:
            print_angle_summary(result.rotation)
        if result.speed is not None:
            print_speed_summary(result.speed)
        if result.signal is not None:
            print_signal_summary(result.signal)
        elif result.signal_problem is not None:
            line("Signal analysis", f"not done: {result.signal_problem}")
        for note in result.notes:
            line("Note", note)
        frames = result.tracking["frames"]
        line("Tracking time", f"{result.tracking_seconds:.1f} s ({frames / max(result.tracking_seconds, 1e-9):.0f} frames per second)")

    if belt is not None:
        print_belt_summary(belt)
    elif belt_problem is not None:
        print()
        line("Belt drive", f"not calculated: {belt_problem}")
    if belt_settings is not None:
        print_slip_summary(slip if slip is not None and slip["available"] else {"available": False, "reason": slip_message})

    print()
    if video_path is not None:
        line("Video time", f"{video_seconds:.1f} s")
    line("Saved", csv_path)
    for path in saved:
        line("Saved", path)
    if video_path is not None:
        line("Saved", video_path)

    # Checked last, so the files above exist even when something failed. They help to find out why.
    for name in PULLEYS:
        try:
            require_enough_detections(results[name].tracking)
        except TrackingError as error:
            raise TrackingError(f"{name} pulley: {error}") from error
    for name in PULLEYS:
        if results[name].analysis_problem is not None:
            raise AnalysisError(f"{name} pulley: {results[name].analysis_problem}")
    if belt_problem is not None:
        raise AnalysisError(belt_problem)
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
