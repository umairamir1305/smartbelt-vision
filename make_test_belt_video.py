"""Make a SIMULATED test video of a belt drive: two discs with a coloured marker on each, turning at known speeds.

The true speeds are known, so run_belt_drive.py can be checked against them before a real belt drive is
recorded. This is simulated data. Never mix it with real results. Without --belt no belt is drawn.

The driven disc has the radius of the driver disc times --ratio, and its ideal speed is the driver speed
divided by the ratio (both surfaces move at the same speed). --slip-percent makes it slower than that.

With --belt an open belt is drawn with a small coloured mark on it. The belt moves at the surface speed of the
driver minus --driver-slip-percent, and the driven disc turns slower than the belt by --slip-percent. This is
the video for trying the belt speed and the slip measurement.

Run from the project folder, for example:
    python make_test_belt_video.py
    python make_test_belt_video.py --driver-rpm 90 --ratio 1.5 --slip-percent 2
    python make_test_belt_video.py --crossed
    python make_test_belt_video.py --belt --driver-slip-percent 1 --slip-percent 2

The last lines show the command to analyse the video.
"""

import argparse
import sys
from pathlib import Path

from src.synthetic import MARKER_COLOURS_BGR, write_synthetic_belt_video
from src.utils import PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Make a simulated belt drive test video.")
    parser.add_argument("--driver-rpm", type=float, default=120.0, help="speed of the driver disc in rpm (default 120)")
    parser.add_argument("--ratio", type=float, default=2.0, help="radius of the driven disc / radius of the driver disc (default 2)")
    parser.add_argument("--slip-percent", type=float, default=0.0, help="how much slower than the ideal the driven disc turns, in percent (default 0)")
    parser.add_argument("--crossed", action="store_true", help="the driven disc turns the opposite way (a crossed belt)")
    parser.add_argument("--belt", action="store_true", help="draw an open belt with a small mark on it (not with --crossed)")
    parser.add_argument(
        "--driver-slip-percent", type=float, default=0.0,
        help="with --belt: how much slower than the driver surface the belt moves, in percent (default 0)",
    )
    parser.add_argument("--belt-colour", default="blue", choices=sorted(MARKER_COLOURS_BGR), help="colour of the mark on the belt (default blue)")
    parser.add_argument(
        "--scale-mm-per-px", type=float, default=0.25,
        help="what one pixel is worth in mm, for the radii and speeds printed (default 0.25)",
    )
    parser.add_argument("--fps", type=float, default=60.0, help="frame rate of the video (default 60)")
    parser.add_argument("--seconds", type=float, default=5.0, help="length of the video in seconds (default 5)")
    parser.add_argument("--width", type=int, default=960, help="frame width in pixels (default 960)")
    parser.add_argument("--height", type=int, default=540, help="frame height in pixels (default 540)")
    parser.add_argument("--driver-colour", default="green", choices=sorted(MARKER_COLOURS_BGR), help="marker colour of the driver (default green)")
    parser.add_argument("--driven-colour", default="orange", choices=sorted(MARKER_COLOURS_BGR), help="marker colour of the driven disc (default orange)")
    parser.add_argument("--noise", type=float, default=0.0, help="random brightness noise, standard deviation in grey levels (default 0)")
    parser.add_argument("--out", default=None, help="output file (default: a name made from the settings, in sample_data)")
    return parser


def default_output_path(args) -> Path:
    """A file name such as sample_data/synthetic_belt_120rpm_ratio2_slip3_60fps.mp4."""
    mark = "_mark" if getattr(args, "belt", False) else ""
    driver_slip = f"_dslip{args.driver_slip_percent:g}" if getattr(args, "driver_slip_percent", 0.0) > 0 else ""
    slip = f"_slip{args.slip_percent:g}" if args.slip_percent > 0 else ""
    crossed = "_crossed" if args.crossed else ""
    name = f"synthetic_belt_{args.driver_rpm:g}rpm_ratio{args.ratio:g}{mark}{driver_slip}{slip}{crossed}_{args.fps:g}fps.mp4"
    return PROJECT_ROOT / "sample_data" / name


def main() -> int:
    args = build_parser().parse_args()
    out_path = Path(args.out) if args.out else default_output_path(args)

    try:
        truth = write_synthetic_belt_video(
            out_path, driver_rpm=args.driver_rpm, ratio=args.ratio, slip_percent=args.slip_percent, crossed=args.crossed,
            fps=args.fps, seconds=args.seconds, width=args.width, height=args.height,
            driver_colour=args.driver_colour, driven_colour=args.driven_colour, noise_sigma=args.noise,
            driver_slip_percent=args.driver_slip_percent, draw_belt=args.belt, belt_colour=args.belt_colour,
            scale_mm_per_px=args.scale_mm_per_px,
        )
    except (ValueError, OSError) as error:
        print(f"Error: {error}")
        return 1

    driver_x, driver_y = truth["driver_centre"]
    driven_x, driven_y = truth["driven_centre"]
    degrees_per_frame = 6.0 * args.driver_rpm / args.fps
    if args.belt:
        driver_radius_mm = truth["driver_effective_radius_mm"]
        driven_radius_mm = truth["driven_effective_radius_mm"]
    else:
        driver_radius_mm = 25.0  # the discs have no real size, so example radii in mm are used for the command below
        driven_radius_mm = driver_radius_mm * args.ratio

    print("SIMULATED belt drive video created (not a real measurement)")
    print(f"File                  : {out_path}")
    print(f"Frames                : {len(truth['times_s'])} ({args.seconds:g} s at {args.fps:g} fps)")
    print(f"Frame size            : {args.width} x {args.height} pixels")
    print(f"Driver disc           : centre ({driver_x}, {driver_y}) pixels, radius {truth['driver_radius_px']:.0f} pixels, {args.driver_colour} marker")
    print(f"Driven disc           : centre ({driven_x}, {driven_y}) pixels, radius {truth['driven_radius_px']:.0f} pixels, {args.driven_colour} marker")
    print(f"True driver speed     : {truth['driver_rpm']:g} rpm")
    print(f"Ideal driven speed    : {abs(truth['ideal_driven_rpm']):g} rpm (driver speed / ratio {args.ratio:g})")
    total_loss = (1.0 - (1.0 - args.driver_slip_percent / 100.0) * (1.0 - args.slip_percent / 100.0)) * 100.0
    print(f"True driven speed     : {abs(truth['driven_rpm']):g} rpm ({'opposite direction' if args.crossed else 'same direction'}), speed loss {total_loss:g} percent")
    if args.belt:
        print(f"Belt                  : open belt on a circle of 0.95 of the disc radius, {args.belt_colour} mark")
        print(f"Scale                 : {args.scale_mm_per_px:g} mm per pixel (the effective radii below are the belt radii in mm)")
        print(f"True belt speed       : {truth['belt_speed_m_s']:.4f} m/s")
        print(f"Driver surface speed  : {truth['driver_surface_speed_m_s']:.4f} m/s, slip on the driver {args.driver_slip_percent:g} percent")
        print(f"Driven surface speed  : {truth['driven_surface_speed_m_s']:.4f} m/s, slip on the driven disc {args.slip_percent:g} percent")
    print(f"Driver rotation/frame : {degrees_per_frame:.1f} degrees")
    if degrees_per_frame > 45:
        print("Warning: the driver marker turns more than 45 degrees per frame. Tracking will probably be unreliable.")

    print()
    note = "the radii in mm are the effective belt radii of the simulated discs" if args.belt else "the radii in mm are examples, the simulated discs have no real size"
    print(f"Command to analyse it ({note}):")
    command = (
        f"python run_belt_drive.py {out_path} --driver-colour {args.driver_colour} --driven-colour {args.driven_colour} "
        f"--driver-centre {driver_x} {driver_y} --driven-centre {driven_x} {driven_y} "
        f"--driver-region-radius {truth['driver_radius_px'] * 1.05:.0f} --driven-region-radius {truth['driven_radius_px'] * 1.05:.0f} "
        f"--driver-radius-mm {driver_radius_mm:.3f} --driven-radius-mm {driven_radius_mm:.3f}"
    )
    if args.belt:
        command += (
            f" --belt-colour {args.belt_colour} --driver-marker-radius-mm {truth['driver_marker_radius_mm']:.3f}"
            f" --driven-marker-radius-mm {truth['driven_marker_radius_mm']:.3f}"
        )
    print(command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
