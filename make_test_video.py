"""Make a SIMULATED test video: a light disc with one coloured marker turning at a known speed.

The true speed is known, so the tracking and rpm code can be checked against it before
a real video is recorded. This is simulated data. Never mix it with real results.

Positive rpm turns the disc counter-clockwise as seen in the video. Use a negative
rpm for clockwise.

Run from the project folder, for example:
    python make_test_video.py
    python make_test_video.py --rpm 90 --fps 30 --seconds 8 --colour red
    python make_test_video.py --rpm 150 --ripple-deg 4 --ripple-hz 3 --noise 6
    python make_test_video.py --rpm 120 --hide 100 10
    python make_test_video.py --rpm 300 --decay-seconds 4 --seconds 10

The extra options are for testing later phases. They can be ignored at first:
    --noise                      random brightness noise (tests tracking on a noisy picture)
    --hide FIRST COUNT           hides the marker for COUNT frames starting at frame FIRST
                                 (tests what happens when the marker is lost)
    --decay-seconds              the speed falls off exponentially, like a hand-spun disc slowing
                                 down (the rpm is then the starting speed)
    --ripple-deg and --ripple-hz a periodic wobble in the speed (tests the frequency analysis)
"""

import argparse
import sys
from pathlib import Path

from src.synthetic import MARKER_COLOURS_BGR, disc_geometry, write_synthetic_video
from src.utils import PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Make a simulated rotating-disc test video.")
    parser.add_argument("--rpm", type=float, default=120.0, help="true speed in rpm (default 120)")
    parser.add_argument("--fps", type=float, default=60.0, help="frame rate of the video (default 60)")
    parser.add_argument("--seconds", type=float, default=5.0, help="length of the video in seconds (default 5)")
    parser.add_argument("--width", type=int, default=640, help="frame width in pixels (default 640)")
    parser.add_argument("--height", type=int, default=480, help="frame height in pixels (default 480)")
    parser.add_argument(
        "--colour",
        default="green",
        choices=sorted(MARKER_COLOURS_BGR),
        help="marker colour (default green)",
    )
    parser.add_argument(
        "--ripple-deg",
        type=float,
        default=0.0,
        help="amplitude of a periodic wobble in the marker angle, in degrees (default 0)",
    )
    parser.add_argument(
        "--ripple-hz",
        type=float,
        default=0.0,
        help="frequency of that wobble in Hz (default 0)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="random brightness noise, standard deviation in grey levels (default 0)",
    )
    parser.add_argument(
        "--decay-seconds",
        type=float,
        default=0.0,
        help="time constant of an exponential slow-down in seconds, 0 means constant speed (default 0)",
    )
    parser.add_argument(
        "--hide",
        type=int,
        nargs=2,
        metavar=("FIRST", "COUNT"),
        default=None,
        help="hide the marker for COUNT frames, starting at frame number FIRST",
    )
    parser.add_argument("--out", default=None, help="output file (default: a name made from the settings, in sample_data)")
    return parser


def default_output_path(args) -> Path:
    """Build a file name such as sample_data/synthetic_disc_120rpm_60fps.mp4."""
    direction = "" if args.rpm >= 0 else "cw_"
    slowing = f"_decay{args.decay_seconds:g}s" if args.decay_seconds > 0 else ""
    name = f"synthetic_disc_{direction}{abs(args.rpm):g}rpm{slowing}_{args.fps:g}fps.mp4"
    return PROJECT_ROOT / "sample_data" / name


def main() -> int:
    args = build_parser().parse_args()
    out_path = Path(args.out) if args.out else default_output_path(args)

    hidden_frames = None
    if args.hide is not None:
        first_hidden, hidden_count = args.hide
        if first_hidden < 0 or hidden_count < 1:
            print("Error: --hide needs a first frame of 0 or more and a count of 1 or more.")
            return 1
        hidden_frames = range(first_hidden, first_hidden + hidden_count)

    try:
        true_angle_rad = write_synthetic_video(
            out_path,
            rpm=args.rpm,
            fps=args.fps,
            seconds=args.seconds,
            width=args.width,
            height=args.height,
            colour=args.colour,
            ripple_deg=args.ripple_deg,
            ripple_hz=args.ripple_hz,
            noise_sigma=args.noise,
            hidden_frames=hidden_frames,
            decay_s=args.decay_seconds,
        )
    except (ValueError, OSError) as error:
        print(f"Error: {error}")
        return 1

    centre, _, orbit_radius, _ = disc_geometry(args.width, args.height)
    degrees_per_frame = 6.0 * abs(args.rpm) / args.fps  # rpm * 360 degrees / 60 s / fps

    direction = "counter-clockwise" if args.rpm >= 0 else "clockwise"
    print("SIMULATED test video created (not a real measurement)")
    print(f"File                  : {out_path}")
    print(f"Frames                : {len(true_angle_rad)} ({args.seconds:g} s at {args.fps:g} fps)")
    print(f"Frame size            : {args.width} x {args.height} pixels")
    print(f"Rotation centre (x, y): ({centre[0]}, {centre[1]}) pixels")
    print(f"Marker distance       : {orbit_radius:.0f} pixels from the centre")
    print(f"Marker colour         : {args.colour}")
    if args.decay_seconds > 0:
        print(
            f"True speed            : {abs(args.rpm):g} rpm at the start, {direction}, falling off "
            f"exponentially (time constant {args.decay_seconds:g} s)"
        )
        print(f"Rotation per frame    : {degrees_per_frame:.1f} degrees at the start")
    else:
        print(f"True speed            : {abs(args.rpm):g} rpm, {direction}")
        print(f"Rotation per frame    : {degrees_per_frame:.1f} degrees")
    if hidden_frames is not None:
        shown_last = min(hidden_frames[-1], len(true_angle_rad) - 1)
        print(f"Marker hidden         : frames {hidden_frames[0]} to {shown_last}")

    if degrees_per_frame > 180:
        print("Warning: the marker turns more than half a turn between frames. The video shows aliasing, "
              "so the true speed cannot be recovered from it.")
    elif degrees_per_frame > 45:
        print("Warning: the marker turns more than 45 degrees per frame. Tracking will probably be unreliable.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
