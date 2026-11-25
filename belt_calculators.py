"""Two calculators for a belt drive. They do NOT use the video. You type in numbers that you measured yourself.

Belt Friction Calculator (Euler-Eytelwein relation): T1 / T2 = exp(mu x beta), so mu = ln(T1 / T2) / beta
    python belt_calculators.py friction --t1 120 --t2 60 --wrap-deg 180
    python belt_calculators.py friction --t1 120 --t2 60 --r-small-mm 25 --r-large-mm 50 --centre-distance-mm 200

Power Difference Calculator: P = T x omega
    python belt_calculators.py power --torque-in 2.0 --rpm-in 120 --torque-out 3.7 --rpm-out 60

The video cannot measure belt tensions, torques, a friction coefficient or a power loss. These calculators
only turn your own measurements into a number. The results are model-based and as good as your inputs.
The speeds in the power calculator can be the mean rpm that run_belt_drive.py prints for each pulley.
"""

import argparse
import math
import sys

from src.slip_analysis import (
    belt_friction_coefficient,
    power_difference,
    wrap_angle_open_belt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Belt Friction Calculator and Power Difference Calculator (not from the video).")
    commands = parser.add_subparsers(dest="command", required=True)

    friction = commands.add_parser("friction", help="Belt Friction Calculator: friction coefficient from the two belt tensions")
    friction.add_argument("--t1", type=float, required=True, help="tension of the tight side in N, measured by you")
    friction.add_argument("--t2", type=float, required=True, help="tension of the slack side in N, measured by you")
    friction.add_argument("--wrap-deg", type=float, default=None, help="wrap angle of the belt on the pulley in degrees")
    friction.add_argument("--r-small-mm", type=float, default=None, help="effective radius of the small pulley in mm (to work out the wrap angle)")
    friction.add_argument("--r-large-mm", type=float, default=None, help="effective radius of the large pulley in mm")
    friction.add_argument("--centre-distance-mm", type=float, default=None, help="distance between the pulley centres in mm")
    friction.add_argument(
        "--pulley", choices=("small", "large"), default="small",
        help="which pulley the friction belongs to when the wrap angle is worked out (default small: it slips first)",
    )

    power = commands.add_parser("power", help="Power Difference Calculator: input power, output power and their difference")
    power.add_argument("--torque-in", type=float, required=True, help="torque on the driver shaft in N m, measured by you")
    power.add_argument("--rpm-in", type=float, required=True, help="speed of the driver in rpm")
    power.add_argument("--torque-out", type=float, required=True, help="torque on the driven shaft in N m, measured by you")
    power.add_argument("--rpm-out", type=float, required=True, help="speed of the driven pulley in rpm")
    return parser


def row(label: str, value) -> None:
    print(f"{label:<28} : {value}")


def run_friction(args) -> int:
    geometry = (args.r_small_mm, args.r_large_mm, args.centre_distance_mm)
    if args.wrap_deg is not None:
        wrap_rad = math.radians(args.wrap_deg)
        wrap_source = "typed in"
    elif all(value is not None for value in geometry):
        small, large = wrap_angle_open_belt(*geometry)
        wrap_rad = small if args.pulley == "small" else large
        wrap_source = f"worked out for the {args.pulley} pulley of an open belt"
    else:
        raise ValueError("Give the wrap angle with --wrap-deg, or all of --r-small-mm, --r-large-mm and --centre-distance-mm.")

    mu = belt_friction_coefficient(args.t1, args.t2, wrap_rad)
    title = "Belt Friction Calculator (model-based, NOT measured from the video)"
    print(title)
    print("-" * len(title))
    row("Tight side tension T1", f"{args.t1:g} N (typed in)")
    row("Slack side tension T2", f"{args.t2:g} N (typed in)")
    row("Wrap angle beta", f"{math.degrees(wrap_rad):.1f} degrees = {wrap_rad:.4f} rad ({wrap_source})")
    row("Tension ratio T1 / T2", f"{args.t1 / args.t2:.3f}")
    row("Friction coefficient mu", f"{mu:.4f}  (mu = ln(T1 / T2) / beta)")
    print()
    print(
        "The coefficient is the one at the point where the belt is just about to slip, so the tensions must be "
        "measured at that point. It is only as good as the two tensions and the wrap angle. For a V-belt it is the "
        "effective coefficient, which includes the wedging of the groove. The video cannot measure it."
    )
    return 0


def run_power(args) -> int:
    result = power_difference(args.torque_in, args.rpm_in, args.torque_out, args.rpm_out)
    title = "Power Difference Calculator (model-based, NOT measured from the video)"
    print(title)
    print("-" * len(title))
    row("Input torque and speed", f"{args.torque_in:g} N m at {args.rpm_in:g} rpm (typed in)")
    row("Output torque and speed", f"{args.torque_out:g} N m at {args.rpm_out:g} rpm (typed in)")
    row("Input power", f"{result['power_in_w']:.2f} W  (P = T x omega)")
    row("Output power", f"{result['power_out_w']:.2f} W")
    row("Estimated power difference", f"{result['power_difference_w']:.2f} W  (input minus output)")
    row("Estimated efficiency", f"{result['efficiency_percent']:.1f} percent  (output / input)")
    print()
    print(
        "The power difference carries the errors of both torques, which are usually bigger than the loss itself, and "
        "it contains everything between the two measuring points, not only the belt. The video cannot measure "
        "torque or power."
    )
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run_friction(args) if args.command == "friction" else run_power(args)
    except ValueError as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
