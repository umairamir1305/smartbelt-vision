"""Text for the console: the summaries that the command line scripts print.

Every function here only prints. The numbers come from the dictionaries made by the analysis
functions (see pipeline.py, tracking.py, kinematics.py and signal_analysis.py).
"""

import contextlib
import io
import math

from src.slip_analysis import SCALE_DISAGREEMENT_WARNING_PERCENT
from src.tracking import describe_settings

# A circle fit centre that differs from the given centre by more than this (in pixels)
# is worth a closer look.
CENTRE_DIFFERENCE_HINT_PX = 2.0


def make_progress_printer(label: str):
    """Return a function that prints progress on one line, only when the percentage changes."""
    last = {"percent": -1}

    def show(done: int, total: int) -> None:
        percent = min(100, int(100 * done / total)) if total else 100
        if percent != last["percent"]:
            last["percent"] = percent
            print(f"\r{label}: {done} of {total} frames ({percent} percent)", end="", flush=True)

    return show


# While collect_rows runs, line() stores (label, value) rows here instead of printing them.
_recorded_rows = None


def line(label: str, value) -> None:
    if _recorded_rows is not None:
        _recorded_rows.append((label, str(value)))
        return
    print(f"{label:<20} : {value}")


def collect_rows(print_function, *args, **kwargs) -> list:
    """Run one of the print functions of this module and return what it prints as (label, value) rows.

    Nothing is printed. The app and the HTML report use this, so that their numbers and their wording are the
    same as in the command line tools. Lines that are not label and value pairs (titles, tables) are left out.
    """
    global _recorded_rows
    previous = _recorded_rows
    _recorded_rows = []
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            print_function(*args, **kwargs)
        return list(_recorded_rows)
    finally:
        _recorded_rows = previous


def print_tracking_summary(
    info, settings, centre, used_centre, region_radius, summary, simulated,
    title="SmartBelt Vision: marker tracking and angle", show_video=True,
) -> None:
    print()
    print(title)
    print("-" * len(title))
    if show_video:  # a script that prints several blocks shows the video once, in its own header
        if simulated:
            line("Data", "SIMULATED video (not a real measurement)")
        line("Video", f"{info.path} ({info.width} x {info.height}, {info.fps:.2f} fps)")
    line("Marker colour", describe_settings(settings))
    if used_centre is None:
        line("Rotation centre", "not given (no angle, and the distance and jump checks are skipped)")
    else:
        text = f"({used_centre[0]:g}, {used_centre[1]:g}) pixels"
        if used_centre != centre:
            text = f"({used_centre[0]:.2f}, {used_centre[1]:.2f}) pixels (from the circle fit, not the one you gave)"
        if region_radius is not None:
            text += f", search region {region_radius:g} pixels"
        line("Rotation centre", text)

    frames = summary["frames"]
    line("Frames tracked", frames)
    line("Marker found", f"{summary['found']} of {frames} ({100 * summary['detection_rate']:.1f} percent)")
    line(
        "Frame quality",
        f"Good {summary['good']}, Moderate {summary['moderate']}, Poor {summary['poor']}, Lost {summary['lost']}",
    )
    line("Longest gap", f"{summary['longest_gap_frames']} frames in a row without the marker")
    line(
        "Failed checks",
        f"area in {summary['area_problems']} frames, distance in {summary['distance_problems']} frames, "
        f"jump in {summary['jump_problems']} frames",
    )
    if summary["median_area_px"] is not None:
        line("Marker area", f"median {summary['median_area_px']:.0f} square pixels")
    if summary["median_radius_px"] is not None:
        line(
            "Distance from centre",
            f"median {summary['median_radius_px']:.1f} pixels, spread {summary['radius_spread_percent']:.2f} percent "
            "(5th to 95th percentile)",
        )
    line("Tracking quality", summary["overall_quality"])


def print_fit_summary(fit, centre, fit_in_use: bool) -> None:
    if fit["centre"] is None:
        line("Circle fit", f"not done: {fit['reason']}")
        return

    fit_x, fit_y = fit["centre"]
    text = (
        f"centre ({fit_x:.2f}, {fit_y:.2f}), radius {fit['radius_px']:.1f} px, "
        f"residual {fit['rms_px']:.2f} px, covers {fit['coverage_deg']:.0f} degrees"
    )
    if not fit["usable"]:
        text += f" (not trusted: {fit['reason']})"
    line("Circle fit", text)

    difference = math.hypot(fit_x - centre[0], fit_y - centre[1])
    text = f"{difference:.2f} px from the centre you gave"
    if fit["usable"] and difference > CENTRE_DIFFERENCE_HINT_PX and not fit_in_use:
        text += " (try --use-fitted-centre)"
    line("Centre difference", text)


def print_angle_summary(rotation) -> None:
    line("Angle frames", f"{rotation['angle_frames']} of {rotation['frames']} frames have an angle")
    degrees = math.degrees(rotation["total_rotation_rad"])
    text = (
        f"{rotation['direction']}, {abs(degrees):.1f} degrees ({abs(rotation['turns']):.3f} turns) "
        f"from frame {rotation['first_frame']} to frame {rotation['last_frame']} ({rotation['duration_s']:.3f} s)"
    )
    if rotation["segments"] > 1:
        text += ", includes the split gaps and may be off by whole turns"
    line("Rotation", text)
    if rotation["segments"] == 1:
        line("Angle segments", "1 (no long gap)")
    else:
        line(
            "Angle segments",
            f"{rotation['segments']} (split at long gaps where the marker could have turned more than 90 degrees)",
        )
        line(
            "Rotation in segments",
            f"{abs(math.degrees(rotation['rotation_in_segments_rad'])):.1f} degrees, not counting the split gaps "
            "(the angle across a split gap can be off by whole turns)",
        )


def print_speed_summary(speed) -> None:
    line("Speed frames", f"{speed['speed_samples']} frames have a speed")
    line(
        "Mean speed",
        f"{speed['mean_rpm']:.2f} rpm, {speed['direction']} (rotation divided by time, the same as counting turns)",
    )
    line(
        "Speed range",
        f"{speed['min_rpm']:.2f} to {speed['max_rpm']:.2f} rpm (single frame values, they include measurement noise)",
    )
    line(
        "Speed variation",
        f"{speed['std_rpm']:.2f} rpm standard deviation ({speed['variation_percent']:.2f} percent of the mean)",
    )
    if abs(speed["trend_rpm"]) > 0.2 * speed["mean_rpm"]:
        line(
            "Note",
            f"the speed changes by about {abs(speed['trend_rpm']):.0f} rpm during the clip (for example a disc "
            "slowing down), so the range and the variation above mostly show that change, not a fluctuation",
        )


def print_signal_summary(result) -> None:
    settings = result["settings"]
    stretch = f"frames {result['first_frame']} to {result['last_frame']} ({result['frames']} frames, {result['duration_s']:.3f} s)"
    if result["interpolated_frames"]:
        stretch += f", {result['interpolated_frames']} lost frames filled in"
    line("Analysis stretch", stretch)

    if settings.remove_outliers:
        line("Outliers", f"{result['outliers']} of {result['frames']} speed values replaced")
    else:
        line("Outliers", "not checked (--keep-outliers)")
    line(
        "Smoothing",
        f"Savitzky-Golay, {result['window_samples']} frames ({result['window_s']:.3f} s), "
        f"polynomial order {settings.smooth_polyorder}",
    )
    line("Speed noise", f"{result['noise_rms_rpm']:.2f} rpm rms (single frame speed minus a smooth curve of 0.2 s)")
    line(
        "Speed fluctuation",
        f"{result['fluctuation_peak_to_peak_rpm']:.2f} rpm peak to peak, {result['fluctuation_rms_rpm']:.2f} rpm rms, "
        f"coefficient (max - min) / mean {result['fluctuation_coefficient_percent']:.2f} percent "
        f"(smoothed, trend of order {settings.trend_order} removed)",
    )
    line(
        "Acceleration",
        f"largest {result['acceleration_max_abs_rad_s2']:.2f} rad/s^2, mean {result['acceleration_mean_rad_s2']:+.2f} rad/s^2 "
        "(positive: counter-clockwise acceleration)",
    )
    line(
        "Spectrum",
        f"resolution {result['frequency_resolution_hz']:.2f} Hz, up to {result['nyquist_hz']:.1f} Hz (half the frame rate), "
        f"rotation frequency {result['rotation_frequency_hz']:.2f} Hz",
    )

    if not result["steady"]:
        line(
            "Note",
            f"the speed is not steady (the trend changes it by about {abs(result['trend_change_rpm']):.0f} rpm), so the "
            "fluctuation spectrum is not reliable and no orders are given. Try a higher --trend-order, "
            "or use a steadier part of the clip",
        )

    peaks = result["peaks"]
    if not peaks:
        line(
            "Dominant frequencies",
            f"none above {result['peak_threshold_rpm']:.2f} rpm (the lowest amplitude listed as a peak. It depends on the "
            "noise, the biggest peak and --min-peak-percent)",
        )
        return

    line("Dominant frequencies", f"{len(peaks)} found. The speed varies periodically at these frequencies:")
    print("    #  Frequency  Period     Amplitude    % of mean  Order  Note")
    for rank, peak in enumerate(peaks, start=1):
        order = "-" if peak["order"] is None else f"{peak['order']:.2f}x"
        print(
            f"  {rank:>3}  {peak['frequency_hz']:>6.2f} Hz  {peak['period_s']:>6.3f} s  {peak['amplitude_rpm']:>6.2f} rpm"
            f"  {peak['percent_of_mean']:>9.2f}  {order:>5}  {peak['note']}"
        )
    line(
        "Caution",
        "a peak shows that the speed varies periodically at that frequency. It does not show the cause. Motor torque "
        "ripple, belt irregularity, misalignment, a rotation centre error, a tilted camera, rolling shutter, frame "
        "timing and tracking noise can all do this",
    )


def print_belt_summary(belt: dict) -> None:
    """Print the comparison of the two pulleys (the dictionary comes from belt_drive.summarise_belt_drive)."""
    title = "BELT DRIVE"
    print()
    print(title)
    print("-" * len(title))
    line("Common frames", f"{belt['common_frames']} frames in {belt['runs']} stretch(es), {belt['duration_s']:.3f} s")
    line("Driver", f"{abs(belt['driver_turns']):.3f} turns, mean {belt['driver_mean_rpm']:.2f} rpm")
    line("Driven", f"{abs(belt['driven_turns']):.3f} turns, mean {belt['driven_mean_rpm']:.2f} rpm")
    line("Direction", belt["direction"])

    spread = ""
    if not math.isnan(belt["ratio_std"]):
        low, high = belt["ratio_range"]
        spread = f" (single frame values: standard deviation {belt['ratio_std']:.3f}, 5th to 95th percentile {low:.3f} to {high:.3f})"
    line("Speed ratio", f"{belt['measured_ratio']:.4f} driver turns per driven turn{spread}")

    if belt["expected_ratio"] is None:
        line(
            "Expected ratio",
            "not known. Give both effective radii (--driver-radius-mm, --driven-radius-mm) or --reference-ratio",
        )
    else:
        line("Expected ratio", f"{belt['expected_ratio']:.4f} (from the {belt['expected_source']})")
        line("Ratio deviation", f"{belt['ratio_deviation_percent']:+.2f} percent (measured ratio against expected ratio)")
        line(
            "Speed loss",
            f"{belt['speed_loss_percent']:+.2f} percent of the ideal driven speed ((v ideal - v actual) / v ideal, "
            "positive: the driven pulley is slower than expected)",
        )

    if belt["driver_surface_speed_m_s"] is not None and belt["driven_surface_speed_m_s"] is not None:
        line(
            "Surface speed",
            f"driver {belt['driver_surface_speed_m_s']:.3f} m/s, driven {belt['driven_surface_speed_m_s']:.3f} m/s (omega x radius)",
        )
    if belt["radius_sensitivity_driver_percent_per_mm"] is not None and belt["radius_sensitivity_driven_percent_per_mm"] is not None:
        line(
            "Radius sensitivity",
            f"an error of 1 mm in the driver radius changes the expected ratio by {belt['radius_sensitivity_driver_percent_per_mm']:.1f} "
            f"percent, in the driven radius by {belt['radius_sensitivity_driven_percent_per_mm']:.1f} percent",
        )
    line(
        "Caution",
        "the speed loss contains belt creep, real slipping, errors in the effective radii and belt stretch, and cannot "
        "separate them. A reference ratio from a recording with no load (--reference-ratio) removes the radius error",
    )


def print_slip_summary(slip: dict) -> None:
    """Print the belt speed and the slip (the dictionary comes from slip_analysis.summarise_slip).

    If the slip could not be calculated, the reason is printed and no number is made up.
    """
    title = "BELT SLIP"
    print()
    print(title)
    print("-" * len(title))
    if not slip["available"]:
        line("Slip", slip["reason"])
        return

    spread = ""
    if slip["belt_speed_spread_percent"] is not None:
        spread = f", spread between the passes {slip['belt_speed_spread_percent']:.2f} percent"
    line(
        "Belt mark",
        f"{slip['passes']} passes along straight spans ({slip['span_frames']} frames), belt speed {slip['belt_speed_m_s']:.4f} m/s{spread}",
    )
    scale_text = f"{slip['scale_mm_per_px']:.4f} mm per pixel"
    if slip.get("scale_source"):
        scale_text += f" ({slip['scale_source']})"
    line("Scale", scale_text)
    if slip["scale_disagreement_percent"] > SCALE_DISAGREEMENT_WARNING_PERCENT:
        line(
            "Note",
            f"the scales from the two pulleys differ by {slip['scale_disagreement_percent']:.1f} percent (perspective or a "
            "measuring error), so the slip is uncertain by at least that much",
        )
    if slip["belt_speed_factor"] != 1.0:
        line("Belt speed factor", f"{slip['belt_speed_factor']:.4f} was applied to the belt speed")
    line("Driver surface", f"{slip['driver_surface_speed_m_s']:.4f} m/s (omega x radius)")
    line("Driven surface", f"{slip['driven_surface_speed_m_s']:.4f} m/s (omega x radius)")
    line(
        "Slip on driver",
        f"{slip['driver_slip_percent']:+.2f} percent ((v driver - v belt) / v driver, positive: the belt is slower than the driver surface)",
    )
    line(
        "Slip on driven",
        f"{slip['driven_slip_percent']:+.2f} percent ((v belt - v driven) / v belt, positive: the driven surface is slower than the belt)",
    )
    line("Speed loss", f"{slip['speed_loss_percent']:+.2f} percent (both together: (v driver - v driven) / v driver)")
    line(
        "Calibration factor",
        f"{slip['calibration_factor_if_no_load']:.4f} (use --belt-speed-factor {slip['calibration_factor_if_no_load']:.4f} "
        "if this recording has no load and no slip)",
    )
    line(
        "Caution",
        "slip is a small difference of speeds that each have errors from the scale and the effective radii: 1 percent "
        "of error in either is 1 percent of slip. The change between no load and load is more reliable than one value. "
        "The tight side of a real belt runs faster than the slack side (belt creep)",
    )
