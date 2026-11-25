"""Belt speed and slip, from a coloured mark on the belt, and two calculators that need your own numbers.

PART 1: belt speed and slip (from the video)

A small coloured mark on the belt is tracked like the pulley markers. Where the belt runs straight between
the pulleys, the mark moves along a straight line at the speed of the belt. A straight line is fitted to the
mark positions over each such pass (a least squares fit of x and y against time), and the length of the
velocity vector is the belt speed in pixels per second. This is much more exact than the displacement
between two frames, which has the full position noise in it. Only frames where the mark is outside both
pulley circles are used, because on the curved parts the mark does not move along a straight line.

The pixels are turned into mm with a scale (mm per pixel). The scale comes from a known distance in the
same plane: either typed in directly, or from the distance in mm between a pulley centre and its marker
(known when the marker was stuck on), divided by the distance in pixels that the tracking measured.

Three speeds are compared. All are in m/s:
    v_driver  = omega_driver x R_driver      surface speed of the driver pulley
    v_driven  = omega_driven x R_driven      surface speed of the driven pulley
    v_belt                                   speed of the belt mark
Definitions of slip used here (the usual form, (v_ideal - v_actual) / v_ideal x 100 percent):
    slip on the driver pulley   = (v_driver - v_belt) / v_driver     the belt is slower than the driver surface
    slip on the driven pulley   = (v_belt - v_driven) / v_belt       the driven surface is slower than the belt
    speed loss of the drive     = (v_driver - v_driven) / v_driver   both together (see belt_drive.py)
Positive values mean the following part is slower.

Limits:
  - Every one of these speeds has an error from the scale and from the effective radii. An error of 1 percent
    in the scale or in a radius is an error of 1 percent in a slip. Slip is a difference of numbers that are
    nearly equal, so it is the least accurate result of this project. The most reliable use is the CHANGE
    between a recording with no load and one with load. The calibration factor that is printed makes
    the driver slip zero for a recording with no load.
  - A real belt stretches elastically, so the tight side runs a little faster than the slack side (belt
    creep). The belt speed is that of the span where the mark was seen, and the straight spans are not equal.
  - The mark must stay on the belt in the plane of the pulleys, and the camera must look straight at it.
  - A pass needs at least 8 frames in a row on a straight span. With a span of L pixels, a belt speed of v
    pixels per second and a frame rate of f, a pass has about L x f / v frames. A fast belt needs a higher
    frame rate.
  - If the mark cannot be tracked on a straight span, the slip is not calculated.

PART 2: two calculators (NOT from the video)

belt_friction_coefficient uses the Euler-Eytelwein relation T1 / T2 = exp(mu x beta), and power_difference uses
P = T x omega. Both need values that you measured yourself (belt tensions and torques). Nothing in the video
gives them, so they are not measurements of the video. The results are model-based.
"""

import math

import numpy as np
import pandas as pd

# A pass needs at least this many frames in a row, and the mark must be on a straight span in at least
# this many frames in total, or no belt speed is calculated.
MIN_PASS_FRAMES = 8
MIN_SPAN_FRAMES = 20

# On a straight span the marks lie on a line. If the typical distance from the fitted line is bigger than
# this (pixels), the pass is not used.
MAX_PASS_RESIDUAL_PX = 2.0

# Two scales (from two pulleys) that differ by more than this are reported, because of perspective or a measuring error.
SCALE_DISAGREEMENT_WARNING_PERCENT = 3.0


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------


def scale_from_marker_radius(marker_radius_mm: float, marker_radius_px: float) -> float:
    """Millimetres per pixel from the distance between a pulley centre and its marker."""
    if marker_radius_mm <= 0 or marker_radius_px <= 0:
        raise ValueError("The marker radius must be positive, in mm and in pixels.")
    return marker_radius_mm / marker_radius_px


def combine_scales(scales: dict):
    """Combine the scales (mm per pixel) found from the pulleys. Returns (scale, disagreement in percent).

    scales is a dictionary such as {"driver": 0.2501, "driven": 0.2497}. The scale is the average. The
    disagreement is the difference between the biggest and the smallest, as a percentage of the average
    (0 if there is only one scale).
    """
    values = [value for value in scales.values() if value is not None]
    if not values:
        return None, 0.0
    average = float(np.mean(values))
    return average, 100.0 * (max(values) - min(values)) / average


# ---------------------------------------------------------------------------
# Belt speed
# ---------------------------------------------------------------------------


def mark_on_straight_span(x, y, centres, radii_px) -> np.ndarray:
    """True where the mark is outside every pulley circle (given by its centre and radius in pixels)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    outside = np.isfinite(x) & np.isfinite(y)
    for centre, radius in zip(centres, radii_px):
        outside &= np.hypot(x - centre[0], y - centre[1]) > radius
    return outside


def find_belt_passes(table: pd.DataFrame, on_span, min_frames: int = MIN_PASS_FRAMES, max_residual_px: float = MAX_PASS_RESIDUAL_PX):
    """Find the passes of the belt mark along a straight span, and the belt speed of each.

    A pass is a run of consecutive frames in which the mark was tracked (tracking_valid) and on a straight span.
    For each pass with at least min_frames frames, x and y are fitted with straight lines against time. The speed
    is the length of the velocity vector. Returns a list of dictionaries (start_index, end_index, start_frame,
    end_frame, frames, time_s, speed_px_s, direction_deg, residual_px, accepted). accepted is False if the typical
    distance of the points from the fitted line is more than max_residual_px.
    """
    x = table["marker_x"].to_numpy(dtype=float)
    y = table["marker_y"].to_numpy(dtype=float)
    time_s = table["time_s"].to_numpy(dtype=float)
    frame = table["frame"].to_numpy()
    usable = table["tracking_valid"].to_numpy(dtype=bool) & np.asarray(on_span, dtype=bool) & np.isfinite(x)

    runs, current = [], []
    for index in range(len(table)):
        if usable[index] and (not current or frame[index] - frame[current[-1]] == 1):
            current.append(index)
            continue
        if current:
            runs.append(current)
        current = [index] if usable[index] else []
    if current:
        runs.append(current)

    passes = []
    for run in runs:
        if len(run) < min_frames:
            continue
        t = time_s[run] - time_s[run].mean()
        vx, x0 = np.polyfit(t, x[run], 1)
        vy, y0 = np.polyfit(t, y[run], 1)
        residual = math.sqrt(float(np.mean((x[run] - (vx * t + x0)) ** 2 + (y[run] - (vy * t + y0)) ** 2)))
        passes.append(
            {
                "start_index": run[0],
                "end_index": run[-1],
                "start_frame": int(frame[run[0]]),
                "end_frame": int(frame[run[-1]]),
                "frames": len(run),
                "time_s": float(time_s[run].mean()),
                "speed_px_s": math.hypot(vx, vy),
                "direction_deg": math.degrees(math.atan2(-vy, vx)),  # image y points down
                "residual_px": residual,
                "accepted": residual <= max_residual_px,
            }
        )
    return passes


def add_belt_speed_columns(merged: pd.DataFrame, belt_table: pd.DataFrame, on_span, passes, scale_mm_per_px, belt_speed_factor: float = 1.0) -> pd.DataFrame:
    """Add the belt mark columns to a merged pulley table, and the slip of every frame that has a belt speed.

    belt_marker_x, belt_marker_y, belt_area_px, belt_tracking_valid, belt_tracking_quality   the tracked mark
    belt_on_span        True where the mark was on a straight span
    belt_speed_m_s      belt speed of the pass that the frame belongs to (empty outside accepted passes)
    driver_slip_percent, driven_slip_percent   slip of that frame (need the surface speeds, see belt_drive.add_belt_columns)

    belt_speed_factor multiplies the belt speed (the calibration factor, see the top of this file).
    Without a scale (scale_mm_per_px is None) only the tracking columns and belt_on_span are added.
    """
    if len(merged) != len(belt_table):
        raise ValueError("The belt table and the pulley table must be for the same frames.")
    merged = merged.copy()
    for column in ("marker_x", "marker_y", "area_px", "tracking_valid", "tracking_quality"):
        merged[f"belt_{column}"] = belt_table[column].to_numpy()
    merged["belt_on_span"] = np.asarray(on_span, dtype=bool)
    if scale_mm_per_px is None:
        return merged

    belt_speed = np.full(len(merged), np.nan)
    for belt_pass in passes:
        if belt_pass["accepted"]:
            belt_speed[belt_pass["start_index"] : belt_pass["end_index"] + 1] = (
                belt_pass["speed_px_s"] * scale_mm_per_px / 1000.0 * belt_speed_factor
            )
    merged["belt_speed_m_s"] = belt_speed

    if "driver_surface_speed_m_s" in merged.columns and "driven_surface_speed_m_s" in merged.columns:
        driver = merged["driver_surface_speed_m_s"].to_numpy(dtype=float)
        driven = merged["driven_surface_speed_m_s"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            merged["driver_slip_percent"] = np.where(driver > 0, (driver - belt_speed) / driver * 100.0, np.nan)
            merged["driven_slip_percent"] = np.where(belt_speed > 0, (belt_speed - driven) / belt_speed * 100.0, np.nan)
    return merged


def summarise_slip(
    passes,
    scale_mm_per_px,
    driver_surface_speed_m_s,
    driven_surface_speed_m_s,
    belt_speed_factor: float = 1.0,
    scale_disagreement_percent: float = 0.0,
) -> dict:
    """The belt speed and the slip, in a dictionary. See the definitions at the top of this file.

    passes comes from find_belt_passes. driver_surface_speed_m_s and driven_surface_speed_m_s are the mean surface
    speeds of the two pulleys. If the belt speed cannot be found, available is False and reason says why.
    """
    accepted = [belt_pass for belt_pass in passes if belt_pass["accepted"]]
    span_frames = sum(belt_pass["frames"] for belt_pass in accepted)
    summary = {
        "available": False,
        "reason": "",
        "passes": len(accepted),
        "rejected_passes": len(passes) - len(accepted),
        "span_frames": span_frames,
        "scale_mm_per_px": scale_mm_per_px,
        "scale_disagreement_percent": scale_disagreement_percent,
        "belt_speed_factor": belt_speed_factor,
        "belt_speed_m_s": None,
        "belt_speed_spread_percent": None,
        "driver_surface_speed_m_s": driver_surface_speed_m_s,
        "driven_surface_speed_m_s": driven_surface_speed_m_s,
        "driver_slip_percent": None,
        "driven_slip_percent": None,
        "speed_loss_percent": None,
        "calibration_factor_if_no_load": None,
    }

    if not accepted or span_frames < MIN_SPAN_FRAMES:
        summary["reason"] = (
            "Slip estimation unavailable because independent belt motion could not be tracked reliably. "
            f"The belt mark was found on a straight part of the belt in {span_frames} frames ({len(accepted)} usable "
            f"passes of at least {MIN_PASS_FRAMES} frames in a row, {len(passes) - len(accepted)} rejected because the "
            f"mark did not move in a straight line). At least {MIN_SPAN_FRAMES} frames are needed."
        )
        return summary

    speeds = np.array([belt_pass["speed_px_s"] for belt_pass in accepted])
    weights = np.array([belt_pass["frames"] for belt_pass in accepted], dtype=float)
    mean_px_s = float(np.average(speeds, weights=weights))
    raw_m_s = mean_px_s * scale_mm_per_px / 1000.0
    belt_speed = raw_m_s * belt_speed_factor

    summary["available"] = True
    summary["belt_speed_m_s"] = belt_speed
    if len(accepted) > 1:
        summary["belt_speed_spread_percent"] = 100.0 * float(speeds.std(ddof=1)) / mean_px_s
    if driver_surface_speed_m_s and driver_surface_speed_m_s > 0:
        summary["driver_slip_percent"] = (driver_surface_speed_m_s - belt_speed) / driver_surface_speed_m_s * 100.0
        summary["calibration_factor_if_no_load"] = driver_surface_speed_m_s / raw_m_s
        if driven_surface_speed_m_s is not None:
            summary["speed_loss_percent"] = (driver_surface_speed_m_s - driven_surface_speed_m_s) / driver_surface_speed_m_s * 100.0
    if driven_surface_speed_m_s is not None and belt_speed > 0:
        summary["driven_slip_percent"] = (belt_speed - driven_surface_speed_m_s) / belt_speed * 100.0
    return summary


# ---------------------------------------------------------------------------
# Geometry of an open belt
# ---------------------------------------------------------------------------


def wrap_angle_open_belt(radius_small_mm: float, radius_large_mm: float, centre_distance_mm: float):
    """Wrap angles of an open belt on the small and on the large pulley, in radians: (small, large).

    wrap on the small pulley = pi - 2 asin((R_large - R_small) / C), on the large one = pi + 2 asin(...).
    The belt is assumed to be thin and straight between the pulleys.
    """
    if radius_small_mm <= 0 or radius_large_mm < radius_small_mm:
        raise ValueError("The radii must be positive and the large radius must not be smaller than the small one.")
    ratio = (radius_large_mm - radius_small_mm) / centre_distance_mm
    if not 0 <= ratio < 1:
        raise ValueError("The centre distance must be larger than the difference of the radii.")
    angle = 2.0 * math.asin(ratio)
    return math.pi - angle, math.pi + angle


# ---------------------------------------------------------------------------
# Calculators (NOT from the video)
# ---------------------------------------------------------------------------


def belt_friction_coefficient(tight_tension_n: float, slack_tension_n: float, wrap_angle_rad: float) -> float:
    """Belt Friction Calculator: mu = ln(T1 / T2) / beta (Euler-Eytelwein: T1 / T2 = exp(mu x beta)).

    T1 and T2 are the tensions of the tight and the slack side, in the same unit (N). beta is the wrap angle in radians.
    The tensions must be measured by you (for example with spring balances). The video cannot give them.
    The result is the coefficient of friction at the point where the belt is just about to slip. For a V-belt it is
    the effective coefficient, which includes the wedging of the groove.
    """
    if tight_tension_n <= 0 or slack_tension_n <= 0:
        raise ValueError("The belt tensions must be positive.")
    if tight_tension_n < slack_tension_n:
        raise ValueError("The tight side tension T1 must not be smaller than the slack side tension T2.")
    if wrap_angle_rad <= 0:
        raise ValueError("The wrap angle must be positive.")
    return math.log(tight_tension_n / slack_tension_n) / wrap_angle_rad


def tension_ratio(friction_coefficient: float, wrap_angle_rad: float) -> float:
    """T1 / T2 = exp(mu x beta): the largest tension ratio that the friction can carry without slipping."""
    if friction_coefficient < 0 or wrap_angle_rad <= 0:
        raise ValueError("The friction coefficient cannot be negative and the wrap angle must be positive.")
    return math.exp(friction_coefficient * wrap_angle_rad)


def power_from_torque(torque_nm: float, rpm: float) -> float:
    """P = T x omega in watts, from a torque in N m and a speed in rpm."""
    if torque_nm < 0:
        raise ValueError("The torque cannot be negative. Give the size of the torque.")
    return torque_nm * 2.0 * math.pi * abs(rpm) / 60.0


def power_difference(torque_in_nm: float, rpm_in: float, torque_out_nm: float, rpm_out: float) -> dict:
    """Power Difference Calculator: power in, power out and their difference (all model-based, from your own numbers).

    The torques have to be measured by you. The speeds can come from this analysis (the mean rpm of each pulley).
    The difference P_in - P_out is only an estimate of the power that the belt drive loses: it carries the errors
    of both torques, which are usually bigger than the loss itself, and it contains everything between the two
    measuring points, not only the belt.
    """
    power_in = power_from_torque(torque_in_nm, rpm_in)
    power_out = power_from_torque(torque_out_nm, rpm_out)
    if power_in <= 0:
        raise ValueError("The input power must be larger than 0.")
    return {
        "power_in_w": power_in,
        "power_out_w": power_out,
        "power_difference_w": power_in - power_out,
        "efficiency_percent": 100.0 * power_out / power_in,
    }
