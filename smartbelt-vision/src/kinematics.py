"""Angle calculations from the tracked marker positions.

Angle convention (the same as in src/synthetic.py):
  - The angle is measured at the rotation centre, from the 3 o'clock direction
    (to the right of the centre).
  - Counter-clockwise on the screen is positive.
  - The image y axis points DOWN, so the y difference is flipped:
        theta = atan2(centre_y - y, x - centre_x)

Why unwrapping is needed: atan2 only gives angles between -pi and +pi. When the marker
passes the 9 o'clock position, the angle jumps from about +pi to about -pi, although the
marker only moved a little. np.unwrap removes these jumps by adding or subtracting 2 pi,
so the angle keeps counting up (or down) as the disc turns. After three turns it reads
about 6 pi. The unwrapped angle is what angular velocity is calculated from.

Frames without a trusted marker position (tracking_valid is False) get no angle.
They are gaps in the angle series.

Angular velocity is the rate of change of the unwrapped angle: omega = d(theta) / dt.
np.gradient approximates it from the angles on both sides of each frame:
    omega[i] = (theta[i + 1] - theta[i - 1]) / (t[i + 1] - t[i - 1])      (inside a segment)
It is given the real time of every frame, so a lost frame inside a segment does not
make the speed wrong: the difference is simply taken across the gap, using the real time
that passed. Speeds are never calculated across a split gap (see below).
Then rpm = 60 * omega / (2 * pi).

Gaps and np.unwrap: np.unwrap assumes the angle changes by less than half a turn (180
degrees) between two samples. Across a gap that may not be true, and the unwrapped angle
after the gap could be off by whole turns. If the marker could have turned more than 90
degrees during a gap (a rule of thumb, half of the 180 degree limit), the angle series is
split there into segments. The segment number goes up by one at that gap. Later phases
never calculate a speed across a split gap.
"""

import math

import numpy as np
import pandas as pd

# A gap is "long" if the marker could have turned more than this during it.
MAX_GAP_ROTATION_RAD = math.pi / 2  # 90 degrees

# The rotation rate before a gap is taken from this many one-frame steps.
RATE_STEPS = 5

# Fewer valid frames than this give no meaningful angle series.
MIN_VALID_FRAMES = 10

# The circle fit needs at least this many points and this much of the circle covered.
MIN_FIT_POINTS = 20
MIN_FIT_COVERAGE_DEG = 270.0


class AnalysisError(Exception):
    """A problem in the angle or speed calculation. The message is meant for the user."""


def angle_from_position(x, y, centre) -> np.ndarray:
    """Angle of the marker about the centre in radians, between -pi and +pi.

    Counter-clockwise on the screen is positive. The y difference is flipped because the
    image y axis points down. Positions that are NaN give NaN.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.arctan2(centre[1] - y, x - centre[0])


def _estimated_gap_rotation(angles: np.ndarray, indices: np.ndarray, k: int) -> float:
    """Estimate (in radians, always positive) how far the marker turned across a gap.

    The gap lies between the valid samples k - 1 and k. The rotation rate per frame is
    the median of up to RATE_STEPS one-frame steps just before the gap (or just after it
    if there are none before). The estimate is that rate times the number of frames across the gap.
    If there are no one-frame steps at all, the observed step across the gap is used.
    """
    frames_apart = indices[k] - indices[k - 1]

    steps = []
    j = k - 1
    while j > 0 and indices[j] - indices[j - 1] == 1 and len(steps) < RATE_STEPS:
        steps.append(angles[j] - angles[j - 1])
        j -= 1
    if not steps:
        j = k
        while j + 1 < len(indices) and indices[j + 1] - indices[j] == 1 and len(steps) < RATE_STEPS:
            steps.append(angles[j + 1] - angles[j])
            j += 1

    if steps:
        return abs(float(np.median(steps))) * frames_apart
    return abs(float(angles[k] - angles[k - 1]))


def add_angle_columns(table: pd.DataFrame, centre) -> pd.DataFrame:
    """Add angle_wrapped_rad, angle_rad and segment to a tracking table.

    angle_wrapped_rad  angle between -pi and +pi, straight from atan2
    angle_rad          unwrapped angle, continuous, in radians (counter-clockwise positive)
    segment            0 for the first stretch of angle data, plus 1 after every long gap.
                       Empty for frames without an angle.

    The table must have the columns marker_x, marker_y and tracking_valid (see tracking.py).
    """
    table = table.copy()
    count = len(table)
    valid = table["tracking_valid"].to_numpy(dtype=bool)
    if int(valid.sum()) < MIN_VALID_FRAMES:
        raise AnalysisError(
            f"Only {int(valid.sum())} frames have a trusted marker position, so no angle is calculated "
            f"(at least {MIN_VALID_FRAMES} are needed). Check the tracking first: use --preview, "
            "and look at the tracking quality picture and the tracked video."
        )

    x = table["marker_x"].to_numpy(dtype=float)
    y = table["marker_y"].to_numpy(dtype=float)

    wrapped = np.full(count, np.nan)
    wrapped[valid] = angle_from_position(x[valid], y[valid], centre)

    # np.unwrap is applied to the valid samples only. Frames without a marker are skipped.
    valid_index = np.flatnonzero(valid)
    unwrapped_valid = np.unwrap(wrapped[valid_index])
    unwrapped = np.full(count, np.nan)
    unwrapped[valid_index] = unwrapped_valid

    segment_valid = np.zeros(len(valid_index), dtype=int)
    current_segment = 0
    for k in range(1, len(valid_index)):
        if valid_index[k] - valid_index[k - 1] > 1:
            if _estimated_gap_rotation(unwrapped_valid, valid_index, k) > MAX_GAP_ROTATION_RAD:
                current_segment += 1
        segment_valid[k] = current_segment

    segment = np.full(count, np.nan)
    segment[valid_index] = segment_valid

    table["angle_wrapped_rad"] = wrapped
    table["angle_rad"] = unwrapped
    table["segment"] = pd.Series(segment).astype("Int64")  # whole numbers, with empty values allowed
    return table


def summarise_rotation(table: pd.DataFrame) -> dict:
    """Summarise the angle columns: how far the disc turned, in which direction, in how many segments."""
    angle = table["angle_rad"].to_numpy(dtype=float)
    time_s = table["time_s"].to_numpy(dtype=float)
    frame = table["frame"].to_numpy()
    segment = table["segment"].to_numpy(dtype=float, na_value=np.nan)

    has_angle = ~np.isnan(angle)
    if not has_angle.any():
        raise AnalysisError("The table has no angle values. Call add_angle_columns first.")

    first, last = np.flatnonzero(has_angle)[[0, -1]]
    total = float(angle[last] - angle[first])

    segment_ids = sorted({int(s) for s in segment[has_angle]})
    rotation_in_segments = 0.0
    longest_segment = 0
    for segment_id in segment_ids:
        in_segment = np.flatnonzero(has_angle & (segment == segment_id))
        rotation_in_segments += float(angle[in_segment[-1]] - angle[in_segment[0]])
        longest_segment = max(longest_segment, len(in_segment))

    if abs(total) < math.radians(5):
        direction = "no clear rotation"
    elif total > 0:
        direction = "counter-clockwise"
    else:
        direction = "clockwise"

    return {
        "frames": len(table),
        "angle_frames": int(has_angle.sum()),
        "first_frame": int(frame[first]),
        "last_frame": int(frame[last]),
        "duration_s": float(time_s[last] - time_s[first]),
        "total_rotation_rad": total,
        "turns": total / (2.0 * math.pi),
        "direction": direction,
        "segments": len(segment_ids),
        "rotation_in_segments_rad": rotation_in_segments,
        "longest_segment_frames": longest_segment,
    }


# ---------------------------------------------------------------------------
# Angular velocity and rpm
# ---------------------------------------------------------------------------


def rad_s_to_rpm(omega_rad_s):
    """Convert angular velocity in rad/s to rotations per minute: rpm = 60 * omega / (2 * pi).

    One turn is 2 * pi radians, so 2 * pi rad/s is 1 turn per second, which is 60 rpm.
    """
    return np.asarray(omega_rad_s, dtype=float) * 60.0 / (2.0 * math.pi)


def rpm_to_rad_s(rpm):
    """Convert rotations per minute to angular velocity in rad/s: omega = 2 * pi * rpm / 60."""
    return np.asarray(rpm, dtype=float) * 2.0 * math.pi / 60.0


def add_speed_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Add angular_velocity_rad_s and rpm to a table that already has the angle columns.

    The speed is calculated for each segment on its own, from the unwrapped angle and the
    real time of each frame. Frames without an angle get no speed, and nothing is
    calculated across a split gap. A segment needs at least 2 frames. At the first and last
    frame of a segment there is only one neighbour, so np.gradient uses a one-sided
    difference there, which is a little less accurate than the two-sided one inside.

    Positive values mean counter-clockwise rotation, negative values clockwise.
    """
    for name in ("time_s", "angle_rad", "segment"):
        if name not in table.columns:
            raise AnalysisError("The table has no angle values yet. Call add_angle_columns first.")

    table = table.copy()
    time_s = table["time_s"].to_numpy(dtype=float)
    angle = table["angle_rad"].to_numpy(dtype=float)
    segment = table["segment"].to_numpy(dtype=float, na_value=np.nan)

    omega = np.full(len(table), np.nan)
    for segment_id in sorted({int(s) for s in segment[~np.isnan(segment)]}):
        in_segment = np.flatnonzero(segment == segment_id)
        if len(in_segment) < 2:
            continue
        # np.gradient accepts the real time of every sample, so unevenly spaced frames are fine.
        edge_order = 2 if len(in_segment) >= 3 else 1
        omega[in_segment] = np.gradient(angle[in_segment], time_s[in_segment], edge_order=edge_order)

    table["angular_velocity_rad_s"] = omega
    table["rpm"] = rad_s_to_rpm(omega)
    return table


def summarise_speed(table: pd.DataFrame) -> dict:
    """Summarise the rotation speed in rpm.

    Speeds are reported as positive numbers in the direction the disc is turning
    (the table keeps the sign: negative means clockwise).

    mean_rpm       total rotation divided by the time it took, inside the segments. This is the
                   same as counting turns and dividing by time. It does not depend on the
                   noise of single frames and it ignores long gaps.
    min_rpm        smallest speed of a single frame
    max_rpm        largest speed of a single frame
    std_rpm        standard deviation of the single frame speeds
    variation_percent   std_rpm as a percentage of mean_rpm
    trend_rpm      change of the speed over the whole clip, from a straight line fitted to
                   the single frame speeds. Close to 0 for a steady speed.

    Single frame speeds include measurement noise, so min, max and the standard deviation
    are overestimates of the real speed variation. For a disc that is slowing down they mostly
    show the slowing down.
    """
    for name in ("time_s", "angle_rad", "segment", "rpm"):
        if name not in table.columns:
            raise AnalysisError("The table has no speed values yet. Call add_speed_columns first.")

    time_s = table["time_s"].to_numpy(dtype=float)
    angle = table["angle_rad"].to_numpy(dtype=float)
    segment = table["segment"].to_numpy(dtype=float, na_value=np.nan)
    rpm = table["rpm"].to_numpy(dtype=float)

    has_speed = ~np.isnan(rpm)
    if not has_speed.any():
        raise AnalysisError("The table has no speed values. The segments are too short to calculate a speed.")

    # Mean speed: total rotation divided by total time, summed over the segments.
    rotation = 0.0
    duration = 0.0
    for segment_id in sorted({int(s) for s in segment[~np.isnan(segment)]}):
        in_segment = np.flatnonzero(segment == segment_id)
        if len(in_segment) >= 2:
            rotation += float(angle[in_segment[-1]] - angle[in_segment[0]])
            duration += float(time_s[in_segment[-1]] - time_s[in_segment[0]])
    mean_signed_rpm = float(rad_s_to_rpm(rotation / duration))

    if abs(rotation) < math.radians(5):
        direction, sign = "no clear rotation", 1.0
    elif mean_signed_rpm > 0:
        direction, sign = "counter-clockwise", 1.0
    else:
        direction, sign = "clockwise", -1.0

    along = rpm[has_speed] * sign  # speeds in the direction of rotation
    std = float(along.std(ddof=1)) if len(along) > 1 else 0.0
    mean_rpm = abs(mean_signed_rpm)

    trend = 0.0
    if len(along) > 2:
        slope = float(np.polyfit(time_s[has_speed], along, 1)[0])  # rpm per second
        trend = slope * float(time_s[has_speed][-1] - time_s[has_speed][0])

    return {
        "speed_samples": int(has_speed.sum()),
        "direction": direction,
        "mean_rpm": mean_rpm,
        "min_rpm": float(along.min()),
        "max_rpm": float(along.max()),
        "std_rpm": std,
        "variation_percent": 100.0 * std / mean_rpm if mean_rpm > 0 else math.nan,
        "trend_rpm": trend,
    }


# ---------------------------------------------------------------------------
# Checking the rotation centre with a circle fit
# ---------------------------------------------------------------------------


def fit_circle(x, y):
    """Least-squares circle through points. Returns (centre_x, centre_y, radius, rms_residual).

    A point on a circle satisfies x^2 + y^2 + D x + E y + F = 0, which is LINEAR in D, E
    and F. So one linear least-squares solve gives the circle: centre = (-D / 2, -E / 2)
    and radius = sqrt(centre_x^2 + centre_y^2 - F). The rms residual is the typical
    distance of the points from the fitted circle, in the same units as x and y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        raise AnalysisError("A circle needs at least 3 points.")

    # Work relative to the mean position, which keeps the numbers small and the solve accurate.
    x0, y0 = x.mean(), y.mean()
    u, v = x - x0, y - y0
    design = np.column_stack([u, v, np.ones_like(u)])
    target = -(u**2 + v**2)
    (d, e, f), *_ = np.linalg.lstsq(design, target, rcond=None)

    centre_u, centre_v = -d / 2.0, -e / 2.0
    radius = math.sqrt(max(centre_u**2 + centre_v**2 - f, 0.0))
    residual = np.hypot(u - centre_u, v - centre_v) - radius
    return x0 + centre_u, y0 + centre_v, radius, float(np.sqrt(np.mean(residual**2)))


def _circle_coverage_deg(x, y, centre_x, centre_y) -> float:
    """How many degrees of the circle have points on them: 360 minus the biggest empty stretch."""
    angles = np.sort(np.arctan2(np.asarray(y) - centre_y, np.asarray(x) - centre_x))
    gaps = np.append(np.diff(angles), angles[0] + 2.0 * math.pi - angles[-1])
    return 360.0 - math.degrees(float(gaps.max()))


def fit_centre_from_track(table: pd.DataFrame) -> dict:
    """Fit a circle to the trusted marker positions to find where the rotation centre really is.

    Returns a dictionary with centre, radius_px, rms_px, coverage_deg, usable and reason.
    usable is False (and reason says why) when there are too few points or the marker only
    covered part of the circle, because then the fitted centre cannot be trusted.
    """
    result = {"centre": None, "radius_px": None, "rms_px": None, "coverage_deg": None, "usable": False, "reason": ""}

    valid = table["tracking_valid"].to_numpy(dtype=bool)
    x = table["marker_x"].to_numpy(dtype=float)[valid]
    y = table["marker_y"].to_numpy(dtype=float)[valid]
    if len(x) < MIN_FIT_POINTS:
        result["reason"] = f"only {len(x)} trusted marker positions (at least {MIN_FIT_POINTS} are needed)"
        return result

    centre_x, centre_y, radius, rms = fit_circle(x, y)
    coverage = _circle_coverage_deg(x, y, centre_x, centre_y)
    result.update(centre=(centre_x, centre_y), radius_px=radius, rms_px=rms, coverage_deg=coverage)

    if coverage < MIN_FIT_COVERAGE_DEG:
        result["reason"] = (
            f"the marker covered only {coverage:.0f} degrees of the circle "
            f"(at least {MIN_FIT_COVERAGE_DEG:.0f} are needed)"
        )
    else:
        result["usable"] = True
    return result
