"""Two-pulley belt drive: the results of the driver and the driven pulley side by side.

Each pulley is analysed on its own (pipeline.analyse_pulley). This module joins the two tables and
compares the pulleys.

Theory used (an open belt that neither slips nor stretches):
  The belt moves with the same speed at both pulleys, so the surface speeds are equal:
      omega_driver x R_driver = omega_driven x R_driven
  R is the effective radius: from the centre to the middle of the belt, not the outside of the pulley.
  So the ideal speed ratio is  driver speed / driven speed = R_driven / R_driver.

Definitions used here:
  speed ratio       driver speed / driven speed, from the video (turns of the driver per turn of the driven pulley)
  expected ratio    R_driven / R_driver from the radii typed in, or a reference ratio measured with no
                    load (which cancels the error of the radii)
  speed loss        (v_ideal - v_actual) / v_ideal x 100 percent, with v_ideal = omega_driver x R_driver and
                    v_actual = omega_driven x R_driven. It is the same as 1 - expected ratio / speed ratio.
                    A positive value means the driven pulley turns slower than the expected ratio says.

Limits:
  - The speed loss is only as good as the expected ratio. A 1 mm error in an effective radius of 25 mm is
    4 percent, which is more than the slip of a normal belt drive. A reference ratio from a recording
    with no load is more reliable than radii measured with a ruler.
  - It contains everything that makes the driven pulley slower than the radii say: belt creep (a small
    elastic effect that is always there under load), real slipping, a wrong effective radius, belt
    stretch. It does not separate them. Tracking the belt itself (the next phase) gives more information.
  - Directions: the same direction is an open belt, opposite directions a crossed belt.
"""

import math
from itertools import groupby

import numpy as np
import pandas as pd

from src.kinematics import AnalysisError, rpm_to_rad_s

# Each pulley must turn at least this much while both are tracked, or no ratio is calculated.
MIN_COMMON_ROTATION_RAD = math.radians(5)

# The speed ratio of a frame is only calculated if the driven pulley turns at least this share of its typical speed.
MIN_SPEED_FRACTION = 0.2


def merge_pulley_tables(driver: pd.DataFrame, driven: pd.DataFrame) -> pd.DataFrame:
    """One table with frame and time_s, and every other column of each pulley with the prefix driver_ or driven_."""
    if len(driver) != len(driven) or not np.array_equal(driver["frame"].to_numpy(), driven["frame"].to_numpy()):
        raise AnalysisError("The two pulley tables are not for the same frames. Track both pulleys with the same video settings.")
    if not np.allclose(driver["time_s"].to_numpy(dtype=float), driven["time_s"].to_numpy(dtype=float)):
        raise AnalysisError("The two pulley tables do not have the same frame times.")

    merged = driver[["frame", "time_s"]].copy()
    for prefix, table in (("driver_", driver), ("driven_", driven)):
        for column in table.columns:
            if column not in ("frame", "time_s"):
                merged[prefix + column] = table[column].to_numpy()
    return merged


def _chosen_speed(merged: pd.DataFrame, prefix: str) -> np.ndarray:
    """The speed in rpm used for the ratio: the smoothed speed if the table has it, otherwise the single frame speed."""
    smooth = prefix + "rpm_smooth"
    if smooth in merged.columns and merged[smooth].notna().any():
        return merged[smooth].to_numpy(dtype=float)
    return merged[prefix + "rpm"].to_numpy(dtype=float)


def common_rotation(merged: pd.DataFrame) -> dict:
    """How far each pulley turned while BOTH had a trusted angle, and how long that took.

    Frames are grouped into runs where neither pulley changed segment (see kinematics.py), because
    the angle can only be compared across frames of the same segment. The rotation of a run is the
    angle at its last frame minus the angle at its first frame, and these are added up over the runs.
    This is the same as counting the turns of each pulley during the same time.
    """
    angle_driver = merged["driver_angle_rad"].to_numpy(dtype=float)
    angle_driven = merged["driven_angle_rad"].to_numpy(dtype=float)
    segment_driver = merged["driver_segment"].to_numpy(dtype=float, na_value=np.nan)
    segment_driven = merged["driven_segment"].to_numpy(dtype=float, na_value=np.nan)
    time_s = merged["time_s"].to_numpy(dtype=float)

    both = ~np.isnan(angle_driver) & ~np.isnan(angle_driven)
    positions = np.flatnonzero(both)
    if len(positions) < 2:
        raise AnalysisError("The two pulleys have fewer than 2 frames in common where both angles are known.")

    rotation_driver = rotation_driven = duration = 0.0
    runs = 0
    for _, group in groupby(positions, key=lambda i: (segment_driver[i], segment_driven[i])):
        indices = list(group)
        if len(indices) < 2:
            continue
        first, last = indices[0], indices[-1]
        rotation_driver += float(angle_driver[last] - angle_driver[first])
        rotation_driven += float(angle_driven[last] - angle_driven[first])
        duration += float(time_s[last] - time_s[first])
        runs += 1

    if runs == 0 or duration <= 0:
        raise AnalysisError("The two pulleys have no stretch of frames in common where both angles can be compared.")
    return {
        "common_frames": int(both.sum()),
        "runs": runs,
        "duration_s": duration,
        "driver_rotation_rad": rotation_driver,
        "driven_rotation_rad": rotation_driven,
    }


def expected_ratio_from(driver_radius_mm=None, driven_radius_mm=None, reference_ratio=None):
    """The expected speed ratio and where it came from: (value, source), or (None, None) if it is not known."""
    if reference_ratio is not None:
        if reference_ratio <= 0:
            raise AnalysisError("The reference ratio must be positive.")
        return float(reference_ratio), "reference ratio"
    if driver_radius_mm is not None and driven_radius_mm is not None:
        if driver_radius_mm <= 0 or driven_radius_mm <= 0:
            raise AnalysisError("The pulley radii must be positive.")
        return float(driven_radius_mm) / float(driver_radius_mm), "pulley radii"
    return None, None


def add_belt_columns(merged: pd.DataFrame, expected_ratio=None, driver_radius_mm=None, driven_radius_mm=None) -> pd.DataFrame:
    """Add speed_ratio, and if possible speed_loss_percent and the surface speeds, to a merged table.

    speed_ratio              driver speed / driven speed (negative for opposite directions)
    speed_loss_percent       the speed loss defined at the top of this file (needs an expected ratio)
    driver_surface_speed_m_s, driven_surface_speed_m_s   omega x radius in m/s (need that pulley's radius)

    These use the smoothed speeds when the table has them. Frames where the driven pulley is
    almost stopped have no ratio.
    """
    merged = merged.copy()
    driver = _chosen_speed(merged, "driver_")
    driven = _chosen_speed(merged, "driven_")

    typical_driven = np.nanmedian(np.abs(driven)) if np.isfinite(driven).any() else math.nan
    usable = np.isfinite(driver) & np.isfinite(driven) & (np.abs(driven) >= MIN_SPEED_FRACTION * typical_driven)

    ratio = np.full(len(merged), np.nan)
    ratio[usable] = driver[usable] / driven[usable]
    merged["speed_ratio"] = ratio

    if expected_ratio is not None:
        loss = np.full(len(merged), np.nan)
        loss[usable] = (1.0 - np.abs(driven[usable]) * expected_ratio / np.abs(driver[usable])) * 100.0
        merged["speed_loss_percent"] = loss

    for name, speed, radius in (("driver", driver, driver_radius_mm), ("driven", driven, driven_radius_mm)):
        if radius is not None:
            merged[f"{name}_surface_speed_m_s"] = np.abs(rpm_to_rad_s(speed)) * radius / 1000.0
    return merged


def summarise_belt_drive(
    merged: pd.DataFrame, expected_ratio=None, expected_source=None, driver_radius_mm=None, driven_radius_mm=None
) -> dict:
    """The numbers of the belt drive comparison, in a dictionary (see the top of this file for the definitions)."""
    rotation = common_rotation(merged)
    driver_rotation, driven_rotation = rotation["driver_rotation_rad"], rotation["driven_rotation_rad"]
    if abs(driver_rotation) < MIN_COMMON_ROTATION_RAD:
        raise AnalysisError("The driver pulley turned less than 5 degrees while both pulleys were tracked, so no speed ratio is calculated.")
    if abs(driven_rotation) < MIN_COMMON_ROTATION_RAD:
        raise AnalysisError("The driven pulley turned less than 5 degrees while both pulleys were tracked, so no speed ratio is calculated.")

    measured = abs(driver_rotation / driven_rotation)
    duration = rotation["duration_s"]
    summary = {
        **rotation,
        "driver_turns": driver_rotation / (2.0 * math.pi),
        "driven_turns": driven_rotation / (2.0 * math.pi),
        "driver_mean_rpm": abs(driver_rotation) / duration * 60.0 / (2.0 * math.pi),
        "driven_mean_rpm": abs(driven_rotation) / duration * 60.0 / (2.0 * math.pi),
        "measured_ratio": measured,
        "direction": "same direction (open belt)" if driver_rotation * driven_rotation > 0 else "opposite directions (crossed belt)",
        "expected_ratio": expected_ratio,
        "expected_source": expected_source,
        "ratio_deviation_percent": None,
        "speed_loss_percent": None,
        "driver_surface_speed_m_s": None,
        "driven_surface_speed_m_s": None,
        "radius_sensitivity_driver_percent_per_mm": None,
        "radius_sensitivity_driven_percent_per_mm": None,
        "ratio_std": math.nan,
        "ratio_range": (math.nan, math.nan),
    }

    if expected_ratio is not None:
        summary["ratio_deviation_percent"] = (measured - expected_ratio) / expected_ratio * 100.0
        summary["speed_loss_percent"] = (1.0 - expected_ratio / measured) * 100.0
    if driver_radius_mm is not None:
        summary["driver_surface_speed_m_s"] = summary["driver_mean_rpm"] * 2.0 * math.pi / 60.0 * driver_radius_mm / 1000.0
        summary["radius_sensitivity_driver_percent_per_mm"] = 100.0 / driver_radius_mm
    if driven_radius_mm is not None:
        summary["driven_surface_speed_m_s"] = summary["driven_mean_rpm"] * 2.0 * math.pi / 60.0 * driven_radius_mm / 1000.0
        summary["radius_sensitivity_driven_percent_per_mm"] = 100.0 / driven_radius_mm

    if "speed_ratio" in merged.columns:
        values = np.abs(merged["speed_ratio"].to_numpy(dtype=float))
        values = values[np.isfinite(values)]
        if len(values) > 1:
            summary["ratio_std"] = float(values.std(ddof=1))
            summary["ratio_range"] = (float(np.percentile(values, 5)), float(np.percentile(values, 95)))
    return summary
