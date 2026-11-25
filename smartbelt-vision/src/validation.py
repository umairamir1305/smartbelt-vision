"""Validation: how close are the measured numbers to a reference?

A reference is a value that was found another way: the true value of a simulated video, a tachometer
reading, or turns counted by hand from the video. This module has the error measures and the checks.
It does not decide what is good enough. A tolerance is given by the person who runs the check.

Definitions (measured m, reference r):
    error                m - r                       (signed: positive means the measurement is too high)
    absolute error       |m - r|
    percent error        (m - r) / |r| x 100         (signed)
    MAE                  mean of |m - r| over a series
    RMSE                 square root of the mean of (m - r)^2 over a series
    bias                 mean of (m - r) over a series
"""

import math

import numpy as np


def error(measured: float, reference: float) -> float:
    """Signed error: measured minus reference."""
    return float(measured) - float(reference)


def absolute_error(measured: float, reference: float) -> float:
    return abs(error(measured, reference))


def percent_error(measured: float, reference: float) -> float:
    """Signed error as a percentage of the reference. The reference must not be zero."""
    if float(reference) == 0.0:
        raise ValueError("A percent error needs a reference that is not zero.")
    return error(measured, reference) / abs(float(reference)) * 100.0


def _pairs(measured, reference):
    """The pairs of values where both are numbers (NaN pairs are skipped)."""
    measured = np.asarray(measured, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if measured.shape != reference.shape:
        raise ValueError(f"The measured series has {measured.size} values and the reference series has {reference.size}. They must match.")
    valid = np.isfinite(measured) & np.isfinite(reference)
    if not valid.any():
        raise ValueError("There is no pair of values that can be compared.")
    return measured[valid], reference[valid]


def mean_absolute_error(measured, reference) -> float:
    m, r = _pairs(measured, reference)
    return float(np.mean(np.abs(m - r)))


def root_mean_square_error(measured, reference) -> float:
    m, r = _pairs(measured, reference)
    return float(math.sqrt(np.mean((m - r) ** 2)))


def bias(measured, reference) -> float:
    """Mean signed error of a series: a steady offset between the measurement and the reference."""
    m, r = _pairs(measured, reference)
    return float(np.mean(m - r))


def compare_value(group, test, quantity, unit, measured, reference, tolerance_percent=None, tolerance_abs=None) -> dict:
    """Compare one measured number with its reference and say whether it is inside a tolerance.

    tolerance_abs is in the unit of the quantity, tolerance_percent is a percentage of the reference.
    If both are None, no result is given (passed is None): the row only reports the error.
    """
    err = error(measured, reference)
    pct = percent_error(measured, reference) if float(reference) != 0.0 else None
    if tolerance_abs is not None:
        passed = abs(err) <= tolerance_abs
        limit = f"{tolerance_abs:g} {unit}"
    elif tolerance_percent is not None:
        if pct is None:
            raise ValueError("A percent tolerance needs a reference that is not zero. Use an absolute tolerance.")
        passed = abs(pct) <= tolerance_percent
        limit = f"{tolerance_percent:g} %"
    else:
        passed = None
        limit = "none"
    return {
        "group": group, "test": test, "quantity": quantity, "unit": unit, "measured": float(measured), "reference": float(reference),
        "error": err, "percent_error": pct, "tolerance": limit, "passed": passed,
    }


def compare_series(test, quantity, unit, measured, reference) -> dict:
    """Error measures of a measured series against a reference series (for example the speed in every frame)."""
    m, r = _pairs(measured, reference)
    return {
        "test": test, "quantity": quantity, "unit": unit, "n": int(m.size),
        "mae": mean_absolute_error(m, r), "rmse": root_mean_square_error(m, r), "bias": bias(m, r),
        "max_abs_error": float(np.max(np.abs(m - r))),
    }


def check_time_axis(times_s, fps: float, tolerance_percent: float = 1.0) -> dict:
    """Check that the frames are evenly spaced in time and that the frame rate matches the frame times.

    times_s are the times of the frames as the video file gives them (see video_io.frame_timestamps).
    The analysis assumes that frame i is at time i / fps. If the file has a variable frame rate or dropped
    frames, or if the frame rate that was used is wrong, the speeds are wrong.
    """
    times = np.asarray(times_s, dtype=float)
    if times.size < 3 or not np.all(np.isfinite(times)):
        return {"available": False, "reason": "The file does not give usable frame times, so the spacing of the frames could not be checked."}
    steps = np.diff(times)
    if not np.any(steps > 0):
        return {"available": False, "reason": "The file does not give frame times (they are all the same), so the spacing of the frames could not be checked."}

    median_step = float(np.median(steps))
    expected_step = 1.0 / float(fps)
    fps_from_times = 1.0 / median_step
    step_error = (median_step - expected_step) / expected_step * 100.0
    irregularity = float(np.max(np.abs(steps - median_step)) / median_step * 100.0)

    problems = []
    if abs(step_error) > tolerance_percent:
        problems.append(f"the frame times say {fps_from_times:.2f} fps but the analysis uses {fps:g} fps")
    if irregularity > tolerance_percent:
        problems.append(
            f"the time between frames varies from {steps.min() * 1000:.1f} to {steps.max() * 1000:.1f} ms "
            "(variable frame rate or dropped frames), so speeds from a constant frame rate are unreliable"
        )
    return {
        "available": True, "frames": int(times.size), "median_step_s": median_step, "expected_step_s": expected_step,
        "fps_from_times": fps_from_times, "step_error_percent": step_error, "max_irregularity_percent": irregularity,
        "shortest_step_s": float(steps.min()), "longest_step_s": float(steps.max()),
        "ok": not problems, "problems": problems,
    }
