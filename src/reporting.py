"""Plots for the analysis. Each plot function builds and returns a Matplotlib figure.
save_figure writes it to a PNG file and closes it.

Figures are only drawn to files (the Agg backend), so no window ever opens.
Plots made from simulated data get "(simulated data)" in their title.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only. This must be set before pyplot is imported.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.utils import ensure_folder  # noqa: E402

# Plot colours for the tracking quality of a frame.
QUALITY_PLOT_COLOURS = {"Good": "#2e9e4f", "Moderate": "#e8a317", "Poor": "#d03030", "Lost": "#9a9a9a"}

SIMULATED_NOTE = " (simulated data)"


def _title(text: str, simulated: bool) -> str:
    return text + SIMULATED_NOTE if simulated else text


def save_figure(figure, path, dpi: int = 150) -> Path:
    """Save a figure as an image file, then close it to free the memory."""
    path = Path(path)
    ensure_folder(path.parent)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_marker_trajectory(table, centre=None, simulated: bool = False):
    """Marker positions in the video frame. The y axis points down, as in the video.

    For a marker on a rotating disc the points should lie on a circle around the centre.
    Positions that were found but are not trusted (tracking quality Poor) are shown as red crosses.
    """
    found = table["marker_x"].notna().to_numpy()
    valid = table["tracking_valid"].to_numpy(dtype=bool)
    x = table["marker_x"].to_numpy(dtype=float)
    y = table["marker_y"].to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(x[valid], y[valid], ".", markersize=4, color="#1f5fa8", label=f"trusted positions ({int(valid.sum())})")
    not_trusted = found & ~valid
    if not_trusted.any():
        axis.plot(
            x[not_trusted], y[not_trusted], "x", markersize=6, color=QUALITY_PLOT_COLOURS["Poor"],
            label=f"not trusted ({int(not_trusted.sum())})",
        )
    if centre is not None:
        axis.plot(centre[0], centre[1], "+", markersize=14, markeredgewidth=2, color="black", label="rotation centre used")

    axis.set_aspect("equal", adjustable="datalim")
    axis.invert_yaxis()  # image coordinates: y increases downwards
    axis.set_xlabel("x (pixels)")
    axis.set_ylabel("y (pixels, increasing downwards)")
    axis.set_title(_title("Marker trajectory in the video frame", simulated))
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9, frameon=False)
    return figure


def plot_angle_vs_time(table, simulated: bool = False):
    """Marker angle against time. Top: straight from atan2 (jumps at +/- pi). Bottom: unwrapped.

    The top panel shows why unwrapping is needed. In the bottom panel the line is
    continuous. If the angle was split into segments (long gaps), each segment is drawn
    as its own line. Frames without a trusted marker are left empty, not filled in.
    """
    time_s = table["time_s"].to_numpy(dtype=float)
    wrapped = table["angle_wrapped_rad"].to_numpy(dtype=float)
    unwrapped = table["angle_rad"].to_numpy(dtype=float)
    segment = table["segment"].to_numpy(dtype=float, na_value=np.nan)
    segment_ids = sorted({int(s) for s in segment[~np.isnan(segment)]})

    figure, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(8, 6.5))

    top.plot(time_s, wrapped, ".", markersize=3, color="#1f5fa8")
    top.set_ylabel("Angle from atan2 (rad)")
    top.set_yticks([-math.pi, 0.0, math.pi])
    top.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    top.set_title(_title("Marker angle against time", simulated))
    top.grid(True, alpha=0.3)

    for segment_id in segment_ids:
        in_segment = segment == segment_id
        label = f"segment {segment_id}" if len(segment_ids) > 1 else None
        bottom.plot(time_s[in_segment], unwrapped[in_segment], "-", linewidth=1.5, label=label)
    bottom.set_xlabel("Time (s)")
    bottom.set_ylabel("Unwrapped angle (rad)")
    bottom.grid(True, alpha=0.3)
    if len(segment_ids) > 1:
        bottom.legend(loc="best", fontsize=9)

    # A second scale in turns makes it easy to count how many times the disc went round.
    turns_axis = bottom.secondary_yaxis("right", functions=(lambda a: a / (2 * np.pi), lambda t: t * 2 * np.pi))
    turns_axis.set_ylabel("Unwrapped angle (turns)")
    return figure


def plot_tracking_timeline(table, simulated: bool = False):
    """One thin line per frame, coloured by tracking quality, along the time axis."""
    time_s = table["time_s"].to_numpy(dtype=float)
    quality = table["tracking_quality"].to_numpy()
    frame_time = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 1.0  # time covered by one frame

    figure, axis = plt.subplots(figsize=(8, 2.4))
    for name in ("Good", "Moderate", "Poor", "Lost"):
        selected = quality == name
        if selected.any():
            # One bar per frame, as wide as the time between frames, so neighbouring frames touch.
            axis.bar(
                time_s[selected], np.ones(int(selected.sum())), width=frame_time,
                color=QUALITY_PLOT_COLOURS[name], linewidth=0, label=f"{name} ({int(selected.sum())})",
            )

    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.set_xlabel("Time (s)")
    axis.set_title(_title("Tracking quality of every frame", simulated))
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=4, fontsize=9, frameon=False)
    return figure


def _direction_sign(values) -> float:
    """-1 if the rotation is clockwise (negative speeds on average), otherwise +1."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return -1.0 if len(values) and values.mean() < 0 else 1.0


def plot_rpm_vs_time(table, mean_rpm=None, simulated: bool = False):
    """Rotation speed in rpm against time, one value per frame.

    The speed is drawn as a positive number in the direction the disc turns (the title says
    when that is clockwise). Frames without a speed are left empty, so the line is broken
    where the marker was lost. mean_rpm, if given, is drawn as a dashed line.
    The line "speed of each frame" is the single frame speed and includes measurement noise.

    If the table has the columns from signal_analysis.py, the figure also shows the smoothed
    speed, the frames that were replaced as outliers, the lost frames that were filled in,
    and a second panel with the angular acceleration (positive means speeding up in the
    direction of rotation, negative means slowing down).
    """
    time_s = table["time_s"].to_numpy(dtype=float)
    rpm = table["rpm"].to_numpy(dtype=float)
    sign = _direction_sign(rpm)
    clockwise = sign < 0

    has_acceleration = "angular_acceleration_rad_s2" in table.columns
    if has_acceleration:
        figure, (axis, acceleration_axis) = plt.subplots(
            2, 1, sharex=True, figsize=(8, 6.5), gridspec_kw={"height_ratios": [2, 1]}
        )
    else:
        figure, axis = plt.subplots(figsize=(8, 4.5))

    axis.plot(time_s, sign * rpm, "-", linewidth=1.0, color="#7fa6d6", label="speed of each frame")

    if "rpm_smooth" in table.columns:
        smooth = sign * table["rpm_smooth"].to_numpy(dtype=float)
        axis.plot(time_s, smooth, "-", linewidth=2.0, color="#1f3f73", label="smoothed")
    if "speed_outlier" in table.columns and table["speed_outlier"].to_numpy(dtype=bool).any():
        outlier = table["speed_outlier"].to_numpy(dtype=bool)
        axis.plot(
            time_s[outlier], sign * rpm[outlier], "x", markersize=7, markeredgewidth=1.5,
            color=QUALITY_PLOT_COLOURS["Poor"], label=f"outlier, replaced ({int(outlier.sum())})",
        )
    if "speed_interpolated" in table.columns and table["speed_interpolated"].to_numpy(dtype=bool).any():
        filled = table["speed_interpolated"].to_numpy(dtype=bool)
        axis.plot(
            time_s[filled], sign * table["rpm_clean"].to_numpy(dtype=float)[filled], "o", markersize=5,
            markerfacecolor="none", color=QUALITY_PLOT_COLOURS["Moderate"], label=f"filled in, no measurement ({int(filled.sum())})",
        )
    if mean_rpm is not None:
        axis.axhline(mean_rpm, linestyle="--", linewidth=1.2, color="black", label=f"mean {mean_rpm:.1f} rpm")

    axis.set_ylabel("Rotation speed (rpm)")
    title = "Rotation speed against time" + (" (clockwise)" if clockwise else "")
    axis.set_title(_title(title, simulated))
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=9)

    if has_acceleration:
        acceleration = sign * table["angular_acceleration_rad_s2"].to_numpy(dtype=float)
        acceleration_axis.plot(time_s, acceleration, "-", linewidth=1.2, color="#1f3f73")
        acceleration_axis.axhline(0.0, color="black", linewidth=0.8)
        acceleration_axis.set_xlabel("Time (s)")
        acceleration_axis.set_ylabel(r"Angular acceleration (rad/s$^2$)")
        acceleration_axis.grid(True, alpha=0.3)
    else:
        axis.set_xlabel("Time (s)")
    return figure


def plot_speed_fluctuation(table, simulated: bool = False):
    """Speed fluctuation: the speed minus its trend, against time.

    The thin line is the cleaned single frame speed minus the trend. The thick line is the
    smoothed version. Positive means faster than the trend (in the direction of rotation).
    The numbers include measurement noise, which is the scatter of the thin line.
    """
    time_s = table["time_s"].to_numpy(dtype=float)
    sign = _direction_sign(table["rpm_clean"].to_numpy(dtype=float))
    fluctuation = sign * table["rpm_fluctuation"].to_numpy(dtype=float)
    smooth_fluctuation = sign * (table["rpm_smooth"].to_numpy(dtype=float) - table["rpm_trend"].to_numpy(dtype=float))

    figure, axis = plt.subplots(figsize=(8, 4.2))
    axis.plot(time_s, fluctuation, "-", linewidth=1.0, color="#7fa6d6", label="speed minus trend, single frames")
    axis.plot(time_s, smooth_fluctuation, "-", linewidth=2.0, color="#1f3f73", label="smoothed")
    axis.axhline(0.0, color="black", linewidth=0.8)

    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Speed fluctuation (rpm)")
    axis.set_title(_title("Angular speed fluctuation about the trend", simulated))
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=9)
    return figure


def plot_spectrum(result, simulated: bool = False):
    """Frequency spectrum of the speed fluctuation: amplitude in rpm against frequency in Hz.

    result is the dictionary from signal_analysis.analyse_speed_signal. The listed peaks are
    marked with their frequency. The dashed horizontal line is the lowest amplitude that is
    listed as a peak. The dotted vertical line is the rotation frequency (once per revolution).
    A peak shows that the speed varies periodically at that frequency. It does not show the cause.
    """
    frequency = result["spectrum_frequency_hz"]
    amplitude = result["spectrum_amplitude_rpm"]

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(frequency, amplitude, "-", linewidth=1.3, color="#1f5fa8", label="amplitude spectrum")

    if not math.isnan(result["peak_threshold_rpm"]):
        axis.axhline(
            result["peak_threshold_rpm"], linestyle="--", linewidth=1.0, color="#7a7a7a",
            label=f"lowest listed peak ({result['peak_threshold_rpm']:.2f} rpm)",
        )
    rotation = result["rotation_frequency_hz"]
    if 0 < rotation < result["nyquist_hz"]:
        axis.axvline(rotation, linestyle=":", linewidth=1.2, color="black", label=f"rotation frequency ({rotation:.2f} Hz)")

    for peak in result["peaks"]:
        axis.plot(peak["frequency_hz"], peak["amplitude_rpm"], "v", color=QUALITY_PLOT_COLOURS["Poor"], markersize=7)
        axis.annotate(
            f"{peak['frequency_hz']:.2f} Hz", (peak["frequency_hz"], peak["amplitude_rpm"]),
            textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9,
        )
    if result["peaks"]:
        axis.plot([], [], "v", color=QUALITY_PLOT_COLOURS["Poor"], label="listed peaks")

    axis.set_xlim(0, result["nyquist_hz"])
    top = float(np.nanmax(amplitude)) if len(amplitude) else 1.0
    if not math.isnan(result["peak_threshold_rpm"]):
        top = max(top, 2.0 * result["peak_threshold_rpm"])
    axis.set_ylim(0, 1.2 * top)  # room above the highest peak for its label
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Amplitude (rpm)")
    axis.set_title(_title("Frequency spectrum of the speed fluctuation", simulated))
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right", fontsize=9)
    if not result["steady"]:
        axis.text(
            0.02, 0.97, "The speed is not steady, so the spectrum is not reliable.",
            transform=axis.transAxes, va="top", fontsize=9, color=QUALITY_PLOT_COLOURS["Poor"],
        )
    return figure


# Plot colours of the two pulleys: a light one for single frame values, a dark one for smoothed values.
PULLEY_PLOT_COLOURS = {"driver": ("#9ec1e8", "#1f3f73"), "driven": ("#f3c28a", "#b35a00")}


def plot_two_pulley_speeds(merged, expected_ratio=None, simulated: bool = False):
    """Speed of the driver and the driven pulley (top) and their speed ratio (bottom), against time.

    merged is the table made by belt_drive.merge_pulley_tables and add_belt_columns. Each speed is drawn as a
    positive number in the direction that pulley turns. The light lines are single frame values and
    the dark lines the smoothed speed (when the table has it). The ratio is driver speed / driven speed.
    If expected_ratio is given it is drawn as a dashed line.
    """
    time_s = merged["time_s"].to_numpy(dtype=float)
    figure, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(8, 6.5), gridspec_kw={"height_ratios": [2, 1]})

    for name in ("driver", "driven"):
        light, dark = PULLEY_PLOT_COLOURS[name]
        raw = merged[f"{name}_rpm"].to_numpy(dtype=float)
        sign = _direction_sign(raw)
        smooth_name = f"{name}_rpm_smooth"
        has_smooth = smooth_name in merged.columns and merged[smooth_name].notna().any()
        top.plot(time_s, sign * raw, "-", linewidth=1.0, color=light, label=None if has_smooth else name)
        if has_smooth:
            top.plot(time_s, sign * merged[smooth_name].to_numpy(dtype=float), "-", linewidth=2.0, color=dark, label=name)
    top.set_ylabel("Rotation speed (rpm)")
    top.set_title(_title("Driver and driven pulley speeds", simulated))
    top.grid(True, alpha=0.3)
    top.legend(loc="best", fontsize=9)

    ratio = merged["speed_ratio"].to_numpy(dtype=float)
    finite = ratio[np.isfinite(ratio)]
    opposite = bool(len(finite) and np.median(finite) < 0)
    magnitude = np.abs(ratio)
    bottom.plot(time_s, magnitude, "-", linewidth=1.5, color="#444444", label="measured")
    if expected_ratio is not None:
        bottom.axhline(expected_ratio, linestyle="--", linewidth=1.2, color="black", label=f"expected {expected_ratio:.3f}")
    if len(finite) > 2:
        low, high = np.percentile(np.abs(finite), [1, 99])
        if expected_ratio is not None:
            low, high = min(low, expected_ratio), max(high, expected_ratio)
        padding = max(0.25 * (high - low), 0.05 * float(np.median(np.abs(finite))))
        bottom.set_ylim(low - padding, high + padding)  # a few wild values must not flatten the curve
    bottom.set_xlabel("Time (s)")
    bottom.set_ylabel("Speed ratio, driver / driven" + (" (opposite directions)" if opposite else ""))
    bottom.grid(True, alpha=0.3)
    bottom.legend(loc="best", fontsize=9)
    return figure


def plot_surface_speeds(merged, simulated: bool = False):
    """Surface speed (omega x radius, in m/s) of the two pulleys against time.

    For a belt that does not slip or stretch the two lines are the same, and also equal to the speed of
    the belt. The surface speed needs the effective radius of each pulley.
    """
    for name in ("driver", "driven"):
        if f"{name}_surface_speed_m_s" not in merged.columns:
            raise ValueError("The surface speeds need the radius of both pulleys (see belt_drive.add_belt_columns).")

    time_s = merged["time_s"].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for name in ("driver", "driven"):
        axis.plot(time_s, merged[f"{name}_surface_speed_m_s"].to_numpy(dtype=float), "-", linewidth=1.8,
                  color=PULLEY_PLOT_COLOURS[name][1], label=name)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Surface speed (m/s)")
    axis.set_title(_title("Surface speed of the two pulleys (omega x radius)", simulated))
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=9)
    return figure


def plot_belt_speed(merged, simulated: bool = False):
    """Belt speed (from the mark on the belt) and the surface speeds of the two pulleys, in m/s, against time.

    The belt speed is only known during the passes of the mark along a straight span, so the thick black
    line comes in pieces. For a belt that does not slip the three speeds are equal.
    """
    for column in ("belt_speed_m_s", "driver_surface_speed_m_s", "driven_surface_speed_m_s"):
        if column not in merged.columns:
            raise ValueError("The belt speed plot needs the belt speed and the surface speeds of both pulleys.")

    time_s = merged["time_s"].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for name in ("driver", "driven"):
        axis.plot(time_s, merged[f"{name}_surface_speed_m_s"].to_numpy(dtype=float), "-", linewidth=1.5,
                  color=PULLEY_PLOT_COLOURS[name][1], label=f"{name} surface")
    axis.plot(time_s, merged["belt_speed_m_s"].to_numpy(dtype=float), "-", linewidth=4.0, color="black", alpha=0.75,
              label="belt mark, straight spans")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Speed (m/s)")
    axis.set_title(_title("Belt speed and pulley surface speeds", simulated))
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=9)
    return figure


def plot_slip(merged, simulated: bool = False):
    """Slip of the belt on the driver pulley and on the driven pulley, in percent, against time.

    Slip on the driver = (v driver - v belt) / v driver. Slip on the driven pulley = (v belt - v driven) / v belt.
    Positive means the following part is slower. Only frames that belong to a pass of the belt mark have a value.
    """
    for column in ("driver_slip_percent", "driven_slip_percent"):
        if column not in merged.columns:
            raise ValueError("The slip plot needs the slip columns (see slip_analysis.add_belt_speed_columns).")

    time_s = merged["time_s"].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(time_s, merged["driver_slip_percent"].to_numpy(dtype=float), "o", markersize=3, color=PULLEY_PLOT_COLOURS["driver"][1],
              label="slip on the driver pulley")
    axis.plot(time_s, merged["driven_slip_percent"].to_numpy(dtype=float), "o", markersize=3, color=PULLEY_PLOT_COLOURS["driven"][1],
              label="slip on the driven pulley")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Slip (percent)")
    axis.set_title(_title("Slip of the belt on the two pulleys", simulated))
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=9)
    return figure


def plot_validation(rows, simulated: bool = True):
    """Errors of a validation run.

    Left: the error of the mean speed in percent, on a scale that stretches small errors so they can be read.
    Right: the error of the speed loss and of the slip in percentage points, grouped by test.
    rows are the dictionaries made by validation.compare_value.
    """
    import re

    speed = [row for row in rows if row["quantity"] == "Mean speed" and row["percent_error"] is not None and row["passed"] is not None]
    belt = [row for row in rows if row["unit"] == "percentage points" and row["group"] == "Belt drive"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})

    positions = list(range(len(speed)))
    ordered = list(reversed(speed))
    values = [row["percent_error"] for row in ordered]
    left.hlines(positions, 0.0, values, color="#2e7fb8", linewidth=2)
    left.scatter(values, positions, color="#2e7fb8", zorder=3, s=36)
    for position, value in zip(positions, values):
        left.text(0.0015 if value >= 0 else -0.0015, position + 0.28, f"{value:+.3f} %", ha="left" if value >= 0 else "right", fontsize=8)
    left.set_yticks(positions)
    left.set_yticklabels([row["test"] for row in ordered], fontsize=8)
    left.set_xscale("symlog", linthresh=1e-3, linscale=0.6)
    left.set_xlim(-1.0, 1.0)
    left.set_xticks([-0.5, -0.1, -0.01, 0.0, 0.01, 0.1, 0.5])
    left.set_xticklabels(["-0.5", "-0.1", "-0.01", "0", "0.01", "0.1", "0.5"])
    if speed:
        limit = max(float(row["tolerance"].split()[0]) for row in speed)
        left.axvspan(-limit, limit, color="grey", alpha=0.18, label=f"tolerance +/- {limit:g} %")
        left.legend(loc="lower right", fontsize=8)
    left.axvline(0.0, color="black", linewidth=0.8)
    left.set_xlabel("Measured minus true speed (percent, log-like scale)")
    left.set_title(_title("Mean speed against the true speed", simulated))
    left.grid(True, axis="x", alpha=0.3)

    quantities = [("Speed loss", "#7b5ea7"), ("Slip on the driver", PULLEY_PLOT_COLOURS["driver"][1]),
                  ("Slip on the driven pulley", PULLEY_PLOT_COLOURS["driven"][1])]
    tests = list(dict.fromkeys(row["test"] for row in belt))
    width = 0.26
    for offset, (quantity, colour) in zip((-width, 0.0, width), quantities):
        values = []
        for test in tests:
            match = [row for row in belt if row["test"] == test and row["quantity"] == quantity]
            values.append(match[0]["error"] if match else float("nan"))
        right.bar([index + offset for index in range(len(tests))], values, width=width * 0.95, color=colour, label=quantity)
    right.set_xticks(range(len(tests)))
    right.set_xticklabels([re.sub(r"^ratio \S+, ", "", test).replace(", ", "\n") for test in tests], fontsize=8)
    limits = [float(row["tolerance"].split()[0]) for row in belt if row["passed"] is not None]
    if limits:
        right.axhspan(-max(limits), max(limits), color="grey", alpha=0.2, label=f"tolerance +/- {max(limits):g} points")
    right.axhline(0.0, color="black", linewidth=0.8)
    right.set_ylabel("Measured minus true (percentage points)")
    right.set_title(_title("Speed loss and slip against the true values", simulated))
    right.legend(loc="lower right", fontsize=8)
    right.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    return figure
