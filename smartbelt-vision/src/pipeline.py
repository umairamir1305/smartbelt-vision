"""The analysis of one pulley, from the video to the numbers, without printing or saving anything.

analyse_pulley does what the single pulley script does in order: track the marker, check the rotation
centre with a circle fit, calculate the angle, the speed and the signal analysis. It returns everything
in a PulleyResult. The command line scripts (and later the app) print and save the results.

This file also holds the helpers that save the plots and the spectrum files and the text drawn on the video.
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.belt_drive import (
    add_belt_columns,
    expected_ratio_from,
    merge_pulley_tables,
    summarise_belt_drive,
)
from src.kinematics import (
    AnalysisError,
    add_angle_columns,
    add_speed_columns,
    fit_centre_from_track,
    summarise_rotation,
    summarise_speed,
)
from src.reporting import (
    plot_angle_vs_time,
    plot_belt_speed,
    plot_marker_trajectory,
    plot_rpm_vs_time,
    plot_slip,
    plot_speed_fluctuation,
    plot_spectrum,
    plot_surface_speeds,
    plot_tracking_timeline,
    plot_two_pulley_speeds,
    save_figure,
)
from src.signal_analysis import SignalSettings, analyse_speed_signal, validate_signal_settings
from src.slip_analysis import (
    add_belt_speed_columns,
    combine_scales,
    find_belt_passes,
    mark_on_straight_span,
    scale_from_marker_radius,
    summarise_slip,
)
from src.tracking import (
    TrackingError,
    add_tracking_quality,
    summarise_tracking,
    track_marker,
    validate_settings,
)
from src.utils import OUTPUT_DIR, ensure_folder
from src.video_io import get_video_info


PULLEYS = ("driver", "driven")

# The two rotation centres of a belt drive must be at least this far apart (in pixels).
MIN_CENTRE_DISTANCE_PX = 10.0


@dataclass
class PulleyResult:
    """Everything that analyse_pulley found out about one pulley."""

    info: object  # VideoInfo of the video
    table: pd.DataFrame  # one row per frame: marker, quality, angle, speed, signal analysis columns
    centre: tuple  # the rotation centre that was given (or None)
    used_centre: tuple  # the centre the angle was calculated about (given, or from the circle fit)
    fit: dict  # circle fit result (or None without a centre)
    tracking: dict  # summary of the tracking (summarise_tracking)
    rotation: dict  # summary of the angle (or None)
    speed: dict  # summary of the speed (or None)
    signal: dict  # result of the signal analysis (or None)
    notes: list = field(default_factory=list)  # short messages for the user
    signal_problem: str = None  # why the signal analysis was not done (not an error)
    analysis_problem: str = None  # why the angle or the speed could not be calculated
    tracking_seconds: float = 0.0


def analyse_pulley(
    video_path,
    settings,
    centre=None,
    region_radius_px=None,
    signal_settings: SignalSettings = None,
    fps_override=None,
    start_frame: int = 0,
    max_frames=None,
    use_fitted_centre: bool = False,
    progress=None,
) -> PulleyResult:
    """Track one marker and analyse its rotation. See the top of this file.

    Without a centre only the tracking is done. A problem in the angle or speed calculation does not
    raise an error: it is stored in analysis_problem (and a problem in the signal analysis in
    signal_problem), so that the tracking results can still be saved. The caller decides what to do.
    """
    signal_settings = signal_settings or SignalSettings()
    validate_settings(settings)
    validate_signal_settings(signal_settings)
    if use_fitted_centre and centre is None:
        raise TrackingError("Using the fitted centre needs a rotation centre to start from.")

    info = get_video_info(video_path, fps_override=fps_override)
    notes = []

    started = time.time()
    table = track_marker(
        video_path,
        settings,
        centre=centre,
        region_radius_px=region_radius_px,
        fps_override=fps_override,
        start_frame=start_frame,
        max_frames=max_frames,
        progress=progress,
    )
    tracking_seconds = time.time() - started

    # Which centre is used for the angle: the one given, or the one found by the circle fit.
    used_centre = centre
    fit = None
    if centre is not None:
        fit = fit_centre_from_track(table)
        if use_fitted_centre:
            if fit["usable"]:
                used_centre = fit["centre"]
                table = add_tracking_quality(table, used_centre)  # rate the frames again with the better centre
            else:
                notes.append(f"The fitted centre was not used: {fit['reason']}.")

    rotation = None
    speed = None
    signal_result = None
    signal_problem = None
    analysis_problem = None
    if used_centre is not None:
        try:
            table = add_angle_columns(table, used_centre)
            rotation = summarise_rotation(table)
            table = add_speed_columns(table)
            speed = summarise_speed(table)
        except AnalysisError as error:
            analysis_problem = str(error)

    if speed is not None:
        try:
            table, signal_result = analyse_speed_signal(table, signal_settings)
        except AnalysisError as error:
            signal_problem = str(error)  # not fatal: the angle and the speed are still there

    return PulleyResult(
        info=info,
        table=table,
        centre=centre,
        used_centre=used_centre,
        fit=fit,
        tracking=summarise_tracking(table, used_centre),
        rotation=rotation,
        speed=speed,
        signal=signal_result,
        notes=notes,
        signal_problem=signal_problem,
        analysis_problem=analysis_problem,
        tracking_seconds=tracking_seconds,
    )


def video_labels(table) -> list:
    """Text for the tracked video: the marker angle (0 to 360 degrees, counter-clockwise from 3 o'clock)
    and, if the table has it, the speed in rpm (negative means clockwise). The smoothed speed is used
    when there is one, otherwise the speed of the single frame."""
    angles = table["angle_wrapped_rad"].to_numpy(dtype=float)
    if "rpm" in table.columns:
        speeds = table["rpm"].to_numpy(dtype=float)
        if "rpm_smooth" in table.columns:
            smooth = table["rpm_smooth"].to_numpy(dtype=float)  # smoothed, so the numbers do not flicker
            speeds = [s if not math.isnan(s) else r for s, r in zip(smooth, speeds)]  # single frame at the ends
    else:
        speeds = [math.nan] * len(table)

    labels = []
    for angle, speed in zip(angles, speeds):
        text = "" if math.isnan(angle) else f"angle {math.degrees(angle) % 360:.1f} deg"
        if not math.isnan(speed):
            text += f"   speed {speed:.0f} rpm"
        labels.append(text)
    return labels


def make_plots(table, centre, simulated: bool, mean_rpm=None, signal_result=None) -> list:
    """The plots that the table allows, as a list of (end of the file name, caption, figure)."""
    plots = [("trajectory", "Marker path in the picture", plot_marker_trajectory(table, centre, simulated))]
    if "angle_rad" in table.columns:
        plots.append(("angle", "Marker angle against time", plot_angle_vs_time(table, simulated)))
    if "rpm" in table.columns:
        plots.append(("rpm", "Speed against time", plot_rpm_vs_time(table, mean_rpm, simulated)))
    if signal_result is not None:
        plots.append(("fluctuation", "Speed fluctuation (smoothed, trend removed)", plot_speed_fluctuation(table, simulated)))
        plots.append(("spectrum", "Frequency spectrum of the speed fluctuation", plot_spectrum(signal_result, simulated)))
    plots.append(("tracking_quality", "Tracking quality in every frame", plot_tracking_timeline(table, simulated)))
    return plots


def save_plots(table, centre, stem: str, simulated: bool, mean_rpm=None, signal_result=None, output_dir=OUTPUT_DIR) -> list:
    """Save the plots that the table allows and return their paths."""
    return [
        save_figure(figure, output_dir / f"{stem}_{suffix}.png")
        for suffix, _, figure in make_plots(table, centre, simulated, mean_rpm, signal_result)
    ]


def make_belt_plots(merged, belt, slip, expected_ratio, simulated: bool) -> list:
    """The plots of a belt drive, as a list of (end of the file name, caption, figure)."""
    plots = []
    if belt is not None:
        plots.append(("belt_speeds", "Speed of both pulleys and the speed ratio", plot_two_pulley_speeds(merged, expected_ratio, simulated)))
        if "driver_surface_speed_m_s" in merged.columns and "driven_surface_speed_m_s" in merged.columns:
            plots.append(("belt_surface_speed", "Surface speed of both pulleys", plot_surface_speeds(merged, simulated)))
    if slip is not None and slip["available"]:
        plots.append(("belt_mark_speed", "Belt speed from the mark and the surface speeds", plot_belt_speed(merged, simulated)))
        plots.append(("belt_slip", "Slip on the driver and on the driven pulley", plot_slip(merged, simulated)))
    return plots


def save_spectrum_files(signal_result, stem: str, output_dir=OUTPUT_DIR) -> list:
    """Save the spectrum and the list of peaks as CSV files and return their paths."""
    ensure_folder(output_dir)
    spectrum = pd.DataFrame(
        {
            "frequency_hz": signal_result["spectrum_frequency_hz"],
            "amplitude_rpm": signal_result["spectrum_amplitude_rpm"],
        }
    )
    spectrum_path = output_dir / f"{stem}_spectrum.csv"
    spectrum.to_csv(spectrum_path, index=False, float_format="%.6f")

    peaks = pd.DataFrame(
        [
            {
                "rank": rank,
                "frequency_hz": peak["frequency_hz"],
                "period_s": peak["period_s"],
                "amplitude_rpm": peak["amplitude_rpm"],
                "percent_of_mean": peak["percent_of_mean"],
                "order": peak["order"],
                "note": peak["note"],
            }
            for rank, peak in enumerate(signal_result["peaks"], start=1)
        ],
        columns=["rank", "frequency_hz", "period_s", "amplitude_rpm", "percent_of_mean", "order", "note"],
    )
    peaks_path = output_dir / f"{stem}_peaks.csv"
    peaks.to_csv(peaks_path, index=False, float_format="%.6f")
    return [spectrum_path, peaks_path]



@dataclass
class BeltDriveResult:
    """Everything that analyse_belt_drive found out about a belt drive with two pulleys."""

    info: object  # VideoInfo of the video
    pulleys: dict  # "driver" and "driven": PulleyResult of each pulley
    merged: pd.DataFrame  # one table with the columns of both pulleys (driver_, driven_), the ratio, the belt mark and the slip
    expected_ratio: float = None
    expected_source: str = None
    belt: dict = None  # comparison of the pulleys (belt_drive.summarise_belt_drive), or None
    belt_problem: str = None  # why the comparison could not be made
    belt_table: pd.DataFrame = None  # tracking table of the mark on the belt, or None
    slip: dict = None  # belt speed and slip (slip_analysis.summarise_slip), or None
    slip_message: str = None  # why the slip is not available (or the reason from the slip summary)


def find_scale(scale_mm_per_px, marker_radii_mm, results):
    """The scale in mm per pixel: typed in, or from the marker distances of the pulleys.

    marker_radii_mm gives the distance in mm from the centre of each pulley to its marker.
    Returns (scale, disagreement of the pulley scales in percent, text that says where it came from).
    """
    if scale_mm_per_px is not None:
        return scale_mm_per_px, 0.0, "typed in"
    scales = {}
    for name in PULLEYS:
        marker_mm = (marker_radii_mm or {}).get(name)
        median_px = results[name].tracking["median_radius_px"]
        if marker_mm is not None and median_px:
            scales[name] = scale_from_marker_radius(marker_mm, median_px)
    scale, disagreement = combine_scales(scales)
    if scale is None:
        return None, 0.0, ""
    return scale, disagreement, "from the distance of the marker from the centre: " + ", ".join(f"{n} {v:.4f}" for n, v in scales.items())


def analyse_belt_drive(
    video_path,
    settings: dict,
    centres: dict,
    region_radii: dict,
    radii_mm: dict,
    marker_radii_mm: dict = None,
    belt_settings=None,
    scale_mm_per_px=None,
    belt_speed_factor: float = 1.0,
    reference_ratio=None,
    signal_settings: SignalSettings = None,
    fps_override=None,
    start_frame: int = 0,
    max_frames=None,
    use_fitted_centres: bool = False,
    progress_factory=None,
) -> BeltDriveResult:
    """Analyse a belt drive: both pulleys, their comparison and, if a belt mark is given, the belt speed and the slip.

    settings, centres, region_radii, radii_mm and marker_radii_mm are dictionaries with the keys "driver" and
    "driven". Values that are not known are None (region_radii and radii_mm may hold None, marker_radii_mm may be
    None). belt_settings are the tracking settings of the mark on the belt (None: no belt tracking).
    progress_factory(label) may return a function that is called as function(done, total) while that marker is tracked.

    A problem in the comparison or in the slip does not raise an error. It is stored (belt_problem, slip_message)
    so that the results of the pulleys can still be used and saved.
    """
    signal_settings = signal_settings or SignalSettings()
    marker_radii_mm = marker_radii_mm or {}
    for name in PULLEYS:
        validate_settings(settings[name])
        if centres.get(name) is None:
            raise TrackingError(f"The {name} pulley needs a rotation centre.")
        region = region_radii.get(name)
        if region is not None and region <= 0:
            raise TrackingError(f"The region radius of the {name} pulley must be larger than 0.")
        for label, values in (("effective radius", radii_mm), ("marker radius", marker_radii_mm)):
            if values.get(name) is not None and values[name] <= 0:
                raise TrackingError(f"The {label} of the {name} pulley must be larger than 0.")
    if math.dist(centres["driver"], centres["driven"]) < MIN_CENTRE_DISTANCE_PX:
        raise TrackingError("The two rotation centres are almost the same point. Each pulley needs its own centre.")
    if belt_settings is not None:
        validate_settings(belt_settings)
    if scale_mm_per_px is not None and scale_mm_per_px <= 0:
        raise TrackingError("The scale must be larger than 0.")
    if belt_speed_factor <= 0:
        raise TrackingError("The belt speed factor must be larger than 0.")
    validate_signal_settings(signal_settings)
    expected_ratio, expected_source = expected_ratio_from(radii_mm.get("driver"), radii_mm.get("driven"), reference_ratio)

    info = get_video_info(video_path, fps_override=fps_override)

    results = {}
    for name in PULLEYS:
        try:
            results[name] = analyse_pulley(
                video_path,
                settings[name],
                centre=centres[name],
                region_radius_px=region_radii.get(name),
                signal_settings=signal_settings,
                fps_override=fps_override,
                start_frame=start_frame,
                max_frames=max_frames,
                use_fitted_centre=use_fitted_centres,
                progress=progress_factory(f"Tracking {name}") if progress_factory else None,
            )
        except TrackingError as error:
            raise TrackingError(f"{name} pulley: {error}") from error

    # Compare the two pulleys.
    merged = merge_pulley_tables(results["driver"].table, results["driven"].table)
    belt = None
    belt_problem = None
    for name in PULLEYS:
        if results[name].speed is None and belt_problem is None:
            belt_problem = f"The speed ratio needs the speed of both pulleys, but the {name} pulley has none. {results[name].analysis_problem or ''}".strip()
    if belt_problem is None:
        try:
            merged = add_belt_columns(merged, expected_ratio, radii_mm.get("driver"), radii_mm.get("driven"))
            belt = summarise_belt_drive(merged, expected_ratio, expected_source, radii_mm.get("driver"), radii_mm.get("driven"))
        except AnalysisError as error:
            belt_problem = str(error)

    # The mark on the belt (only if belt settings were given): belt speed and slip.
    belt_table = None
    slip = None
    slip_message = None
    if belt_settings is not None:
        try:
            belt_table = track_marker(
                video_path, belt_settings, fps_override=fps_override, start_frame=start_frame, max_frames=max_frames,
                progress=progress_factory("Tracking belt mark") if progress_factory else None,
            )
        except TrackingError as error:
            raise TrackingError(f"belt mark: {error}") from error

        scale, scale_disagreement, scale_source = find_scale(scale_mm_per_px, marker_radii_mm, results)
        reasons = []
        if belt is None:
            reasons.append("the speeds of both pulleys are needed first (see the message above)")
        if radii_mm.get("driver") is None or radii_mm.get("driven") is None:
            reasons.append("both effective radii are needed")
        if scale is None:
            reasons.append("a scale is needed (typed in, or from the distance in mm between the centre of a pulley and its marker)")
        if region_radii.get("driver") is None or region_radii.get("driven") is None:
            reasons.append("both region radii are needed to find the straight part of the belt")

        if region_radii.get("driver") is None or region_radii.get("driven") is None:
            on_span = np.zeros(len(belt_table), dtype=bool)
        else:
            on_span = mark_on_straight_span(
                belt_table["marker_x"], belt_table["marker_y"],
                [results[name].used_centre for name in PULLEYS], [region_radii[name] for name in PULLEYS],
            )
        passes = [] if reasons else find_belt_passes(belt_table, on_span)
        merged = add_belt_speed_columns(merged, belt_table, on_span, passes, scale, belt_speed_factor)
        if reasons:
            slip_message = "Slip estimation unavailable: " + "; ".join(reasons) + "."
        else:
            slip = summarise_slip(
                passes, scale, belt["driver_surface_speed_m_s"], belt["driven_surface_speed_m_s"],
                belt_speed_factor, scale_disagreement,
            )
            slip["scale_source"] = scale_source
            if not slip["available"]:
                slip_message = slip["reason"]

    return BeltDriveResult(
        info=info, pulleys=results, merged=merged, expected_ratio=expected_ratio, expected_source=expected_source,
        belt=belt, belt_problem=belt_problem, belt_table=belt_table, slip=slip, slip_message=slip_message,
    )
