"""Helpers for app.py. They do not need Streamlit, so they can be tested without a browser.

app.py only reads the widgets and shows the results. Everything that calculates, draws or builds a file is here
or in the modules that this file uses.
"""

import io
import tempfile
import time
import zipfile
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from src import report
from src.pipeline import (
    PULLEYS,
    analyse_belt_drive,
    analyse_pulley,
    make_belt_plots,
    make_plots,
)
from src.signal_analysis import SignalSettings
from src.tracking import (
    annotate_layers,
    colour_range_from_patch,
    describe_settings,
    draw_pulley_region,
    make_mask_preview,
    measure_blob_at,
    suggest_area_limits,
    write_annotated_video_layers,
)
from src.utils import draw_pixel_grid, looks_simulated
from src.video_io import check_video_info, get_video_info, read_frame

VIDEO_TYPES = ("mp4", "mov", "avi", "mkv", "m4v")

PULLEY_LABELS = {"pulley": "Pulley", "driver": "Driver pulley", "driven": "Driven pulley"}


# ---------------------------------------------------------------------------
# The video
# ---------------------------------------------------------------------------


def list_sample_videos(directory) -> list:
    """The video files in a folder (for example sample_data), sorted by name."""
    folder = Path(directory)
    if not folder.is_dir():
        return []
    return sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower().lstrip(".") in VIDEO_TYPES
    )


def save_uploaded_video(name: str, data: bytes, directory=None) -> Path:
    """Write an uploaded video to a folder on this computer and return its path. Nothing leaves the computer."""
    suffix = Path(name).suffix.lower().lstrip(".")
    if suffix not in VIDEO_TYPES:
        raise ValueError(f"'{name}' does not look like a video file. Use one of: {', '.join(VIDEO_TYPES)}.")
    if not data:
        raise ValueError(f"The uploaded file '{name}' is empty.")
    folder = Path(directory) if directory else Path(tempfile.mkdtemp(prefix="smartbelt_"))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / Path(name).name
    path.write_bytes(data)
    return path


def video_info(video_path, fps_override=None):
    """Basic facts about the video: size, frame rate, number of frames and duration."""
    return get_video_info(video_path, fps_override=fps_override)


def video_rows(info, name: str) -> list:
    """The facts about the video as (label, value) rows."""
    return [
        ("File", name),
        ("Picture size", f"{info.width} x {info.height} pixels"),
        ("Frame rate", f"{info.fps:g} fps ({info.fps_source})"),
        ("Frames", f"{info.frame_count}"),
        ("Duration", f"{info.duration_s:.2f} s"),
    ]


def video_warnings(info) -> list:
    return check_video_info(info)


# ---------------------------------------------------------------------------
# Pictures for checking the settings
# ---------------------------------------------------------------------------


def to_rgb(picture):
    return cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)


def frame_preview(video_path, frame_number: int, markers=(), grid_step=None):
    """One frame as an RGB picture, with an optional grid and the rotation centres and search regions.

    markers is a list of (centre, region radius) pairs. A centre that is None is skipped.
    """
    frame = read_frame(video_path, frame_number)
    picture = draw_pixel_grid(frame, grid_step) if grid_step else frame.copy()
    for centre, region in markers:
        if centre is not None:
            draw_pulley_region(picture, centre, region)
    return to_rgb(picture)


def mask_preview(video_path, frame_number: int, settings, centre=None, region_radius=None):
    """The picture for checking the colour settings of one marker: (RGB picture, MarkerDetection or None)."""
    frame = read_frame(video_path, frame_number)
    picture, detection = make_mask_preview(frame, settings, centre, region_radius)
    return to_rgb(picture), detection


def settings_from_pixel(video_path, frame_number: int, x: float, y: float):
    """Colour settings from the pixels around a point on the marker: (settings, detection or None).

    The area limits are set from the size of the blob at that point, as pick_points.py does.
    """
    frame = read_frame(video_path, frame_number)
    settings = colour_range_from_patch(frame, x, y)
    detection = measure_blob_at(frame, settings, x, y)
    if detection is not None:
        smallest, largest = suggest_area_limits(detection.area_px)
        settings = replace(settings, min_area_px=float(smallest), max_area_px=float(largest))
    return settings, detection


# ---------------------------------------------------------------------------
# Running the analysis
# ---------------------------------------------------------------------------


@dataclass
class PulleyInput:
    """What is known about one pulley: the marker colour, the rotation centre and the sizes."""

    settings: object  # TrackingSettings
    centre: tuple  # (x, y) in pixels
    region_radius: float = None  # search region in pixels
    radius_mm: float = None  # effective radius of the pulley
    marker_radius_mm: float = None  # distance from the centre to the marker


@dataclass
class RunConfig:
    video_path: str
    kind: str  # "single" or "belt"
    pulleys: dict  # "pulley" for a single pulley, "driver" and "driven" for a belt drive: PulleyInput
    signal_settings: SignalSettings = field(default_factory=SignalSettings)
    fps_override: float = None
    start_frame: int = 0
    max_frames: int = None
    use_fitted_centres: bool = False
    belt_settings: object = None  # TrackingSettings of a mark on the belt, or None
    scale_mm_per_px: float = None
    belt_speed_factor: float = 1.0
    reference_ratio: float = None


@dataclass
class AppResult:
    """Everything the app shows and offers for download."""

    config: RunConfig
    video_name: str
    simulated: bool
    info: object
    table: pd.DataFrame  # the table that is saved as CSV
    pulleys: dict  # name: PulleyResult
    outcome: object = None  # BeltDriveResult of a belt drive
    figures: dict = field(default_factory=dict)  # name: list of (end of file name, caption, PNG bytes)
    seconds: float = 0.0


def figure_png(figure, dpi: int = 150) -> bytes:
    """The picture of a matplotlib figure as PNG bytes. The figure is closed afterwards."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def _png_list(plots) -> list:
    return [(suffix, caption, figure_png(figure)) for suffix, caption, figure in plots]


def run_analysis(config: RunConfig, progress_factory=None) -> AppResult:
    """Run the analysis that the settings ask for. Errors of the analysis are raised with plain messages."""
    started = time.perf_counter()
    path = Path(config.video_path)
    simulated = looks_simulated(path)

    if config.kind == "single":
        item = config.pulleys["pulley"]
        result = analyse_pulley(
            path, item.settings, centre=item.centre, region_radius_px=item.region_radius,
            signal_settings=config.signal_settings, fps_override=config.fps_override, start_frame=config.start_frame,
            max_frames=config.max_frames, use_fitted_centre=config.use_fitted_centres,
            progress=progress_factory("Tracking the marker") if progress_factory else None,
        )
        mean_rpm = result.speed["mean_rpm"] if result.speed else None
        return AppResult(
            config=config, video_name=path.name, simulated=simulated, info=result.info, table=result.table,
            pulleys={"pulley": result},
            figures={"pulley": _png_list(make_plots(result.table, result.used_centre, simulated, mean_rpm, result.signal))},
            seconds=time.perf_counter() - started,
        )

    items = {name: config.pulleys[name] for name in PULLEYS}
    outcome = analyse_belt_drive(
        path,
        {name: item.settings for name, item in items.items()},
        {name: item.centre for name, item in items.items()},
        {name: item.region_radius for name, item in items.items()},
        {name: item.radius_mm for name, item in items.items()},
        {name: item.marker_radius_mm for name, item in items.items()},
        belt_settings=config.belt_settings, scale_mm_per_px=config.scale_mm_per_px,
        belt_speed_factor=config.belt_speed_factor, reference_ratio=config.reference_ratio,
        signal_settings=config.signal_settings, fps_override=config.fps_override, start_frame=config.start_frame,
        max_frames=config.max_frames, use_fitted_centres=config.use_fitted_centres, progress_factory=progress_factory,
    )
    figures = {}
    for name in PULLEYS:
        pulley = outcome.pulleys[name]
        mean_rpm = pulley.speed["mean_rpm"] if pulley.speed else None
        figures[name] = _png_list(make_plots(pulley.table, pulley.used_centre, simulated, mean_rpm, pulley.signal))
    figures["belt"] = _png_list(make_belt_plots(outcome.merged, outcome.belt, outcome.slip, outcome.expected_ratio, simulated))
    return AppResult(
        config=config, video_name=path.name, simulated=simulated, info=outcome.info, table=outcome.merged,
        pulleys=outcome.pulleys, outcome=outcome, figures=figures, seconds=time.perf_counter() - started,
    )


# ---------------------------------------------------------------------------
# What the results show
# ---------------------------------------------------------------------------


def pulley_names(result: AppResult) -> list:
    return list(result.pulleys.keys())


def result_notes(result: AppResult) -> list:
    """Short messages for the user: notes of the pulleys, why something was not calculated, and the slip message."""
    messages = []
    for name, pulley in result.pulleys.items():
        prefix = "" if result.config.kind == "single" else f"{PULLEY_LABELS[name]}: "
        for note in pulley.notes:
            messages.append(prefix + note)
        for text in (pulley.analysis_problem, pulley.signal_problem):
            if text:
                messages.append(prefix + text)
    if result.outcome is not None:
        if result.outcome.belt_problem:
            messages.append(f"Belt drive: {result.outcome.belt_problem}")
        if result.outcome.slip_message:
            messages.append(result.outcome.slip_message)
    return messages


def settings_rows(result: AppResult) -> list:
    """The settings that were used, as (label, value) rows. They go into the report."""
    config = result.config
    rows = [("Analysis", "Single pulley" if config.kind == "single" else "Two-pulley belt drive")]
    rows.append(("Frame rate", f"{result.info.fps:g} fps ({result.info.fps_source})"))
    rows.append(("Frames used", f"from frame {config.start_frame}" + (f", at most {config.max_frames} frames" if config.max_frames else ", to the end")))
    for name, item in config.pulleys.items():
        label = PULLEY_LABELS[name]
        rows.append((f"{label}: marker colour", describe_settings(item.settings)))
        place = f"({item.centre[0]:g}, {item.centre[1]:g}) pixels"
        if item.region_radius is not None:
            place += f", search region {item.region_radius:g} pixels"
        rows.append((f"{label}: rotation centre", place))
        if item.radius_mm is not None:
            rows.append((f"{label}: effective radius", f"{item.radius_mm:g} mm"))
        if item.marker_radius_mm is not None:
            rows.append((f"{label}: marker distance from centre", f"{item.marker_radius_mm:g} mm"))
    if config.use_fitted_centres:
        rows.append(("Rotation centre used", "from the circle fit of the marker path"))
    if config.belt_settings is not None:
        rows.append(("Belt mark: colour", describe_settings(config.belt_settings)))
    if config.scale_mm_per_px is not None:
        rows.append(("Scale", f"{config.scale_mm_per_px:g} mm per pixel (typed in)"))
    if config.belt_speed_factor != 1.0:
        rows.append(("Belt speed factor", f"{config.belt_speed_factor:g}"))
    if config.reference_ratio is not None:
        rows.append(("Reference ratio", f"{config.reference_ratio:g} (typed in)"))
    signal = config.signal_settings
    rows.append(("Smoothing", f"Savitzky-Golay, window {signal.smooth_window_s:g} s, polynomial order {signal.smooth_polyorder}"))
    rows.append(("Outlier removal", "on" if signal.remove_outliers else "off"))
    rows.append(("Trend removed", f"polynomial of order {signal.trend_order}"))
    rows.append(("Lost frames filled in", f"up to {signal.max_gap_frames} in a row"))
    rows.append(("Peaks listed", f"up to {signal.max_peaks}, at least {signal.min_peak_percent:g} percent of the mean speed"))
    return rows


# ---------------------------------------------------------------------------
# The annotated frame, the tracked video and the downloads
# ---------------------------------------------------------------------------


def _layers(result: AppResult) -> list:
    layers = []
    for name, pulley in result.pulleys.items():
        item = result.config.pulleys[name]
        layers.append({"name": PULLEY_LABELS[name].lower(), "table": pulley.table, "centre": pulley.used_centre,
                       "region_radius_px": item.region_radius, "extra_text": None})
    if result.outcome is not None and result.outcome.belt_table is not None:
        layers.append({"name": "belt", "table": result.outcome.belt_table, "centre": None, "region_radius_px": None, "extra_text": None})
    return layers


def frame_count_of(result: AppResult) -> int:
    return len(result.table)


def annotated_frame(result: AppResult, position: int, trail_length: int = 40):
    """The frame at a position in the table (0 is the first frame used) with the tracked markers drawn on it.

    Returns (RGB picture, frame number, time in seconds).
    """
    layers = _layers(result)
    position = int(min(max(position, 0), len(result.table) - 1))
    first = layers[0]["table"].iloc[position]
    frame_number = int(first["frame"])
    states = []
    for layer in layers:
        table = layer["table"]
        row = table.iloc[position]
        quality = str(row["tracking_quality"])
        marker = None if quality == "Lost" else (float(row["marker_x"]), float(row["marker_y"]), float(row["area_px"]))
        recent = table.iloc[max(0, position - trail_length + 1): position + 1]
        trail = [
            (float(r["marker_x"]), float(r["marker_y"]), str(r["tracking_quality"]))
            for _, r in recent.iterrows() if str(r["tracking_quality"]) != "Lost"
        ]
        states.append({"name": layer["name"], "marker": marker, "quality": quality, "centre": layer["centre"],
                       "region_radius_px": layer["region_radius_px"], "trail": trail, "text": ""})
    picture = annotate_layers(read_frame(result.info.path, frame_number), frame_number, float(first["time_s"]), states)
    return to_rgb(picture), frame_number, float(first["time_s"])


def write_tracked_video(result: AppResult, out_path) -> Path:
    """Write the video with the tracked markers drawn on it. It is an mp4v file: play it with a desktop player, not a browser."""
    layers = _layers(result)
    return write_annotated_video_layers(
        result.info.path, layers, out_path, result.info.fps, start_frame=result.config.start_frame,
    )


def csv_bytes(result: AppResult) -> bytes:
    return result.table.to_csv(index=False).encode("utf-8")


def file_stem(result: AppResult) -> str:
    return Path(result.video_name).stem


def csv_name(result: AppResult) -> str:
    return f"{file_stem(result)}_analysis.csv" if result.config.kind == "single" else f"{file_stem(result)}_belt_analysis.csv"


def plot_files(result: AppResult) -> list:
    """All plots as (file name, PNG bytes)."""
    files = []
    for name, plots in result.figures.items():
        for suffix, _, png in plots:
            middle = "" if result.config.kind == "single" else f"_{name}" if name in PULLEYS else ""
            files.append((f"{file_stem(result)}{middle}_{suffix}.png", png))
    return files


def plots_zip(result: AppResult) -> bytes:
    """A zip file with every plot as a PNG picture."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, png in plot_files(result):
            archive.writestr(name, png)
    return buffer.getvalue()


def report_html(result: AppResult, notes: str = "") -> str:
    """The HTML report: video information, settings, results, graphs, definitions and limitations."""
    blocks = []
    for name, pulley in result.pulleys.items():
        item = result.config.pulleys[name]
        block = report.ReportBlock(
            title=("Results" if result.config.kind == "single" else f"Results: {PULLEY_LABELS[name]}"),
            sections=report.pulley_sections(pulley, item.settings, item.region_radius, result.simulated),
            peaks=report.peak_table(pulley.signal),
            figures=[(caption, png) for _, caption, png in result.figures.get(name, [])],
        )
        blocks.append(block)
    if result.outcome is not None:
        sections = []
        if result.outcome.belt is not None:
            sections.append(("Comparison of the two pulleys", report.belt_rows(result.outcome.belt)))
        elif result.outcome.belt_problem:
            sections.append(("Comparison of the two pulleys", [("Belt drive", f"not calculated: {result.outcome.belt_problem}")]))
        if result.config.belt_settings is not None:
            sections.append(("Belt slip", report.slip_rows(result.outcome.slip, result.outcome.slip_message)))
        blocks.append(
            report.ReportBlock(
                title="Belt drive", sections=sections,
                figures=[(caption, png) for _, caption, png in result.figures.get("belt", [])],
            )
        )
    data = report.ReportData(
        title="SmartBelt Vision: analysis report",
        simulated=result.simulated,
        video_rows=video_rows(result.info, result.video_name),
        settings_rows=settings_rows(result),
        blocks=blocks,
        notes=notes,
        generated=f"Made on {datetime.now().strftime('%Y-%m-%d at %H:%M')}.",
    )
    return report.build_html_report(data)


def engineering_summary(result: AppResult) -> list:
    """A few plain sentences about what was measured. They only repeat the numbers. They never name a cause or a fault."""
    sentences = []
    single = result.config.kind == "single"
    for name, pulley in result.pulleys.items():
        who = "The pulley" if single else PULLEY_LABELS[name]
        tracking = pulley.tracking
        sentences.append(
            f"{who}: the marker was found in {tracking['found']} of {tracking['frames']} frames and the tracking quality is "
            f"{tracking['overall_quality']}."
        )
        speed = pulley.speed
        if speed is None:
            sentences.append(f"{who}: no speed could be calculated.")
            continue
        sentences.append(
            f"{who} turned {speed['direction']} at a mean speed of {speed['mean_rpm']:.2f} rpm. Single frame values ranged from "
            f"{speed['min_rpm']:.1f} to {speed['max_rpm']:.1f} rpm (standard deviation {speed['variation_percent']:.2f} percent of the "
            "mean, which includes measurement noise)."
        )
        signal = pulley.signal
        if signal is None:
            continue
        if not signal["steady"]:
            sentences.append(f"{who}: the speed is not steady during the clip, so the frequency analysis is not reliable.")
        elif signal["peaks"]:
            biggest = max(signal["peaks"], key=lambda peak: peak["amplitude_rpm"])
            sentences.append(
                f"{who}: the speed varies periodically at {biggest['frequency_hz']:.2f} Hz ({biggest['amplitude_rpm']:.2f} rpm, "
                f"{biggest['percent_of_mean']:.2f} percent of the mean). This shows that the speed varies at that frequency, not what causes it."
            )
        else:
            sentences.append(f"{who}: no dominant frequency was found above {signal['peak_threshold_rpm']:.2f} rpm.")

    outcome = result.outcome
    if outcome is not None:
        belt = outcome.belt
        if belt is not None:
            text = f"The measured speed ratio is {belt['measured_ratio']:.3f} driver turns per driven turn"
            if belt["expected_ratio"] is not None:
                direction = "slower" if belt["speed_loss_percent"] > 0 else "faster"
                text += (
                    f", against an expected ratio of {belt['expected_ratio']:.3f}. The driven pulley is "
                    f"{abs(belt['speed_loss_percent']):.2f} percent {direction} than expected."
                )
            else:
                text += ". The expected ratio is not known (no effective radii or reference ratio were given)."
            sentences.append(text)
        slip = outcome.slip
        if slip is not None and slip["available"]:
            sentences.append(
                f"From the mark on the belt, the slip is {slip['driver_slip_percent']:+.2f} percent on the driver pulley and "
                f"{slip['driven_slip_percent']:+.2f} percent on the driven pulley. Slip is a small difference of speeds, so it is only "
                "as good as the scale and the effective radii."
            )
        elif outcome.slip_message:
            sentences.append(outcome.slip_message)
    return sentences
