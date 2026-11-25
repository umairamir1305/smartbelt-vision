"""SmartBelt Vision: the Streamlit app.

Start it from the project folder with:

    streamlit run app.py

The app only reads the widgets and shows the results. The calculations are in src/ (src/app_support.py,
src/pipeline.py and the modules it uses), the same ones that the command line scripts use. The video is
processed on this computer and is not uploaded anywhere.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src import app_support as support
from src import report
from src.kinematics import AnalysisError
from src.signal_analysis import SignalSettings
from src.tracking import COLOUR_PRESETS, TrackingError, TrackingSettings, settings_for_colour
from src.video_io import VideoError

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = Path(os.environ.get("SMARTBELT_SAMPLE_DIR", PROJECT_ROOT / "sample_data"))

UPLOAD_SOURCE = "Upload a video"
SAMPLE_SOURCE = "Use a file from sample_data"
SINGLE = "Single pulley"
BELT_DRIVE = "Two-pulley belt drive"
ANALYSIS_ERRORS = (VideoError, TrackingError, AnalysisError, ValueError, OSError)

st.set_page_config(page_title="SmartBelt Vision", layout="wide")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def ensure(key: str, default):
    """Give a widget key its first value (so the widget needs no default of its own). Returns the key."""
    if key not in st.session_state:
        st.session_state[key] = default
    return key


def show_rows(rows) -> None:
    """Show (label, value) rows as a table."""
    if rows:
        st.table(pd.DataFrame({"Value": [value for _, value in rows]}, index=[label for label, _ in rows]))


def optional(value: float):
    """A number input where 0 means 'not given'."""
    return value if value and value > 0 else None


def apply_pixel_colour(prefix: str) -> None:
    """Button callback: set the colour limits of a marker from the pixels around a point."""
    path = st.session_state.get("video_path")
    frame_number = int(st.session_state.get("preview_frame", 0))
    try:
        settings, detection = support.settings_from_pixel(
            path, frame_number, st.session_state[f"{prefix}_pick_x"], st.session_state[f"{prefix}_pick_y"]
        )
    except ANALYSIS_ERRORS as error:
        st.session_state[f"{prefix}_pick_message"] = f"Not used: {error}"
        return
    st.session_state[f"{prefix}_manual"] = True
    st.session_state[f"{prefix}_hue_low"] = float(settings.hue_low)
    st.session_state[f"{prefix}_hue_high"] = float(settings.hue_high)
    st.session_state[f"{prefix}_sat_min"] = float(settings.sat_min)
    st.session_state[f"{prefix}_val_min"] = float(settings.val_min)
    st.session_state[f"{prefix}_min_area"] = float(settings.min_area_px)
    st.session_state[f"{prefix}_max_area"] = float(settings.max_area_px)
    found = "a marker blob was measured there" if detection is not None else "no blob was found at that point, the area limits are the defaults"
    st.session_state[f"{prefix}_pick_message"] = (
        f"Colour limits set from frame {frame_number}: hue {settings.hue_low} to {settings.hue_high}, "
        f"saturation from {settings.sat_min}, brightness from {settings.val_min}; {found}."
    )


# ---------------------------------------------------------------------------
# The sidebar
# ---------------------------------------------------------------------------


def choose_video():
    """The video source. Returns the path of the video, or None."""
    st.sidebar.header("1. Video")
    source = st.sidebar.radio("Where is the video?", [UPLOAD_SOURCE, SAMPLE_SOURCE], key="source")
    if source == UPLOAD_SOURCE:
        uploaded = st.sidebar.file_uploader("Video file", type=list(support.VIDEO_TYPES), key="upload")
        if uploaded is None:
            return None
        key = (uploaded.name, uploaded.size)
        if st.session_state.get("uploaded_key") != key:
            try:
                st.session_state["uploaded_path"] = str(support.save_uploaded_video(uploaded.name, uploaded.getvalue()))
            except ValueError as error:
                st.sidebar.error(str(error))
                return None
            st.session_state["uploaded_key"] = key
        return st.session_state["uploaded_path"]

    samples = support.list_sample_videos(SAMPLE_DIR)
    if not samples:
        st.sidebar.info(f"There are no video files in {SAMPLE_DIR.name}. Make one with make_test_belt_video.py or upload your own.")
        return None
    choice = st.sidebar.selectbox("File in sample_data", [path.name for path in samples], key="sample_choice")
    return str(SAMPLE_DIR / choice)


def reset_defaults(info) -> None:
    """When the video changes, put the rotation centres at a sensible start, in the picture of the new video."""
    for prefix, share in (("pulley", 0.5), ("driver", 0.3), ("driven", 0.7)):
        st.session_state[f"{prefix}_cx"] = float(round(info.width * share))
        st.session_state[f"{prefix}_cy"] = float(round(info.height * 0.5))
    st.session_state["preview_frame"] = 0
    st.session_state["fps_value"] = float(info.fps)


def colour_inputs(prefix: str, default_colour: str):
    """Marker colour: a named colour, or limits typed in (or taken from a pixel). Returns TrackingSettings."""
    names = list(COLOUR_PRESETS)
    colour = st.selectbox("Marker colour", names, index=names.index(default_colour), key=f"{prefix}_colour")
    manual = st.checkbox("Set the colour limits myself", key=f"{prefix}_manual")
    settings = settings_for_colour(colour)
    if manual:
        low = st.number_input("Hue from (0 to 179)", 0.0, 179.0, step=1.0, key=ensure(f"{prefix}_hue_low", float(settings.hue_low)))
        high = st.number_input("Hue to (0 to 179)", 0.0, 179.0, step=1.0, key=ensure(f"{prefix}_hue_high", float(settings.hue_high)))
        sat = st.number_input("Saturation from (0 to 255)", 0.0, 255.0, step=1.0, key=ensure(f"{prefix}_sat_min", float(settings.sat_min)))
        val = st.number_input("Brightness from (0 to 255)", 0.0, 255.0, step=1.0, key=ensure(f"{prefix}_val_min", float(settings.val_min)))
        small = st.number_input("Smallest marker area (square pixels)", 1.0, step=10.0, key=ensure(f"{prefix}_min_area", float(settings.min_area_px)))
        large = st.number_input("Largest marker area (square pixels)", 1.0, step=100.0, key=ensure(f"{prefix}_max_area", float(settings.max_area_px)))
        settings = TrackingSettings(
            hue_low=int(low), hue_high=int(high), sat_min=int(sat), val_min=int(val), min_area_px=small, max_area_px=large
        )
    with st.expander("Pick the colour from a pixel"):
        st.caption("Read the position of the middle of the marker from the grid in the picture, type it here and press the button.")
        st.number_input("Marker pixel X", 0.0, step=1.0, key=ensure(f"{prefix}_pick_x", 0.0))
        st.number_input("Marker pixel Y", 0.0, step=1.0, key=ensure(f"{prefix}_pick_y", 0.0))
        st.button("Use the colour at this pixel", key=f"{prefix}_pick", on_click=apply_pixel_colour, args=(prefix,))
        message = st.session_state.get(f"{prefix}_pick_message")
        if message:
            st.caption(message)
    return settings


def pulley_inputs(prefix: str, label: str, default_colour: str):
    """The inputs of one pulley. Returns a PulleyInput."""
    with st.sidebar.expander(label, expanded=True):
        settings = colour_inputs(prefix, default_colour)
        cx = st.number_input("Rotation centre X (pixels)", 0.0, step=1.0, key=ensure(f"{prefix}_cx", 0.0))
        cy = st.number_input("Rotation centre Y (pixels)", 0.0, step=1.0, key=ensure(f"{prefix}_cy", 0.0))
        region = st.number_input(
            "Search region radius (pixels, 0 = whole picture)", 0.0, step=1.0, key=ensure(f"{prefix}_region", 0.0),
            help="A circle around the rotation centre, a little bigger than the pulley. The marker is only searched inside it.",
        )
        radius = st.number_input(
            "Effective radius (mm, 0 = not known)", 0.0, step=0.5, key=ensure(f"{prefix}_radius_mm", 0.0),
            help="From the centre of the pulley to the middle of the belt. Needed for the expected ratio and the surface speed.",
        )
        marker_radius = st.number_input(
            "Marker distance from the centre (mm, 0 = not known)", 0.0, step=0.5, key=ensure(f"{prefix}_marker_mm", 0.0),
            help="Gives the scale (mm per pixel) for the belt speed, together with the distance in the picture.",
        )
    return support.PulleyInput(settings, (cx, cy), optional(region), optional(radius), optional(marker_radius))


def signal_inputs():
    with st.sidebar.expander("Signal analysis"):
        window = st.number_input("Smoothing window (s)", 0.02, 5.0, step=0.05, key=ensure("smooth_window", 0.2))
        order = st.number_input("Smoothing polynomial order", 1, 5, step=1, key=ensure("smooth_order", 2))
        trend = st.number_input("Trend order (0 = mean, 1 = straight line)", 0, 4, step=1, key=ensure("trend_order", 1))
        gap = st.number_input("Lost frames filled in (in a row)", 0, 30, step=1, key=ensure("max_gap", 3))
        outliers = st.checkbox("Remove outliers", key=ensure("remove_outliers", True))
        min_peak = st.number_input("Smallest peak listed (percent of mean speed)", 0.0, 100.0, step=0.1, key=ensure("min_peak", 0.5))
        fitted = st.checkbox("Use the centre from the circle fit of the marker path", key=ensure("use_fitted", False))
    settings = SignalSettings(
        smooth_window_s=float(window), smooth_polyorder=int(order), remove_outliers=bool(outliers),
        max_gap_frames=int(gap), trend_order=int(trend), min_peak_percent=float(min_peak),
    )
    return settings, bool(fitted)


# ---------------------------------------------------------------------------
# The main area
# ---------------------------------------------------------------------------


def show_video_and_preview(path, info, kind, pulleys, belt_settings):
    st.header("Video")
    show_rows(support.video_rows(info, Path(path).name))
    if support.looks_simulated(path):
        st.warning("This is a SIMULATED video made by the program. The numbers are not a measurement of a real machine.")
    for warning in support.video_warnings(info):
        st.warning(warning)

    st.subheader("Check the picture")
    frame_number = 0
    if info.frame_count > 1:
        frame_number = st.slider("Frame", 0, info.frame_count - 1, key=ensure("preview_frame", 0))
    grid = st.checkbox("Show a pixel grid (to read the positions)", key=ensure("show_grid", True))
    markers = [(item.centre, item.region_radius) for item in pulleys.values()]
    try:
        st.image(support.frame_preview(path, frame_number, markers, 100 if grid else None),
                 caption="The cross is the rotation centre you typed in. The circle is the search region.")
    except ANALYSIS_ERRORS as error:
        st.error(str(error))
        return frame_number

    with st.expander("Check the marker colours in this frame"):
        st.caption("Left: the frame. Right: the pixels that count as the marker (white). The marker should be one clean white blob.")
        checks = [(support.PULLEY_LABELS[name], item.settings, item.centre, item.region_radius) for name, item in pulleys.items()]
        if belt_settings is not None:
            checks.append(("Belt mark", belt_settings, None, None))
        for label, settings, centre, region in checks:
            try:
                picture, detection = support.mask_preview(path, frame_number, settings, centre, region)
            except ANALYSIS_ERRORS as error:
                st.error(f"{label}: {error}")
                continue
            if detection is None:
                st.warning(f"{label}: no marker found in this frame.")
            else:
                st.success(f"{label}: marker found at ({detection.x:.1f}, {detection.y:.1f}) pixels, area {detection.area_px:.0f} square pixels.")
            st.image(picture)
    return frame_number


def show_figure(result, name: str, suffix: str) -> None:
    for end, caption, png in result.figures.get(name, []):
        if end == suffix:
            st.image(png, caption=caption)


def show_metrics(pairs) -> None:
    columns = st.columns(len(pairs))
    for column, (label, value) in zip(columns, pairs):
        column.metric(label, value)


def show_pulley(result, name: str) -> None:
    pulley = result.pulleys[name]
    item = result.config.pulleys[name]
    show_metrics(report.pulley_headline(pulley))
    left, right = st.columns(2)
    with left:
        show_figure(result, name, "rpm")
        show_figure(result, name, "tracking_quality")
    with right:
        show_figure(result, name, "fluctuation")
        show_figure(result, name, "spectrum")
    peaks = report.peak_table(pulley.signal)
    st.subheader("Dominant frequencies")
    if peaks:
        st.table(pd.DataFrame(peaks).set_index("#"))
        st.caption("A peak shows that the speed varies periodically at that frequency. It does not show the cause.")
    else:
        st.write("No dominant frequency was listed. See the signal analysis details below for the level that a peak has to reach.")
    show_figure(result, name, "angle")
    show_figure(result, name, "trajectory")
    st.subheader("Details")
    for heading, rows in report.pulley_sections(pulley, item.settings, item.region_radius, result.simulated):
        with st.expander(heading):
            show_rows(rows)


def show_belt_drive(result) -> None:
    outcome = result.outcome
    show_metrics(report.belt_headline(outcome.belt, outcome.slip))
    if outcome.belt is None:
        st.warning(f"The comparison of the two pulleys could not be made: {outcome.belt_problem}")
    left, right = st.columns(2)
    with left:
        show_figure(result, "belt", "belt_speeds")
        show_figure(result, "belt", "belt_mark_speed")
    with right:
        show_figure(result, "belt", "belt_surface_speed")
        show_figure(result, "belt", "belt_slip")
    if outcome.belt is not None:
        with st.expander("Comparison of the two pulleys", expanded=True):
            show_rows(report.belt_rows(outcome.belt))
    if result.config.belt_settings is not None:
        with st.expander("Belt slip", expanded=True):
            show_rows(report.slip_rows(outcome.slip, outcome.slip_message))
    else:
        st.caption("No mark on the belt was given, so the belt speed and the slip were not measured. Only the speed loss of the driven pulley is calculated.")


def show_annotated_frames(result, result_id: int) -> None:
    count = support.frame_count_of(result)
    position = 0
    if count > 1:
        position = st.slider("Frame in the analysis", 0, count - 1, key=f"annot_pos_{result_id}")
    picture, frame_number, time_s = support.annotated_frame(result, position)
    st.image(picture, caption=f"frame {frame_number}, t = {time_s:.3f} s. Ring colour: green Good, yellow Moderate, orange Poor.")
    st.caption(
        "A video file with the markers drawn on it can be made below. It is an mp4v file that most desktop players open "
        "(for example VLC), but a web browser may not play it."
    )
    if st.button("Make the tracked video file", key="make_video"):
        with st.spinner("Writing the video"):
            folder = Path(tempfile.mkdtemp(prefix="smartbelt_"))
            path = support.write_tracked_video(result, folder / f"{support.file_stem(result)}_tracked.mp4")
            st.session_state["tracked_video"] = (result_id, path.name, path.read_bytes())
    saved = st.session_state.get("tracked_video")
    if saved is not None and saved[0] == result_id:
        st.download_button("Download the tracked video", saved[2], file_name=saved[1], mime="video/mp4", key="dl_video")


def show_downloads(result) -> None:
    st.subheader("Table")
    st.caption(f"{len(result.table)} rows, one per frame. The same table is saved by the command line scripts.")
    st.dataframe(result.table)
    st.subheader("Downloads")
    notes = st.text_area("Notes for the report (optional)", key="report_notes")
    left, middle, right = st.columns(3)
    left.download_button("Download the table (CSV)", support.csv_bytes(result), file_name=support.csv_name(result), mime="text/csv", key="dl_csv")
    middle.download_button("Download the plots (zip of PNG)", support.plots_zip(result), file_name=f"{support.file_stem(result)}_plots.zip", mime="application/zip", key="dl_plots")
    right.download_button(
        "Download the report (HTML)", support.report_html(result, notes), file_name=f"{support.file_stem(result)}_report.html",
        mime="text/html", key="dl_report",
    )


def show_limitations() -> None:
    st.write("What to keep in mind when you read the numbers:")
    for text in report.LIMITATIONS:
        st.markdown(f"- {text}")
    st.subheader("Definitions")
    show_rows(report.DEFINITIONS)


def show_results(result, result_id: int) -> None:
    st.header("Results")
    st.caption(f"Video: {result.video_name}. The analysis took {result.seconds:.1f} s.")
    if result.simulated:
        st.warning("SIMULATED data. These numbers are not a measurement of a real machine.")
    for message in support.result_notes(result):
        st.warning(message)

    st.subheader("Summary")
    for sentence in support.engineering_summary(result):
        st.markdown(f"- {sentence}")

    # A row of radio buttons instead of st.tabs: a tab selection is lost when a widget inside a tab (the frame slider)
    # makes the script run again, but the value of a radio button with a key is kept.
    if result.config.kind == "single":
        names = ["Speed and frequency", "Annotated frames", "Data and downloads", "Limitations"]
    else:
        names = ["Belt drive", "Driver pulley", "Driven pulley", "Annotated frames", "Data and downloads", "Limitations"]
    view = st.radio("Show", names, horizontal=True, key="results_view", label_visibility="collapsed")
    if view == "Speed and frequency":
        show_pulley(result, "pulley")
    elif view == "Belt drive":
        show_belt_drive(result)
    elif view == "Driver pulley":
        show_pulley(result, "driver")
    elif view == "Driven pulley":
        show_pulley(result, "driven")
    elif view == "Annotated frames":
        show_annotated_frames(result, result_id)
    elif view == "Data and downloads":
        show_downloads(result)
    else:
        show_limitations()


def run_clicked_analysis(path, kind, pulleys, belt_ui, signal_settings, fitted, fps_override, start_frame, max_frames):
    """Run the analysis with a progress bar. The result is kept in the session. Problems are shown as plain messages."""
    st.session_state.pop("result", None)
    bar = st.progress(0.0, text="Starting")

    def progress_factory(label):
        def callback(done, total):
            bar.progress(min(1.0, done / max(total, 1)), text=f"{label}: {done} of {total} frames")
        return callback

    try:
        config = support.RunConfig(
            video_path=path, kind="single" if kind == SINGLE else "belt", pulleys=pulleys, signal_settings=signal_settings,
            fps_override=fps_override, start_frame=start_frame, max_frames=max_frames, use_fitted_centres=fitted,
            belt_settings=belt_ui["settings"], scale_mm_per_px=belt_ui["scale"], belt_speed_factor=belt_ui["factor"],
            reference_ratio=belt_ui["reference_ratio"],
        )
        st.session_state["result"] = support.run_analysis(config, progress_factory)
        st.session_state["result_id"] = st.session_state.get("result_id", 0) + 1
    except ANALYSIS_ERRORS as error:
        st.error(str(error))
    finally:
        bar.empty()


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

st.title("SmartBelt Vision")
st.caption("Video-based analysis of rotating pulley motion and belt slip")
st.write(
    "SmartBelt Vision uses smartphone video to estimate rotational motion and investigate belt-drive behaviour. "
    "It tracks coloured markers, calculates speed and its frequency content, and compares the two pulleys of a belt drive. "
    "The video is processed on this computer. Nothing is uploaded anywhere."
)

video_path = choose_video()
if video_path is None:
    st.info("Upload a video, or choose a file from sample_data, in the sidebar. Markers of different colours are needed on the pulleys.")
    st.stop()

st.session_state["video_path"] = video_path
try:
    video_info = support.video_info(video_path)
except ANALYSIS_ERRORS as error:
    st.error(str(error))
    st.stop()

if st.session_state.get("last_video") != video_path:
    reset_defaults(video_info)
    st.session_state["last_video"] = video_path

st.sidebar.header("2. Analysis")
analysis_kind = st.sidebar.radio("Type of analysis", [SINGLE, BELT_DRIVE], key="kind")

with st.sidebar.expander("Frame rate and frames"):
    manual_fps = st.checkbox("Enter the frame rate myself", key=ensure("fps_manual", False),
                             help=f"The file says {video_info.fps:g} fps. Use this if that is wrong.")
    fps_value = st.number_input("Frame rate (fps)", 1.0, 1000.0, step=1.0, key=ensure("fps_value", float(video_info.fps)))
    first_frame = st.number_input("First frame", 0, max(video_info.frame_count - 1, 0), step=1, key=ensure("start_frame", 0))
    frame_limit = st.number_input("Number of frames (0 = all)", 0, video_info.frame_count, step=1, key=ensure("max_frames", 0))
fps_override = float(fps_value) if manual_fps else None
if manual_fps:
    video_info = support.video_info(video_path, fps_override)

st.sidebar.header("3. Markers")
if analysis_kind == SINGLE:
    pulley_items = {"pulley": pulley_inputs("pulley", "Pulley", "green")}
    belt_ui = {"settings": None, "scale": None, "factor": 1.0, "reference_ratio": None}
else:
    pulley_items = {
        "driver": pulley_inputs("driver", "Driver pulley", "green"),
        "driven": pulley_inputs("driven", "Driven pulley", "orange"),
    }
    with st.sidebar.expander("Belt drive and belt mark"):
        reference = st.number_input(
            "Reference ratio (0 = from the effective radii)", 0.0, step=0.01, key=ensure("reference_ratio", 0.0),
            help="The measured speed ratio of a recording with no load. It removes the error of the effective radii.",
        )
        has_mark = st.checkbox("There is a mark on the belt", key=ensure("belt_enabled", False))
        belt_settings_ui = None
        scale_ui = None
        factor_ui = 1.0
        if has_mark:
            belt_settings_ui = colour_inputs("belt", "blue")
            scale_ui = st.number_input("Scale (mm per pixel, 0 = from the marker distances)", 0.0, step=0.01, format="%.4f", key=ensure("scale", 0.0))
            factor_ui = st.number_input(
                "Belt speed factor", 0.5, 2.0, step=0.001, format="%.4f", key=ensure("belt_factor", 1.0),
                help="The calibration factor that the analysis prints for a recording with no load and no slip.",
            )
    belt_ui = {"settings": belt_settings_ui, "scale": optional(scale_ui), "factor": float(factor_ui), "reference_ratio": optional(reference)}

signal_settings_ui, use_fitted = signal_inputs()
st.sidebar.header("4. Run")
run_clicked = st.sidebar.button("Run analysis", type="primary", key="run")
st.sidebar.caption("Every setting can be checked in the picture first (right).")

show_video_and_preview(video_path, video_info, analysis_kind, pulley_items, belt_ui["settings"])

if run_clicked:
    run_clicked_analysis(
        video_path, analysis_kind, pulley_items, belt_ui, signal_settings_ui, use_fitted, fps_override,
        int(first_frame), int(frame_limit) or None,
    )

saved_result = st.session_state.get("result")
if saved_result is not None:
    show_results(saved_result, st.session_state.get("result_id", 1))
else:
    st.info("Set the markers and the rotation centres in the sidebar, check them in the picture, then press Run analysis.")
