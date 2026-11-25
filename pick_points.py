"""Click on the marker, the pulley centre and the pulley edge to get the numbers run_analysis.py needs.

Run from the project folder, for example:
    python pick_points.py sample_data/disc_60fps_handspin_01.mp4
    python pick_points.py sample_data/disc_60fps_handspin_01.mp4 --frame 30

A window opens with one frame of the video. Click three times, in this order:
    1. the middle of the coloured marker
    2. the centre of the pulley (the axle)
    3. any point on the edge of the pulley

Keys:  u = undo the last click    r = start again    q, Esc or Enter = finish

When you finish, the numbers are printed together with a ready-made run_analysis.py command,
and a picture of what was picked is saved in the outputs folder.

For a belt drive with two pulleys, run it once for each pulley and add --pulley driver (or --pulley driven):
    python pick_points.py sample_data/belt_60fps_01.mp4 --pulley driver
    python pick_points.py sample_data/belt_60fps_01.mp4 --pulley driven
Then the numbers are printed as options for run_belt_drive.py (--driver-hue, --driven-centre and so on).
Put both sets of options in one run_belt_drive.py command.

For the mark on the belt, add --pulley belt. Click only on the mark and press q. The colour options are
printed as --belt-hue and so on (a centre and an edge are not needed for the belt mark).

This window needs the normal opencv-python package, not opencv-python-headless.
If no window is possible, read the centre from the grid picture made by inspect_video.py
and check the colour with:  python run_analysis.py <video> --preview
"""

import argparse
import math
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import cv2

from src.tracking import (
    CENTRE_COLOUR,
    QUALITY_COLOURS,
    TrackingError,
    colour_range_from_patch,
    describe_settings,
    draw_marker_ring,
    draw_pulley_region,
    make_colour_mask,
    measure_blob_at,
    suggest_area_limits,
)
from src.utils import OUTPUT_DIR, put_label, save_image
from src.video_io import VideoError, read_frame

WINDOW_NAME = "SmartBelt Vision: pick points"
MAX_DISPLAY_WIDTH = 1200  # big frames are shown smaller than this so the window fits on the screen
MAX_DISPLAY_HEIGHT = 650
REGION_MARGIN = 1.05  # the search region is made 5 percent larger than the clicked pulley edge
MIN_PULLEY_RADIUS_PX = 20  # an edge click closer to the centre than this is treated as a mistake

STEPS = ("marker", "centre", "edge")
STEP_TEXT = {
    "marker": "Click 1 of 3: the middle of the coloured marker",
    "centre": "Click 2 of 3: the centre of the pulley (the axle)",
    "edge": "Click 3 of 3: any point on the edge of the pulley",
}

INSTRUCTIONS = """A window will open. Click, in this order:
  1. the middle of the coloured marker
  2. the centre of the pulley (the axle)
  3. any point on the edge of the pulley
Keys: u = undo the last click, r = start again, q or Esc or Enter = finish."""


# ---------------------------------------------------------------------------
# State: the three clicks. A click is stored as (x, y) in full-size image pixels.
# ---------------------------------------------------------------------------


def new_state() -> dict:
    return {step: None for step in STEPS}


def next_step(state: dict):
    """The name of the next click that is needed, or None when all three are done."""
    for step in STEPS:
        if state[step] is None:
            return step
    return None


def handle_click(state: dict, x: int, y: int) -> None:
    step = next_step(state)
    if step is not None:
        state[step] = (int(x), int(y))


def undo_last_click(state: dict) -> None:
    for step in reversed(STEPS):
        if state[step] is not None:
            state[step] = None
            return


def reset_state(state: dict) -> None:
    for step in STEPS:
        state[step] = None


# ---------------------------------------------------------------------------
# Working out the numbers from the clicks
# ---------------------------------------------------------------------------


def compute_results(frame, state: dict) -> dict:
    """Turn the clicks into colour settings, a marker size, the centre and the search region."""
    results = {
        "settings": None,
        "blob": None,
        "centre": None,
        "pulley_radius_px": None,
        "region_radius_px": None,
        "message": "",
    }

    if state["marker"] is not None:
        x, y = state["marker"]
        try:
            settings = colour_range_from_patch(frame, x, y)
        except TrackingError as error:
            results["message"] = str(error)
        else:
            blob = measure_blob_at(frame, settings, x, y)
            if blob is None:
                results["message"] = (
                    "The colour was picked but no region of that colour was found at the click. "
                    "Press u and click nearer the middle of the marker."
                )
            else:
                smallest, largest = suggest_area_limits(blob.area_px)
                results["settings"] = replace(settings, min_area_px=smallest, max_area_px=largest)
                results["blob"] = blob

    if state["centre"] is not None:
        results["centre"] = state["centre"]
        if state["edge"] is not None:
            radius = math.hypot(state["edge"][0] - state["centre"][0], state["edge"][1] - state["centre"][1])
            if radius < MIN_PULLEY_RADIUS_PX:
                results["message"] = "The edge click is very close to the centre. Press u and click on the pulley edge."
            else:
                results["pulley_radius_px"] = radius
                results["region_radius_px"] = REGION_MARGIN * radius
    return results


def format_results(video_path, results: dict, pulley=None) -> list:
    """Text lines that describe the picked values, ending with a ready-made command.

    pulley is None for run_analysis.py. For run_belt_drive.py it is "driver" or "driven", and the options
    then start with --driver- or --driven- and the command only holds the options of that pulley.
    """
    lines = []
    video_text = f'"{video_path}"' if " " in str(video_path) else str(video_path)
    prefix = f"--{pulley}-" if pulley else "--"
    is_belt = pulley == "belt"
    command = f"python run_belt_drive.py {video_text}" if pulley else f"python run_analysis.py {video_text}"

    settings = results["settings"]
    if settings is not None:
        blob = results["blob"]
        lines.append(f"Marker colour   : {describe_settings(settings)}")
        lines.append(f"Marker size     : about {blob.area_px:.0f} square pixels, found at ({blob.x:.1f}, {blob.y:.1f})")
        command += (
            f" {prefix}hue {settings.hue_low} {settings.hue_high} {prefix}sat-min {settings.sat_min}"
            f" {prefix}val-min {settings.val_min} {prefix}min-area {settings.min_area_px:g} {prefix}max-area {settings.max_area_px:g}"
        )

    if results["centre"] is not None and not is_belt:
        cx, cy = results["centre"]
        lines.append(f"Rotation centre : ({cx}, {cy}) pixels")
        command += f" {prefix}centre {cx} {cy}"
    if results["region_radius_px"] is not None and not is_belt:
        lines.append(f"Pulley radius   : about {results['pulley_radius_px']:.0f} pixels (from the edge click)")
        lines.append(f"Search region   : {results['region_radius_px']:.0f} pixels around the centre")
        command += f" {prefix}region-radius {results['region_radius_px']:.0f}"

    if results["message"]:
        lines.append(f"Note            : {results['message']}")
    lines.append("")
    if is_belt:
        lines.append("Options for the mark on the belt (put them in the run_belt_drive.py command):")
        lines.append(command.replace(f"python run_belt_drive.py {video_text} ", "", 1))
    elif pulley:
        lines.append(f"Options for the {pulley} pulley (put them in the run_belt_drive.py command, with the options of the other pulley):")
        lines.append(command.replace(f"python run_belt_drive.py {video_text} ", "", 1))
    else:
        lines.append("Command to run the tracking with these values:")
        lines.append(command)
    return lines


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_view(frame, state: dict, results: dict, scale: float, with_text: bool = True, pulley=None):
    """Draw what has been picked so far. The picture is scaled down by scale for the window."""
    picture = frame.copy()

    settings = results["settings"]
    if settings is not None:
        mask = make_colour_mask(picture, settings)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(picture, contours, -1, CENTRE_COLOUR, 2)  # everything the colour limits pick up
        blob = results["blob"]
        draw_marker_ring(picture, blob.x, blob.y, blob.area_px, QUALITY_COLOURS["Good"])

    if results["centre"] is not None:
        draw_pulley_region(picture, results["centre"], results["pulley_radius_px"])
    if state["edge"] is not None:
        ex, ey = state["edge"]
        cv2.drawMarker(picture, (ex, ey), CENTRE_COLOUR, cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)

    if scale < 1.0:
        picture = cv2.resize(picture, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    if with_text:
        step = next_step(state)
        if pulley == "belt":
            text = (
                "Click the middle of the mark on the belt, then press q or Enter"
                if state["marker"] is None
                else "Mark picked. Press q or Enter to finish, u to undo."
            )
        else:
            text = STEP_TEXT[step] if step is not None else "All three points picked. Press q or Enter to finish, u to undo."
        put_label(picture, text, (10, 24), 0.6)
        put_label(picture, "u = undo   r = start again   q = finish", (10, 48), 0.5)
        y_position = 72
        for message_line in textwrap.wrap(results["message"], 80):
            put_label(picture, message_line, (10, y_position), 0.5)
            y_position += 22
    return picture


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


def run_picker(video_path, frame_index: int = 0, output_dir=OUTPUT_DIR, pulley=None) -> int:
    """Show one frame, collect the clicks, print the numbers and save a picture."""
    frame = read_frame(video_path, frame_index)
    height, width = frame.shape[:2]
    scale = min(1.0, MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height)
    state = new_state()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # The window may show a scaled-down picture, so convert back to full-size pixels.
            full_x = min(width - 1, max(0, int(round(x / scale))))
            full_y = min(height - 1, max(0, int(round(y / scale))))
            handle_click(state, full_x, full_y)

    print(INSTRUCTIONS)
    try:
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)

        shown_state = None
        view = None
        loops = 0
        while True:
            if state != shown_state:  # only redraw when a click has changed something
                results = compute_results(frame, state)
                view = draw_view(frame, state, results, scale, pulley=pulley)
                shown_state = dict(state)
            cv2.imshow(WINDOW_NAME, view)

            key = cv2.waitKey(30) & 0xFF
            if key in (27, 13, 10, ord("q")):
                break
            if key == ord("u"):
                undo_last_click(state)
            if key == ord("r"):
                reset_state(state)

            # Stop when the window is closed with the X button. The check starts after a
            # few loops so a slow-to-appear window is not mistaken for a closed one.
            loops += 1
            if loops > 10 and cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    except cv2.error as error:
        raise OSError(
            "OpenCV could not open a window. Check that opencv-python (not opencv-python-headless) is installed: "
            "run 'pip uninstall opencv-python-headless' and then 'pip install opencv-python'."
        ) from error
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    if all(point is None for point in state.values()):
        print("Nothing was picked.")
        return 1

    results = compute_results(frame, state)
    print()
    for text_line in format_results(video_path, results, pulley):
        print(text_line)

    picture_path = save_image(
        Path(output_dir) / (f"{Path(video_path).stem}_picked_{pulley}.png" if pulley else f"{Path(video_path).stem}_picked.png"),
        draw_view(frame, state, results, 1.0, with_text=False),
    )
    print()
    print(f"Saved: {picture_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Click on the marker, the pulley centre and the pulley edge.")
    parser.add_argument("video", help="path to the video file")
    parser.add_argument("--frame", type=int, default=0, help="frame to show (default 0, the first frame)")
    parser.add_argument(
        "--pulley",
        choices=("driver", "driven", "belt"),
        default=None,
        help="print the options for this part of a belt drive (for run_belt_drive.py) instead of a run_analysis.py command",
    )
    args = parser.parse_args()

    try:
        return run_picker(args.video, args.frame, pulley=args.pulley)
    except (VideoError, TrackingError, OSError, ValueError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
