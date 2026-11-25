"""Finding the coloured marker in video frames, frame by frame.

Method (classical computer vision, no machine learning):
  1. Blur the frame a little to reduce noise.
  2. Convert it from BGR to HSV. HSV separates the colour itself (hue) from how
     strong the colour is (saturation) and how bright it is (value). That makes it
     easier to pick one colour when the brightness changes a little.
  3. Keep only the pixels inside the chosen hue, saturation and value limits.
     The result is a mask: white where the marker colour is, black everywhere else.
  4. Clean the mask. Opening removes tiny white specks. Closing fills tiny black holes.
  5. Find the outlines (contours) of the white regions. Ignore regions that are too
     small or too big and take the biggest one that is left.
  6. The centroid (centre of mass) of that region is the marker position in pixels.

OpenCV HSV scale: hue 0 to 179 (the colour wheel divided by 2), saturation 0 to 255,
value 0 to 255. Red sits at both ends of the hue scale (about 170 to 179 and 0 to 10),
so a hue range with low > high is treated as wrapping around 0.

The result of tracking a whole video is one pandas table with one row per frame.
Positions are in pixels. The image origin is the top-left corner and y points DOWN.
Frames where the marker was not found have NaN (empty) position values, never 0.
"""

import math
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.utils import ensure_folder, put_label
from src.video_io import get_video_info, iter_frames

# Hue ranges (OpenCV scale 0 to 179) for common marker colours.
# low > high means the range wraps around 0, which is needed for red.
COLOUR_PRESETS = {
    "green": (35, 85),
    "red": (170, 10),
    "blue": (95, 130),
    "orange": (8, 22),
    "yellow": (22, 35),
    "magenta": (140, 170),
}

# Rules for the Tracking Quality of each frame. These are simple working rules, not
# a statistical confidence. Each frame with a marker is given three checks:
#   area      the marker area must be within +/- 50 percent of the median area
#   distance  the distance from the centre must be within +/- 10 percent of the median distance
#   jump      the marker must not move more than 1.0 x the median distance between two frames
# (The distance and jump checks need the rotation centre.)
AREA_TOLERANCE = 0.5
RADIUS_TOLERANCE = 0.10
MAX_JUMP_FRACTION = 1.0

# Frame quality: no failed check = Good, one = Moderate, two or more = Poor.
# Frames that are Good or Moderate are marked tracking_valid. Lost frames have no marker.
# Whole video: Good if at least 90 percent of frames are Good, Moderate if at least
# 80 percent are Good or Moderate, otherwise Poor.
GOOD_FRACTION = 0.90
USABLE_FRACTION = 0.80

# Below this share of frames with a marker, the analysis stops with an error message.
MIN_DETECTION_RATE = 0.5

# A clicked point with less saturation than this is treated as grey (not a coloured marker).
MIN_PICK_SATURATION = 50

# Colours used when drawing on the video (BGR order).
QUALITY_COLOURS = {"Good": (0, 200, 0), "Moderate": (0, 190, 255), "Poor": (0, 0, 255)}
CENTRE_COLOUR = (255, 255, 0)  # cyan


class TrackingError(Exception):
    """A problem with the tracking settings or the tracking result. The message is meant for the user."""


@dataclass
class TrackingSettings:
    """Everything that decides which pixels count as the marker."""

    hue_low: int = 35  # hue limits, 0 to 179. low > high wraps around 0 (red)
    hue_high: int = 85
    sat_min: int = 100  # minimum saturation, 0 to 255 (grey and white have low saturation)
    val_min: int = 70  # minimum brightness (value), 0 to 255 (dark areas have low value)
    min_area_px: float = 30.0  # smallest region accepted as the marker, in square pixels
    max_area_px: float = 20000.0  # largest region accepted as the marker, in square pixels
    blur_size: int = 5  # size of the blur before the colour conversion, 0 (off) or odd
    morph_size: int = 3  # size of the mask cleaning, 0 means no cleaning


@dataclass
class MarkerDetection:
    """Where the marker was found in one frame."""

    x: float  # pixels from the left edge
    y: float  # pixels from the top edge (y points down)
    area_px: float  # area of the detected region in square pixels


def settings_for_colour(colour_name: str, **changes) -> TrackingSettings:
    """Settings for one of the named colours. Extra keyword arguments replace single values.

    Example: settings_for_colour("green", sat_min=120, min_area_px=200)
    """
    if colour_name not in COLOUR_PRESETS:
        raise TrackingError(
            f"Unknown colour '{colour_name}'. Choose one of: {', '.join(COLOUR_PRESETS)}, "
            "or give the hue range yourself."
        )
    hue_low, hue_high = COLOUR_PRESETS[colour_name]
    # dataclasses.replace makes a copy of the settings with the listed values changed.
    return replace(TrackingSettings(hue_low=hue_low, hue_high=hue_high), **changes)


def validate_settings(settings: TrackingSettings) -> None:
    """Raise a TrackingError with a plain message if any setting is out of range."""
    if not (0 <= settings.hue_low <= 179 and 0 <= settings.hue_high <= 179):
        raise TrackingError("The hue limits must be between 0 and 179 (OpenCV hue scale).")
    if not (0 <= settings.sat_min <= 255 and 0 <= settings.val_min <= 255):
        raise TrackingError("The saturation and brightness limits must be between 0 and 255.")
    if settings.min_area_px < 1 or settings.max_area_px <= settings.min_area_px:
        raise TrackingError("The area limits must satisfy 1 <= min area < max area (in square pixels).")
    if settings.blur_size != 0 and (settings.blur_size < 3 or settings.blur_size % 2 == 0):
        raise TrackingError("The blur size must be 0 (off) or an odd number such as 3, 5 or 7.")
    if settings.morph_size < 0:
        raise TrackingError("The mask cleaning size cannot be negative.")


def describe_settings(settings: TrackingSettings) -> str:
    """One line that describes the colour and size limits, for printing."""
    if settings.hue_low <= settings.hue_high:
        hue_text = f"hue {settings.hue_low} to {settings.hue_high}"
    else:
        hue_text = f"hue {settings.hue_low} to {settings.hue_high} (wraps around 0)"
    return (
        f"{hue_text}, saturation from {settings.sat_min}, brightness from {settings.val_min}, "
        f"area {settings.min_area_px:g} to {settings.max_area_px:g} pixels"
    )


def check_centre(centre, width: int, height: int) -> None:
    """Raise a TrackingError if the rotation centre is not inside the frame."""
    x, y = centre
    if not (0 <= x < width and 0 <= y < height):
        raise TrackingError(
            f"The rotation centre ({x:g}, {y:g}) is outside the frame, which is {width} x {height} pixels. "
            "Read the centre from the grid picture made by inspect_video.py or use pick_points.py."
        )


def make_colour_mask(frame_bgr: np.ndarray, settings: TrackingSettings) -> np.ndarray:
    """Return a mask (uint8, 255 where the pixel has the marker colour, 0 elsewhere)."""
    image = frame_bgr
    if settings.blur_size > 0:
        image = cv2.GaussianBlur(image, (settings.blur_size, settings.blur_size), 0)

    # cv2.cvtColor changes the colour system of an image. OpenCV frames are BGR.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # cv2.inRange gives 255 where all three channels are inside the limits, otherwise 0.
    if settings.hue_low <= settings.hue_high:
        mask = cv2.inRange(
            hsv,
            (settings.hue_low, settings.sat_min, settings.val_min),
            (settings.hue_high, 255, 255),
        )
    else:
        # The hue range wraps around 0 (red): take the top end and the bottom end and join them.
        upper_part = cv2.inRange(hsv, (settings.hue_low, settings.sat_min, settings.val_min), (179, 255, 255))
        lower_part = cv2.inRange(hsv, (0, settings.sat_min, settings.val_min), (settings.hue_high, 255, 255))
        mask = cv2.bitwise_or(upper_part, lower_part)

    if settings.morph_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (settings.morph_size, settings.morph_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # removes small white specks
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fills small black holes
    return mask


def find_marker(frame_bgr, settings: TrackingSettings, centre=None, region_radius_px=None):
    """Find the marker in one frame. Returns a MarkerDetection, or None if it was not found.

    centre and region_radius_px (both optional) describe the pulley region. When both are
    given, only the square around the pulley is searched and regions whose centre lies
    outside the circle are ignored. This is faster and avoids picking up objects of the
    same colour elsewhere in the picture.
    """
    height, width = frame_bgr.shape[:2]
    search_area = frame_bgr
    x_offset = y_offset = 0
    use_region = centre is not None and region_radius_px is not None

    if use_region:
        x0 = max(0, int(math.floor(centre[0] - region_radius_px)))
        y0 = max(0, int(math.floor(centre[1] - region_radius_px)))
        x1 = min(width, int(math.ceil(centre[0] + region_radius_px)) + 1)
        y1 = min(height, int(math.ceil(centre[1] + region_radius_px)) + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        search_area = frame_bgr[y0:y1, x0:x1]
        x_offset, y_offset = x0, y0

    mask = make_colour_mask(search_area, settings)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for contour in contours:
        # Image moments: m00 is the area, m10 / m00 and m01 / m00 give the centroid.
        moments = cv2.moments(contour)
        area = moments["m00"]
        if area < settings.min_area_px or area > settings.max_area_px:
            continue
        x = moments["m10"] / area + x_offset
        y = moments["m01"] / area + y_offset
        if use_region and math.hypot(x - centre[0], y - centre[1]) > region_radius_px:
            continue
        if best is None or area > best.area_px:
            best = MarkerDetection(x, y, area)
    return best


def measure_blob_at(frame_bgr, settings: TrackingSettings, x: float, y: float, max_distance: float = 3.0):
    """Measure the coloured region that contains the point (x, y), or lies within a few pixels of it.

    Used by pick_points.py to find out how big the clicked marker is. The area limits in
    settings are ignored here. Returns a MarkerDetection or None.
    """
    mask = make_colour_mask(frame_bgr, settings)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_distance = -math.inf
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        # pointPolygonTest with True gives the distance to the outline: positive inside, negative outside.
        distance = cv2.pointPolygonTest(contour, (float(x), float(y)), True)
        if distance >= -max_distance and distance > best_distance:
            best_distance = distance
            best = MarkerDetection(
                moments["m10"] / moments["m00"], moments["m01"] / moments["m00"], moments["m00"]
            )
    return best


def colour_range_from_patch(frame_bgr, x: float, y: float, patch_radius: int = 3, hue_margin: int = 12):
    """Work out colour settings from the pixels around a clicked point.

    The hue limits are the typical hue of the patch plus and minus hue_margin.
    The saturation and brightness minimums are half of the typical values of the patch
    (never below 50), so the soft edge of the marker is still included.
    The area limits are left at their defaults.
    """
    height, width = frame_bgr.shape[:2]
    x, y = int(round(x)), int(round(y))
    if not (0 <= x < width and 0 <= y < height):
        raise TrackingError("The clicked point is outside the picture.")

    x0, x1 = max(0, x - patch_radius), min(width, x + patch_radius + 1)
    y0, y1 = max(0, y - patch_radius), min(height, y + patch_radius + 1)
    patch_hsv = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).reshape(-1, 3)

    saturation = float(np.median(patch_hsv[:, 1]))
    value = float(np.median(patch_hsv[:, 2]))
    if saturation < MIN_PICK_SATURATION:
        raise TrackingError(
            f"The clicked point is almost grey (saturation {saturation:.0f} out of 255). "
            "Click on the middle of the coloured marker."
        )

    # Hue is a circle (179 is next to 0), so it is averaged as an angle, not as a plain number.
    hue_angles = patch_hsv[:, 0].astype(float) * (2.0 * math.pi / 180.0)
    mean_angle = math.atan2(np.sin(hue_angles).mean(), np.cos(hue_angles).mean())
    hue_centre = (mean_angle % (2.0 * math.pi)) * 180.0 / (2.0 * math.pi)

    return TrackingSettings(
        hue_low=int(round(hue_centre - hue_margin)) % 180,
        hue_high=int(round(hue_centre + hue_margin)) % 180,
        sat_min=max(50, int(0.5 * saturation)),
        val_min=max(50, int(0.5 * value)),
    )


def suggest_area_limits(area_px: float):
    """Suggest (min_area, max_area) from the measured area of the marker: 0.3 x and 3 x, rounded."""
    smallest = max(10, int(round(0.3 * area_px, -1)))
    largest = int(round(3.0 * area_px, -1))
    return smallest, largest


def track_marker(
    video_path,
    settings: TrackingSettings,
    centre=None,
    region_radius_px=None,
    fps_override=None,
    start_frame: int = 0,
    max_frames=None,
    progress=None,
) -> pd.DataFrame:
    """Track the marker through a video and return one table row per frame.

    Columns: frame, time_s, marker_x, marker_y, area_px, radius_px, tracking_valid,
    tracking_quality, quality_notes.
    time_s is frame / fps. radius_px (distance from the centre) is only filled in when
    centre is given. progress is an optional function called as progress(frames_done, frames_total).
    """
    validate_settings(settings)
    info = get_video_info(video_path, fps_override=fps_override)
    if centre is not None:
        check_centre(centre, info.width, info.height)
    if region_radius_px is not None and (centre is None or region_radius_px <= 0):
        raise TrackingError("A region radius needs a rotation centre and must be larger than 0.")
    if start_frame >= info.frame_count:
        raise TrackingError(f"The start frame ({start_frame}) is beyond the end of the video ({info.frame_count} frames).")

    total = info.frame_count - start_frame
    if max_frames is not None:
        total = min(total, max_frames)

    rows = []
    for frame_index, frame in iter_frames(video_path, start_frame, max_frames):
        found = find_marker(frame, settings, centre, region_radius_px)
        if found is None:
            rows.append((frame_index, np.nan, np.nan, np.nan))
        else:
            rows.append((frame_index, found.x, found.y, found.area_px))
        if progress is not None:
            progress(len(rows), total)

    if not rows:
        raise TrackingError("No frames could be read from the video.")

    table = pd.DataFrame(rows, columns=["frame", "marker_x", "marker_y", "area_px"])
    table.insert(1, "time_s", table["frame"] / info.fps)
    return add_tracking_quality(table, centre)


def add_tracking_quality(table: pd.DataFrame, centre=None) -> pd.DataFrame:
    """Add radius_px, tracking_valid, tracking_quality and quality_notes to a tracking table.

    The three checks (area, distance from centre, jump) are described near the top of this file.
    The median values used by the checks are taken from all frames where the marker was found.
    """
    table = table.copy()
    count = len(table)
    x = table["marker_x"].to_numpy(dtype=float)
    y = table["marker_y"].to_numpy(dtype=float)
    area = table["area_px"].to_numpy(dtype=float)
    found = ~np.isnan(x)

    if centre is not None:
        radius = np.hypot(x - centre[0], y - centre[1])  # NaN where the marker was not found
    else:
        radius = np.full(count, np.nan)

    median_area = float(np.nanmedian(area)) if found.any() else math.nan
    median_radius = float(np.nanmedian(radius)) if (found.any() and centre is not None) else math.nan

    quality, notes, valid = [], [], []
    for i in range(count):
        if not found[i]:
            quality.append("Lost")
            notes.append("marker not found")
            valid.append(False)
            continue

        problems = []
        if abs(area[i] - median_area) > AREA_TOLERANCE * median_area:
            problems.append("area")
        if centre is not None:
            if abs(radius[i] - median_radius) > RADIUS_TOLERANCE * median_radius:
                problems.append("distance from centre")
            # The jump check only makes sense when the marker was also found in the previous frame.
            if i > 0 and found[i - 1]:
                step = math.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
                if step > MAX_JUMP_FRACTION * median_radius:
                    problems.append("jump")

        if not problems:
            quality.append("Good")
        elif len(problems) == 1:
            quality.append("Moderate")
        else:
            quality.append("Poor")
        notes.append(", ".join(problems))
        valid.append(len(problems) <= 1)

    table["radius_px"] = radius
    table["tracking_valid"] = valid
    table["tracking_quality"] = quality
    table["quality_notes"] = notes
    return table


def _longest_run(flags) -> int:
    """Length of the longest run of True values in a sequence."""
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def summarise_tracking(table: pd.DataFrame, centre=None) -> dict:
    """Summarise a tracking table in a dictionary (counts, rates, medians, overall Tracking Quality)."""
    total = len(table)
    if total == 0:
        raise TrackingError("The tracking table is empty.")

    quality = table["tracking_quality"].to_numpy()
    good = int(np.sum(quality == "Good"))
    moderate = int(np.sum(quality == "Moderate"))
    poor = int(np.sum(quality == "Poor"))
    lost = int(np.sum(quality == "Lost"))
    found = total - lost

    if good / total >= GOOD_FRACTION:
        overall = "Good"
    elif (good + moderate) / total >= USABLE_FRACTION:
        overall = "Moderate"
    else:
        overall = "Poor"

    notes = table["quality_notes"].astype(str)
    summary = {
        "frames": total,
        "found": found,
        "detection_rate": found / total,
        "good": good,
        "moderate": moderate,
        "poor": poor,
        "lost": lost,
        "longest_gap_frames": _longest_run(quality == "Lost"),
        "area_problems": int(notes.str.contains("area").sum()),
        "distance_problems": int(notes.str.contains("distance").sum()),
        "jump_problems": int(notes.str.contains("jump").sum()),
        "overall_quality": overall,
        "median_area_px": None,
        "median_radius_px": None,
        "radius_spread_percent": None,
    }

    if found > 0:
        summary["median_area_px"] = float(table["area_px"].median())
        if centre is not None:
            radius = table["radius_px"]
            median_radius = float(radius.median())
            spread = float(radius.quantile(0.95) - radius.quantile(0.05))
            summary["median_radius_px"] = median_radius
            summary["radius_spread_percent"] = 100.0 * spread / median_radius
    return summary


def require_enough_detections(summary: dict, minimum_rate: float = MIN_DETECTION_RATE) -> None:
    """Raise a TrackingError with advice if the marker was found in too few frames."""
    if summary["detection_rate"] < minimum_rate:
        raise TrackingError(
            f"Marker could not be detected in enough frames ({summary['found']} of {summary['frames']}). "
            "Things to check: (1) look at the picture made by --preview, "
            "(2) adjust the colour limits (--hue, --sat-min, --val-min) or use pick_points.py, "
            "(3) make sure --min-area and --max-area fit the size of the marker, "
            "(4) improve the lighting and the contrast between the marker and the background."
        )


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_marker_ring(image: np.ndarray, x: float, y: float, area_px: float, colour) -> None:
    """Draw a ring around a marker. A dark ring underneath keeps it visible on any background."""
    centre = (int(round(x)), int(round(y)))
    radius = int(round(math.sqrt(area_px / math.pi))) + 6
    cv2.circle(image, centre, radius, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.circle(image, centre, radius, colour, 2, cv2.LINE_AA)


def draw_pulley_region(image: np.ndarray, centre, region_radius_px=None) -> None:
    """Draw the rotation centre as a cross and, if given, the pulley region as a circle."""
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    cv2.line(image, (cx - 12, cy), (cx + 12, cy), CENTRE_COLOUR, 1, cv2.LINE_AA)
    cv2.line(image, (cx, cy - 12), (cx, cy + 12), CENTRE_COLOUR, 1, cv2.LINE_AA)
    if region_radius_px is not None:
        cv2.circle(image, (cx, cy), int(round(region_radius_px)), CENTRE_COLOUR, 1, cv2.LINE_AA)


def annotate_frame(
    frame, frame_number, time_s, marker, quality, centre, region_radius_px, trail, extra_text=""
) -> np.ndarray:
    """Return a copy of the frame with the tracking drawn on it.

    marker is (x, y, area) or None. trail is a list of (x, y, quality) of recent positions.
    extra_text is an optional second line of text, for example the angle.
    """
    picture = frame.copy()
    if centre is not None:
        draw_pulley_region(picture, centre, region_radius_px)

    for trail_x, trail_y, trail_quality in trail:
        cv2.circle(picture, (int(round(trail_x)), int(round(trail_y))), 2, QUALITY_COLOURS[trail_quality], -1, cv2.LINE_AA)

    if marker is not None:
        draw_marker_ring(picture, marker[0], marker[1], marker[2], QUALITY_COLOURS[quality])

    label = f"frame {frame_number}   t = {time_s:.3f} s   tracking: {quality}"
    put_label(picture, label, (10, 24), 0.6)
    if extra_text:
        put_label(picture, extra_text, (10, 50), 0.6)
    return picture


def write_annotated_video(
    video_path,
    table: pd.DataFrame,
    out_path,
    fps: float,
    centre=None,
    region_radius_px=None,
    trail_length: int = 40,
    start_frame: int = 0,
    extra_text=None,
) -> Path:
    """Write a copy of the video with the tracking drawn on every frame.

    The table must come from track_marker for the same video and the same start_frame.
    The video is read a second time here, one frame at a time.
    extra_text: optional list with one text per table row, drawn as a second line (for example the angle).
    """
    if len(table) == 0:
        raise TrackingError("There are no tracked frames to draw.")
    if extra_text is not None and len(extra_text) != len(table):
        raise TrackingError("extra_text needs exactly one text for every row of the table.")
    out_path = Path(out_path)
    ensure_folder(out_path.parent)

    frame_numbers = table["frame"].to_numpy()
    times = table["time_s"].to_numpy(dtype=float)
    x_values = table["marker_x"].to_numpy(dtype=float)
    y_values = table["marker_y"].to_numpy(dtype=float)
    areas = table["area_px"].to_numpy(dtype=float)
    qualities = table["tracking_quality"].to_numpy()

    trail = deque(maxlen=trail_length)  # the most recent marker positions
    writer = None
    try:
        for position, (_, frame) in enumerate(iter_frames(video_path, start_frame, max_frames=len(table))):
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
                if not writer.isOpened():
                    raise OSError(f"Could not create the video file {out_path}. Check that the folder can be written to.")

            quality = str(qualities[position])
            marker = None
            if quality != "Lost":
                marker = (x_values[position], y_values[position], areas[position])
                trail.append((x_values[position], y_values[position], quality))

            writer.write(
                annotate_frame(
                    frame, int(frame_numbers[position]), float(times[position]),
                    marker, quality, centre, region_radius_px, trail,
                    extra_text[position] if extra_text is not None else "",
                )
            )
    finally:
        if writer is not None:
            writer.release()
    return out_path


def annotate_layers(frame, frame_number, time_s, layers, header_text="") -> np.ndarray:
    """Return a copy of the frame with several tracked markers drawn on it (for example two pulleys).

    layers is a list of dictionaries with the keys name, marker ((x, y, area) or None), quality,
    centre, region_radius_px, trail (list of (x, y, quality)) and text (extra text for that marker).
    Each marker gets its name next to the ring. The top lines of the picture give the frame number,
    the time and, for each layer, the tracking quality and its text.
    """
    picture = frame.copy()
    for layer in layers:
        if layer["centre"] is not None:
            draw_pulley_region(picture, layer["centre"], layer["region_radius_px"])
        for trail_x, trail_y, trail_quality in layer["trail"]:
            cv2.circle(picture, (int(round(trail_x)), int(round(trail_y))), 2, QUALITY_COLOURS[trail_quality], -1, cv2.LINE_AA)
        if layer["marker"] is not None:
            x, y, area = layer["marker"]
            draw_marker_ring(picture, x, y, area, QUALITY_COLOURS[layer["quality"]])
            offset = int(round(math.sqrt(area / math.pi))) + 12
            put_label(picture, layer["name"], (int(round(x)) + offset, int(round(y)) - offset), 0.5)

    header = f"frame {frame_number}   t = {time_s:.3f} s"
    if header_text:
        header += f"   {header_text}"
    put_label(picture, header, (10, 24), 0.6)
    for number, layer in enumerate(layers):
        text = f"{layer['name']}: tracking {layer['quality']}"
        if layer["text"]:
            text += f"   {layer['text']}"
        put_label(picture, text, (10, 50 + 26 * number), 0.6)
    return picture


def write_annotated_video_layers(
    video_path,
    layers,
    out_path,
    fps: float,
    trail_length: int = 40,
    start_frame: int = 0,
    header_texts=None,
) -> Path:
    """Write a copy of the video with several tracked markers drawn on every frame.

    layers is a list of dictionaries with the keys name, table, centre, region_radius_px and extra_text
    (a list with one text per table row, or None). All the tables must be for the same frames.
    header_texts is an optional list with one text per frame, drawn in the top line.
    """
    if not layers or len(layers[0]["table"]) == 0:
        raise TrackingError("There are no tracked frames to draw.")
    count = len(layers[0]["table"])
    for layer in layers:
        if len(layer["table"]) != count:
            raise TrackingError("All the tables must have the same number of frames.")
        if layer.get("extra_text") is not None and len(layer["extra_text"]) != count:
            raise TrackingError("extra_text needs exactly one text for every row of the table.")
    if header_texts is not None and len(header_texts) != count:
        raise TrackingError("header_texts needs exactly one text for every row of the table.")

    out_path = Path(out_path)
    ensure_folder(out_path.parent)
    columns = []
    for layer in layers:
        table = layer["table"]
        columns.append(
            {
                "frames": table["frame"].to_numpy(),
                "times": table["time_s"].to_numpy(dtype=float),
                "x": table["marker_x"].to_numpy(dtype=float),
                "y": table["marker_y"].to_numpy(dtype=float),
                "area": table["area_px"].to_numpy(dtype=float),
                "quality": table["tracking_quality"].to_numpy(),
                "trail": deque(maxlen=trail_length),
            }
        )

    writer = None
    try:
        for position, (_, frame) in enumerate(iter_frames(video_path, start_frame, max_frames=count)):
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
                if not writer.isOpened():
                    raise OSError(f"Could not create the video file {out_path}. Check that the folder can be written to.")

            states = []
            for layer, data in zip(layers, columns):
                quality = str(data["quality"][position])
                marker = None
                if quality != "Lost":
                    marker = (data["x"][position], data["y"][position], data["area"][position])
                    data["trail"].append((data["x"][position], data["y"][position], quality))
                extra = layer.get("extra_text")
                states.append(
                    {
                        "name": layer["name"],
                        "marker": marker,
                        "quality": quality,
                        "centre": layer["centre"],
                        "region_radius_px": layer["region_radius_px"],
                        "trail": data["trail"],
                        "text": extra[position] if extra is not None else "",
                    }
                )
            writer.write(
                annotate_layers(
                    frame, int(columns[0]["frames"][position]), float(columns[0]["times"][position]), states,
                    header_texts[position] if header_texts is not None else "",
                )
            )
    finally:
        if writer is not None:
            writer.release()
    return out_path


def make_mask_preview(frame_bgr, settings: TrackingSettings, centre=None, region_radius_px=None, max_width: int = 1600):
    """Make a picture for checking the colour settings. Returns (picture, detection).

    Left panel: the frame with an outline around every region the colour mask picked up,
    and a ring around the region that was chosen as the marker.
    Right panel: the mask itself (white is the marker colour).
    detection is a MarkerDetection, or None if no marker was found.
    """
    mask = make_colour_mask(frame_bgr, settings)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detection = find_marker(frame_bgr, settings, centre, region_radius_px)

    left = frame_bgr.copy()
    cv2.drawContours(left, contours, -1, CENTRE_COLOUR, 1)
    right = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    if centre is not None:
        draw_pulley_region(left, centre, region_radius_px)
        draw_pulley_region(right, centre, region_radius_px)
    if detection is not None:
        draw_marker_ring(left, detection.x, detection.y, detection.area_px, QUALITY_COLOURS["Good"])

    put_label(left, "frame with mask outlines", (10, 24), 0.6)
    put_label(right, "colour mask (white = marker colour)", (10, 24), 0.6)

    picture = np.hstack([left, right])
    scale = min(1.0, max_width / picture.shape[1])
    if scale < 1.0:
        picture = cv2.resize(picture, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return picture, detection
