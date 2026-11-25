"""Simulated data for testing the code. NOTHING in this file is a real measurement.

The video generator draws a light disc with one coloured marker that turns at a
known speed. Because the true speed is known, the tracking and rpm code can be
checked against the right answer before any real video is recorded.

A simulated video has no motion blur, no rolling shutter, no lens distortion and
no lighting changes. Passing a test on it shows the maths and the code are
correct. It does NOT show how accurate the method is on real phone footage.

Conventions (the same ones the analysis code uses):
  - The image origin is the top-left corner, x goes right and y goes DOWN.
  - Positive angle and positive rpm mean counter-clockwise rotation as seen in the video.
"""

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Marker colours in BGR order (OpenCV uses blue, green, red, not RGB).
MARKER_COLOURS_BGR = {
    "green": (0, 200, 0),
    "red": (0, 0, 220),
    "blue": (220, 80, 0),
    "orange": (0, 140, 255),
    "yellow": (0, 220, 255),
    "magenta": (200, 0, 200),
}

# The marker starts 30 degrees counter-clockwise from the 3 o'clock position.
DEFAULT_START_ANGLE_RAD = math.radians(30.0)

# Circles are drawn with 4 fractional bits (1/16 of a pixel) so that the motion
# is smooth and the marker position is not rounded to whole pixels.
_SHIFT = 4
_SCALE = 1 << _SHIFT


def disc_geometry(width: int, height: int):
    """Return (centre, disc_radius, orbit_radius, marker_radius) in pixels.

    centre is the rotation centre as (x, y). orbit_radius is the distance from the
    centre to the middle of the marker. All sizes scale with the frame height.
    """
    centre = (width // 2, height // 2)
    disc_radius = 0.38 * height
    orbit_radius = 0.30 * height
    marker_radius = 0.03 * height
    return centre, disc_radius, orbit_radius, marker_radius


def true_marker_angle(
    times_s,
    rpm: float,
    start_angle_rad: float = DEFAULT_START_ANGLE_RAD,
    ripple_amplitude_rad: float = 0.0,
    ripple_hz: float = 0.0,
    decay_s: float = 0.0,
) -> np.ndarray:
    """True marker angle in radians (counter-clockwise positive) at the given times.

    Constant speed:
        theta(t) = start_angle + omega * t + A * sin(2 * pi * f * t)

    omega = 2 * pi * rpm / 60 is the angular velocity in rad/s. The last term is an
    optional periodic ripple with amplitude A (rad) and frequency f (Hz). It is used
    later to test the frequency analysis.

    Slowing down (decay_s greater than 0): the speed falls off exponentially,
    omega(t) = omega0 * exp(-t / decay_s), like a disc that was flicked and is
    slowing down by itself. rpm is then the starting speed and
        theta(t) = start_angle + omega0 * decay_s * (1 - exp(-t / decay_s)) + ripple
    This is only a simple test signal. A real disc also has other kinds of friction.
    """
    times_s = np.asarray(times_s, dtype=float)
    omega = 2.0 * math.pi * rpm / 60.0
    if decay_s > 0:
        turning = omega * decay_s * (1.0 - np.exp(-times_s / decay_s))
    else:
        turning = omega * times_s
    ripple = ripple_amplitude_rad * np.sin(2.0 * math.pi * ripple_hz * times_s)
    return start_angle_rad + turning + ripple


def true_marker_speed_rpm(
    times_s,
    rpm: float,
    ripple_amplitude_rad: float = 0.0,
    ripple_hz: float = 0.0,
    decay_s: float = 0.0,
) -> np.ndarray:
    """True rotation speed in rpm (counter-clockwise positive) at the given times.

    This is the derivative of true_marker_angle, converted to rpm:
        constant speed:  omega(t) = omega0 + A * 2 * pi * f * cos(2 * pi * f * t)
        slowing down:    omega(t) = omega0 * exp(-t / decay_s) + A * 2 * pi * f * cos(2 * pi * f * t)
    with omega0 = 2 * pi * rpm / 60. It is the answer that the speed calculation should find.
    """
    times_s = np.asarray(times_s, dtype=float)
    omega0 = 2.0 * math.pi * rpm / 60.0
    if decay_s > 0:
        steady_part = omega0 * np.exp(-times_s / decay_s)
    else:
        steady_part = np.full_like(times_s, omega0)
    ripple_part = ripple_amplitude_rad * 2.0 * math.pi * ripple_hz * np.cos(2.0 * math.pi * ripple_hz * times_s)
    return (steady_part + ripple_part) * 60.0 / (2.0 * math.pi)


def make_synthetic_table(
    rpm: float = 120.0,
    fps: float = 60.0,
    seconds: float = 5.0,
    centre=(320.0, 240.0),
    radius_px: float = 144.0,
    ripple_deg: float = 0.0,
    ripple_hz: float = 0.0,
    decay_s: float = 0.0,
    noise_px: float = 0.1,
    lost_frames=(),
    seed: int = 1,
) -> pd.DataFrame:
    """Simulated marker positions, one row per frame, without making or tracking any video.

    The table has the same columns that track_marker makes before the quality rating:
    frame, time_s, marker_x, marker_y, area_px. The marker moves on a circle of radius_px
    around centre, at the true angle from true_marker_angle. Random position noise with a
    standard deviation of noise_px pixels is added to x and y (0.1 pixels is about what the
    tracking gives on the simulated videos). Frames in lost_frames have no marker (NaN).

    To go on from here:
        table = add_tracking_quality(table, centre)
        table = add_speed_columns(add_angle_columns(table, centre))
    This is SIMULATED data, only for testing the maths.
    """
    if fps <= 0 or seconds <= 0:
        raise ValueError("fps and seconds must both be positive.")
    frame_count = int(round(fps * seconds))
    times_s = np.arange(frame_count) / fps
    angle = true_marker_angle(
        times_s, rpm, ripple_amplitude_rad=math.radians(ripple_deg), ripple_hz=ripple_hz, decay_s=decay_s
    )

    rng = np.random.default_rng(seed)
    marker_x = centre[0] + radius_px * np.cos(angle) + rng.normal(0.0, noise_px, frame_count)
    marker_y = centre[1] - radius_px * np.sin(angle) + rng.normal(0.0, noise_px, frame_count)  # image y points down
    area = np.full(frame_count, math.pi * (0.1 * radius_px) ** 2)

    lost = np.zeros(frame_count, dtype=bool)
    lost[[frame for frame in lost_frames if 0 <= frame < frame_count]] = True
    marker_x[lost] = np.nan
    marker_y[lost] = np.nan
    area[lost] = np.nan

    return pd.DataFrame(
        {"frame": np.arange(frame_count), "time_s": times_s, "marker_x": marker_x, "marker_y": marker_y, "area_px": area}
    )


def belt_drive_geometry(width: int, height: int, ratio: float):
    """Positions and sizes (in pixels) of the two discs of the simulated belt drive.

    ratio is the radius of the driven disc divided by the radius of the driver disc.
    Returns (driver_centre, driver_radius, driven_centre, driven_radius, marker_radius).
    The driver is on the left, the driven disc on the right, both at half the frame height.
    """
    if ratio <= 0:
        raise ValueError("The radius ratio must be positive.")
    driver_radius = 0.18 * height
    driven_radius = ratio * driver_radius
    margin = 0.08 * height
    gap = 0.06 * height
    if 2 * driven_radius + 2 * margin > height:
        raise ValueError("The radius ratio is too big for this frame height. Use a smaller ratio or a taller frame.")
    driver_x = margin + driver_radius
    driven_x = driver_x + driver_radius + gap + driven_radius
    if driven_x + driven_radius + margin > width:
        raise ValueError("The frame is too narrow for the two discs. Use a wider frame or a smaller ratio.")
    centre_y = height // 2
    marker_radius = max(5.0, 0.12 * driver_radius)
    return (round(driver_x), centre_y), driver_radius, (round(driven_x), centre_y), driven_radius, marker_radius


# The simulated belt runs on a circle of this fraction of the disc radius, so it shows as a band near the rim.
BELT_RADIUS_FRACTION = 0.95


class BeltLoop:
    """The path of an open belt around two discs, for the simulated belt drive.

    The belt is two straight spans (the external tangents of the two circles) and two arcs. The belt
    travels counter-clockwise around the loop, which is right for two discs that both turn counter-clockwise:
    along the lower span to the right, round the driven disc, back along the upper span to the left and round
    the driver. Positions are given by the distance u along the belt (arc length, in pixels).
    """

    def __init__(self, driver_centre, driver_radius, driven_centre, driven_radius):
        self.c1, self.a1 = driver_centre, driver_radius
        self.c2, self.a2 = driven_centre, driven_radius
        distance = driven_centre[0] - driver_centre[0]  # the centres are at the same height
        # Normal angle of the upper tangent line, seen from each centre (maths convention, y up).
        self.alpha = math.acos((self.a1 - self.a2) / distance)
        self.straight = math.sqrt(distance**2 - (self.a2 - self.a1) ** 2)
        self.arc_driven = 2.0 * self.alpha * self.a2  # wraps 2 alpha of the driven disc
        self.arc_driver = (2.0 * math.pi - 2.0 * self.alpha) * self.a1
        self.length = 2.0 * self.straight + self.arc_driven + self.arc_driver
        self.wrap_driver_rad = 2.0 * math.pi - 2.0 * self.alpha
        self.wrap_driven_rad = 2.0 * self.alpha

    def _arc_point(self, centre, radius, angle):
        return centre[0] + radius * math.cos(angle), centre[1] - radius * math.sin(angle)  # image y points down

    def point_at(self, u: float):
        """Pixel position (x, y) at distance u along the belt, counting from the start of the lower span."""
        u = u % self.length
        alpha = self.alpha
        if u < self.straight:  # lower span, from the driver to the driven disc
            start = self._arc_point(self.c1, self.a1, -alpha)
            end = self._arc_point(self.c2, self.a2, -alpha)
            fraction = u / self.straight
            return start[0] + fraction * (end[0] - start[0]), start[1] + fraction * (end[1] - start[1])
        u -= self.straight
        if u < self.arc_driven:  # round the driven disc, from -alpha to +alpha
            return self._arc_point(self.c2, self.a2, -alpha + u / self.a2)
        u -= self.arc_driven
        if u < self.straight:  # upper span, from the driven disc back to the driver
            start = self._arc_point(self.c2, self.a2, alpha)
            end = self._arc_point(self.c1, self.a1, alpha)
            fraction = u / self.straight
            return start[0] + fraction * (end[0] - start[0]), start[1] + fraction * (end[1] - start[1])
        u -= self.straight  # round the driver, from alpha to 2 pi - alpha
        return self._arc_point(self.c1, self.a1, alpha + u / self.a1)

    def polyline(self, count: int = 600):
        """Points along the whole belt, for drawing it."""
        return np.array([self.point_at(self.length * i / count) for i in range(count)])


def belt_loop(width: int, height: int, ratio: float) -> BeltLoop:
    """The belt of the simulated belt drive (see belt_drive_geometry for the discs)."""
    driver_centre, driver_radius, driven_centre, driven_radius, _ = belt_drive_geometry(width, height, ratio)
    return BeltLoop(driver_centre, BELT_RADIUS_FRACTION * driver_radius, driven_centre, BELT_RADIUS_FRACTION * driven_radius)


def make_belt_drive_frame(
    width,
    height,
    ratio,
    driver_angle_rad,
    driven_angle_rad,
    driver_colour="green",
    driven_colour="orange",
    noise_sigma=0.0,
    rng=None,
    show_driver_marker=True,
    show_driven_marker=True,
    belt=None,
    belt_mark_xy=None,
    belt_colour="blue",
) -> np.ndarray:
    """Draw one frame of the simulated belt drive: two light discs, each with a coloured marker.

    Angles are counter-clockwise positive, as everywhere else. The markers sit at 0.75 of the
    disc radius. If belt (a BeltLoop) is given, the belt is drawn as a dark band near the rim of the
    discs and along the two straight spans. belt_mark_xy, if given, is the pixel position of a small
    coloured mark on the belt.
    """
    for colour in (driver_colour, driven_colour, belt_colour):
        if colour not in MARKER_COLOURS_BGR:
            raise ValueError(f"Unknown marker colour '{colour}'. Choose one of: {', '.join(MARKER_COLOURS_BGR)}.")

    driver_centre, driver_radius, driven_centre, driven_radius, marker_radius = belt_drive_geometry(width, height, ratio)
    frame = np.full((height, width, 3), 45, dtype=np.uint8)

    discs = (
        (driver_centre, driver_radius, driver_angle_rad, driver_colour, show_driver_marker, (235, 235, 235)),
        (driven_centre, driven_radius, driven_angle_rad, driven_colour, show_driven_marker, (222, 222, 222)),
    )
    for centre, radius, _, _, _, fill in discs:
        _draw_circle(frame, centre, radius, fill)
        _draw_circle(frame, centre, radius, (130, 130, 130), thickness=2)
        _draw_circle(frame, centre, 4, (0, 0, 0))

    if belt is not None:
        points = np.round(belt.polyline() * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [points], True, (70, 70, 70), 6, cv2.LINE_AA, 4)

    for centre, radius, angle_rad, colour, show, _ in discs:
        if show:
            orbit = 0.75 * radius
            marker_x = centre[0] + orbit * math.cos(angle_rad)
            marker_y = centre[1] - orbit * math.sin(angle_rad)  # minus: image y points down
            _draw_circle(frame, (marker_x, marker_y), marker_radius, MARKER_COLOURS_BGR[colour])

    if belt_mark_xy is not None:
        _draw_circle(frame, belt_mark_xy, marker_radius, MARKER_COLOURS_BGR[belt_colour])

    if noise_sigma > 0:
        if rng is None:
            rng = np.random.default_rng(1)
        noisy = frame.astype(float) + rng.normal(0.0, noise_sigma, frame.shape)
        frame = np.clip(noisy, 0, 255).astype(np.uint8)
    return frame


def write_synthetic_belt_video(
    out_path,
    driver_rpm: float = 120.0,
    ratio: float = 2.0,
    slip_percent: float = 0.0,
    crossed: bool = False,
    fps: float = 60.0,
    seconds: float = 5.0,
    width: int = 960,
    height: int = 540,
    driver_colour: str = "green",
    driven_colour: str = "orange",
    noise_sigma: float = 0.0,
    hidden_driver_frames=None,
    hidden_driven_frames=None,
    driver_slip_percent: float = 0.0,
    draw_belt: bool = False,
    belt_colour: str = "blue",
    scale_mm_per_px: float = 0.25,
    hidden_belt_frames=None,
) -> dict:
    """Write a SIMULATED video of two rotating discs, with or without a belt around them.

    The ideal speed of the driven disc is driver_rpm / ratio, where ratio is the radius of the driven
    disc divided by the radius of the driver disc (the surface speeds are equal). slip_percent makes the
    driven disc slower than that by this percentage. crossed=True turns it the opposite way
    (a crossed belt). Returns a dictionary with the true times, angles, speeds and the geometry,
    which is the right answer that the analysis should find.

    With draw_belt=True an open belt is drawn with a small coloured mark on it. The belt moves at the surface
    speed of the driver disc minus driver_slip_percent (the belt slips on the driver). The driven disc
    then runs slower than the belt by slip_percent (it slips on the driven disc). The surface speeds are
    taken on the belt circle, which has 0.95 of the disc radius. scale_mm_per_px only decides what the
    pixels are worth in mm in the returned radii and speeds.
    """
    if fps <= 0 or seconds <= 0:
        raise ValueError("fps and seconds must both be positive.")
    if not 0 <= slip_percent < 100 or not 0 <= driver_slip_percent < 100:
        raise ValueError("The slip percentages must be from 0 to below 100.")
    if draw_belt and crossed:
        raise ValueError("The simulated belt is only drawn for an open belt, not for a crossed belt.")
    if scale_mm_per_px <= 0:
        raise ValueError("The scale must be positive.")
    geometry = belt_drive_geometry(width, height, ratio)  # also checks the sizes
    frame_count = int(round(fps * seconds))
    if frame_count < 2:
        raise ValueError("The video would have fewer than 2 frames. Increase fps or seconds.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ideal_driven_rpm = driver_rpm / ratio * (-1.0 if crossed else 1.0)
    driven_rpm = ideal_driven_rpm * (1.0 - driver_slip_percent / 100.0) * (1.0 - slip_percent / 100.0)
    times_s = np.arange(frame_count) / fps
    driver_angle = true_marker_angle(times_s, driver_rpm)
    driven_angle = true_marker_angle(times_s, driven_rpm, start_angle_rad=math.radians(150.0))

    hidden_driver = set(hidden_driver_frames) if hidden_driver_frames is not None else set()
    hidden_driven = set(hidden_driven_frames) if hidden_driven_frames is not None else set()
    hidden_belt = set(hidden_belt_frames) if hidden_belt_frames is not None else set()

    loop = belt_loop(width, height, ratio) if draw_belt else None
    belt_speed_px_s = None
    belt_positions = None
    if loop is not None:
        omega_driver = 2.0 * math.pi * driver_rpm / 60.0
        belt_speed_px_s = omega_driver * loop.a1 * (1.0 - driver_slip_percent / 100.0)
        start = 0.15 * loop.length
        belt_positions = np.array([loop.point_at(start + belt_speed_px_s * time) for time in times_s])

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise OSError(f"Could not create the video file {out_path}. Check that the folder can be written to.")
    rng = np.random.default_rng(1)
    try:
        for index in range(frame_count):
            writer.write(
                make_belt_drive_frame(
                    width, height, ratio, driver_angle[index], driven_angle[index], driver_colour, driven_colour,
                    noise_sigma, rng, index not in hidden_driver, index not in hidden_driven,
                    belt=loop,
                    belt_mark_xy=belt_positions[index] if loop is not None and index not in hidden_belt else None,
                    belt_colour=belt_colour,
                )
            )
    finally:
        writer.release()

    result = {
        "times_s": times_s,
        "driver_angle_rad": driver_angle,
        "driven_angle_rad": driven_angle,
        "driver_rpm": driver_rpm,
        "driven_rpm": driven_rpm,
        "ideal_driven_rpm": ideal_driven_rpm,
        "ratio": ratio,
        "slip_percent": slip_percent,
        "driver_slip_percent": driver_slip_percent,
        "driver_centre": geometry[0],
        "driver_radius_px": geometry[1],
        "driven_centre": geometry[2],
        "driven_radius_px": geometry[3],
        "scale_mm_per_px": scale_mm_per_px,
        "driver_marker_radius_mm": 0.75 * geometry[1] * scale_mm_per_px,
        "driven_marker_radius_mm": 0.75 * geometry[3] * scale_mm_per_px,
    }
    if loop is not None:
        result.update(
            {
                "belt_loop": loop,
                "belt_positions": belt_positions,
                "belt_speed_px_s": belt_speed_px_s,
                "belt_speed_m_s": belt_speed_px_s * scale_mm_per_px / 1000.0,
                "driver_effective_radius_mm": loop.a1 * scale_mm_per_px,
                "driven_effective_radius_mm": loop.a2 * scale_mm_per_px,
                "driver_surface_speed_m_s": 2.0 * math.pi * driver_rpm / 60.0 * loop.a1 * scale_mm_per_px / 1000.0,
                "driven_surface_speed_m_s": abs(2.0 * math.pi * driven_rpm / 60.0) * loop.a2 * scale_mm_per_px / 1000.0,
            }
        )
    return result


def _draw_circle(image, centre_xy, radius, colour, thickness=-1) -> None:
    """Draw an anti-aliased circle at a position given in fractional pixels."""
    centre = (int(round(centre_xy[0] * _SCALE)), int(round(centre_xy[1] * _SCALE)))
    cv2.circle(image, centre, int(round(radius * _SCALE)), colour, thickness, cv2.LINE_AA, _SHIFT)


def make_disc_frame(
    width, height, angle_rad, colour="green", noise_sigma=0.0, rng=None, show_marker=True
) -> np.ndarray:
    """Draw one frame: dark background, light disc, black axle dot and the marker at angle_rad.

    noise_sigma is the standard deviation of random brightness noise (0 means no noise).
    show_marker=False leaves the marker out, to simulate a frame where it is hidden.
    """
    if colour not in MARKER_COLOURS_BGR:
        raise ValueError(f"Unknown marker colour '{colour}'. Choose one of: {', '.join(MARKER_COLOURS_BGR)}.")

    centre, disc_radius, orbit_radius, marker_radius = disc_geometry(width, height)

    frame = np.full((height, width, 3), 45, dtype=np.uint8)  # dark background
    _draw_circle(frame, centre, disc_radius, (235, 235, 235))  # light disc
    _draw_circle(frame, centre, disc_radius, (130, 130, 130), thickness=2)  # disc edge
    _draw_circle(frame, centre, 4, (0, 0, 0))  # axle at the rotation centre

    if show_marker:
        marker_x = centre[0] + orbit_radius * math.cos(angle_rad)
        marker_y = centre[1] - orbit_radius * math.sin(angle_rad)  # minus: image y points down
        _draw_circle(frame, (marker_x, marker_y), marker_radius, MARKER_COLOURS_BGR[colour])

    if noise_sigma > 0:
        if rng is None:
            rng = np.random.default_rng(1)
        noisy = frame.astype(float) + rng.normal(0.0, noise_sigma, frame.shape)
        frame = np.clip(noisy, 0, 255).astype(np.uint8)
    return frame


def write_synthetic_video(
    out_path,
    rpm: float = 120.0,
    fps: float = 60.0,
    seconds: float = 5.0,
    width: int = 640,
    height: int = 480,
    colour: str = "green",
    ripple_deg: float = 0.0,
    ripple_hz: float = 0.0,
    noise_sigma: float = 0.0,
    hidden_frames=None,
    decay_s: float = 0.0,
) -> np.ndarray:
    """Write a simulated rotating-disc video and return the true marker angle of every frame.

    The returned array is in radians, counter-clockwise positive, one value per frame.
    hidden_frames: frame numbers in which the marker is left out (simulates the marker
    being hidden for a few frames). The returned angles still contain the true angle there.
    decay_s: if greater than 0 the speed falls off exponentially (see true_marker_angle).
    """
    if fps <= 0 or seconds <= 0:
        raise ValueError("fps and seconds must both be positive.")
    if decay_s < 0:
        raise ValueError("decay_s cannot be negative.")
    if width < 160 or height < 120:
        raise ValueError("The frame must be at least 160 x 120 pixels.")
    frame_count = int(round(fps * seconds))
    if frame_count < 2:
        raise ValueError("The video would have fewer than 2 frames. Increase fps or seconds.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    times_s = np.arange(frame_count) / fps
    true_angle_rad = true_marker_angle(
        times_s,
        rpm,
        ripple_amplitude_rad=math.radians(ripple_deg),
        ripple_hz=ripple_hz,
        decay_s=decay_s,
    )

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise OSError(f"Could not create the video file {out_path}. Check that the folder can be written to.")

    hidden = set(hidden_frames) if hidden_frames is not None else set()
    rng = np.random.default_rng(1)  # fixed seed, so the same noise is made every time
    try:
        for frame_index, angle_rad in enumerate(true_angle_rad):
            frame = make_disc_frame(
                width, height, angle_rad, colour, noise_sigma, rng, show_marker=frame_index not in hidden
            )
            writer.write(frame)
    finally:
        writer.release()

    return true_angle_rad
