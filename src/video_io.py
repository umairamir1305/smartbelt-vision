"""Reading video files with OpenCV.

A video is a sequence of frames. OpenCV gives each frame as a NumPy array with
shape (height, width, 3). The three numbers in each pixel are in BGR order
(blue, green, red), not RGB. The origin (0, 0) is the top-left corner of the
image and the y axis points DOWN. This matters later when angles are calculated.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class VideoError(Exception):
    """A problem with a video file. The message is written so it can be shown to the user."""


@dataclass
class VideoInfo:
    """Basic facts about a video, kept together so a function can return them in one go."""

    path: str
    width: int  # pixels, measured from the first frame that was read
    height: int  # pixels, measured from the first frame that was read
    fps: float  # frames per second, used for the time axis
    fps_source: str  # "read from file" or "entered by user"
    frame_count: int  # number of frames reported by the file
    duration_s: float  # frame_count / fps, in seconds


def open_video(video_path) -> cv2.VideoCapture:
    """Open a video file and return an OpenCV VideoCapture object.

    The caller must call .release() on the returned object when finished.
    """
    path = Path(video_path)
    if not path.exists():
        raise VideoError(f"Video file not found: {path}. Check the file name and the folder.")
    if not path.is_file():
        raise VideoError(f"{path} is a folder, not a video file.")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoError(
            f"OpenCV could not open {path.name}. The file may be damaged or may use a codec "
            "that OpenCV cannot read. Try recording again in H.264 format (called 'Most "
            "Compatible' on iPhone), and avoid special characters in the file path."
        )
    return capture


def _check_fps(fps_value) -> float:
    """Check a frame rate typed in by the user and return it as a float."""
    try:
        fps = float(fps_value)
    except (TypeError, ValueError):
        raise VideoError("The frame rate must be a number, for example 30 or 59.94.") from None
    if not np.isfinite(fps) or fps <= 0:
        raise VideoError("The frame rate must be a positive number, for example 30 or 59.94.")
    return fps


def get_video_info(video_path, fps_override=None) -> VideoInfo:
    """Read the basic facts about a video.

    fps_override: a frame rate typed in by the user. If given, it replaces the value
    stored in the file (useful when the file reports the wrong frame rate).
    """
    if fps_override is not None:
        fps_override = _check_fps(fps_override)

    capture = open_video(video_path)
    try:
        file_fps = capture.get(cv2.CAP_PROP_FPS)
        reported_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        got_frame, first_frame = capture.read()
    finally:
        capture.release()

    if not got_frame:
        raise VideoError("The video opened but no frame could be read from it. The file may be empty or damaged.")

    # Width and height come from the frame itself. The numbers in the file header can
    # disagree with the real frames when a phone video carries rotation information.
    height, width = first_frame.shape[:2]

    if fps_override is not None:
        fps, fps_source = fps_override, "entered by user"
    elif np.isfinite(file_fps) and file_fps > 0:
        fps, fps_source = float(file_fps), "read from file"
    else:
        raise VideoError(
            "The frame rate could not be read from the file. Enter the frame rate yourself (for example 60)."
        )

    frame_count = int(reported_frames) if np.isfinite(reported_frames) else 0
    if frame_count <= 0:
        # Some files do not report a frame count, so count by reading the video.
        frame_count = count_frames(video_path)

    return VideoInfo(
        path=str(video_path),
        width=width,
        height=height,
        fps=fps,
        fps_source=fps_source,
        frame_count=frame_count,
        duration_s=frame_count / fps,
    )


def read_frame(video_path, frame_index: int = 0) -> np.ndarray:
    """Read one frame (a BGR image) from the video. Frame 0 is the first frame."""
    if frame_index < 0:
        raise VideoError("The frame number cannot be negative.")

    capture = open_video(video_path)
    try:
        if frame_index > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        got_frame, frame = capture.read()
    finally:
        capture.release()

    if not got_frame:
        raise VideoError(f"Could not read frame {frame_index}. The video may have fewer frames than that.")
    return frame


def iter_frames(video_path, start_frame: int = 0, max_frames=None):
    """Go through the video one frame at a time. Gives (frame_index, frame) pairs.

    This is a generator: the word 'yield' hands over one frame and then waits.
    The whole video is never held in memory at once. One 1920 x 1080 frame needs
    about 6 MB, so a few hundred frames in a list would already use gigabytes.

    Use it in a for loop:
        for frame_index, frame in iter_frames("video.mp4"):
            ...
    """
    if start_frame < 0:
        raise VideoError("The start frame cannot be negative.")

    capture = open_video(video_path)
    try:
        if start_frame > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_index = start_frame
        frames_given = 0
        while max_frames is None or frames_given < max_frames:
            got_frame, frame = capture.read()
            if not got_frame:  # end of the video (or a frame that could not be decoded)
                break
            yield frame_index, frame
            frame_index += 1
            frames_given += 1
    finally:
        capture.release()


def count_frames(video_path) -> int:
    """Count the frames by reading the whole video. Slower than the file header but reliable."""
    return sum(1 for _ in iter_frames(video_path))


def frame_times(frame_count: int, fps: float) -> np.ndarray:
    """Time of each frame in seconds: frame 0 is at 0 s, frame 1 at 1/fps, and so on.

    This assumes the frames are evenly spaced in time (constant frame rate).
    If the real frame rate differs from the value used here, every speed
    calculated later is wrong by the same factor.
    """
    return np.arange(frame_count) / fps


def check_video_info(info: VideoInfo) -> list:
    """Return a list of warning messages. An empty list means nothing suspicious was found."""
    warnings = []
    if info.fps < 5 or info.fps > 1000:
        warnings.append(
            f"The frame rate ({info.fps:g} fps) is unusual for a phone video. "
            "Compare it with your camera setting and correct it with --fps if needed."
        )
    if info.duration_s < 2.0:
        warnings.append(
            f"The video is only {info.duration_s:.2f} s long. A few seconds of footage or more "
            "is needed for the speed and frequency analysis."
        )
    return warnings


def frame_timestamps(video_path, start_frame: int = 0, max_frames=None) -> np.ndarray:
    """The time of every frame in seconds, as the video file gives it.

    Phone videos can have a frame rate that changes. The analysis assumes that every frame is 1 / fps after the
    one before. validation.check_time_axis compares that with these times. A file or a backend that gives no
    times returns values that are all 0.
    """
    capture = open_video(video_path)
    times = []
    try:
        if start_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while max_frames is None or len(times) < max_frames:
            if not capture.grab():
                break
            times.append(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    finally:
        capture.release()
    return np.asarray(times, dtype=float)
