"""Small helper functions shared by the other modules."""

from pathlib import Path

import cv2
import numpy as np

# The project folder is one level above this file (src/utils.py -> project folder).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def looks_simulated(video_path) -> bool:
    """True if the file name says the video is simulated, that is, it starts with 'synthetic'.

    make_test_video.py names its files like that. Plots made from such a video are
    labelled as simulated so they cannot be mistaken for real results.
    """
    return Path(video_path).stem.lower().startswith("synthetic")


def ensure_folder(folder) -> Path:
    """Create a folder (and any missing parent folders) and return it as a Path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_image(path, image: np.ndarray) -> Path:
    """Save an image to disk. Raises an error if OpenCV could not write the file."""
    path = Path(path)
    ensure_folder(path.parent)
    saved = cv2.imwrite(str(path), image)
    if not saved:
        raise OSError(
            f"Could not save the image to {path}. "
            "Check that the folder can be written to and the path has no special characters."
        )
    return path


def put_label(image: np.ndarray, text: str, position: tuple, scale: float = 0.45) -> None:
    """Write white text with a dark outline so it can be read on both light and dark areas.

    position is the bottom-left corner of the text, as (x, y) in pixels.
    Only plain ASCII characters are supported by OpenCV text drawing.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = int(position[0]), int(position[1])
    # The outline is the same text drawn one pixel away in every direction, with the same thickness as the white text.
    # (A thicker black text would be wider than the white text in OpenCV 5, because the thickness changes the letter shapes.)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                cv2.putText(image, text, (x + dx, y + dy), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_pixel_grid(frame: np.ndarray, step: int = 100) -> np.ndarray:
    """Return a copy of the frame with labelled grid lines.

    The labels are pixel coordinates: x values along the top, y values down the left side.
    The origin (0, 0) is the top-left corner of the image and y increases downwards.
    This makes it easy to read the approximate position of the pulley centre by eye.
    """
    if step < 10:
        raise ValueError("The grid step must be at least 10 pixels.")

    gridded = frame.copy()
    height, width = gridded.shape[:2]
    line_colour = (0, 140, 255)  # orange, in BGR order

    for x in range(0, width, step):
        cv2.line(gridded, (x, 0), (x, height - 1), line_colour, 1)
        put_label(gridded, str(x), (x + 3, 14))

    # Start at y = step so the first y label does not sit on top of the x labels.
    for y in range(step, height, step):
        cv2.line(gridded, (0, y), (width - 1, y), line_colour, 1)
        put_label(gridded, str(y), (3, y - 4))

    return gridded
