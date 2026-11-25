"""Tests for src/utils.py."""

import numpy as np
import pytest

from src.utils import draw_pixel_grid, ensure_folder, put_label, save_image


def test_ensure_folder_creates_nested_folders(tmp_path):
    target = tmp_path / "first" / "second"
    result = ensure_folder(target)
    assert result.is_dir()


def test_draw_pixel_grid_returns_a_marked_copy():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    gridded = draw_pixel_grid(frame, step=100)
    assert gridded.shape == frame.shape
    assert frame.max() == 0  # the original frame is not changed
    assert gridded.max() > 0  # the copy has grid lines on it


def test_draw_pixel_grid_rejects_a_tiny_step():
    with pytest.raises(ValueError):
        draw_pixel_grid(np.zeros((50, 50, 3), dtype=np.uint8), step=5)


def test_save_image_writes_a_file(tmp_path):
    image = np.full((20, 30, 3), 128, dtype=np.uint8)
    path = save_image(tmp_path / "sub" / "image.png", image)
    assert path.exists()


def test_looks_simulated_goes_by_the_file_name():
    from src.utils import looks_simulated

    assert looks_simulated("sample_data/synthetic_disc_120rpm_60fps.mp4")
    assert looks_simulated("SYNTHETIC_test.mp4")
    assert not looks_simulated("sample_data/disc_60fps_handspin_01.mp4")
    assert not looks_simulated("sample_data/my_synthetic_idea.mp4")  # only a name that STARTS with synthetic counts


def test_a_label_has_its_outline_directly_around_the_letters_in_every_opencv_version():
    """In OpenCV 5 thick text is wider than thin text. The outline must not run past the end of the white text."""
    image = np.full((60, 520, 3), 128, dtype=np.uint8)
    put_label(image, "driver pulley: tracking Good   frame 150", (10, 35), 0.6)
    white = np.flatnonzero((image.min(axis=2) > 200).any(axis=0))
    dark = np.flatnonzero((image.max(axis=2) < 60).any(axis=0))
    assert len(white) and len(dark)
    assert dark.max() - white.max() <= 2 and white.min() - dark.min() <= 2
