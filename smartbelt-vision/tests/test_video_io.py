"""Tests for src/video_io.py. A tiny video is written to a temporary folder for each test."""

import cv2
import numpy as np
import pytest

from src.video_io import (
    VideoError,
    VideoInfo,
    check_video_info,
    count_frames,
    frame_times,
    get_video_info,
    iter_frames,
    read_frame,
)


def write_tiny_video(path, fps=20.0, frame_count=10, size=(64, 48)):
    """Write a small grey video. Frame i has brightness 25 * i, so frames can be told apart."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frame_count):
        frame = np.full((size[1], size[0], 3), 25 * i, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def tiny_video(tmp_path):
    return write_tiny_video(tmp_path / "tiny.mp4")


def test_get_video_info_reads_basic_values(tiny_video):
    info = get_video_info(tiny_video)
    assert info.width == 64
    assert info.height == 48
    assert info.fps == pytest.approx(20.0)
    assert info.fps_source == "read from file"
    assert info.frame_count == 10
    assert info.duration_s == pytest.approx(0.5)


def test_fps_can_be_overridden(tiny_video):
    info = get_video_info(tiny_video, fps_override=120)
    assert info.fps == 120.0
    assert info.fps_source == "entered by user"
    assert info.duration_s == pytest.approx(10 / 120)


@pytest.mark.parametrize("bad_fps", [0, -30, "abc", float("nan")])
def test_bad_fps_is_rejected(tiny_video, bad_fps):
    with pytest.raises(VideoError):
        get_video_info(tiny_video, fps_override=bad_fps)


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(VideoError, match="not found"):
        get_video_info(tmp_path / "does_not_exist.mp4")


def test_file_that_is_not_a_video_gives_clear_error(tmp_path):
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"this is not a video")
    with pytest.raises(VideoError):
        get_video_info(fake)


def test_read_frame_returns_the_requested_frame(tiny_video):
    frame = read_frame(tiny_video, 4)
    assert frame.shape == (48, 64, 3)
    # Frame 4 was written with brightness 100. Video compression changes it slightly.
    assert frame.mean() == pytest.approx(100, abs=8)


def test_read_frame_beyond_the_end_gives_error(tiny_video):
    with pytest.raises(VideoError):
        read_frame(tiny_video, 500)


def test_iter_frames_gives_every_frame_in_order(tiny_video):
    indices = [index for index, _ in iter_frames(tiny_video)]
    assert indices == list(range(10))


def test_iter_frames_start_frame_and_max_frames(tiny_video):
    indices = [index for index, _ in iter_frames(tiny_video, start_frame=3, max_frames=4)]
    assert indices == [3, 4, 5, 6]

    first_index, first_frame = next(iter_frames(tiny_video, start_frame=3))
    assert first_index == 3
    # Frame 3 was written with brightness 75.
    assert first_frame.mean() == pytest.approx(75, abs=8)


def test_count_frames(tiny_video):
    assert count_frames(tiny_video) == 10


def test_frame_times():
    times = frame_times(5, 10.0)
    assert np.allclose(times, [0.0, 0.1, 0.2, 0.3, 0.4])


def test_check_video_info_flags_short_video_and_odd_frame_rate():
    short = VideoInfo("a.mp4", 64, 48, 30.0, "read from file", 30, 1.0)
    assert len(check_video_info(short)) == 1

    odd_fps = VideoInfo("a.mp4", 64, 48, 2000.0, "read from file", 4000, 2.0)
    assert len(check_video_info(odd_fps)) == 1

    fine = VideoInfo("a.mp4", 1920, 1080, 60.0, "read from file", 600, 10.0)
    assert check_video_info(fine) == []
