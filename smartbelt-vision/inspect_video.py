"""Check a video before analysing it: print the basic information and save the first frame.

Run from the project folder, for example:
    python inspect_video.py sample_data/synthetic_disc_120rpm_60fps.mp4
    python inspect_video.py sample_data/disc_60fps_handspin_01.mp4 --fps 60
    python inspect_video.py sample_data/disc_60fps_handspin_01.mp4 --count

Two pictures are saved in the outputs folder:
    <video name>_first_frame.png        the first frame as it is
    <video name>_first_frame_grid.png   the same frame with a pixel grid, for reading
                                        the approximate position of the pulley centre
"""

import argparse
import sys
from pathlib import Path

from src.utils import OUTPUT_DIR, draw_pixel_grid, save_image
from src.video_io import VideoError, check_video_info, count_frames, get_video_info, read_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print basic information about a video and save its first frame."
    )
    parser.add_argument("video", help="path to the video file")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="frame rate to use instead of the one stored in the file",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="read the whole video to count the frames (slower, but more reliable)",
    )
    parser.add_argument(
        "--grid-step",
        type=int,
        default=100,
        help="distance between grid lines in pixels (default 100)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        info = get_video_info(args.video, fps_override=args.fps)
        first_frame = read_frame(args.video, 0)
        frames_read = count_frames(args.video) if args.count else None

        stem = Path(args.video).stem
        plain_path = save_image(OUTPUT_DIR / f"{stem}_first_frame.png", first_frame)
        grid_path = save_image(
            OUTPUT_DIR / f"{stem}_first_frame_grid.png",
            draw_pixel_grid(first_frame, args.grid_step),
        )
    except (VideoError, OSError, ValueError) as error:
        print(f"Error: {error}")
        return 1

    # Rule of thumb used in this project: less than 45 degrees of rotation per frame.
    # 45 degrees is 1/8 of a turn, so the limit is (1/8) turn * fps * 60 s = 7.5 * fps rpm.
    rule_of_thumb_rpm = (45.0 / 360.0) * info.fps * 60.0

    print("SmartBelt Vision: video check")
    print("-----------------------------")
    print(f"File          : {info.path}")
    print(f"Frame size    : {info.width} x {info.height} pixels")
    print(f"Frame rate    : {info.fps:.2f} fps ({info.fps_source})")
    print(f"Time step     : {1000.0 / info.fps:.3f} ms between frames")
    print(f"Frames        : {info.frame_count} (reported by the file)")
    if frames_read is not None:
        print(f"Frames read   : {frames_read} (counted by reading the whole video)")
    print(f"Duration      : {info.duration_s:.3f} s")
    print(
        f"Rule of thumb : keep the rotation below about {rule_of_thumb_rpm:.0f} rpm at this frame rate "
        "(less than 45 degrees of rotation per frame)"
    )
    print(f"Saved         : {plain_path}")
    print(f"Saved         : {grid_path}")

    if frames_read is not None and frames_read != info.frame_count:
        print(
            "Note: the frame count in the file header and the number of frames actually read are "
            "different. This can happen with some phone files. The number actually read is the "
            "one to trust."
        )

    for message in check_video_info(info):
        print(f"Warning: {message}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
