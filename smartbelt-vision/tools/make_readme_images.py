"""Makes the pictures for the README from simulated videos. Run it from the project folder:

    python tools/make_readme_images.py

Everything is made by the program itself: it writes simulated videos with known values, runs the analysis on them,
and saves the plots, the annotated frames, a short GIF of the tracking, the validation plot and a diagram of the
pipeline into the assets folder. The file names start with "simulated" because the data is simulated.
The screenshots of the app are made by tools/app_screenshots.py (it needs Playwright).
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

import run_validation  # noqa: E402
from src import app_support as support  # noqa: E402
from src.reporting import plot_validation, save_figure  # noqa: E402
from src.synthetic import write_synthetic_belt_video, write_synthetic_video  # noqa: E402
from src.tracking import settings_for_colour  # noqa: E402

ASSETS = ROOT / "assets"
DOCS = ROOT / "docs"


def save_rgb(path: Path, picture) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(picture, cv2.COLOR_RGB2BGR))
    print("saved", path.relative_to(ROOT))


def save_png(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    print("saved", path.relative_to(ROOT))


def belt_inputs(info) -> dict:
    items = {}
    for name, colour in (("driver", "green"), ("driven", "orange")):
        items[name] = support.PulleyInput(
            settings_for_colour(colour), tuple(info[f"{name}_centre"]), round(info[f"{name}_radius_px"] * 1.05),
            info[f"{name}_effective_radius_mm"], info[f"{name}_marker_radius_mm"],
        )
    return items


def draw_pipeline(path: Path) -> None:
    """A diagram of the steps of the analysis."""
    figure, axis = plt.subplots(figsize=(13, 5.6))
    axis.set_xlim(0, 13)
    axis.set_ylim(0, 5.6)
    axis.axis("off")
    width, height = 2.2, 1.35
    colours = {"main": "#dbe9f6", "belt": "#fde8cf", "out": "#dff1e1"}

    def box(x, y, title, detail, kind="main"):
        axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.03,rounding_size=0.12",
                                      facecolor=colours[kind], edgecolor="#444444", linewidth=1.2))
        axis.text(x + width / 2, y + height - 0.32, title, ha="center", va="center", fontsize=12, fontweight="bold")
        axis.text(x + width / 2, y + 0.42, detail, ha="center", va="center", fontsize=9.5, linespacing=1.35)
        return x, y

    def arrow(start, end):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.5, color="#333333", shrinkA=2, shrinkB=2))

    xs = [0.2, 2.8, 5.4, 8.0, 10.6]
    top, bottom = 3.7, 0.5
    box(xs[0], top, "Phone video", "mp4 or mov\ncoloured markers\non the pulleys")
    box(xs[1], top, "Video reader", "frames, frame rate,\nframe times")
    box(xs[2], top, "Marker tracking", "HSV colour mask,\ncentroid, quality\nof every frame")
    box(xs[3], top, "Angle and speed", "unwrapped angle,\nrpm in every frame,\ncircle fit of the centre")
    box(xs[4], top, "Signal analysis", "smoothing, trend,\nspectrum, dominant\nfrequencies")
    box(xs[2], bottom, "Belt mark", "belt speed on the\nstraight spans, scale", "belt")
    box(xs[3], bottom, "Belt drive", "speed ratio, speed loss,\nsurface speeds, slip", "belt")
    box(xs[4], bottom, "Outputs", "CSV, plots, HTML report,\nStreamlit app", "out")
    for index in range(4):
        arrow((xs[index] + width, top + height / 2), (xs[index + 1], top + height / 2))
    arrow((xs[2] + width / 2, top), (xs[2] + width / 2, bottom + height))      # belt mark is tracked like a marker
    arrow((xs[2] + width, bottom + height / 2), (xs[3], bottom + height / 2))  # belt mark to belt drive
    arrow((xs[3] + width / 2, top), (xs[3] + width / 2, bottom + height))      # speeds to belt drive
    arrow((xs[3] + width, bottom + height / 2), (xs[4], bottom + height / 2))  # belt drive to outputs
    arrow((xs[4] + width / 2, top), (xs[4] + width / 2, bottom + height))      # signal analysis to outputs
    figure.tight_layout()
    save_figure(figure, path, dpi=140)
    print("saved", path.relative_to(ROOT))


def make_gif(result, path: Path, frames: int = 60, step: int = 2, width: int = 640) -> None:
    pictures = []
    for index in range(frames):
        picture, _, _ = support.annotated_frame(result, 20 + index * step)
        image = Image.fromarray(picture)
        image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
        pictures.append(image.quantize(colors=128, method=Image.MEDIANCUT, dither=Image.Dither.NONE))
    pictures[0].save(path, save_all=True, append_images=pictures[1:], duration=50, loop=0, optimize=True, disposal=2)
    print("saved", path.relative_to(ROOT), f"({path.stat().st_size // 1024} KB)")


def validation_markdown(rows) -> str:
    lines = [
        "# Validation on simulated videos",
        "",
        "Made by `python run_validation.py`. The videos are simulated, so the true value of every quantity is known.",
        "A row with no limit only shows the size of an error (a frame rate that is typed wrong on purpose, and the turns counted across a long gap).",
        "",
        "| Group | Test | Quantity | Measured | Reference | Error | Tolerance | Result |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        result = {True: "within", False: "outside", None: "no limit"}[row["passed"]]
        lines.append(
            f"| {row['group']} | {row['test']} | {row['quantity']} ({row['unit']}) | {row['measured']:.3f} | {row['reference']:.3f} | "
            f"{row['error']:+.3f} | {row['tolerance']} | {result} |"
        )
    judged = [row for row in rows if row["passed"] is not None]
    lines += ["", f"{sum(1 for row in judged if row['passed'])} of {len(judged)} checks are within their tolerance.", ""]
    return "\n".join(lines)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smartbelt_images_") as folder:
        folder = Path(folder)

        # A belt drive with a mark on the belt: the driver slips 1 percent, the driven pulley 2 percent more.
        belt_path = folder / "synthetic_belt.mp4"
        info = write_synthetic_belt_video(belt_path, driver_rpm=120.0, ratio=2.0, driver_slip_percent=1.0, slip_percent=2.0,
                                          draw_belt=True, seconds=5.0, scale_mm_per_px=0.25)
        items = belt_inputs(info)
        belt = support.run_analysis(support.RunConfig(str(belt_path), "belt", items, belt_settings=settings_for_colour("blue")))

        markers = [(item.centre, item.region_radius) for item in items.values()]
        save_rgb(ASSETS / "simulated_frame_preview.png", support.frame_preview(belt_path, 0, markers, 100))
        save_rgb(ASSETS / "simulated_annotated_frame.png", support.annotated_frame(belt, 150)[0])
        picture, _ = support.mask_preview(belt_path, 0, items["driver"].settings, items["driver"].centre, items["driver"].region_radius)
        save_rgb(ASSETS / "simulated_mask_check_driver.png", picture)
        picture, _ = support.mask_preview(belt_path, 0, settings_for_colour("blue"))
        save_rgb(ASSETS / "simulated_mask_check_belt_mark.png", picture)
        make_gif(belt, ASSETS / "simulated_tracking.gif")
        for suffix, _, png in belt.figures["belt"]:
            save_png(ASSETS / f"simulated_{suffix}.png", png)

        # One pulley that wobbles at 3 Hz, for the speed plot and the spectrum.
        ripple_path = folder / "synthetic_ripple.mp4"
        write_synthetic_video(ripple_path, rpm=150.0, fps=60.0, seconds=6.0, ripple_deg=4.0, ripple_hz=3.0, noise_sigma=6.0)
        single = support.run_analysis(
            support.RunConfig(str(ripple_path), "single", {"pulley": support.PulleyInput(settings_for_colour("green"), (320.0, 240.0))})
        )
        for suffix, _, png in single.figures["pulley"]:
            save_png(ASSETS / f"simulated_{suffix}.png", png)

    # The validation on simulated videos.
    rows, _ = run_validation.run_simulated_suite(quick=False)
    save_figure(plot_validation(rows), ASSETS / "simulated_validation_summary.png", dpi=150)
    print("saved assets/simulated_validation_summary.png")
    (DOCS / "validation.md").write_text(validation_markdown(rows), encoding="utf-8")
    print("saved docs/validation.md")

    draw_pipeline(ASSETS / "pipeline.png")


if __name__ == "__main__":
    main()
