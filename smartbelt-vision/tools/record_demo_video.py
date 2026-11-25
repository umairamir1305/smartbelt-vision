"""Records the demo video of the app: a silent screen recording with captions. Optional: it needs Playwright and ffmpeg.

    pip install playwright
    playwright install chromium
    python tools/record_demo_video.py

It makes a simulated belt drive video, starts the app on port 8603 with that video in the sample folder, and drives
the app like a user: choose the video, set the markers, check the colours, run the analysis and open the views.
A red dot shows the mouse and a caption bar says what is happening. The recording is converted to
assets/demo_app_walkthrough.mp4 (H.264).
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright  # noqa: E402

from src.synthetic import write_synthetic_belt_video  # noqa: E402

ASSETS = ROOT / "assets"
PORT = 8603
URL = f"http://localhost:{PORT}"
VIEWPORT = {"width": 1280, "height": 800}

# A red dot that follows the mouse, and a caption bar at the bottom of the page.
OVERLAY_JS = """
(() => {
  const install = () => {
    if (document.getElementById('demo-cursor')) return;
    const dot = document.createElement('div');
    dot.id = 'demo-cursor';
    dot.style.cssText = 'position:fixed;z-index:2147483647;width:20px;height:20px;border-radius:50%;background:rgba(224,49,49,0.6);' +
      'border:2px solid #ffffff;box-shadow:0 0 4px rgba(0,0,0,0.5);pointer-events:none;transform:translate(-50%,-50%);left:-60px;top:-60px;';
    const bar = document.createElement('div');
    bar.id = 'demo-caption';
    bar.style.cssText = 'position:fixed;z-index:2147483646;left:50%;bottom:22px;transform:translateX(-50%);max-width:82%;padding:10px 18px;' +
      'border-radius:6px;background:rgba(20,20,20,0.88);color:#ffffff;font:600 18px Arial,Helvetica,sans-serif;text-align:center;' +
      'display:none;pointer-events:none;';
    document.body.appendChild(dot);
    document.body.appendChild(bar);
    document.addEventListener('mousemove', e => { dot.style.left = e.clientX + 'px'; dot.style.top = e.clientY + 'px'; }, true);
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', install); else install();
})();
"""


def wait_for_server(timeout_s: float = 60.0) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            if urllib.request.urlopen(f"{URL}/_stcore/health", timeout=2).read() == b"ok":
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("The Streamlit server did not start.")


def settle(page, ms: int = 600) -> None:
    page.wait_for_timeout(ms)
    page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=120000)
    page.wait_for_timeout(200)


def caption(page, text: str) -> None:
    page.evaluate(
        "text => { const bar = document.getElementById('demo-caption'); if (!bar) return; bar.textContent = text; bar.style.display = text ? 'block' : 'none'; }",
        text,
    )


def glide_to(page, locator, steps: int = 16) -> None:
    locator.scroll_into_view_if_needed()
    box = locator.bounding_box()
    if box:
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=steps)


def click(page, locator, wait_ms: int = 700) -> None:
    glide_to(page, locator)
    locator.click()
    settle(page, wait_ms)


def set_number(page, label: str, value, index: int = 0) -> None:
    box = page.get_by_label(label, exact=True).nth(index)
    glide_to(page, box, steps=10)
    box.fill(str(value))
    box.press("Enter")
    settle(page, 450)


def scroll_main(page, pixels: int) -> None:
    page.mouse.move(820, 420, steps=8)
    page.mouse.wheel(0, pixels)
    page.wait_for_timeout(700)


def record(folder: Path, info, sample_dir: Path):
    """Record the walkthrough. Returns (path of the webm file, seconds to cut from its start)."""
    env = dict(os.environ, SMARTBELT_SAMPLE_DIR=str(sample_dir))
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", str(PORT),
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    video_dir = folder / "recording"
    try:
        wait_for_server()
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context(viewport=VIEWPORT, record_video_dir=str(video_dir), record_video_size=VIEWPORT)
            context.add_init_script(OVERLAY_JS)
            page = context.new_page()
            started = time.time()  # the recording starts with the page
            page.goto(URL)
            page.get_by_role("heading", name="SmartBelt Vision").wait_for(timeout=60000)
            settle(page)
            caption(page, "SmartBelt Vision: speed, ratio and slip from a phone video")
            trim_s = max(time.time() - started - 0.4, 0.0)  # cut the loading time from the start of the video
            page.wait_for_timeout(2600)

            caption(page, "1. Choose the video (a simulated belt drive)")
            click(page, page.get_by_text("Use a file from sample_data", exact=True), 1200)
            page.wait_for_timeout(1200)
            caption(page, "2. Choose the analysis: two-pulley belt drive")
            click(page, page.get_by_text("Two-pulley belt drive", exact=True), 1200)

            caption(page, "3. Type the rotation centre, search region and sizes of each pulley")
            for index, name in enumerate(("driver", "driven")):
                set_number(page, "Rotation centre X (pixels)", info[f"{name}_centre"][0], index)
                set_number(page, "Rotation centre Y (pixels)", info[f"{name}_centre"][1], index)
                set_number(page, "Search region radius (pixels, 0 = whole picture)", round(info[f"{name}_radius_px"] * 1.05), index)
                set_number(page, "Effective radius (mm, 0 = not known)", round(info[f"{name}_effective_radius_mm"], 3), index)
                set_number(page, "Marker distance from the centre (mm, 0 = not known)", round(info[f"{name}_marker_radius_mm"], 3), index)

            caption(page, "The grid, the crosses and the circles show where the pulleys are")
            scroll_main(page, 380)
            page.wait_for_timeout(2800)

            caption(page, "4. Tick the mark on the belt")
            click(page, page.get_by_text("Belt drive and belt mark", exact=True), 600)
            click(page, page.get_by_text("There is a mark on the belt", exact=True), 1200)

            caption(page, "5. Check that each marker is one clean white blob")
            click(page, page.get_by_text("Check the marker colours in this frame", exact=True), 1200)
            scroll_main(page, 560)
            page.wait_for_timeout(3000)

            caption(page, "6. Run the analysis")
            click(page, page.get_by_role("button", name="Run analysis"), 800)
            page.get_by_role("heading", name="Results").wait_for(timeout=120000)
            settle(page, 2000)
            caption(page, "Summary, speed ratio, speed loss and slip")
            page.get_by_role("heading", name="Results").scroll_into_view_if_needed()
            page.wait_for_timeout(1500)
            scroll_main(page, 260)
            page.wait_for_timeout(4500)

            main = page.locator('[data-testid="stMainBlockContainer"]')
            caption(page, "One view for each pulley: speed, spectrum, tracking quality")
            click(page, main.get_by_text("Driver pulley", exact=True), 1500)
            scroll_main(page, 300)
            page.wait_for_timeout(3800)

            caption(page, "Annotated frames: what the program tracked")
            click(page, main.get_by_text("Annotated frames", exact=True), 1500)
            scroll_main(page, 380)
            page.wait_for_timeout(1500)
            slider = page.get_by_role("slider").last
            glide_to(page, slider)
            slider.focus()
            caption(page, "Move the slider to step through the frames")
            for key in ("PageDown", "PageDown", "PageDown", "End"):
                slider.press(key)
                settle(page, 1300)
            page.wait_for_timeout(1500)

            caption(page, "Data and downloads: table, plots and report")
            click(page, main.get_by_text("Data and downloads", exact=True), 1500)
            page.wait_for_timeout(3000)

            caption(page, "All numbers here are from a simulated video. Everything ran on this computer.")
            page.wait_for_timeout(3500)
            context.close()
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
    recordings = sorted(video_dir.glob("*.webm"))
    if not recordings:
        raise RuntimeError("No recording was made.")
    return recordings[-1], trim_s


def main() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is needed to make the mp4 file. Install it and run this again.")
    ASSETS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smartbelt_demo_") as folder:
        folder = Path(folder)
        sample_dir = folder / "sample_data"
        sample_dir.mkdir()
        info = write_synthetic_belt_video(
            sample_dir / "synthetic_belt_120rpm_ratio2_mark.mp4", driver_rpm=120.0, ratio=2.0, driver_slip_percent=1.0,
            slip_percent=2.0, draw_belt=True, seconds=5.0, scale_mm_per_px=0.25,
        )
        recording, trim_s = record(folder, info, sample_dir)
        out = ASSETS / "demo_app_walkthrough.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{trim_s:.2f}", "-i", str(recording), "-c:v", "libx264", "-preset", "medium", "-crf", "27",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)],
            check=True,
        )
        print("saved", out.relative_to(ROOT), f"({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
