"""Takes the screenshots of the Streamlit app for the README. Optional: it needs Playwright.

    pip install playwright
    playwright install chromium
    python tools/app_screenshots.py

It makes a simulated belt drive video, starts the app on port 8601 with that video in the sample folder, drives the
app like a user (choose the video, set the markers, run the analysis, open the tabs) and saves screenshots into
the assets folder. It also downloads the HTML report from the app and takes a screenshot of that.
"""

import os
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
PORT = 8601
URL = f"http://localhost:{PORT}"


def wait_for_server(timeout_s: float = 60.0) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            if urllib.request.urlopen(f"{URL}/_stcore/health", timeout=2).read() == b"ok":
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("The Streamlit server did not start.")


def settle(page, ms: int = 900) -> None:
    """Wait until the app has finished its run."""
    page.wait_for_timeout(ms)
    page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=120000)
    page.wait_for_timeout(300)


def set_number(page, label: str, value, index: int = 0) -> None:
    box = page.get_by_label(label, exact=True).nth(index)
    box.fill(str(value))
    box.press("Enter")
    settle(page, 700)


def shot(page, name: str, height=None) -> None:
    path = ASSETS / name
    if height:
        page.screenshot(path=str(path), clip={"x": 0, "y": 0, "width": 1500, "height": height})
    else:
        page.screenshot(path=str(path))
    print("saved", path.relative_to(ROOT), f"({path.stat().st_size // 1024} KB)")


def shot_results(page, name: str, max_height: int = 2600) -> None:
    """A screenshot of the main area from the Results heading down (the sidebar is left out)."""
    box = page.get_by_role("heading", name="Results").bounding_box()
    container = page.locator('[data-testid="stMainBlockContainer"]').bounding_box()
    top = max(box["y"] - 24, 0)
    height = min(container["y"] + container["height"] - top + 24, max_height)
    path = ASSETS / name
    page.screenshot(path=str(path), clip={"x": 300, "y": top, "width": 1200, "height": height})
    print("saved", path.relative_to(ROOT), f"({path.stat().st_size // 1024} KB)")


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smartbelt_shots_") as folder:
        folder = Path(folder)
        sample_dir = folder / "sample_data"
        sample_dir.mkdir()
        video = sample_dir / "synthetic_belt_120rpm_ratio2_mark.mp4"
        info = write_synthetic_belt_video(video, driver_rpm=120.0, ratio=2.0, driver_slip_percent=1.0, slip_percent=2.0,
                                          draw_belt=True, seconds=5.0, scale_mm_per_px=0.25)
        env = dict(os.environ, SMARTBELT_SAMPLE_DIR=str(sample_dir))
        server = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", str(PORT),
             "--browser.gatherUsageStats", "false"],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            wait_for_server()
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                page = browser.new_page(viewport={"width": 1500, "height": 1200}, device_scale_factor=1)
                page.goto(URL)
                page.get_by_role("heading", name="SmartBelt Vision").wait_for(timeout=60000)
                settle(page)
                shot(page, "app_01_start.png")

                page.set_viewport_size({"width": 1500, "height": 2500})
                page.get_by_text("Use a file from sample_data", exact=True).click()
                settle(page, 1500)
                page.get_by_text("Two-pulley belt drive", exact=True).click()
                settle(page, 1500)
                for index, name in enumerate(("driver", "driven")):
                    set_number(page, "Rotation centre X (pixels)", info[f"{name}_centre"][0], index)
                    set_number(page, "Rotation centre Y (pixels)", info[f"{name}_centre"][1], index)
                    set_number(page, "Search region radius (pixels, 0 = whole picture)", round(info[f"{name}_radius_px"] * 1.05), index)
                    set_number(page, "Effective radius (mm, 0 = not known)", round(info[f"{name}_effective_radius_mm"], 3), index)
                    set_number(page, "Marker distance from the centre (mm, 0 = not known)", round(info[f"{name}_marker_radius_mm"], 3), index)
                page.get_by_text("Belt drive and belt mark", exact=True).click()
                page.wait_for_timeout(500)
                page.get_by_text("There is a mark on the belt", exact=True).click()
                settle(page, 1500)
                page.get_by_text("Check the marker colours in this frame", exact=True).click()
                settle(page, 1200)
                shot(page, "app_02_setup.png", 1900)

                page.get_by_text("Check the marker colours in this frame", exact=True).click()  # close it again: the results come first
                settle(page, 1000)
                page.set_viewport_size({"width": 1500, "height": 5200})
                page.get_by_role("button", name="Run analysis").click()
                page.get_by_role("heading", name="Results").wait_for(timeout=120000)
                settle(page, 2500)
                shot_results(page, "app_03_results_belt.png", 1950)

                main = page.locator('[data-testid="stMainBlockContainer"]')
                main.get_by_text("Driver pulley", exact=True).click()
                page.get_by_text("Dominant frequencies", exact=True).first.wait_for(state="visible", timeout=30000)
                settle(page, 1500)
                shot_results(page, "app_04_driver_pulley.png", 1950)

                main.get_by_text("Annotated frames", exact=True).click()
                page.get_by_text("Frame in the analysis", exact=True).wait_for(state="visible", timeout=30000)
                settle(page, 1500)
                slider = page.get_by_role("slider").last
                slider.focus()
                slider.press("End")
                settle(page, 2000)
                shot_results(page, "app_05_annotated_frames.png")

                main.get_by_text("Data and downloads", exact=True).click()
                page.get_by_role("button", name="Download the table (CSV)").wait_for(state="visible", timeout=30000)
                settle(page, 1500)
                shot_results(page, "app_06_downloads.png", 1500)

                with page.expect_download(timeout=60000) as download_info:
                    page.get_by_role("button", name="Download the report (HTML)").click()
                report_path = folder / "report.html"
                download_info.value.save_as(str(report_path))
                report_page = browser.new_page(viewport={"width": 1000, "height": 1500})
                report_page.goto(report_path.as_uri())
                report_page.wait_for_timeout(800)
                path = ASSETS / "app_07_html_report.png"
                report_page.screenshot(path=str(path))
                print("saved", path.relative_to(ROOT), f"({path.stat().st_size // 1024} KB)")
                browser.close()
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    main()
