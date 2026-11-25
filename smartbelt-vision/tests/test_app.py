"""Tests of the Streamlit app (app.py), run without a browser with Streamlit's own test tool.

The file uploader cannot be driven by this tool, so the tests use the other video source of the app:
a file in the sample_data folder (here a temporary folder, set with SMARTBELT_SAMPLE_DIR).
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.synthetic import write_synthetic_belt_video  # noqa: E402
from src.tracking import find_marker, settings_for_colour  # noqa: E402
from src.video_io import read_frame  # noqa: E402

APP_FILE = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(scope="module")
def sample_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sample_data")
    info = write_synthetic_belt_video(
        folder / "synthetic_marked.mp4", driver_rpm=120.0, ratio=2.0, driver_slip_percent=1.0, slip_percent=2.0,
        draw_belt=True, width=640, height=360, fps=60.0, seconds=3.0, scale_mm_per_px=0.25,
    )
    return folder, info


@pytest.fixture
def app(sample_folder, monkeypatch):
    folder, _ = sample_folder
    monkeypatch.setenv("SMARTBELT_SAMPLE_DIR", str(folder))
    test = AppTest.from_file(str(APP_FILE), default_timeout=120)
    test.run()
    return test


def use_sample(app):
    app.radio(key="source").set_value("Use a file from sample_data").run()
    return app


def set_pulley(app, prefix, info, region=True, mm=True):
    app.number_input(key=f"{prefix}_cx").set_value(float(info[f"{prefix}_centre"][0]))
    app.number_input(key=f"{prefix}_cy").set_value(float(info[f"{prefix}_centre"][1]))
    if region:
        app.number_input(key=f"{prefix}_region").set_value(float(round(info[f"{prefix}_radius_px"] * 1.05)))
    if mm:
        app.number_input(key=f"{prefix}_radius_mm").set_value(float(info[f"{prefix}_effective_radius_mm"]))
        app.number_input(key=f"{prefix}_marker_mm").set_value(float(info[f"{prefix}_marker_radius_mm"]))


def metrics(app):
    return [(metric.label, metric.value) for metric in app.metric]


def test_the_app_starts_and_asks_for_a_video(app):
    assert not app.exception
    assert app.title[0].value == "SmartBelt Vision"
    assert any("Upload a video" in info.value for info in app.info)
    assert not app.metric
    text = " ".join(item.value for item in app.markdown)
    assert "processed on this computer" in text and "Nothing is uploaded anywhere" in text


def test_a_sample_video_shows_its_facts_and_the_picture_checks(app):
    use_sample(app)
    assert not app.exception and not app.error
    assert any("SIMULATED video" in warning.value for warning in app.warning)
    assert [header.value for header in app.header][0] == "Video"
    assert any("marker found at" in success.value for success in app.success)
    assert app.slider[0].label == "Frame"


def test_a_single_pulley_run_shows_the_speed_and_the_tabs(app, sample_folder):
    _, info = sample_folder
    use_sample(app)
    set_pulley(app, "pulley", {"pulley_centre": info["driver_centre"], "pulley_radius_px": info["driver_radius_px"]}, mm=False)
    app.run()
    app.button(key="run").click().run()
    assert not app.exception and not app.error
    values = dict(metrics(app))
    assert values["Mean speed"] == "120.00 rpm" and values["Tracking quality"] == "Good"
    assert app.radio(key="results_view").options == ["Speed and frequency", "Annotated frames", "Data and downloads", "Limitations"]
    app.radio(key="results_view").set_value("Data and downloads").run()
    assert not app.exception and len(app.get("download_button")) == 3


def test_a_belt_drive_run_with_a_belt_mark_shows_the_slip(app, sample_folder):
    _, info = sample_folder
    use_sample(app)
    app.radio(key="kind").set_value("Two-pulley belt drive").run()
    set_pulley(app, "driver", info)
    set_pulley(app, "driven", info)
    app.checkbox(key="belt_enabled").set_value(True).run()
    app.button(key="run").click().run()
    assert not app.exception and not app.error
    values = dict(metrics(app))
    assert float(values["Speed ratio"]) == pytest.approx(2.06, abs=0.02)
    assert float(values["Speed loss"].split()[0]) == pytest.approx(2.98, abs=0.3)
    assert float(values["Slip on driver"].split()[0]) == pytest.approx(1.0, abs=0.3)
    assert float(values["Slip on driven"].split()[0]) == pytest.approx(2.0, abs=0.3)
    assert app.radio(key="results_view").options[:3] == ["Belt drive", "Driver pulley", "Driven pulley"]
    assert any(warning.value.startswith("SIMULATED data") for warning in app.warning)
    summary = " ".join(item.value for item in app.markdown if item.value.startswith("- "))
    assert "speed ratio is 2.06" in summary


def test_without_a_belt_mark_the_app_says_the_slip_is_not_measured(app, sample_folder):
    _, info = sample_folder
    use_sample(app)
    app.radio(key="kind").set_value("Two-pulley belt drive").run()
    set_pulley(app, "driver", info)
    set_pulley(app, "driven", info)
    app.button(key="run").click().run()
    assert not app.exception and not app.error
    values = dict(metrics(app))
    assert values["Belt slip"] == "not available" and "Slip on driver" not in values
    assert any("No mark on the belt was given" in caption.value for caption in app.caption)


def test_a_belt_mark_without_a_scale_gives_a_message_and_no_slip(app, sample_folder):
    _, info = sample_folder
    use_sample(app)
    app.radio(key="kind").set_value("Two-pulley belt drive").run()
    set_pulley(app, "driver", info, mm=False)
    set_pulley(app, "driven", info, mm=False)
    app.checkbox(key="belt_enabled").set_value(True).run()
    app.button(key="run").click().run()
    assert not app.exception and not app.error
    assert any("Slip estimation unavailable" in warning.value for warning in app.warning)
    assert "Slip on driver" not in dict(metrics(app))


def test_a_rotation_centre_outside_the_picture_is_a_plain_error_message(app):
    use_sample(app)
    app.number_input(key="pulley_cx").set_value(5000.0)
    app.run()
    app.button(key="run").click().run()
    assert not app.exception
    assert len(app.error) == 1 and "centre" in app.error[0].value.lower()
    assert not app.metric


def test_the_colour_can_be_taken_from_a_pixel_on_the_marker(app, sample_folder):
    folder, _ = sample_folder
    use_sample(app)
    marker = find_marker(read_frame(folder / "synthetic_marked.mp4", 0), settings_for_colour("green"))
    app.number_input(key="pulley_pick_x").set_value(float(round(marker.x)))
    app.number_input(key="pulley_pick_y").set_value(float(round(marker.y)))
    app.run()
    app.button(key="pulley_pick").click().run()
    assert not app.exception
    assert app.checkbox(key="pulley_manual").value is True
    assert 35 <= app.number_input(key="pulley_hue_low").value <= 60
    assert any("Colour limits set from frame 0" in caption.value for caption in app.caption)


def test_a_grey_pixel_is_refused_with_a_message(app):
    use_sample(app)
    app.number_input(key="pulley_pick_x").set_value(3.0)
    app.number_input(key="pulley_pick_y").set_value(3.0)
    app.run()
    app.button(key="pulley_pick").click().run()
    assert not app.exception
    assert app.checkbox(key="pulley_manual").value is False
    assert any("almost grey" in caption.value for caption in app.caption)


def test_the_limitations_are_part_of_the_results(app, sample_folder):
    _, info = sample_folder
    use_sample(app)
    set_pulley(app, "pulley", {"pulley_centre": info["driver_centre"], "pulley_radius_px": info["driver_radius_px"]}, mm=False)
    app.run()
    app.button(key="run").click().run()
    app.radio(key="results_view").set_value("Limitations").run()
    text = " ".join(item.value for item in app.markdown)
    assert "Nothing in this report is a diagnosis of a fault" in text


def test_the_chosen_view_and_the_frame_slider_are_kept_when_the_script_runs_again(app, sample_folder):
    """Moving the frame slider runs the script again. The view that was chosen must stay (st.tabs would jump back to the first tab)."""
    _, info = sample_folder
    use_sample(app)
    set_pulley(app, "pulley", {"pulley_centre": info["driver_centre"], "pulley_radius_px": info["driver_radius_px"]}, mm=False)
    app.run()
    app.button(key="run").click().run()
    app.radio(key="results_view").set_value("Annotated frames").run()
    slider = [item for item in app.slider if item.label == "Frame in the analysis"][0]
    slider.set_value(120).run()
    assert not app.exception
    assert app.radio(key="results_view").value == "Annotated frames"
    assert [item for item in app.slider if item.label == "Frame in the analysis"][0].value == 120
