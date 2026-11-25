"""Tests for src/report.py and for the row recorder in src/console.py."""

import pytest

from src import console, report


def test_collect_rows_returns_the_rows_and_prints_nothing(capsys):
    def printer():
        print("A TITLE")
        console.line("Mean speed", "120.00 rpm")
        console.line("Frames", 300)
        print("    # a table line")

    rows = console.collect_rows(printer)
    assert rows == [("Mean speed", "120.00 rpm"), ("Frames", "300")]
    assert capsys.readouterr().out == ""


def test_line_prints_again_after_collecting_even_if_the_printer_failed(capsys):
    def broken():
        console.line("Before", "1")
        raise RuntimeError("something went wrong")

    with pytest.raises(RuntimeError):
        console.collect_rows(broken)
    console.line("After", "2")
    assert capsys.readouterr().out == f"{'After':<20} : 2\n"


def test_collect_rows_can_be_nested():
    def inner():
        console.line("Inner", "x")

    def outer():
        console.line("Outer one", "1")
        assert console.collect_rows(inner) == [("Inner", "x")]
        console.line("Outer two", "2")

    assert console.collect_rows(outer) == [("Outer one", "1"), ("Outer two", "2")]


def data(simulated=False, notes="", peaks=None, figures=None):
    block = report.ReportBlock(
        title="Results", sections=[("Speed", [("Mean speed", "120.00 rpm")])], peaks=peaks, figures=figures or [],
    )
    return report.ReportData(
        title="SmartBelt Vision: analysis report", simulated=simulated, video_rows=[("File", "clip.mp4")],
        settings_rows=[("Analysis", "Single pulley")], blocks=[block], notes=notes, generated="Made on 2026-11-23 at 10:00.",
    )


def test_the_report_has_all_its_parts():
    html = report.build_html_report(data())
    for part in ("Video information", "Analysis settings", "Results", "Definitions", "Limitations", "Notes",
                 "Mean speed", "120.00 rpm", "clip.mp4"):
        assert part in html
    assert html.startswith("<!DOCTYPE html>") and html.rstrip().endswith("</html>")
    for text in report.LIMITATIONS:
        assert text.replace("&", "&amp;") in html or text in html.replace("&#x27;", "'")


def test_the_simulated_banner_is_only_in_reports_of_simulated_data():
    assert "SIMULATED data" in report.build_html_report(data(simulated=True))
    assert "SIMULATED data" not in report.build_html_report(data(simulated=False))


def test_text_from_the_user_is_escaped():
    html = report.build_html_report(data(notes="Belt <script>alert(1)</script> & more"))
    assert "<script>" not in html
    assert "&lt;script&gt;" in html and "&amp; more" in html
    assert "No notes were written." in report.build_html_report(data(notes="   "))


def test_pictures_are_inside_the_file_and_nothing_is_loaded_from_the_internet():
    png = b"\x89PNG\r\n\x1a\nfake"
    html = report.build_html_report(data(figures=[("Speed against time", png)]))
    assert "data:image/png;base64," in html and "Speed against time" in html
    assert "http://" not in html and "https://" not in html and "<script" not in html


def test_the_peak_table_comes_with_the_caution():
    peaks = [{"#": 1, "Frequency (Hz)": 3.0, "Period (s)": 0.333, "Amplitude (rpm)": 2.4, "Percent of mean": 2.0, "Order": "1.00x", "Note": "x"}]
    html = report.build_html_report(data(peaks=peaks))
    assert "Dominant frequencies" in html and "3.0" in html and "does not show the cause" in html
    assert "Dominant frequencies" not in report.build_html_report(data(peaks=None))


def test_the_report_text_is_plain_ascii_without_dashes():
    html = report.build_html_report(data(simulated=True))
    assert html.isascii()
    assert "\u2014" not in html and "\u2013" not in html


def test_headline_of_a_belt_drive_without_slip_says_not_available():
    belt = {"measured_ratio": 2.0618, "expected_ratio": 2.0, "speed_loss_percent": 3.0}
    assert report.belt_headline(belt, None) == [
        ("Speed ratio", "2.062"), ("Expected ratio", "2.000"), ("Speed loss", "+3.00 %"), ("Belt slip", "not available"),
    ]
    assert report.belt_headline(None, None) == [("Speed ratio", "not calculated")]
    slip = {"available": True, "driver_slip_percent": 1.07, "driven_slip_percent": 1.93}
    assert ("Slip on driver", "+1.07 %") in report.belt_headline(belt, slip)


def test_the_limitations_do_not_promise_a_diagnosis_or_a_measured_friction():
    text = " ".join(report.LIMITATIONS).lower()
    assert "not a diagnosis" in text or "nothing in this report is a diagnosis" in text
    assert "cannot be measured from the video" in text
