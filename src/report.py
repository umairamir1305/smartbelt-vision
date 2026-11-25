"""Text for the app and the HTML report.

The rows (label, value) come from the same print functions as the command line tools (console.collect_rows),
so the numbers and the wording are the same everywhere. The HTML report is one file with the pictures inside.
"""

import base64
import html
from dataclasses import dataclass, field

from src import console

LIMITATIONS = [
    "The time axis uses the frame rate that is stored in the video file, or the one you typed. If it is wrong, every speed is wrong "
    "by the same factor. Some phones record with a frame rate that changes, which gives uneven frame times.",
    "Motion blur and a rolling shutter smear a fast marker and move its centre by a few pixels.",
    "Finding the marker depends on its colour and on the light. Changing light, reflections, another object of a similar colour or a "
    "marker that is hidden for a while give wrong or missing positions.",
    "The rotation centre has to be right. An error in the centre makes the angle wrong and shows up as a false fluctuation once per turn.",
    "The camera has to look straight along the axis of the pulley. A tilt or perspective turns circles into ellipses, which changes the "
    "angle and the speed.",
    "Video compression changes the picture from frame to frame and adds small errors to the marker position.",
    "The settings (colour limits, search region, smoothing window, trend order) are chosen by hand. Smoothing and trend removal change "
    "the numbers, so the settings that were used are listed in this report.",
    "A peak in the spectrum shows that the speed varies at that frequency. It does not show the cause. Nothing in this report is a "
    "diagnosis of a fault.",
    "Slip and speed loss are small differences between speeds that each have errors. Errors in the effective radii and in the scale, "
    "belt creep and belt stretch are all inside the number.",
    "The friction coefficient and the power loss cannot be measured from the video. The calculators (belt_calculators.py) are models "
    "that use values that you measure yourself.",
]

DEFINITIONS = [
    ("Speed", "angular velocity in rad/s times 60 / (2 pi) gives rpm. The mean speed is the total rotation divided by the time, the same as counting turns."),
    ("Speed ratio", "driver turns divided by driven turns. The expected ratio is the driven effective radius divided by the driver effective radius."),
    ("Speed loss", "(v ideal - v actual) / v ideal in percent, where v ideal is the driver speed divided by the expected ratio. Positive: the driven pulley is slower than expected."),
    ("Surface speed", "angular velocity times the effective radius of the pulley."),
    ("Slip on the driver", "(v driver - v belt) / v driver. The belt speed is measured with a mark on the belt where the belt runs straight."),
    ("Slip on the driven pulley", "(v belt - v driven) / v belt."),
    ("Fluctuation coefficient", "(max - min) / mean of the smoothed speed after the trend is removed."),
]


# ---------------------------------------------------------------------------
# Rows for one pulley and for the belt drive
# ---------------------------------------------------------------------------


def pulley_sections(result, settings, region_radius, simulated: bool) -> list:
    """The summary of one pulley as a list of (heading, rows). It repeats what run_analysis.py prints."""
    sections = [
        (
            "Tracking",
            console.collect_rows(
                console.print_tracking_summary, result.info, settings, result.centre, result.used_centre, region_radius,
                result.tracking, simulated, show_video=False,
            ),
        )
    ]
    if result.fit is not None:
        sections.append(
            ("Circle fit", console.collect_rows(console.print_fit_summary, result.fit, result.centre, result.used_centre != result.centre))
        )
    if result.rotation is not None:
        sections.append(("Angle", console.collect_rows(console.print_angle_summary, result.rotation)))
    if result.speed is not None:
        sections.append(("Speed", console.collect_rows(console.print_speed_summary, result.speed)))
    if result.signal is not None:
        sections.append(("Signal analysis", console.collect_rows(console.print_signal_summary, result.signal)))
    return [(heading, rows) for heading, rows in sections if rows]


def peak_table(signal) -> list:
    """The dominant frequencies as a list of dictionaries (one per peak), for a table."""
    rows = []
    if signal is None:
        return rows
    for rank, peak in enumerate(signal["peaks"], start=1):
        rows.append(
            {
                "#": rank,
                "Frequency (Hz)": round(peak["frequency_hz"], 2),
                "Period (s)": round(peak["period_s"], 3),
                "Amplitude (rpm)": round(peak["amplitude_rpm"], 2),
                "Percent of mean": round(peak["percent_of_mean"], 2),
                "Order": "-" if peak["order"] is None else f"{peak['order']:.2f}x",
                "Note": peak["note"],
            }
        )
    return rows


def pulley_headline(result) -> list:
    """The main numbers of one pulley as (label, value) pairs, for the big numbers at the top."""
    speed = result.speed
    if speed is None:
        return [("Mean speed", "not calculated"), ("Tracking quality", result.tracking["overall_quality"])]
    if result.signal is None:
        frequency = "not analysed"
    elif result.signal["peaks"]:
        biggest = max(result.signal["peaks"], key=lambda peak: peak["amplitude_rpm"])
        frequency = f"{biggest['frequency_hz']:.2f} Hz"
    else:
        frequency = "none found"
    return [
        ("Mean speed", f"{speed['mean_rpm']:.2f} rpm"),
        ("Speed range", f"{speed['min_rpm']:.1f} to {speed['max_rpm']:.1f} rpm"),
        ("Speed variation", f"{speed['variation_percent']:.2f} %"),
        ("Tracking quality", result.tracking["overall_quality"]),
        ("Dominant frequency", frequency),
    ]


def belt_headline(belt, slip) -> list:
    """The main numbers of a belt drive as (label, value) pairs."""
    if belt is None:
        return [("Speed ratio", "not calculated")]
    rows = [("Speed ratio", f"{belt['measured_ratio']:.3f}")]
    if belt["expected_ratio"] is not None:
        rows.append(("Expected ratio", f"{belt['expected_ratio']:.3f}"))
        rows.append(("Speed loss", f"{belt['speed_loss_percent']:+.2f} %"))
    if slip is not None and slip["available"]:
        rows.append(("Slip on driver", f"{slip['driver_slip_percent']:+.2f} %"))
        rows.append(("Slip on driven", f"{slip['driven_slip_percent']:+.2f} %"))
    else:
        rows.append(("Belt slip", "not available"))
    return rows


def belt_rows(belt) -> list:
    return console.collect_rows(console.print_belt_summary, belt) if belt is not None else []


def slip_rows(slip, message) -> list:
    """The rows of the slip summary. If the slip is not available, one row gives the reason."""
    if slip is not None and slip["available"]:
        return console.collect_rows(console.print_slip_summary, slip)
    return [("Slip", message or "not calculated")]


# ---------------------------------------------------------------------------
# The HTML report
# ---------------------------------------------------------------------------


@dataclass
class ReportBlock:
    """One part of the report: a pulley, or the belt drive."""

    title: str
    sections: list = field(default_factory=list)  # (heading, rows)
    peaks: list = None  # rows of peak_table, or None
    figures: list = field(default_factory=list)  # (caption, PNG bytes)


@dataclass
class ReportData:
    title: str
    simulated: bool
    video_rows: list
    settings_rows: list
    blocks: list
    notes: str = ""
    generated: str = ""  # date and time as text


CSS = """
body { font-family: Arial, Helvetica, sans-serif; color: #1a1a1a; max-width: 900px; margin: 2em auto; padding: 0 1em; line-height: 1.45; }
h1 { font-size: 1.6em; margin-bottom: 0.2em; }
h2 { font-size: 1.25em; margin-top: 2em; border-bottom: 1px solid #999; padding-bottom: 0.2em; }
h3 { font-size: 1.05em; margin-top: 1.4em; }
table { border-collapse: collapse; width: 100%; margin: 0.5em 0 1em 0; }
th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; vertical-align: top; font-size: 0.92em; }
table.rows th { width: 25%; background: #f2f2f2; font-weight: bold; }
table.peaks th { background: #f2f2f2; }
figure { margin: 1em 0; }
img { max-width: 100%; height: auto; border: 1px solid #ddd; }
figcaption { font-size: 0.9em; color: #444; }
.simulated { border: 2px solid #b00020; padding: 0.6em 0.8em; font-weight: bold; }
.small { font-size: 0.9em; color: #444; }
pre.notes { white-space: pre-wrap; font-family: inherit; }
"""


def _escape(value) -> str:
    return html.escape(str(value))


def _rows_table(rows) -> str:
    cells = "".join(f"<tr><th>{_escape(label)}</th><td>{_escape(value)}</td></tr>" for label, value in rows)
    return f'<table class="rows">{cells}</table>'


def _peaks_table(peaks) -> str:
    if not peaks:
        return ""
    columns = list(peaks[0].keys())
    header = "".join(f"<th>{_escape(column)}</th>" for column in columns)
    body = "".join("<tr>" + "".join(f"<td>{_escape(row[column])}</td>" for column in columns) + "</tr>" for row in peaks)
    caution = (
        '<p class="small">A peak shows that the speed varies periodically at that frequency. It does not show the cause: '
        "motor torque ripple, belt irregularity, misalignment, a rotation centre error, a tilted camera, rolling shutter, frame "
        "timing and tracking noise can all do this.</p>"
    )
    return f'<table class="peaks"><tr>{header}</tr>{body}</table>{caution}'


def _figure(caption: str, png: bytes) -> str:
    encoded = base64.b64encode(png).decode("ascii")
    return f'<figure><img alt="{_escape(caption)}" src="data:image/png;base64,{encoded}"><figcaption>{_escape(caption)}</figcaption></figure>'


def build_html_report(data: ReportData) -> str:
    """One HTML page with the video information, the settings, the results, the graphs and the limitations."""
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{_escape(data.title)}</title>",
        f"<style>{CSS}</style></head><body>",
        f"<h1>{_escape(data.title)}</h1>",
        f'<p class="small">Made by SmartBelt Vision, a student project. {_escape(data.generated)}</p>'.replace("  ", " "),
    ]
    if data.simulated:
        parts.append('<p class="simulated">SIMULATED data. This video was made by the program. The numbers are not a measurement of a real machine.</p>')

    parts.append("<h2>Video information</h2>")
    parts.append(_rows_table(data.video_rows))
    parts.append("<h2>Analysis settings</h2>")
    parts.append(_rows_table(data.settings_rows))

    for block in data.blocks:
        parts.append(f"<h2>{_escape(block.title)}</h2>")
        for heading, rows in block.sections:
            parts.append(f"<h3>{_escape(heading)}</h3>")
            parts.append(_rows_table(rows))
        if block.peaks:
            parts.append("<h3>Dominant frequencies</h3>")
            parts.append(_peaks_table(block.peaks))
        for caption, png in block.figures:
            parts.append(_figure(caption, png))

    parts.append("<h2>Definitions</h2>")
    parts.append(_rows_table(DEFINITIONS))
    parts.append("<h2>Limitations</h2>")
    parts.append("<ul>" + "".join(f"<li>{_escape(text)}</li>" for text in LIMITATIONS) + "</ul>")
    parts.append("<h2>Notes</h2>")
    if data.notes.strip():
        parts.append(f'<pre class="notes">{_escape(data.notes.strip())}</pre>')
    else:
        parts.append('<p class="small">No notes were written.</p>')
    parts.append("</body></html>")
    return "\n".join(parts)
