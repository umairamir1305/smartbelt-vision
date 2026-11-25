"""Tests for src/reporting.py. The plots are checked for titles, labels, saving and odd data."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.kinematics import add_angle_columns, add_speed_columns
from src.belt_drive import add_belt_columns, merge_pulley_tables
from src.reporting import (
    plot_angle_vs_time,
    plot_belt_speed,
    plot_marker_trajectory,
    plot_rpm_vs_time,
    plot_speed_fluctuation,
    plot_slip,
    plot_spectrum,
    plot_surface_speeds,
    plot_tracking_timeline,
    plot_two_pulley_speeds,
    save_figure,
)
from src.signal_analysis import SignalSettings, analyse_speed_signal
from src.synthetic import make_synthetic_table
from src.tracking import add_tracking_quality

CENTRE = (200.0, 150.0)


@pytest.fixture(autouse=True)
def close_all_figures():
    yield
    plt.close("all")


def make_table(count=60, step_deg=12.0, lost=(), wild=(), with_angles=True):
    """A rated tracking table, with angles unless with_angles is False.

    lost frames have no marker, wild frames have a far-away position.
    """
    angles = np.radians(step_deg) * np.arange(count)
    table = pd.DataFrame(
        {
            "frame": np.arange(count),
            "time_s": np.arange(count) / 60.0,
            "marker_x": CENTRE[0] + 100.0 * np.cos(angles),
            "marker_y": CENTRE[1] - 100.0 * np.sin(angles),
            "area_px": 500.0,
        }
    )
    table.loc[list(lost), ["marker_x", "marker_y", "area_px"]] = np.nan
    for frame in wild:
        table.loc[frame, ["marker_x", "marker_y"]] = [10.0, 10.0]
    table = add_tracking_quality(table, CENTRE)
    return add_angle_columns(table, CENTRE) if with_angles else table


def test_trajectory_plot_labels():
    figure = plot_marker_trajectory(make_table(), CENTRE)
    axis = figure.axes[0]
    assert axis.get_title() == "Marker trajectory in the video frame"
    assert axis.get_xlabel() == "x (pixels)"
    assert "y (pixels" in axis.get_ylabel()
    assert axis.yaxis_inverted()  # y points down, as in the video


def test_angle_plot_has_two_panels_with_units():
    figure = plot_angle_vs_time(make_table())
    top, bottom = figure.axes[0], figure.axes[1]
    assert top.get_title() == "Marker angle against time"
    assert "rad" in top.get_ylabel()
    assert bottom.get_xlabel() == "Time (s)"
    assert "rad" in bottom.get_ylabel()


def test_timeline_plot_counts_every_quality_in_the_legend():
    table = make_table(lost=[10, 11, 12], wild=[30])
    figure = plot_tracking_timeline(table)
    legend_texts = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    quality = table["tracking_quality"]
    for name in ("Good", "Moderate", "Poor", "Lost"):
        count = int((quality == name).sum())
        if count:
            assert f"{name} ({count})" in legend_texts
    assert "Lost (3)" in legend_texts


@pytest.mark.parametrize("plot_function", [plot_angle_vs_time, plot_tracking_timeline])
def test_simulated_data_is_marked_in_the_title(plot_function):
    table = make_table()
    assert plot_function(table, simulated=True).axes[0].get_title().endswith("(simulated data)")
    assert "simulated" not in plot_function(table, simulated=False).axes[0].get_title()


def test_trajectory_plot_marks_simulated_data_too():
    figure = plot_marker_trajectory(make_table(), CENTRE, simulated=True)
    assert figure.axes[0].get_title().endswith("(simulated data)")


def test_save_figure_writes_a_png_and_closes_the_figure(tmp_path):
    figure = plot_angle_vs_time(make_table())
    number = figure.number
    path = save_figure(figure, tmp_path / "plots" / "angle.png")

    assert path.exists()
    assert path.read_bytes()[:4] == b"\x89PNG"
    assert not plt.fignum_exists(number)


def test_plots_work_with_lost_frames_poor_frames_and_two_segments(tmp_path):
    # 20 degrees per frame and 6 lost frames in a row: the angle is split into two segments.
    table = make_table(count=60, step_deg=20.0, lost=range(20, 26), wild=[40])
    assert table["segment"].dropna().nunique() == 2

    angle_figure = plot_angle_vs_time(table)
    legend = angle_figure.axes[1].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["segment 0", "segment 1"]

    trajectory_axis = plot_marker_trajectory(table, CENTRE).axes[0]
    labels = trajectory_axis.get_legend_handles_labels()[1]
    assert any(label.startswith("not trusted") for label in labels)

    for index, figure in enumerate([angle_figure, plot_marker_trajectory(table, CENTRE), plot_tracking_timeline(table)]):
        assert save_figure(figure, tmp_path / f"plot{index}.png").stat().st_size > 1000


def test_trajectory_and_timeline_work_when_nothing_was_found(tmp_path):
    table = pd.DataFrame(
        {
            "frame": np.arange(20),
            "time_s": np.arange(20) / 30.0,
            "marker_x": np.nan,
            "marker_y": np.nan,
            "area_px": np.nan,
        }
    )
    table = add_tracking_quality(table, CENTRE)
    assert (table["tracking_quality"] == "Lost").all()
    save_figure(plot_marker_trajectory(table, CENTRE), tmp_path / "trajectory.png")
    save_figure(plot_tracking_timeline(table), tmp_path / "timeline.png")


def test_timeline_works_for_a_single_frame(tmp_path):
    table = make_table(count=1, with_angles=False)  # a single frame is too few for an angle, but fine for this plot
    save_figure(plot_tracking_timeline(table), tmp_path / "one_frame.png")


def test_the_angle_plot_shows_the_unwrapped_values():
    table = make_table(count=80, step_deg=17.0)
    figure = plot_angle_vs_time(table)
    bottom = figure.axes[1]
    plotted_y = np.concatenate([line.get_ydata() for line in bottom.get_lines()])
    assert plotted_y.max() == pytest.approx(math.radians(17.0) * 79, rel=1e-6)  # about 3.7 turns, not wrapped


# ---------------------------------------------------------------------------
# rpm plot
# ---------------------------------------------------------------------------


def make_speed_table(**kwargs):
    return add_speed_columns(make_table(**kwargs))


def test_rpm_plot_labels_and_values():
    # 12 degrees per frame at 60 fps is 120 rpm.
    table = make_speed_table(count=60, step_deg=12.0)
    figure = plot_rpm_vs_time(table, mean_rpm=120.0)
    axis = figure.axes[0]

    assert axis.get_title() == "Rotation speed against time"
    assert axis.get_xlabel() == "Time (s)"
    assert axis.get_ylabel() == "Rotation speed (rpm)"
    speed_line = axis.get_lines()[0]
    assert np.allclose(speed_line.get_ydata(), 120.0)
    legend_texts = [text.get_text() for text in axis.get_legend().get_texts()]
    assert "mean 120.0 rpm" in legend_texts


def test_rpm_plot_without_a_mean_has_no_mean_line():
    figure = plot_rpm_vs_time(make_speed_table(), mean_rpm=None)
    legend_texts = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert not any(text.startswith("mean") for text in legend_texts)


def test_clockwise_speed_is_drawn_as_a_positive_number_and_the_title_says_clockwise():
    figure = plot_rpm_vs_time(make_speed_table(count=60, step_deg=-12.0))
    axis = figure.axes[0]
    assert axis.get_title() == "Rotation speed against time (clockwise)"
    assert np.allclose(axis.get_lines()[0].get_ydata(), 120.0)


def test_rpm_plot_marks_simulated_data():
    figure = plot_rpm_vs_time(make_speed_table(), simulated=True)
    assert figure.axes[0].get_title().endswith("(simulated data)")


def test_rpm_plot_leaves_a_break_where_frames_are_lost(tmp_path):
    table = make_speed_table(count=60, step_deg=12.0, lost=[20, 21, 22])
    figure = plot_rpm_vs_time(table, mean_rpm=120.0)
    plotted = figure.axes[0].get_lines()[0].get_ydata()
    assert np.isnan(plotted[[20, 21, 22]]).all()  # empty, not filled in
    assert save_figure(figure, tmp_path / "rpm.png").stat().st_size > 1000


def test_rpm_plot_works_when_there_are_no_speed_values(tmp_path):
    table = make_speed_table()
    table["rpm"] = np.nan
    figure = plot_rpm_vs_time(table)
    assert save_figure(figure, tmp_path / "empty_rpm.png").exists()


# ---------------------------------------------------------------------------
# Plots of the signal analysis
# ---------------------------------------------------------------------------

SIM_CENTRE = (320.0, 240.0)


def make_signal_result(settings=None, mutate=None, **kwargs):
    """Simulated table taken through angle, speed and signal analysis. Returns (table, result).

    mutate is an optional function that changes the raw table (for example to add an error) before the analysis.
    """
    centre = kwargs.get("centre", SIM_CENTRE)  # the angle is calculated about the centre the data was made with
    raw = make_synthetic_table(noise_px=0.05, **kwargs)
    if mutate is not None:
        mutate(raw)
    table = add_speed_columns(add_angle_columns(add_tracking_quality(raw, centre), centre))
    return analyse_speed_signal(table, settings)


def test_rpm_plot_with_the_signal_columns_has_smoothing_and_an_acceleration_panel():
    table, _ = make_signal_result(rpm=120.0)
    figure = plot_rpm_vs_time(table, mean_rpm=120.0)

    assert len(figure.axes) == 2
    speed_axis, acceleration_axis = figure.axes
    assert speed_axis.get_title() == "Rotation speed against time"
    legend_texts = [text.get_text() for text in speed_axis.get_legend().get_texts()]
    assert legend_texts[0] == "speed of each frame"
    assert "smoothed" in legend_texts and "mean 120.0 rpm" in legend_texts
    assert "Angular acceleration" in acceleration_axis.get_ylabel()
    assert acceleration_axis.get_xlabel() == "Time (s)"


def test_rpm_plot_marks_outliers_and_filled_in_frames():
    def add_errors(raw):
        raw.loc[150, ["marker_x", "marker_y"]] += [6.0, 6.0]  # a wrong marker position in one frame

    table, result = make_signal_result(rpm=120.0, lost_frames=[60, 61], mutate=add_errors)
    assert result["outliers"] >= 1 and result["interpolated_frames"] == 2

    figure = plot_rpm_vs_time(table)
    legend_texts = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert f"outlier, replaced ({result['outliers']})" in legend_texts
    assert "filled in, no measurement (2)" in legend_texts


def test_a_clockwise_disc_that_slows_down_shows_negative_acceleration():
    table, _ = make_signal_result(rpm=-300.0, fps=60.0, seconds=8.0, decay_s=3.0)
    figure = plot_rpm_vs_time(table)
    speed_axis, acceleration_axis = figure.axes

    assert speed_axis.get_title() == "Rotation speed against time (clockwise)"
    smoothed = [line for line in speed_axis.get_lines() if line.get_linewidth() == 2.0][0].get_ydata()
    assert np.nanmax(smoothed) > 250.0  # shown as a positive speed
    plotted_acceleration = acceleration_axis.get_lines()[0].get_ydata()
    assert np.nanmean(plotted_acceleration) < 0  # slowing down in the direction of rotation


def test_fluctuation_plot():
    table, _ = make_signal_result(rpm=150.0, ripple_deg=4.0, ripple_hz=3.0)
    figure = plot_speed_fluctuation(table)
    axis = figure.axes[0]

    assert axis.get_title() == "Angular speed fluctuation about the trend"
    assert axis.get_xlabel() == "Time (s)"
    assert axis.get_ylabel() == "Speed fluctuation (rpm)"
    thick = [line for line in axis.get_lines() if line.get_linewidth() == 2.0][0].get_ydata()
    assert np.nanmax(thick) == pytest.approx(11.8, abs=1.0)  # 12.4 rpm ripple, a little lost by the smoothing
    assert np.isnan(thick[:6]).all()  # the ends of the smooth curve are empty


def test_spectrum_plot_marks_the_peak_the_threshold_and_the_rotation_frequency(tmp_path):
    _, result = make_signal_result(rpm=150.0, ripple_deg=4.0, ripple_hz=3.0)
    figure = plot_spectrum(result)
    axis = figure.axes[0]

    assert axis.get_title() == "Frequency spectrum of the speed fluctuation"
    assert axis.get_xlabel() == "Frequency (Hz)" and axis.get_ylabel() == "Amplitude (rpm)"
    assert axis.get_xlim() == pytest.approx((0.0, 30.0))
    legend_texts = [text.get_text() for text in axis.get_legend().get_texts()]
    assert any(text.startswith("lowest listed peak") for text in legend_texts)
    assert "rotation frequency (2.50 Hz)" in legend_texts
    assert "listed peaks" in legend_texts
    annotations = [text.get_text() for text in axis.texts]
    assert "3.00 Hz" in annotations
    assert save_figure(figure, tmp_path / "spectrum.png").stat().st_size > 1000


def test_spectrum_plot_without_peaks_and_with_the_simulated_mark():
    _, result = make_signal_result(rpm=120.0)
    assert result["peaks"] == []
    figure = plot_spectrum(result, simulated=True)
    axis = figure.axes[0]
    assert axis.get_title().endswith("(simulated data)")
    legend_texts = [text.get_text() for text in axis.get_legend().get_texts()]
    assert "listed peaks" not in legend_texts
    assert not axis.texts  # no note, no annotations


def test_spectrum_plot_warns_when_the_speed_is_not_steady():
    _, result = make_signal_result(rpm=300.0, fps=60.0, seconds=8.0, decay_s=3.0)
    assert not result["steady"]
    texts = [text.get_text() for text in plot_spectrum(result).axes[0].texts]
    assert any("not steady" in text for text in texts)


def test_all_the_signal_plots_can_be_saved(tmp_path):
    table, result = make_signal_result(rpm=150.0, ripple_deg=4.0, ripple_hz=3.0, lost_frames=[100, 101])
    for index, figure in enumerate([plot_rpm_vs_time(table, 150.0, True), plot_speed_fluctuation(table, True), plot_spectrum(result, True)]):
        assert save_figure(figure, tmp_path / f"signal{index}.png").stat().st_size > 1000


def test_signal_plots_work_with_a_different_trend_order_and_a_short_window(tmp_path):
    table, result = make_signal_result(
        SignalSettings(smooth_window_s=0.1, trend_order=3), rpm=300.0, fps=60.0, seconds=8.0, decay_s=3.0
    )
    save_figure(plot_speed_fluctuation(table), tmp_path / "fluctuation.png")
    save_figure(plot_spectrum(result), tmp_path / "spectrum.png")


# ---------------------------------------------------------------------------
# Two pulleys
# ---------------------------------------------------------------------------


def make_belt_table(driver_rpm=120.0, driven_rpm=60.0, with_surface=True, **kwargs):
    driver, _ = make_signal_result(rpm=driver_rpm, centre=(200.0, 150.0), radius_px=80.0, **kwargs)
    driven, _ = make_signal_result(rpm=driven_rpm, centre=(500.0, 150.0), radius_px=120.0, **kwargs)
    merged = merge_pulley_tables(driver, driven)
    radii = (25.0, 50.0) if with_surface else (None, None)
    return add_belt_columns(merged, 2.0, *radii)


def test_two_pulley_speed_plot_has_both_pulleys_and_the_ratio():
    figure = plot_two_pulley_speeds(make_belt_table(), expected_ratio=2.0)
    top, bottom = figure.axes

    assert top.get_title() == "Driver and driven pulley speeds"
    assert top.get_ylabel() == "Rotation speed (rpm)"
    assert [text.get_text() for text in top.get_legend().get_texts()] == ["driver", "driven"]
    dark_lines = [line for line in top.get_lines() if line.get_linewidth() == 2.0]
    assert np.nanmean(dark_lines[0].get_ydata()) == pytest.approx(120.0, abs=0.5)
    assert np.nanmean(dark_lines[1].get_ydata()) == pytest.approx(60.0, abs=0.5)

    assert bottom.get_ylabel() == "Speed ratio, driver / driven"
    assert bottom.get_xlabel() == "Time (s)"
    assert [text.get_text() for text in bottom.get_legend().get_texts()] == ["measured", "expected 2.000"]
    low, high = bottom.get_ylim()
    assert low < 2.0 < high


def test_a_crossed_belt_is_shown_with_positive_speeds_and_a_note():
    figure = plot_two_pulley_speeds(make_belt_table(driven_rpm=-60.0), expected_ratio=2.0)
    top, bottom = figure.axes
    dark_lines = [line for line in top.get_lines() if line.get_linewidth() == 2.0]
    assert np.nanmean(dark_lines[1].get_ydata()) == pytest.approx(60.0, abs=0.5)  # shown as a positive speed
    assert bottom.get_ylabel().endswith("(opposite directions)")


def test_two_pulley_plot_without_expected_ratio_or_smoothing_and_with_the_simulated_mark(tmp_path):
    merged = make_belt_table()
    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("rpm_smooth")])
    figure = plot_two_pulley_speeds(merged, simulated=True)
    top, bottom = figure.axes
    assert top.get_title().endswith("(simulated data)")
    assert [text.get_text() for text in bottom.get_legend().get_texts()] == ["measured"]
    assert save_figure(figure, tmp_path / "two.png").stat().st_size > 1000


def test_the_ratio_panel_is_not_flattened_by_a_few_wild_values():
    merged = make_belt_table()
    merged.loc[100, "speed_ratio"] = 40.0
    low, high = plot_two_pulley_speeds(merged, expected_ratio=2.0).axes[1].get_ylim()
    assert high < 5.0


def test_surface_speed_plot(tmp_path):
    figure = plot_surface_speeds(make_belt_table())
    axis = figure.axes[0]
    assert axis.get_title() == "Surface speed of the two pulleys (omega x radius)"
    assert axis.get_ylabel() == "Surface speed (m/s)"
    assert [text.get_text() for text in axis.get_legend().get_texts()] == ["driver", "driven"]
    assert np.nanmean(axis.get_lines()[0].get_ydata()) == pytest.approx(0.314, rel=0.01)
    assert save_figure(figure, tmp_path / "surface.png").exists()

    with pytest.raises(ValueError, match="radius of both pulleys"):
        plot_surface_speeds(make_belt_table(with_surface=False))


# ---------------------------------------------------------------------------
# Belt speed and slip
# ---------------------------------------------------------------------------


def make_slip_table(with_slip=True):
    merged = make_belt_table()
    count = len(merged)
    belt = np.full(count, np.nan)
    belt[40:80] = 0.2875  # one pass of the belt mark
    belt[150:190] = 0.2880
    merged["belt_speed_m_s"] = belt
    if with_slip:
        driver = merged["driver_surface_speed_m_s"].to_numpy()
        driven = merged["driven_surface_speed_m_s"].to_numpy()
        merged["driver_slip_percent"] = (driver - belt) / driver * 100
        merged["driven_slip_percent"] = (belt - driven) / belt * 100
    return merged


def test_belt_speed_plot(tmp_path):
    figure = plot_belt_speed(make_slip_table())
    axis = figure.axes[0]
    assert axis.get_title() == "Belt speed and pulley surface speeds"
    assert axis.get_ylabel() == "Speed (m/s)" and axis.get_xlabel() == "Time (s)"
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "driver surface", "driven surface", "belt mark, straight spans"
    ]
    belt_line = axis.get_lines()[2].get_ydata()
    assert np.isnan(belt_line[:40]).all() and belt_line[50] == pytest.approx(0.2875)  # in pieces: nothing between the passes
    assert save_figure(figure, tmp_path / "belt.png").stat().st_size > 1000


def test_slip_plot(tmp_path):
    figure = plot_slip(make_slip_table(), simulated=True)
    axis = figure.axes[0]
    assert axis.get_title() == "Slip of the belt on the two pulleys (simulated data)"
    assert axis.get_ylabel() == "Slip (percent)"
    assert [text.get_text() for text in axis.get_legend().get_texts()] == ["slip on the driver pulley", "slip on the driven pulley"]
    # driver surface 0.314 m/s against a belt at 0.2875 m/s: slip about 8.5 percent
    assert np.nanmean(axis.get_lines()[0].get_ydata()) == pytest.approx((0.3142 - 0.2877) / 0.3142 * 100, abs=0.3)
    assert save_figure(figure, tmp_path / "slip.png").exists()


def test_belt_plots_need_their_columns():
    with pytest.raises(ValueError, match="belt speed plot needs"):
        plot_belt_speed(make_belt_table())
    with pytest.raises(ValueError, match="slip plot needs"):
        plot_slip(make_slip_table(with_slip=False))
