"""Tests for src/signal_analysis.py. Simulated signals with a known right answer are used."""

import math

import numpy as np
import pandas as pd
import pytest

from src.kinematics import AnalysisError, add_angle_columns, add_speed_columns
from src.signal_analysis import (
    MIN_STRETCH_FRAMES,
    SignalSettings,
    analyse_speed_signal,
    compute_spectrum,
    describe_peak,
    fill_short_gaps,
    estimate_noise,
    find_analysis_stretch,
    find_outliers,
    find_spectrum_peaks,
    fit_trend,
    smooth_signal,
    smoothing_window_samples,
    validate_signal_settings,
)
from src.synthetic import make_synthetic_table, true_marker_speed_rpm, write_synthetic_video
from src.tracking import add_tracking_quality, settings_for_colour, track_marker

CENTRE = (320.0, 240.0)


def speed_table(centre_used=CENTRE, noise_px=0.05, **kwargs):
    """A simulated table that has gone through the angle and speed calculation.

    kwargs go to make_synthetic_table. centre_used is the centre given to the angle calculation,
    which can differ from the true centre (320, 240) to simulate a centre error.
    """
    table = make_synthetic_table(noise_px=noise_px, **kwargs)
    table = add_tracking_quality(table, centre_used)
    return add_speed_columns(add_angle_columns(table, centre_used))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_default_settings_are_valid():
    validate_signal_settings(SignalSettings())


@pytest.mark.parametrize(
    "bad_values",
    [
        {"smooth_window_s": 0},
        {"smooth_polyorder": 0},
        {"smooth_polyorder": 6},
        {"outlier_threshold": 1.0},
        {"max_gap_frames": -1},
        {"trend_order": -1},
        {"trend_order": 6},
        {"max_peaks": 0},
        {"min_peak_percent": -0.1},
    ],
)
def test_bad_settings_are_rejected(bad_values):
    with pytest.raises(AnalysisError):
        validate_signal_settings(SignalSettings(**bad_values))


# ---------------------------------------------------------------------------
# Stretch and gaps
# ---------------------------------------------------------------------------


def test_stretch_covers_everything_when_all_frames_have_a_speed():
    assert find_analysis_stretch(np.ones(50, dtype=bool), 3) == (0, 49)


def test_short_gaps_do_not_break_the_stretch_but_long_ones_do():
    has_speed = np.ones(100, dtype=bool)
    has_speed[10:13] = False  # 3 lost frames in a row: filled in
    assert find_analysis_stretch(has_speed, 3) == (0, 99)

    has_speed[50:54] = False  # 4 lost frames in a row: breaks the stretch. The longer side is frames 0 to 49.
    assert find_analysis_stretch(has_speed, 3) == (0, 49)

    has_speed[:] = True
    has_speed[40:44] = False  # now the longer side is frames 0 to 39 (40 frames) against 44 to 99 (56 frames)
    assert find_analysis_stretch(has_speed, 3) == (44, 99)


def test_frames_without_a_speed_at_the_start_and_end_are_left_out():
    has_speed = np.ones(60, dtype=bool)
    has_speed[:5] = False
    has_speed[-7:] = False
    assert find_analysis_stretch(has_speed, 3) == (5, 52)


def test_no_gap_is_allowed_when_max_gap_frames_is_zero():
    has_speed = np.ones(30, dtype=bool)
    has_speed[10] = False
    assert find_analysis_stretch(has_speed, 0) == (11, 29)


def test_no_speed_at_all_is_an_error():
    with pytest.raises(AnalysisError, match="nothing to analyse"):
        find_analysis_stretch(np.zeros(10, dtype=bool), 3)


def test_gaps_are_filled_with_straight_lines_over_time():
    filled, mask = fill_short_gaps([0.0, 1.0, 2.0, 3.0], [1.0, np.nan, np.nan, 4.0])
    assert filled.tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert mask.tolist() == [False, True, True, False]

    # The real time is used, so uneven spacing is handled correctly.
    filled, _ = fill_short_gaps([0.0, 1.0, 3.0], [0.0, np.nan, 6.0])
    assert filled[1] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def test_smoothing_window_is_odd_and_fits_the_data():
    assert smoothing_window_samples(0.2, 60.0, 300, 2) == 13  # 12 samples made odd
    assert smoothing_window_samples(0.2, 30.0, 300, 2) == 7  # 6 samples made odd
    assert smoothing_window_samples(0.2, 240.0, 300, 2) == 49
    assert smoothing_window_samples(0.01, 60.0, 300, 2) == 5  # never smaller than polyorder + 2 (made odd)
    assert smoothing_window_samples(2.0, 60.0, 30, 2) == 29  # cut down to fit 30 samples
    with pytest.raises(AnalysisError):
        smoothing_window_samples(0.2, 60.0, 2, 2)  # two samples are too few for any smoothing


def test_smoothing_leaves_a_polynomial_unchanged():
    t = np.linspace(0.0, 1.0, 100)
    quadratic = 100.0 + 20.0 * t - 15.0 * t**2
    assert np.allclose(smooth_signal(quadratic, 13, 2), quadratic)  # exact, also at the ends


def test_smoothing_reduces_white_noise_by_the_expected_factor():
    # For window 13 and polynomial order 2 the noise variance is reduced to 0.175 of the original,
    # so the standard deviation to 0.418.
    noise = np.random.default_rng(2).normal(0.0, 1.0, 40000)
    smoothed = smooth_signal(noise, 13, 2)
    assert smoothed[20:-20].std() == pytest.approx(0.418, abs=0.02)


def test_default_smoothing_reduces_a_3_hz_ripple_at_60_fps_by_about_5_percent():
    # Smoothing always costs some of a fast fluctuation. With a 13 sample window and polynomial order 2,
    # a 3 Hz ripple at 60 fps (fps / 20) keeps 95 percent of its height. The spectrum is made from the
    # unsmoothed signal, so it does not have this loss.
    t = np.arange(0, 20, 1 / 60.0)
    ripple = 10.0 * np.sin(2 * math.pi * 3.0 * t)
    smoothed = smooth_signal(ripple, 13, 2)
    assert smoothed[30:-30].max() == pytest.approx(9.5, abs=0.1)
    # A shorter window (7 samples) keeps more of it.
    assert smooth_signal(ripple, 7, 2)[30:-30].max() > 9.8


# ---------------------------------------------------------------------------
# Noise estimate
# ---------------------------------------------------------------------------


def test_noise_estimate_of_white_noise():
    # For independent noise and a 13 sample window, the difference from the smooth curve has a standard
    # deviation of sqrt(1 - 0.1748) = 0.908 times the noise.
    noise = np.random.default_rng(12).normal(0.0, 0.7, 20000)
    assert estimate_noise(noise, 60.0) == pytest.approx(0.7 * 0.908, rel=0.03)
    assert estimate_noise(120.0 + noise, 60.0) == pytest.approx(0.7 * 0.908, rel=0.03)  # a constant level makes no difference


def test_noise_estimate_ignores_slow_changes_and_counts_only_a_little_of_a_moderate_ripple():
    t = np.arange(3000) / 60.0
    noise = np.random.default_rng(13).normal(0.0, 0.2, 3000)
    base = estimate_noise(120.0 + noise, 60.0)

    assert estimate_noise(120.0 + 12.0 * np.sin(2 * math.pi * 0.5 * t) + noise, 60.0) == pytest.approx(base, rel=0.05)
    assert estimate_noise(300.0 * np.exp(-t / 4.0) + noise, 60.0) == pytest.approx(base, rel=0.05)
    # A 3 Hz ripple of 12 rpm at 60 fps is followed by the fit except for about 5 percent of it (0.6 rpm).
    with_ripple = estimate_noise(120.0 + 12.0 * np.sin(2 * math.pi * 3.0 * t) + noise, 60.0)
    assert base < with_ripple < 0.6


def test_noise_estimate_of_a_clean_signal_is_tiny():
    t = np.arange(600) / 60.0
    assert estimate_noise(np.full(600, 120.0), 60.0) == pytest.approx(0.0, abs=1e-9)
    assert estimate_noise(50.0 + 10.0 * t, 60.0) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------


def noisy_signal(seed=0, count=1500, ripple_hz=0.0, ripple_rpm=0.0, noise=0.2, fps=60.0):
    t = np.arange(count) / fps
    rng = np.random.default_rng(seed)
    return 120.0 + ripple_rpm * np.sin(2 * math.pi * ripple_hz * t) + rng.normal(0.0, noise, count)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("ripple_hz", [0.0, 3.0, 6.0])
def test_clean_noisy_signals_have_no_false_outliers(seed, ripple_hz):
    values = noisy_signal(seed, ripple_hz=ripple_hz, ripple_rpm=12.0)
    found, _ = find_outliers(values, 13, 2, 5.0)
    assert not found.any()


def test_noise_free_signals_have_no_outliers():
    t = np.arange(600) / 60.0
    for values in (np.full(600, 120.0), 120.0 + 12.0 * np.sin(2 * math.pi * 3.0 * t), 50.0 + 10.0 * t):
        found, _ = find_outliers(values, 13, 2, 5.0)
        assert not found.any()


def test_injected_errors_are_found_exactly_and_their_neighbours_are_not():
    values = noisy_signal(7, ripple_hz=3.0, ripple_rpm=12.0)
    truth = values.copy()
    errors = {300: 8.0, 700: -9.0, 701: 7.0, 1100: 12.0, 1101: 10.0}  # a lone error, an opposite pair, a same-sign pair
    for position, size in errors.items():
        values[position] += size

    found, reference = find_outliers(values, 13, 2, 5.0)
    assert np.flatnonzero(found).tolist() == sorted(errors)
    # The smooth reference at those positions is close to the value without the error.
    for position in errors:
        assert reference[position] == pytest.approx(truth[position], abs=1.0)


def test_errors_are_found_when_the_noise_is_large():
    values = noisy_signal(8, noise=2.0)
    values[[200, 900]] += [25.0, -30.0]
    found, _ = find_outliers(values, 13, 2, 5.0)
    assert np.flatnonzero(found).tolist() == [200, 900]


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


def test_trend_of_order_0_is_the_mean():
    t = np.arange(100) / 60.0
    values = 100.0 + 5.0 * np.sin(2 * math.pi * 3.0 * t)
    assert fit_trend(t, values, 0) == pytest.approx(np.full(100, values.mean()))


def test_trend_of_order_1_removes_a_drift_and_leaves_the_ripple():
    t = np.arange(0, 10, 1 / 60.0)
    ripple = 3.0 * np.sin(2 * math.pi * 2.0 * t)
    values = 100.0 - 4.0 * t + ripple
    fluctuation = values - fit_trend(t, values, 1)
    assert fluctuation == pytest.approx(ripple, abs=0.2)  # a small error remains because the ripple is not a straight line


def test_trend_of_order_2_is_exact_for_a_quadratic():
    t = np.arange(50) / 30.0
    values = 80.0 + 10.0 * t - 3.0 * t**2
    assert fit_trend(t, values, 2) == pytest.approx(values)


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frequency", [2.0, 3.0, 3.07, 7.3])
def test_spectrum_gives_the_amplitude_and_frequency_of_a_sine(frequency):
    fps = 60.0
    t = np.arange(300) / fps
    values = 5.0 * np.sin(2 * math.pi * frequency * t + 0.4)  # amplitude 5 rpm
    frequency_hz, amplitude = compute_spectrum(values, fps)

    peak = amplitude.argmax()
    assert frequency_hz[peak] == pytest.approx(frequency, abs=0.02)
    assert amplitude[peak] == pytest.approx(5.0, rel=0.01)


def test_spectrum_axis_goes_up_to_half_the_frame_rate():
    frequency_hz, _ = compute_spectrum(np.random.default_rng(1).normal(size=300), 60.0)
    assert frequency_hz[0] == 0.0
    assert frequency_hz[-1] == pytest.approx(30.0)


def test_spectrum_shows_two_separate_tones():
    fps = 60.0
    t = np.arange(600) / fps  # 10 s, resolution 0.1 Hz
    values = 4.0 * np.sin(2 * math.pi * 2.5 * t) + 1.5 * np.sin(2 * math.pi * 7.0 * t + 1.0)
    frequency_hz, amplitude = compute_spectrum(values, fps)
    peaks, _, _ = find_spectrum_peaks(frequency_hz, amplitude, 10.0, 2.0, 120.0, True, 5)

    assert [round(p["frequency_hz"], 1) for p in peaks] == [2.5, 7.0]
    assert [p["amplitude_rpm"] for p in peaks] == pytest.approx([4.0, 1.5], rel=0.02)


# ---------------------------------------------------------------------------
# Peaks
# ---------------------------------------------------------------------------


def test_peak_order_and_notes():
    resolution = 0.2
    order, note = describe_peak(2.0, 2.0, resolution, True)
    assert order == pytest.approx(1.0)
    assert "once per revolution" in note and "centre error" in note

    order, note = describe_peak(4.05, 2.0, resolution, True)
    assert order == pytest.approx(2.025)
    assert note == "2 times per revolution"

    order, note = describe_peak(3.0, 2.0, resolution, True)
    assert order == pytest.approx(1.5)
    assert note == "periodic speed variation, source not identified"

    order, note = describe_peak(1.0, 2.0, resolution, True)  # half the rotation frequency: not a multiple
    assert order == pytest.approx(0.5)
    assert "source not identified" in note

    order, note = describe_peak(2.0, 2.0, resolution, False)  # speed not steady: no order
    assert order is None and "not steady" in note


def test_notes_do_not_name_a_cause():
    for note in (
        describe_peak(2.0, 2.0, 0.2, True)[1],
        describe_peak(4.0, 2.0, 0.2, True)[1],
        describe_peak(3.1, 2.0, 0.2, True)[1],
    ):
        for word in ("bearing", "fault", "damage", "failure", "unbalance", "imbalance", "defect"):
            assert word not in note.lower()


def test_peaks_are_sorted_limited_and_below_the_minimum_amplitude_are_left_out():
    fps = 60.0
    t = np.arange(600) / fps
    values = (
        6.0 * np.sin(2 * math.pi * 3.0 * t)
        + 3.0 * np.sin(2 * math.pi * 5.0 * t)
        + 1.5 * np.sin(2 * math.pi * 8.0 * t)
        + 0.8 * np.sin(2 * math.pi * 11.0 * t)
    )
    frequency_hz, amplitude = compute_spectrum(values, fps)

    peaks, _, threshold = find_spectrum_peaks(frequency_hz, amplitude, 10.0, 2.0, 120.0, True, 3)
    assert [round(p["frequency_hz"]) for p in peaks] == [3, 5, 8]  # biggest first, only 3 allowed

    peaks, _, threshold = find_spectrum_peaks(frequency_hz, amplitude, 10.0, 2.0, 120.0, True, 5, min_amplitude_rpm=2.0)
    assert [round(p["frequency_hz"]) for p in peaks] == [3, 5]  # the 1.5 rpm and 0.8 rpm tones are below 2 rpm
    assert threshold >= 2.0


def test_the_smallest_listed_peak_is_a_percentage_of_the_mean_speed_and_can_be_changed():
    # A ripple of 1 rpm at 3 Hz on 120 rpm (0.8 percent of the speed). The angle ripple for that is
    # 1 rpm / (60 x 3 Hz) = 0.0056 rad = 0.318 degrees.
    table = speed_table(rpm=120.0, ripple_deg=0.318, ripple_hz=3.0)

    _, default = analyse_speed_signal(table)  # limit: 0.5 percent of 120 rpm = 0.6 rpm
    assert default["peak_threshold_rpm"] >= 0.6
    assert len(default["peaks"]) == 1
    assert default["peaks"][0]["frequency_hz"] == pytest.approx(3.0, abs=0.05)
    assert default["peaks"][0]["amplitude_rpm"] == pytest.approx(0.98, rel=0.1)

    _, strict = analyse_speed_signal(table, SignalSettings(min_peak_percent=1.0))  # limit: 1.2 rpm
    assert strict["peak_threshold_rpm"] >= 1.2
    assert strict["peaks"] == []


def test_the_pixel_grid_peaks_of_a_clean_steady_video_are_not_listed(tmp_path):
    # In a rendered video the marker position is rounded by the pixel grid in a way that repeats every
    # revolution. That makes peaks of about 0.14 rpm (0.1 percent) at multiples of the rotation frequency.
    path = tmp_path / "steady.mp4"
    write_synthetic_video(path, rpm=120.0, fps=60.0, seconds=5.0, width=640, height=480)
    table = track_marker(path, settings_for_colour("green"), centre=(320, 240))
    _, result = analyse_speed_signal(add_speed_columns(add_angle_columns(table, (320, 240))))
    assert result["peaks"] == []
    assert result["peak_threshold_rpm"] == pytest.approx(0.6, rel=0.01)  # 0.5 percent of the mean speed


def test_a_strong_peak_does_not_produce_side_lobe_peaks():
    fps = 60.0
    t = np.arange(300) / fps
    values = 100.0 * np.sin(2 * math.pi * 4.37 * t) + np.random.default_rng(4).normal(0.0, 0.01, 300)
    frequency_hz, amplitude = compute_spectrum(values, fps)
    peaks, _, _ = find_spectrum_peaks(frequency_hz, amplitude, 5.0, 2.0, 120.0, True, 5)
    assert len(peaks) == 1
    assert peaks[0]["frequency_hz"] == pytest.approx(4.37, abs=0.02)


def test_frequencies_with_fewer_than_two_cycles_in_the_clip_are_not_looked_at():
    fps = 60.0
    t = np.arange(300) / fps  # 5 s
    values = 5.0 * np.sin(2 * math.pi * 0.2 * t)  # one cycle in 5 s
    frequency_hz, amplitude = compute_spectrum(values, fps)
    peaks, _, _ = find_spectrum_peaks(frequency_hz, amplitude, 5.0, 2.0, 120.0, True, 5)
    assert peaks == []


def test_noise_only_gives_no_peaks():
    values = np.random.default_rng(6).normal(0.0, 1.0, 600)
    frequency_hz, amplitude = compute_spectrum(values, 60.0)
    noise = float(values.std())
    peaks, _, _ = find_spectrum_peaks(frequency_hz, amplitude, 10.0, 2.0, 120.0, True, 5, min_amplitude_rpm=noise)
    assert peaks == []


# ---------------------------------------------------------------------------
# The whole analysis, on simulated tables
# ---------------------------------------------------------------------------


def test_steady_speed_has_no_outliers_no_peaks_and_the_right_mean():
    table, result = analyse_speed_signal(speed_table(rpm=120.0))

    assert (result["first_frame"], result["last_frame"], result["frames"]) == (0, 299, 300)
    assert result["duration_s"] == pytest.approx(5.0)
    assert result["frequency_resolution_hz"] == pytest.approx(0.2)
    assert result["nyquist_hz"] == pytest.approx(30.0)
    assert result["mean_rpm"] == pytest.approx(120.0, abs=0.05)
    assert result["rotation_frequency_hz"] == pytest.approx(2.0, abs=0.001)
    assert result["steady"]
    assert result["outliers"] == 0 and result["interpolated_frames"] == 0
    assert result["peaks"] == []
    assert result["window_samples"] == 13
    assert result["fluctuation_coefficient_percent"] < 1.0

    for name in ("speed_outlier", "speed_interpolated", "rpm_clean", "rpm_smooth", "rpm_trend", "rpm_fluctuation",
                 "angular_acceleration_rad_s2"):
        assert name in table.columns
    assert table["rpm_fluctuation"].abs().max() < 1.5


def test_a_ripple_shows_as_a_peak_with_the_right_frequency_and_amplitude():
    # 150 rpm with an angle ripple of 4 degrees at 3 Hz. The speed ripple amplitude is 4 degrees x 2 pi x 3 Hz = 12.57 rpm.
    # The central difference reduces a 3 Hz ripple at 60 fps by 1.6 percent: 12.57 x 0.9836 = 12.36 rpm.
    table, result = analyse_speed_signal(speed_table(rpm=150.0, ripple_deg=4.0, ripple_hz=3.0))

    assert len(result["peaks"]) == 1
    peak = result["peaks"][0]
    assert peak["frequency_hz"] == pytest.approx(3.0, abs=0.03)
    assert peak["amplitude_rpm"] == pytest.approx(12.36, rel=0.03)
    assert peak["percent_of_mean"] == pytest.approx(100 * peak["amplitude_rpm"] / 150.0, rel=1e-3)
    assert peak["order"] == pytest.approx(3.0 / 2.5, abs=0.02)
    assert "source not identified" in peak["note"]
    assert peak["period_s"] == pytest.approx(1 / 3.0, abs=0.005)

    # In the time domain the speed swings by about 2 x 12.4 rpm (a little more because of the noise).
    assert result["fluctuation_peak_to_peak_rpm"] == pytest.approx(24.7, rel=0.15)
    assert result["fluctuation_coefficient_percent"] == pytest.approx(
        100 * result["fluctuation_peak_to_peak_rpm"] / 150.0, rel=1e-3
    )


def test_two_ripples_are_both_found():
    table = make_synthetic_table(rpm=120.0, seconds=8.0, noise_px=0.05)
    # Add a second ripple by hand: shift the marker angle by 2 degrees at 7 Hz on top of the 3 Hz one.
    first = make_synthetic_table(rpm=120.0, seconds=8.0, ripple_deg=4.0, ripple_hz=3.0, noise_px=0.05)
    extra = np.radians(2.0) * np.sin(2 * math.pi * 7.0 * first["time_s"].to_numpy())
    angle = np.arctan2(CENTRE[1] - first["marker_y"], first["marker_x"] - CENTRE[0]) + extra
    table["marker_x"] = CENTRE[0] + 144.0 * np.cos(angle)
    table["marker_y"] = CENTRE[1] - 144.0 * np.sin(angle)

    table = add_speed_columns(add_angle_columns(add_tracking_quality(table, CENTRE), CENTRE))
    _, result = analyse_speed_signal(table)
    frequencies = sorted(round(p["frequency_hz"], 1) for p in result["peaks"])
    assert frequencies == [3.0, 7.0]


def test_a_wrong_centre_shows_as_a_once_per_revolution_peak_and_the_right_centre_does_not():
    # The marker really moves on a circle around (320, 240). The centre 6 pixels to the right is wrong.
    _, wrong = analyse_speed_signal(speed_table(centre_used=(326.0, 240.0), rpm=120.0))
    _, right = analyse_speed_signal(speed_table(centre_used=(320.0, 240.0), rpm=120.0))

    assert right["peaks"] == []
    first = wrong["peaks"][0]
    assert first["frequency_hz"] == pytest.approx(2.0, abs=0.03)  # the rotation frequency
    assert first["order"] == pytest.approx(1.0, abs=0.02)
    assert "once per revolution" in first["note"] and "centre error" in first["note"]
    # The speed ripple is omega x (error / radius) = 12.57 rad/s x 6 / 144 = 0.52 rad/s = 5.0 rpm.
    assert first["amplitude_rpm"] == pytest.approx(5.0, rel=0.05)


def test_errors_in_single_frames_are_replaced_and_marked():
    table = make_synthetic_table(rpm=120.0, noise_px=0.05)
    # A wrong marker position in frames 100 and 200 (the marker jumps 5 and 8 degrees away).
    for frame, degrees in ((100, 5.0), (200, -8.0)):
        angle = math.atan2(CENTRE[1] - table.loc[frame, "marker_y"], table.loc[frame, "marker_x"] - CENTRE[0])
        angle += math.radians(degrees)
        table.loc[frame, "marker_x"] = CENTRE[0] + 144.0 * math.cos(angle)
        table.loc[frame, "marker_y"] = CENTRE[1] - 144.0 * math.sin(angle)
    table = add_speed_columns(add_angle_columns(add_tracking_quality(table, CENTRE), CENTRE))

    cleaned, result = analyse_speed_signal(table)
    # One wrong position makes the two neighbouring frames wrong (the speed uses the frames on both sides).
    assert result["outlier_frames"] == [99, 101, 199, 201]
    assert result["outliers"] == 4
    assert cleaned["speed_outlier"].sum() == 4
    assert result["peaks"] == []
    assert (cleaned["rpm_clean"] - 120.0).abs().max() < 1.5  # no spike is left

    # Without the outlier step the errors stay in the data.
    kept, kept_result = analyse_speed_signal(table, SignalSettings(remove_outliers=False))
    assert kept_result["outliers"] == 0
    assert (kept["rpm_clean"] - 120.0).abs().max() > 10.0


def test_a_few_lost_frames_are_filled_in_and_marked():
    table = speed_table(rpm=120.0, lost_frames=[100, 101, 102])
    result_table, result = analyse_speed_signal(table)

    assert (result["first_frame"], result["last_frame"]) == (0, 299)
    assert result["interpolated_frames"] == 3
    assert result_table.loc[[100, 101, 102], "speed_interpolated"].all()
    assert result_table["speed_interpolated"].sum() == 3
    assert result_table["rpm_clean"].notna().all()
    assert result_table.loc[100:102, "rpm_clean"].to_numpy() == pytest.approx([120.0] * 3, abs=1.0)
    assert result["peaks"] == []


def test_a_long_gap_limits_the_analysis_to_the_longer_stretch():
    table = speed_table(rpm=120.0, lost_frames=range(100, 110))  # 10 lost frames in a row
    result_table, result = analyse_speed_signal(table)

    assert (result["first_frame"], result["last_frame"], result["frames"]) == (110, 299, 190)
    assert result["frequency_resolution_hz"] == pytest.approx(1.0 / (190 / 60.0))
    assert result_table.loc[:109, "rpm_clean"].isna().all()  # outside the stretch the columns are empty
    assert not result_table.loc[:109, "speed_outlier"].any()
    assert result_table.loc[110:, "rpm_clean"].notna().all()

    # A larger allowance fills the gap in instead.
    _, wide = analyse_speed_signal(table, SignalSettings(max_gap_frames=10))
    assert (wide["first_frame"], wide["last_frame"], wide["interpolated_frames"]) == (0, 299, 10)


def test_too_short_a_stretch_gives_a_clear_error():
    table = speed_table(rpm=120.0, seconds=0.3)  # 18 frames
    with pytest.raises(AnalysisError, match=f"at least {MIN_STRETCH_FRAMES} are needed"):
        analyse_speed_signal(table)


def test_the_speed_columns_must_exist():
    table = make_synthetic_table()
    with pytest.raises(AnalysisError, match="add_speed_columns"):
        analyse_speed_signal(table)


def test_uneven_frame_times_are_rejected():
    table = speed_table(rpm=120.0, seconds=3.0)
    table.loc[50:, "time_s"] += 0.01  # a jump in the frame times
    with pytest.raises(AnalysisError, match="not evenly spaced"):
        analyse_speed_signal(table)


def test_the_input_table_is_not_changed():
    table = speed_table(rpm=120.0, seconds=3.0)
    columns_before = list(table.columns)
    analyse_speed_signal(table)
    assert list(table.columns) == columns_before


def test_clockwise_rotation_gives_the_same_analysis():
    _, clockwise = analyse_speed_signal(speed_table(rpm=-150.0, ripple_deg=-4.0, ripple_hz=3.0))
    _, counter = analyse_speed_signal(speed_table(rpm=150.0, ripple_deg=4.0, ripple_hz=3.0))
    assert clockwise["mean_rpm"] == pytest.approx(counter["mean_rpm"], rel=1e-3)
    assert clockwise["peaks"][0]["frequency_hz"] == pytest.approx(counter["peaks"][0]["frequency_hz"], abs=0.01)
    assert clockwise["peaks"][0]["amplitude_rpm"] == pytest.approx(counter["peaks"][0]["amplitude_rpm"], rel=0.02)


def test_a_slowing_disc_is_not_steady_and_a_higher_trend_order_removes_the_slow_down():
    table = speed_table(rpm=300.0, fps=60.0, seconds=10.0, decay_s=4.0, noise_px=0.05)
    _, simple = analyse_speed_signal(table)
    assert not simple["steady"]
    assert abs(simple["trend_change_rpm"]) > 100.0
    assert all(peak["order"] is None for peak in simple["peaks"])

    _, higher = analyse_speed_signal(table, SignalSettings(trend_order=4))
    assert higher["fluctuation_peak_to_peak_rpm"] < 2.0
    assert higher["peaks"] == []


def test_acceleration_of_a_slowing_disc_matches_the_true_value():
    # omega(t) = omega0 exp(-t / tau), so the acceleration is -omega0 / tau x exp(-t / tau).
    rpm0, tau = 300.0, 4.0
    omega0 = 2 * math.pi * rpm0 / 60.0
    table, _ = analyse_speed_signal(speed_table(rpm=rpm0, fps=60.0, seconds=10.0, decay_s=tau, noise_px=0.0))

    time_s = table["time_s"].to_numpy()
    expected = -omega0 / tau * np.exp(-time_s / tau)
    measured = table["angular_acceleration_rad_s2"].to_numpy()
    interior = slice(30, -30)
    assert measured[interior] == pytest.approx(expected[interior], rel=0.03, abs=0.05)
    # The window is 13 frames, so the acceleration starts at frame 6. There it is already about -7.85 rad/s^2 x exp(-0.1 / 4).
    assert measured[6] == pytest.approx(expected[6], rel=0.03)
    assert expected[6] == pytest.approx(-omega0 / tau, rel=0.03)


def test_acceleration_of_a_steady_speed_is_close_to_zero():
    table, result = analyse_speed_signal(speed_table(rpm=120.0))
    # Differentiating a noisy speed gives noise of a few tenths of rad/s^2, which is small next to
    # the real accelerations of a disc that is speeding up or slowing down (several rad/s^2).
    assert result["acceleration_max_abs_rad_s2"] < 1.5
    assert abs(result["acceleration_mean_rad_s2"]) < 0.1


def test_smoothing_window_is_reported_and_does_not_change_the_noise_estimate():
    table = speed_table(rpm=150.0, ripple_deg=4.0, ripple_hz=3.0)
    _, short = analyse_speed_signal(table, SignalSettings(smooth_window_s=0.1))
    _, default = analyse_speed_signal(table)
    _, long = analyse_speed_signal(table, SignalSettings(smooth_window_s=0.4))

    assert (short["window_samples"], default["window_samples"], long["window_samples"]) == (7, 13, 25)
    assert long["window_s"] == pytest.approx(25 / 60.0)
    # The noise is measured with a short fixed fit, so a longer smoothing window does not make it bigger,
    # and the 3 Hz ripple (12 rpm) is not counted as noise.
    assert short["noise_rms_rpm"] == default["noise_rms_rpm"] == long["noise_rms_rpm"]
    assert default["noise_rms_rpm"] < 0.6
    # The spectrum does not depend on the smoothing either.
    assert long["peaks"][0]["amplitude_rpm"] == pytest.approx(default["peaks"][0]["amplitude_rpm"])


def test_frame_rate_is_taken_from_the_times():
    _, result = analyse_speed_signal(speed_table(rpm=60.0, fps=30.0, seconds=6.0))
    assert result["fps"] == pytest.approx(30.0)
    assert result["nyquist_hz"] == pytest.approx(15.0)
    assert result["window_samples"] == 7


# ---------------------------------------------------------------------------
# End to end, with a simulated video
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ripple_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("videos") / "ripple.mp4"
    write_synthetic_video(
        path, rpm=150.0, fps=60.0, seconds=5.0, width=640, height=480, ripple_deg=4.0, ripple_hz=3.0
    )
    return path


def test_ripple_in_a_simulated_video_is_found_at_3_hz(ripple_video):
    table = track_marker(ripple_video, settings_for_colour("green"), centre=(320, 240))
    table = add_speed_columns(add_angle_columns(table, (320, 240)))
    result_table, result = analyse_speed_signal(table)

    assert len(result["peaks"]) == 1
    assert result["peaks"][0]["frequency_hz"] == pytest.approx(3.0, abs=0.03)
    assert result["peaks"][0]["amplitude_rpm"] == pytest.approx(12.36, rel=0.03)

    # The smooth speed follows the true speed (including the ripple) closely.
    truth = true_marker_speed_rpm(
        result_table["time_s"].to_numpy(), 150.0, ripple_amplitude_rad=math.radians(4.0), ripple_hz=3.0
    )
    error = (result_table["rpm_smooth"] - truth)[30:-30]
    assert error.abs().max() < 1.5
    assert isinstance(result_table, pd.DataFrame)


def test_the_smooth_speed_and_acceleration_are_left_empty_in_the_first_and_last_half_window():
    # A 3 Hz ripple would make the smoothing overshoot at the two ends, so those values are not given.
    table, result = analyse_speed_signal(speed_table(rpm=150.0, ripple_deg=4.0, ripple_hz=3.0))
    half = result["window_samples"] // 2  # 6 frames

    for column in ("rpm_smooth", "angular_acceleration_rad_s2"):
        values = table[column].to_numpy()
        assert np.isnan(values[:half]).all() and np.isnan(values[-half:]).all()
        assert not np.isnan(values[half:-half]).any()
    assert table["rpm_clean"].notna().all()  # the cleaned speed itself has no gaps
    assert result["acceleration_max_abs_rad_s2"] < 30.0  # no edge overshoot (the real peak is about 25 rad/s^2)


def test_in_the_first_and_last_half_window_only_very_big_errors_are_outliers():
    # In the end regions the smooth curve is less reliable, so the limit is 3 times higher (15 noise units).
    # The noise here is 0.2 rpm, so the limit is about 3 rpm at the ends and 1 rpm in the middle.
    values = noisy_signal(3, count=300)
    values[[2, 297]] += 2.0  # 10 noise units: an outlier in the middle, but not at the ends
    values[[100, 200]] += 2.0
    found, _ = find_outliers(values, 13, 2, 5.0)
    assert np.flatnonzero(found).tolist() == [100, 200]

    values = noisy_signal(3, count=300)
    values[[2, 297, 100]] += 40.0  # very big errors are found everywhere, and their neighbours are not blamed
    found, _ = find_outliers(values, 13, 2, 5.0)
    assert np.flatnonzero(found).tolist() == [2, 100, 297]


def test_a_smoothing_window_that_is_too_long_for_the_stretch_gives_a_clear_error():
    table = speed_table(rpm=120.0, seconds=0.45)  # 27 frames
    with pytest.raises(AnalysisError, match="too long"):
        analyse_speed_signal(table, SignalSettings(smooth_window_s=0.5))
