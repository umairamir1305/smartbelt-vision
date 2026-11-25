"""Signal processing of the rotation speed: cleaning, smoothing, fluctuation and frequency spectrum.

The speed in the table (the rpm column) has one value per frame. It contains measurement
noise, and sometimes a wrong value for a single frame. The steps here are:

  1. Stretch   Pick the longest stretch of frames where the speed is known, allowing short gaps
               (a few lost frames) that are filled in by straight-line interpolation. These
               filled-in frames are marked, they are not measurements. Frequency analysis
               needs evenly spaced samples without long holes.
  2. Outliers  A single frame whose speed is far from the smooth curve (more than 5 times the
               typical noise) is treated as a measurement error, for example a wrong blob in
               one frame. It is replaced by the smooth value and marked.
  3. Smoothing A Savitzky-Golay filter fits a small polynomial in a moving window. It keeps
               peaks better than a moving average. It is only used for the smooth curve, the
               acceleration and the outlier check, NOT for the spectrum. In the first and last
               half window it has no neighbours on one side and is not reliable (it can
               overshoot), so the smooth speed and the acceleration are left empty there.
  4. Trend     A polynomial fitted to the speed (order 0 is the mean, order 1 a straight line).
               The fluctuation is the speed minus the trend: delta_rpm = rpm - trend.
  5. Spectrum  scipy.signal.periodogram of the fluctuation (cleaned but not smoothed, so the
               smoothing cannot hide or change a peak). The result is an amplitude spectrum in rpm.

A peak in the spectrum shows that the speed varies periodically at that frequency. It does NOT
show the cause. Motor torque ripple, belt irregularity, misalignment, an error in the rotation
centre, a tilted camera, rolling shutter, frame timing and tracking noise can all do that.
The program only reports what it sees and gives no diagnosis.

Limits that come from the video:
  - The highest frequency that can be seen is half the frame rate (the Nyquist frequency).
  - The frequency resolution is 1 / duration. Peaks closer than about 2 / duration are not separated.
  - The speed comes from differences of neighbouring angles, which reduces a fast fluctuation
    by sin(x) / x with x = 2 pi f / fps (1.6 percent at fps / 20, 6.5 percent at fps / 10,
    36 percent at fps / 4).
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal

from src.kinematics import AnalysisError

# At least this many frames in a row are needed for any signal analysis.
MIN_STRETCH_FRAMES = 20

# The spectrum is calculated with zeros added up to at least this many times the number of
# samples. This does not add information, it only gives a smoother curve and a better
# reading of where a peak is and how high it is.
SPECTRUM_PADDING = 8

# A spectrum peak is listed only if it is higher than ALL of these:
#   - 5 times the typical (median) level of the spectrum
#   - 5 percent of the biggest peak (smaller ones cannot be told apart from the side lobes of the window)
#   - the noise of a single frame (the typical scatter of single frame speeds, see estimate_noise)
#   - 0.5 percent of the mean speed (the setting min_peak_percent)
# The last two are limits of the method. A fluctuation smaller than the noise is not trustworthy, and
# the pixel grid and the tracking cause small fluctuations of their own. Those repeat every
# revolution, so they show up as small peaks at multiples of the rotation frequency (about 0.1
# percent of the speed in the simulated videos, which are the best case).
PEAK_NOISE_RATIO = 5.0
PEAK_FRACTION_OF_MAX = 0.05

# The window used to measure the noise. It is fixed, so the noise does not depend on the smoothing window.
NOISE_WINDOW_S = 0.2

# Frequencies with fewer than this many full cycles in the clip are not looked at.
MIN_CYCLES_IN_CLIP = 2.0

# If the trend changes the speed by more than this share of the mean speed over the clip,
# the speed is not steady, the spectrum is not reliable and orders are not given.
STEADY_LIMIT = 0.2


@dataclass
class SignalSettings:
    """Settings for the signal analysis."""

    smooth_window_s: float = 0.2  # length of the smoothing window in seconds
    smooth_polyorder: int = 2  # order of the polynomial fitted in each window
    remove_outliers: bool = True
    outlier_threshold: float = 5.0  # in robust standard deviations of the noise
    max_gap_frames: int = 3  # lost frames in a row that are filled in by interpolation
    trend_order: int = 1  # 0 = subtract the mean, 1 = subtract a straight line, and so on
    max_peaks: int = 5  # number of spectrum peaks listed
    min_peak_percent: float = 0.5  # smallest peak that is listed, as a percentage of the mean speed


def validate_signal_settings(settings: SignalSettings) -> None:
    """Raise an AnalysisError with a plain message if a setting is out of range."""
    if settings.smooth_window_s <= 0:
        raise AnalysisError("The smoothing window must be longer than 0 seconds.")
    if not 1 <= settings.smooth_polyorder <= 5:
        raise AnalysisError("The smoothing polynomial order must be between 1 and 5.")
    if settings.outlier_threshold < 2:
        raise AnalysisError("The outlier threshold must be at least 2 (it is in standard deviations of the noise).")
    if settings.max_gap_frames < 0:
        raise AnalysisError("The number of frames that may be filled in cannot be negative.")
    if not 0 <= settings.trend_order <= 5:
        raise AnalysisError("The trend order must be between 0 and 5.")
    if settings.max_peaks < 1:
        raise AnalysisError("At least one spectrum peak must be allowed.")
    if settings.min_peak_percent < 0:
        raise AnalysisError("The smallest listed peak cannot be a negative percentage of the speed.")


def find_analysis_stretch(has_speed, max_gap_frames: int):
    """Find the longest stretch of frames where the speed is known, allowing short gaps.

    has_speed is True for frames that have a speed. Lost frames in a row, up to
    max_gap_frames of them, do not break a stretch. A longer gap does. Returns (first, last),
    the positions of the first and last frame of the longest stretch, both included.
    """
    positions = np.flatnonzero(np.asarray(has_speed, dtype=bool))
    if len(positions) == 0:
        raise AnalysisError("No frame has a speed, so there is nothing to analyse.")

    lost_between = np.diff(positions) - 1  # lost frames between two neighbouring known frames
    breaks = np.flatnonzero(lost_between > max_gap_frames)
    first_of_stretch = np.concatenate(([0], breaks + 1))
    last_of_stretch = np.concatenate((breaks, [len(positions) - 1]))

    spans = positions[last_of_stretch] - positions[first_of_stretch]
    best = int(np.argmax(spans))
    return int(positions[first_of_stretch[best]]), int(positions[last_of_stretch[best]])


def fill_short_gaps(time_s, values):
    """Fill NaN values by straight-line interpolation over time. Returns (filled values, filled mask).

    The values must start and end with a known value. The filled mask is True for the frames
    that were filled in, which are not measurements.
    """
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    missing = np.isnan(values)
    filled = values.copy()
    if missing.any():
        filled[missing] = np.interp(time_s[missing], time_s[~missing], values[~missing])
    return filled, missing


def smoothing_window_samples(window_s: float, fps: float, sample_count: int, polyorder: int) -> int:
    """Number of samples in the smoothing window: odd, at least polyorder + 2, at most the data length."""
    window = int(round(window_s * fps))
    window = max(window, polyorder + 2)
    if window % 2 == 0:
        window += 1
    if window > sample_count:
        window = sample_count if sample_count % 2 == 1 else sample_count - 1
    if window <= polyorder:
        raise AnalysisError("There are too few frames for the smoothing settings. Use a shorter smoothing window.")
    return window


def smooth_signal(values, window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay smoothing: a polynomial is fitted to the samples in a moving window.

    Near the two ends the polynomial is fitted to the first (or last) window and evaluated there.
    """
    return signal.savgol_filter(np.asarray(values, dtype=float), window, polyorder, mode="interp")


def estimate_noise(values, fps: float) -> float:
    """Typical size of the single frame scatter of the speed: the root mean square (rms) of the
    difference between the values and a smooth curve.

    The smooth curve is a Savitzky-Golay fit with a fixed window of 0.2 s and polynomial order 2, so
    the result does not depend on the smoothing window chosen for the plots. A slow fluctuation
    is followed by the curve and hardly counts. A fast one (about a tenth of the frame rate or more)
    is only partly followed and adds a little, because it cannot be told apart from noise.
    For noise that is independent from frame to frame, the result is about 0.91 times the noise
    at 60 fps. The real speed noise is not independent, so the number is a practical size, not
    an exact standard deviation.
    """
    values = np.asarray(values, dtype=float)
    window = smoothing_window_samples(NOISE_WINDOW_S, fps, len(values), 2)
    half = window // 2
    residual = (values - smooth_signal(values, window, 2))[half : len(values) - half]  # not the two end regions
    return float(np.sqrt(np.mean(residual**2)))


def find_outliers(values, window: int, polyorder: int, threshold: float):
    """Find single values that are far from the smooth curve. Returns (is_outlier, smooth curve).

    The noise is the typical size of the difference between the values and the smooth curve,
    measured with the median absolute deviation (MAD), which a few big errors cannot spoil.
    A value is an outlier if its difference is more than threshold times that noise (the noise
    is never taken as less than 0.1 percent of the typical value).

    A big error pulls the smooth curve towards itself and makes its neighbours look like
    outliers too. So the check is repeated: the values found so far are replaced by straight
    lines between their neighbours, the curve is made again from the original values with
    those replaced, and the check is run again, until nothing changes (at most 4 times).

    In the first and last half window the smooth curve has no neighbours on one side and is
    less reliable. There a value must be 3 times further from the curve to count as an outlier
    (so only very big errors are found), and those values are not used to measure the noise.
    """
    values = np.asarray(values, dtype=float)
    positions = np.arange(len(values))
    half = window // 2
    interior = np.zeros(len(values), dtype=bool)
    interior[half : len(values) - half] = True
    suspect = np.zeros(len(values), dtype=bool)
    reference = smooth_signal(values, window, polyorder)

    for _ in range(4):
        working = values.copy()
        if suspect.any():
            working[suspect] = np.interp(positions[suspect], positions[~suspect], values[~suspect])
        reference = smooth_signal(working, window, polyorder)

        difference = values - reference
        trusted = difference[~suspect & interior]
        noise = 1.4826 * np.median(np.abs(trusted - np.median(trusted)))
        # A floor of 0.1 percent of the typical value. Without it, data with almost no noise would get a limit
        # near zero, and tiny differences (for example at the two ends, where the smooth curve is less
        # accurate) would be called outliers.
        noise = max(noise, 1e-3 * float(np.median(np.abs(values))), 1e-9)

        limit = np.where(interior, threshold * noise, 3.0 * threshold * noise)
        found = np.abs(difference) > limit
        if np.array_equal(found, suspect):
            break
        suspect = found
    return suspect, reference


def fit_trend(time_s, values, order: int) -> np.ndarray:
    """A polynomial of the given order fitted to the values (order 0 is the mean, 1 a straight line)."""
    time_s = np.asarray(time_s, dtype=float)
    centred_time = time_s - time_s.mean()  # keeps the numbers small, which makes the fit accurate
    coefficients = np.polyfit(centred_time, np.asarray(values, dtype=float), order)
    return np.polyval(coefficients, centred_time)


def compute_spectrum(fluctuation, fps: float):
    """Amplitude spectrum of the fluctuation. Returns (frequency in Hz, amplitude in the unit of the signal).

    scipy.signal.periodogram gives the power spectrum. With scaling="spectrum" a sine wave of
    amplitude A has a peak power of A^2 / 2, so the amplitude is sqrt(2 x power). A Hann window
    is used, which avoids the large side lobes that a plain cut-off of the data would give.
    """
    fluctuation = np.asarray(fluctuation, dtype=float)
    fft_length = 1 << int(math.ceil(math.log2(len(fluctuation) * SPECTRUM_PADDING)))
    frequency, power = signal.periodogram(
        fluctuation, fs=fps, window="hann", nfft=fft_length, detrend=False, scaling="spectrum"
    )
    return frequency, np.sqrt(2.0 * power)


def describe_peak(frequency_hz: float, rotation_hz: float, resolution_hz: float, steady: bool):
    """Relate a peak to the rotation frequency. Returns (order, neutral note).

    order is the peak frequency divided by the rotation frequency (1 means once per revolution).
    The note only states what the number shows. It does not name a cause.
    """
    if not steady or rotation_hz <= 0:
        return None, "periodic speed variation (the speed is not steady, so no order is given)"

    order = frequency_hz / rotation_hz
    nearest = round(order)
    if nearest >= 1 and abs(frequency_hz - nearest * rotation_hz) <= resolution_hz:
        if nearest == 1:
            return order, "once per revolution (also possible: centre error, tilted camera)"
        return order, f"{nearest} times per revolution"
    return order, "periodic speed variation, source not identified"


def find_spectrum_peaks(
    frequency_hz,
    amplitude,
    duration_s: float,
    rotation_hz: float,
    mean_rpm: float,
    steady: bool,
    max_peaks: int,
    min_amplitude_rpm: float = 0.0,
):
    """Find the dominant peaks of an amplitude spectrum. Returns (list of peaks, noise floor, threshold).

    min_amplitude_rpm is the larger of the two limits of the method described near the top of this file:
    the noise of a single frame and the smallest listed percentage of the mean speed.

    Each peak is a dictionary with frequency_hz, period_s, amplitude_rpm, percent_of_mean,
    order and note. The list is sorted with the biggest peak first.
    """
    resolution_hz = 1.0 / duration_s
    in_band = frequency_hz >= MIN_CYCLES_IN_CLIP / duration_s
    if not in_band.any():
        return [], math.nan, math.nan

    band_frequency = frequency_hz[in_band]
    band_amplitude = amplitude[in_band]
    noise_floor = float(np.median(band_amplitude))
    # The biggest peak anywhere (not counting frequency 0) sets the size of the side lobes.
    # This includes slow components below the lowest frequency that is looked at.
    biggest = float(amplitude[1:].max())
    threshold = max(PEAK_NOISE_RATIO * noise_floor, PEAK_FRACTION_OF_MAX * biggest, min_amplitude_rpm)

    # find_peaks gives the positions of all local maxima that reach the threshold.
    positions, _ = signal.find_peaks(band_amplitude, height=threshold)
    positions = positions[np.argsort(band_amplitude[positions])[::-1]][:max_peaks]

    peaks = []
    for position in positions:
        peak_frequency = float(band_frequency[position])
        peak_amplitude = float(band_amplitude[position])
        order, note = describe_peak(peak_frequency, rotation_hz, resolution_hz, steady)
        peaks.append(
            {
                "frequency_hz": peak_frequency,
                "period_s": 1.0 / peak_frequency,
                "amplitude_rpm": peak_amplitude,
                "percent_of_mean": 100.0 * peak_amplitude / mean_rpm if mean_rpm > 0 else math.nan,
                "order": order,
                "note": note,
            }
        )
    return peaks, noise_floor, threshold


def analyse_speed_signal(table: pd.DataFrame, settings: SignalSettings = None):
    """Clean, smooth and analyse the speed in a table that has the speed columns (see kinematics.py).

    Returns (new table, result dictionary). The new table has these extra columns, which are
    empty outside the analysed stretch:
      speed_outlier                 True for frames whose speed was replaced as an outlier
      speed_interpolated            True for lost frames that were filled in (not measurements)
      rpm_clean                     speed after the outlier and gap handling
      rpm_smooth                    Savitzky-Golay smoothed speed (empty in the first and last half window)
      rpm_trend                     the fitted trend
      rpm_fluctuation               rpm_clean minus rpm_trend
      angular_acceleration_rad_s2   derivative of the speed, from the smoothing polynomial (also empty at the ends)
    The result dictionary holds the numbers, the spectrum and the list of peaks.
    """
    settings = settings or SignalSettings()
    validate_signal_settings(settings)
    for name in ("time_s", "rpm"):
        if name not in table.columns:
            raise AnalysisError("The table has no speed values yet. Call add_speed_columns first.")

    table = table.copy()
    count = len(table)
    time_all = table["time_s"].to_numpy(dtype=float)
    rpm_all = table["rpm"].to_numpy(dtype=float)

    first, last = find_analysis_stretch(~np.isnan(rpm_all), settings.max_gap_frames)
    stretch = slice(first, last + 1)
    frames_in_stretch = last - first + 1
    if frames_in_stretch < MIN_STRETCH_FRAMES:
        raise AnalysisError(
            f"Only {frames_in_stretch} frames in a row have a speed (at least {MIN_STRETCH_FRAMES} are needed for "
            f"smoothing and a frequency spectrum). Lost frames break a stretch when more than "
            f"{settings.max_gap_frames} are missing in a row. Improve the tracking, or record a longer video."
        )

    time_s = time_all[stretch]
    step = np.diff(time_s)
    if np.abs(step - np.median(step)).max() > 0.01 * np.median(step):
        raise AnalysisError("The frame times are not evenly spaced, so the spectrum cannot be calculated.")
    fps = 1.0 / float(np.median(step))
    duration_s = float(time_s[-1] - time_s[0] + 1.0 / fps)  # each frame covers 1 / fps of time

    # 1. fill short gaps
    filled, interpolated = fill_short_gaps(time_s, rpm_all[stretch])

    # 2. outliers
    window = smoothing_window_samples(settings.smooth_window_s, fps, frames_in_stretch, settings.smooth_polyorder)
    if settings.remove_outliers:
        is_outlier, reference = find_outliers(filled, window, settings.smooth_polyorder, settings.outlier_threshold)
        is_outlier &= ~interpolated  # filled-in frames are not measurements, so they cannot be measurement errors
        clean = np.where(is_outlier, reference, filled)
    else:
        is_outlier = np.zeros(frames_in_stretch, dtype=bool)
        clean = filled

    # 3. smoothing and acceleration. The first and last half window are left empty (NaN).
    half = window // 2
    if frames_in_stretch - 2 * half < 10:
        raise AnalysisError(
            f"The smoothing window ({window} frames) is too long for {frames_in_stretch} frames. "
            "Use a shorter --smooth-window or a longer video."
        )
    smooth = smooth_signal(clean, window, settings.smooth_polyorder)
    acceleration_rpm_s = signal.savgol_filter(
        clean, window, settings.smooth_polyorder, deriv=1, delta=1.0 / fps, mode="interp"
    )
    acceleration = acceleration_rpm_s * 2.0 * math.pi / 60.0  # rpm per second to rad/s^2
    for values in (smooth, acceleration):
        values[:half] = np.nan
        values[frames_in_stretch - half :] = np.nan

    # 4. trend and fluctuation
    trend = fit_trend(time_s, clean, settings.trend_order)
    fluctuation = clean - trend
    smooth_fluctuation = smooth - trend

    mean_rpm = abs(float(clean.mean()))
    trend_change = float(trend[-1] - trend[0])
    steady = mean_rpm > 0 and abs(trend_change) <= STEADY_LIMIT * mean_rpm
    peak_to_peak = float(np.nanmax(smooth_fluctuation) - np.nanmin(smooth_fluctuation))

    # 5. spectrum
    frequency_hz, amplitude_rpm = compute_spectrum(fluctuation, fps)
    rotation_hz = mean_rpm / 60.0
    noise_rms = estimate_noise(clean, fps)
    min_amplitude = max(noise_rms, settings.min_peak_percent / 100.0 * mean_rpm)
    peaks, noise_floor, threshold = find_spectrum_peaks(
        frequency_hz, amplitude_rpm, duration_s, rotation_hz, mean_rpm, steady, settings.max_peaks, min_amplitude
    )

    # new columns, empty outside the stretch
    def full_column(values, fill=np.nan, dtype=float):
        column = np.full(count, fill, dtype=dtype)
        column[stretch] = values
        return column

    table["speed_outlier"] = full_column(is_outlier, False, bool)
    table["speed_interpolated"] = full_column(interpolated, False, bool)
    table["rpm_clean"] = full_column(clean)
    table["rpm_smooth"] = full_column(smooth)
    table["rpm_trend"] = full_column(trend)
    table["rpm_fluctuation"] = full_column(fluctuation)
    table["angular_acceleration_rad_s2"] = full_column(acceleration)

    frame_numbers = table["frame"].to_numpy()
    result = {
        "settings": settings,
        "fps": fps,
        "first_frame": int(frame_numbers[first]),
        "last_frame": int(frame_numbers[last]),
        "frames": frames_in_stretch,
        "duration_s": duration_s,
        "interpolated_frames": int(interpolated.sum()),
        "outliers": int(is_outlier.sum()),
        "outlier_frames": [int(frame) for frame in frame_numbers[stretch][is_outlier]],
        "window_samples": window,
        "window_s": window / fps,
        "mean_rpm": mean_rpm,
        "trend_change_rpm": trend_change,
        "steady": steady,
        "noise_rms_rpm": noise_rms,
        "fluctuation_peak_to_peak_rpm": peak_to_peak,
        "fluctuation_rms_rpm": float(np.sqrt(np.nanmean(smooth_fluctuation**2))),
        "fluctuation_coefficient_percent": 100.0 * peak_to_peak / mean_rpm if mean_rpm > 0 else math.nan,
        "acceleration_max_abs_rad_s2": float(np.nanmax(np.abs(acceleration))),
        "acceleration_mean_rad_s2": float(np.nanmean(acceleration)),
        "frequency_resolution_hz": 1.0 / duration_s,
        "nyquist_hz": fps / 2.0,
        "rotation_frequency_hz": rotation_hz,
        "spectrum_frequency_hz": frequency_hz,
        "spectrum_amplitude_rpm": amplitude_rpm,
        "noise_floor_rpm": noise_floor,
        "peak_threshold_rpm": threshold,
        "peaks": peaks,
    }
    return table, result
