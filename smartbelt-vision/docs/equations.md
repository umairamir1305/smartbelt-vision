# Equations

The formulas that the program uses, and where they are in the code. Symbols: `x, y` marker position in pixels, `c` rotation centre, `t` time of a frame, `N` speed in rpm, `R` effective radius (from the centre of a pulley to the middle of the belt).

## 1. Marker angle (`src/kinematics.py`: `angle_from_position`, `add_angle_columns`)

$$\theta = \operatorname{atan2}(c_y - y,\; x - c_x)$$

The image y axis points down, so the y difference is turned around and counter-clockwise is positive. `np.unwrap` removes the jumps of 2 pi. If the marker is lost for so long that it could have turned more than 90 degrees, the angle starts a new segment, because the number of whole turns in the gap is no longer certain.

## 2. Angular velocity and rpm (`add_speed_columns`, `summarise_speed`)

$$\omega_i = \frac{\theta_{i+1} - \theta_{i-1}}{t_{i+1} - t_{i-1}}, \qquad N = \omega \cdot \frac{60}{2\pi}$$

This is `np.gradient` with the real frame times, done inside each segment. The mean speed is the total rotation inside the segments divided by the time of those segments. That is the same as counting turns and dividing by time, and it does not depend on the noise of single frames.

## 3. Rotation centre from the marker path (`fit_circle`, `fit_centre_from_track`)

A point on a circle satisfies $x^2 + y^2 + Dx + Ey + F = 0$, which is linear in D, E and F. One least squares solve gives the circle:

$$c = \left(-\tfrac{D}{2},\; -\tfrac{E}{2}\right), \qquad r = \sqrt{c_x^2 + c_y^2 - F}$$

The rms residual is the typical distance of the points from the circle. A centre that is wrong by a few pixels shows up as a false speed fluctuation once per turn.

## 4. Signal analysis (`src/signal_analysis.py`)

| Step | Rule | Function |
|---|---|---|
| Short gaps | filled by straight lines, up to 3 frames in a row | `fill_short_gaps` |
| Outliers | a value is replaced if it is more than 5 noise units from a Savitzky-Golay fit (3 times that in the end regions), repeated up to 4 times | `find_outliers` |
| Smoothing | Savitzky-Golay, window 0.2 s, polynomial order 2 | `smooth_signal` |
| Trend | least squares polynomial, order 1 by default | `fit_trend` |
| Fluctuation | smoothed speed minus trend | `analyse_speed_signal` |
| Coefficient | (max - min) / mean of the smoothed speed | `analyse_speed_signal` |
| Noise | rms difference between the speed and a fixed 0.2 s fit | `estimate_noise` |

## 5. Spectrum (`compute_spectrum`, `find_spectrum_peaks`)

`scipy.signal.periodogram` with a Hann window and zero padding to at least 8 times the length. With `scaling="spectrum"` a sine wave of amplitude A has a peak power of $A^2/2$, so

$$\text{amplitude} = \sqrt{2 \cdot \text{power}}$$

in rpm. The order of a peak is $f_\text{peak} / f_\text{rot}$ with $f_\text{rot} = N_\text{mean} / 60$. A peak is listed if it is above 5 times the median level, above 5 percent of the biggest peak, above the speed noise, and above 0.5 percent of the mean speed. The note on a peak only says that the speed varies periodically at that frequency. It does not name a cause.

## 6. Belt drive (`src/belt_drive.py`)

$$i = \frac{N_\text{driver}}{N_\text{driven}}\ (\text{turns over the same frames}), \qquad i_0 = \frac{R_\text{driven}}{R_\text{driver}}$$

$$L = \frac{v_\text{ideal} - v_\text{actual}}{v_\text{ideal}} \cdot 100, \qquad v_\text{ideal} = \frac{N_\text{driver}}{i_0}, \quad v_\text{actual} = N_\text{driven}$$

L is the speed loss in percent. It equals $(1 - i_0/i) \cdot 100$. A positive value means the driven pulley is slower than the radii say. The expected ratio $i_0$ can also be a reference ratio from a recording with no load, which removes the error of the radii.

Surface speed: $v = \omega R$.

Sensitivity of the expected ratio to a radius error: an error of 1 mm in R changes $i_0$ by $1/R$ (4.3 percent for a 23.1 mm driver radius, 2.2 percent for a 46.2 mm driven radius).

## 7. Belt speed and slip (`src/slip_analysis.py`)

Scale from the marker distance, averaged over the pulleys (a note appears if they differ by more than 3 percent):

$$s = \frac{r_\text{marker,mm}}{\tilde r_\text{marker,px}} \quad [\text{mm per pixel}]$$

where the tilde is the median tracked distance from the centre. Where the belt runs straight, the mark moves along a line. For every pass of at least 8 frames outside both pulley circles, a line is fitted by least squares, $x = x_0 + v_x t$ and $y = y_0 + v_y t$, and a pass with a residual above 2 px is dropped:

$$v_\text{belt} = s\,\sqrt{v_x^2 + v_y^2}$$

The belt speed is the mean over the passes, weighted by the number of frames in each pass, and the spread between passes (standard deviation divided by the mean) is reported. Then

$$s_\text{driver} = \frac{v_\text{driver} - v_\text{belt}}{v_\text{driver}}, \qquad s_\text{driven} = \frac{v_\text{belt} - v_\text{driven}}{v_\text{belt}}$$

with $v_\text{driver} = \omega_\text{driver} R_\text{driver}$ and $v_\text{driven} = \omega_\text{driven} R_\text{driven}$. The calibration factor $k = v_\text{driver} / v_\text{belt}$ of a recording with no load and no slip is applied to later recordings with `--belt-speed-factor`.

Slip is a small difference of speeds, so an error in the scale goes straight into it (simulated belt, true slips 1 and 2 percent):

| Scale typed in (true value 0.25 mm per pixel) | Slip on the driver | Slip on the driven pulley |
|---|---|---|
| 0.2475 (1 percent too small) | +2.06 percent | +0.94 percent |
| 0.2500 | +1.07 percent | +1.93 percent |
| 0.2525 (1 percent too big) | +0.09 percent | +2.90 percent |

## 8. Calculators that do not use the video (`belt_calculators.py`)

Euler-Eytelwein for a belt that is about to slip, and the wrap angle of an open belt (R_large, R_small, centre distance C):

$$\frac{T_1}{T_2} = e^{\mu \beta} \;\Rightarrow\; \mu = \frac{\ln(T_1/T_2)}{\beta}, \qquad \beta_\text{small} = \pi - 2\arcsin\frac{R_\text{large} - R_\text{small}}{C}, \quad \beta_\text{large} = \pi + 2\arcsin\frac{R_\text{large} - R_\text{small}}{C}$$

$$P = T\,\omega, \qquad \omega = \frac{2\pi N}{60}, \qquad \text{efficiency} = \frac{P_\text{out}}{P_\text{in}}$$

T1 and T2 are tensions and T is torque. They have to be measured by you, because the video cannot measure them.

## 9. Validation measures (`src/validation.py`)

| Measure | Definition |
|---|---|
| Error | measured - reference |
| Percent error | (measured - reference) / abs(reference) * 100 |
| MAE | mean of abs(measured - reference) over a series |
| RMSE | square root of the mean of (measured - reference)^2 |
| Bias | mean of (measured - reference) |
