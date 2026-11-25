# Validation on simulated videos

Made by `python run_validation.py`. The videos are simulated, so the true value of every quantity is known.
A row with no limit only shows the size of an error (a frame rate that is typed wrong on purpose, and the turns counted across a long gap).

| Group | Test | Quantity | Measured | Reference | Error | Tolerance | Result |
|---|---|---|---:|---:|---:|---|---|
| Speed | 30 rpm, 60 fps | Mean speed (rpm) | 30.000 | 30.000 | +0.000 | 0.5 % | within |
| Speed | 30 rpm, 60 fps | Turns (turns) | 1.992 | 1.992 | +0.000 | 0.05 turns | within |
| Speed | 60 rpm, 60 fps | Mean speed (rpm) | 59.999 | 60.000 | -0.001 | 0.5 % | within |
| Speed | 60 rpm, 60 fps | Turns (turns) | 3.983 | 3.983 | -0.000 | 0.05 turns | within |
| Speed | 120 rpm, 60 fps | Mean speed (rpm) | 120.000 | 120.000 | -0.000 | 0.5 % | within |
| Speed | 120 rpm, 60 fps | Turns (turns) | 7.967 | 7.967 | -0.000 | 0.05 turns | within |
| Time axis | 120 rpm, 60 fps | Largest irregularity of the frame spacing (percent) | 0.000 | 0.000 | +0.000 | 1 percent | within |
| Time axis | 120 rpm, 60 fps | Frame rate from the frame times (fps) | 60.000 | 60.000 | +0.000 | 1 % | within |
| Wrong frame rate | 120 rpm video read as 30 fps | Mean speed (rpm) | 60.000 | 120.000 | -60.000 | none | no limit |
| Speed | 240 rpm, 60 fps | Mean speed (rpm) | 240.001 | 240.000 | +0.001 | 0.5 % | within |
| Speed | 240 rpm, 60 fps | Turns (turns) | 15.933 | 15.933 | +0.000 | 0.05 turns | within |
| Speed | 480 rpm, 60 fps | Mean speed (rpm) | 479.999 | 480.000 | -0.001 | 0.5 % | within |
| Speed | 480 rpm, 60 fps | Turns (turns) | 31.867 | 31.867 | -0.000 | 0.05 turns | within |
| Frame rate | 120 rpm, 30 fps | Mean speed (rpm) | 120.001 | 120.000 | +0.001 | 0.5 % | within |
| Frame rate | 120 rpm, 30 fps | Turns (turns) | 7.933 | 7.933 | +0.000 | 0.05 turns | within |
| Frame rate | 120 rpm, 120 fps | Mean speed (rpm) | 119.999 | 120.000 | -0.001 | 0.5 % | within |
| Frame rate | 120 rpm, 120 fps | Turns (turns) | 7.983 | 7.983 | -0.000 | 0.05 turns | within |
| Noise | 120 rpm, noise 6 | Mean speed (rpm) | 119.999 | 120.000 | -0.001 | 0.5 % | within |
| Noise | 120 rpm, noise 6 | Turns (turns) | 7.967 | 7.967 | -0.000 | 0.05 turns | within |
| Noise | 120 rpm, noise 12 | Mean speed (rpm) | 120.000 | 120.000 | -0.000 | 0.5 % | within |
| Noise | 120 rpm, noise 12 | Turns (turns) | 7.967 | 7.967 | -0.000 | 0.05 turns | within |
| Lost marker | 120 rpm, hidden 15 frames | Mean speed (rpm) | 120.001 | 120.000 | +0.001 | 0.5 % | within |
| Lost marker | 120 rpm, hidden 15 frames | Turns inside the stretches with a marker (turns) | 7.433 | 7.433 | +0.000 | 0.05 turns | within |
| Lost marker | 120 rpm, hidden 15 frames | Turns including the gap (known limit) (turns) | 6.967 | 7.967 | -1.000 | none | no limit |
| Frequency | 150 rpm, wobble 4 deg at 3 Hz | Dominant frequency (Hz) | 3.003 | 3.000 | +0.003 | 0.1 Hz | within |
| Frequency | 150 rpm, wobble 4 deg at 3 Hz | Fluctuation amplitude (rpm) | 12.378 | 12.566 | -0.188 | 5 % | within |
| Belt drive | ratio 2, driver slip 0 %, driven slip 0 % | Speed ratio (driver turns per driven turn) | 2.000 | 2.000 | +0.000 | 0.5 % | within |
| Belt drive | ratio 2, driver slip 0 %, driven slip 0 % | Speed loss (percentage points) | 0.003 | 0.000 | +0.003 | 0.3 percentage points | within |
| Belt drive | ratio 2, driver slip 0 %, driven slip 0 % | Slip on the driver (percentage points) | 0.075 | 0.000 | +0.075 | 0.3 percentage points | within |
| Belt drive | ratio 2, driver slip 0 %, driven slip 0 % | Slip on the driven pulley (percentage points) | -0.072 | 0.000 | -0.072 | 0.3 percentage points | within |
| Belt drive | ratio 2, driver slip 1 %, driven slip 2 % | Speed ratio (driver turns per driven turn) | 2.061 | 2.061 | +0.000 | 0.5 % | within |
| Belt drive | ratio 2, driver slip 1 %, driven slip 2 % | Speed loss (percentage points) | 2.980 | 2.980 | +0.000 | 0.3 percentage points | within |
| Belt drive | ratio 2, driver slip 1 %, driven slip 2 % | Slip on the driver (percentage points) | 1.067 | 1.000 | +0.067 | 0.3 percentage points | within |
| Belt drive | ratio 2, driver slip 1 %, driven slip 2 % | Slip on the driven pulley (percentage points) | 1.934 | 2.000 | -0.066 | 0.3 percentage points | within |
| Belt drive | ratio 2, driver slip 3 %, driven slip 0 % | Speed ratio (driver turns per driven turn) | 2.062 | 2.062 | -0.000 | 0.5 % | within |
| Belt drive | ratio 2, driver slip 3 %, driven slip 0 % | Speed loss (percentage points) | 2.999 | 3.000 | -0.001 | 0.3 percentage points | within |
| Belt drive | ratio 2, driver slip 3 %, driven slip 0 % | Slip on the driver (percentage points) | 3.075 | 3.000 | +0.075 | 0.3 percentage points | within |
| Belt drive | ratio 2, driver slip 3 %, driven slip 0 % | Slip on the driven pulley (percentage points) | -0.078 | 0.000 | -0.078 | 0.3 percentage points | within |
| Belt drive | ratio 2, driver slip 0 %, driven slip 3 % | Speed ratio (driver turns per driven turn) | 2.062 | 2.062 | -0.000 | 0.5 % | within |
| Belt drive | ratio 2, driver slip 0 %, driven slip 3 % | Speed loss (percentage points) | 2.999 | 3.000 | -0.001 | 0.3 percentage points | within |
| Belt drive | ratio 2, driver slip 0 %, driven slip 3 % | Slip on the driver (percentage points) | 0.068 | 0.000 | +0.068 | 0.3 percentage points | within |
| Belt drive | ratio 2, driver slip 0 %, driven slip 3 % | Slip on the driven pulley (percentage points) | 2.933 | 3.000 | -0.067 | 0.3 percentage points | within |

40 of 40 checks are within their tolerance.
