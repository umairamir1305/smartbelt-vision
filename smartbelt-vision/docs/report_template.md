# Report template

An engineering report for the project. Write each section in plain English. The lines in italics say what goes in. Delete them when the section is written. Figures and tables come from the program: the HTML report and the plots from the app, or the files in `outputs/`.

---

# SmartBelt Vision: video-based analysis of rotating pulley motion and belt slip

Author: Umair
Course, supervisor, date:

## Abstract

*150 words. The aim, the method in one sentence, the main numbers (speed, ratio, speed loss, slip, validation errors), and the main limit.*

## 1. Introduction

*Belt drives and why slip and speed loss matter. What is normally used to measure them. The question: how far can a phone video and a computer go. The aim and 3 to 4 objectives.*

## 2. Theory

*Short. Give the equations that the results use and refer to `docs/equations.md` for the rest.*

- 2.1 Marker angle and speed: angle from atan2, angular velocity, rpm
- 2.2 Speed ratio, speed loss and surface speed of a belt drive
- 2.3 Belt speed and slip
- 2.4 Frequency analysis of the speed
- 2.5 Belt friction (Euler-Eytelwein) and power, only if the calculators were used with measured tensions and torques

## 3. Method

### 3.1 Test rig

*Pulleys, belt, drive, load. A table of the effective radii and the marker distances in mm, and how they were measured.*

| Item | Driver | Driven |
|---|---|---|
| Effective radius (mm) | | |
| Marker distance from the centre (mm) | | |
| Marker colour | | |

### 3.2 Recording

*Phone, frame rate, resolution, distance, lighting, camera support, exposure and focus. A list of the recordings (see `docs/real_recording_protocol.md`).*

### 3.3 Processing

*The pipeline in `assets/pipeline.png`: tracking, angle, speed, signal analysis, belt drive, slip. The settings that were used: copy the settings table of the HTML report.*

## 4. Results

*Use the plots from the app or from `outputs/`. One figure for one point, with a caption that says what it shows.*

### 4.1 Tracking quality

*Frames found, quality counts, the longest gap. Figure: marker path and tracking quality in every frame.*

### 4.2 Speed

*Mean speed and range of each pulley, speed variation, figure of speed against time.*

| Recording | Driver rpm | Driven rpm | Variation (percent) |
|---|---|---|---|
| | | | |

### 4.3 Frequency content

*Spectrum figure and the table of dominant frequencies. Say what the numbers are. Do not name a cause.*

### 4.4 Belt drive: speed ratio and speed loss

| Recording | Measured ratio | Expected ratio | Speed loss (percent) |
|---|---|---|---|
| | | | |

### 4.5 Belt speed and slip

*No load against load first. Then the values. State the scale and where it came from, and the calibration factor.*

| Recording | Belt speed (m/s) | Slip on driver (percent) | Slip on driven (percent) |
|---|---|---|---|
| | | | |

## 5. Validation

### 5.1 Simulated videos

*Summarise `docs/validation.md`: how many checks, the tolerances, the largest errors, the wrong frame rate example and the known limit of a long gap. Figure: `assets/simulated_validation_summary.png`.*

### 5.2 Real recordings

*Counted turns against the measured turns. Time base check with a stopwatch. The agreement between 30 fps and 60 fps. Use the error measures in `docs/equations.md`, section 9.*

| Check | Measured | Reference | Error | Tolerance | Result |
|---|---|---|---|---|---|
| | | | | | |

## 6. Discussion

*What the numbers mean. The sources of error one by one: frame rate and frame timing, marker detection, rotation centre, perspective, scale, effective radii, belt creep, compression. What the change between no load and load shows.*

## 7. Conclusions

*3 to 5 bullets that answer the aim.*

## 8. Future work

*What would improve it: more recordings, a second camera angle, automatic centre detection, another type of belt.*

## References

- Budynas, R. G. and Nisbett, J. K., Shigley's Mechanical Engineering Design, McGraw-Hill.
- Savitzky, A. and Golay, M. J. E., Smoothing and differentiation of data by simplified least squares procedures, Analytical Chemistry 36(8), 1964, pp. 1627-1639.
- Bradski, G., The OpenCV Library, Dr. Dobb's Journal of Software Tools, 2000.
- Virtanen, P. et al., SciPy 1.0: fundamental algorithms for scientific computing in Python, Nature Methods 17, 2020, pp. 261-272.

## Appendices

- A. Equations: `docs/equations.md`
- B. Settings used: the settings table of the HTML report
- C. Full validation table: `docs/validation.md`
- D. Source code: the GitHub repository
