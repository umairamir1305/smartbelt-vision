# GitHub and portfolio text

## GitHub About box

Measure pulley speed, belt speed and belt slip from a smartphone video. Python, OpenCV, Streamlit.

Topics: `computer-vision`, `opencv`, `streamlit`, `mechanical-engineering`, `belt-drive`, `rpm`, `signal-processing`, `python`

## Short description (pinned repository, CV line)

SmartBelt Vision tracks coloured markers in a phone video of a belt drive and calculates the rpm of each pulley, the frequency content of the speed, the speed ratio, the speed loss and the slip of the belt.

## Portfolio project description

**SmartBelt Vision: video-based analysis of rotating pulley motion and belt slip**

Belt drives lose a little speed to slip and creep, and measuring it normally needs a tachometer or an encoder. I wanted to see how far a phone video and a computer can go. SmartBelt Vision tracks a coloured marker on each pulley and a small mark on the belt, turns the positions into angle and speed, and compares the two pulleys.

What I built:

- Marker tracking with an HSV colour mask, a quality rating for every frame, and a circle fit of the rotation centre
- Speed in rpm for every frame, smoothing, trend removal and a frequency spectrum of the speed fluctuation
- Speed ratio, speed loss and surface speeds for a two-pulley belt drive
- Belt speed from a line fit of the belt mark on the straight spans, and the slip on the driver and on the driven pulley
- A Streamlit app, command line tools, a self-contained HTML report, and calculators for belt friction and power difference that use measured tension and torque

Results on simulated videos with known values:

- 40 of 40 validation checks within tolerance
- Mean speed from 30 to 480 rpm, at 30, 60 and 120 fps, with picture noise and lost frames: largest error 0.001 rpm
- Belt drive with known slip: speed loss within 0.003 points, slip within 0.08 points
- A 3 Hz speed wobble found at 3.003 Hz, amplitude 12.38 rpm against 12.57 rpm
- 511 automated tests

Tools: Python, OpenCV, NumPy, SciPy, Pandas, Matplotlib, Streamlit, pytest.

Links: GitHub repository, demo video, report.

## CV bullets

- Built a Python and OpenCV tool that measures pulley rpm, speed ratio, speed loss and belt slip from a smartphone video, with a Streamlit app and an HTML report
- Validated it on simulated videos with known values: 40 of 40 checks within tolerance, mean speed error at most 0.001 rpm, slip error at most 0.08 points
- Wrote 511 automated tests and a validation script that compares results with a tachometer reading or counted turns

## Questions you may get

- **What is it based on?** Colour marker tracking, an angle around the rotation centre, and a speed from central differences with the real frame times. The belt speed comes from a straight line fit of a mark on the belt
- **How accurate is it?** On simulated videos the numbers above. On a real belt drive it depends on the frame rate, the light, the effective radii and the scale. The plan for real recordings is in `docs/real_recording_protocol.md`
- **Why is slip hard?** It is a small difference between speeds that each have errors. 1 percent of error in the scale is about 1 percentage point of slip, so the change between no load and load is more reliable than one value
- **Can it find faults from the spectrum?** No. A peak says that the speed varies at that frequency. It does not say why
- **Does it measure friction or power loss?** No. The calculators use tension and torque that you measure yourself
- **What did you learn?** Frame rate and time base have to be checked first, a rotation centre error looks like a fluctuation once per turn, and a result needs a reference
