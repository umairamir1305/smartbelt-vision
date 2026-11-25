# Presentation outline (about 10 minutes, 12 slides)

Pictures are files in `assets/`. All of them are from simulated videos, so say so on the slide.

| # | Slide | Show | Say | Time |
|---|---|---|---|---|
| 1 | SmartBelt Vision | `simulated_tracking.gif` | One sentence: a phone video of a belt drive goes in, speed, speed ratio and slip come out. Hardware is a phone and a computer | 0:30 |
| 2 | The question | `simulated_annotated_frame.png` | Belt drives lose speed to slip and creep. Tachometers and encoders need hardware that a student lab often does not have. How much can a phone video tell? | 0:45 |
| 3 | Scope | a two column list | Measured: rpm in every frame, frequency content of the speed, speed ratio, speed loss, belt speed, slip. Not measured from video: friction coefficient, power loss (calculators that use your own tension and torque), causes of faults | 0:30 |
| 4 | Method | `pipeline.png` | Read the video, track coloured markers, turn positions into angles and speed, analyse the signal, compare the two pulleys, follow a mark on the belt | 1:00 |
| 5 | Tracking | `simulated_mask_check_driver.png`, `simulated_mask_check_belt_mark.png` | HSV colour mask, largest blob, centroid. Every frame gets a quality from size, distance and jump checks. The mask check shows the marker as one clean white blob | 1:00 |
| 6 | Angle and speed | `simulated_rpm.png` | atan2 around the rotation centre, unwrapped, central differences with the real frame times. The mean speed is total rotation divided by time, the same as counting turns | 1:00 |
| 7 | Signal analysis | `simulated_spectrum.png` | Smoothing, trend removal, spectrum with a Hann window. The simulated disc has a 3 Hz wobble and the spectrum finds 3.00 Hz. A peak says the speed varies at that frequency, not why | 0:45 |
| 8 | Belt drive | `simulated_belt_speeds.png` | Measured speed ratio against the ratio of the effective radii. Speed loss and surface speeds. Radius error goes straight into the expected ratio | 0:45 |
| 9 | Belt mark and slip | `simulated_belt_mark_speed.png`, `simulated_belt_slip.png` | Belt speed from a straight line fit on the straight spans. Slip on the driver and on the driven pulley. Slip is a small difference of speeds: 1 percent of scale error is about 1 point of slip | 1:00 |
| 10 | The app | `app_03_results_belt.png` or `demo_app_walkthrough.mp4` | Upload, check the markers on the frame, run, read the results, download the table, the plots and the report | 0:45 |
| 11 | Validation | `simulated_validation_summary.png` | 40 of 40 checks within tolerance on simulated videos. A wrong frame rate gives a wrong speed by the same factor. A long gap makes the turn count ambiguous | 1:00 |
| 12 | Limits and next steps | `docs/real_recording_protocol.md` | Next: real recordings with no load and with load, compared with counted turns and a stopwatch. Report the change between no load and load first | 0:30 |

## Backup slides

- Equations: `docs/equations.md` (angle, speed, circle fit, spectrum, ratio, slip)
- Scale sensitivity of the slip (table in `docs/equations.md`, section 7)
- Radius sensitivity of the expected ratio: 1 mm of radius error is 4.3 percent of expected ratio for a 23.1 mm radius
- How the tracking quality is decided (size, distance from the centre, jump between frames)
- Validation table: `docs/validation.md`

## Questions to expect

- Why colour markers and not feature tracking? Colour markers are simple, fast and easy to check by eye. The program shows the colour mask so a wrong setting is visible
- How accurate is it on a real machine? Simulated videos have known answers and the numbers on slide 11 are from them. Real recordings are the next step, with the plan in `docs/real_recording_protocol.md`
- What if the phone has a variable frame rate? `run_validation.py --video` checks the frame times of a file and says if the spacing varies
