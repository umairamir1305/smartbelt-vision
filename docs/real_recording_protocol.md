# Recording a real belt drive

A plan for the first real recordings, using only a phone and a computer. Everything here is a checklist. The program does the rest.

## Before you record

| Item | What to do |
|---|---|
| Markers | A small matte marker of a saturated colour on each pulley, away from the centre (about 10 to 15 mm wide). A mark of a third colour on the outside of the belt. Green, orange and blue are the defaults |
| Effective radii | Measure from the centre of each pulley to the middle of the belt, in mm. Write them down |
| Marker distances | Measure from the centre of each pulley to the middle of its marker, in mm. This gives the scale for the belt speed |
| Camera | Fixed on a tripod or a stack of books, square on to the pulleys, no zoom, no stabilisation, exposure and focus locked. Both pulleys and a straight part of the belt in the picture |
| Light | Bright and even, no strong shadows or reflections on the markers |
| Frame rate | 60 fps if you can. Check the file with `python inspect_video.py <video>`. Some phones store slow motion with a playback frame rate that is lower than the capture rate: if the file says 30 fps but the phone recorded at 240 fps, every speed is 8 times too low |

## Recordings

| Name | What | Why |
|---|---|---|
| R1 | Turn the driver slowly by hand for exactly 10 turns, counting aloud. Film it | Checks the turn count: `--reference-turns 10` |
| R2 | Film a running stopwatch (a second phone or the computer screen) for 10 s | Checks the time base: the number of frames divided by the frame rate should match the stopwatch |
| R3 | Run the belt drive with no load for 30 s at a steady speed | Gives the reference ratio and the belt speed calibration factor |
| R4 | The same speed with a load (brake the driven pulley a little) for 30 s | The slip and the speed loss with load |
| R5 | Repeat R3 at 30 fps and at 60 fps | The two results should agree. If not, suspect the frame timing |

Write in a notebook: date, radii, marker distances, frame rate, resolution, distance to the camera, lighting, file names, and what the load was.

## Analysis

1. Check the markers on one frame: `python run_belt_drive.py <video> --preview ...` or the Check the picture section of the app.
2. Run R3. Read the measured speed ratio, then use it as the reference ratio (`--reference-ratio`) and use the printed calibration factor (`--belt-speed-factor`) for the other recordings.
3. Run R4 with the same settings and read the speed loss and the slip.
4. Check the frame timing of every file: `python run_validation.py --video <video>`. If it says that the frame spacing varies, do not trust the speeds of that file.
5. For R1 compare the turns: `python run_validation.py --table outputs/<name>_analysis.csv --reference-turns 10`.

## What to report

| Recording | Driver rpm | Driven rpm | Speed ratio | Speed loss | Slip on driver | Slip on driven | Tracking quality |
|---|---|---|---|---|---|---|---|
| R3 no load | | | | | | | |
| R4 load | | | | | | | |

Report the change between R3 and R4 first. It is more reliable than one value, because the errors of the scale and of the radii are the same in both.

## If something looks wrong

- Tracking quality Poor or Lost: the colour limits or the search region are wrong, or the light changed. Check the colour mask on several frames
- A speed ratio far from the radii: an effective radius is wrong, or the marker is not at the radius that you typed in
- A slip that is large at no load: the scale is wrong. Use the calibration factor
- A speed with a wobble once per turn: the rotation centre is off. Use the circle fit centre (`--use-fitted-centre` in `run_analysis.py`, `--use-fitted-centres` in `run_belt_drive.py`) and compare
- Speeds that are the same factor off in every recording: the frame rate is wrong
