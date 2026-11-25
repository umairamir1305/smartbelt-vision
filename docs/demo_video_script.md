# Demo video script (about 2 minutes)

`assets/demo_app_walkthrough.mp4` is a silent screen recording of the app with captions, made by `tools/record_demo_video.py`. This script is for a version with a voice. Record it with Xbox Game Bar (Windows key + G) or OBS, at 1280 x 800 or larger, and read the narration at a calm pace.

| Time | On the screen | Narration |
|---|---|---|
| 0:00 | The app, title and the first paragraph | This is SmartBelt Vision. A phone video of a belt drive goes in. Speed, speed ratio and slip come out. Everything runs on this computer |
| 0:10 | Choose the video | I use a simulated belt drive with a green marker on the small pulley, an orange one on the large pulley and a blue mark on the belt. The true slip is known, so I can check the result |
| 0:20 | Choose Two-pulley belt drive | The type of analysis is a two-pulley belt drive |
| 0:25 | Type the centres, search regions and sizes | For each pulley I type the rotation centre, the search region, the effective radius and the distance from the centre to the marker |
| 0:50 | The picture with the grid and circles | The picture shows the grid, the centres and the search regions, so I can see that they sit on the pulleys |
| 1:00 | Tick the mark on the belt | The blue mark on the belt gives the belt speed and the slip |
| 1:05 | Check the marker colours | The colour check shows which pixels count as the marker. Each one should be a single white blob |
| 1:20 | Press Run analysis | Now I run the analysis. It takes a few seconds |
| 1:30 | The summary and the numbers | The speed ratio is 2.06 against 2.00 expected. The speed loss is 2.98 percent. The slip is 1.07 percent on the driver and 1.93 percent on the driven pulley |
| 1:45 | Driver pulley, annotated frames | Each pulley has its own view with speed, spectrum and tracking quality. The annotated frames show what the program tracked |
| 2:00 | Downloads | The table, the plots and a report are available as downloads. These numbers are from a simulated video. The next step is real recordings |

Check before you record: a clean browser window, the app started with `streamlit run app.py`, the video made with `python make_test_belt_video.py --belt --driver-slip-percent 1 --slip-percent 2`.
