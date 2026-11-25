# sample_data

Put test videos in this folder.

- Videos recorded with a phone: name them like `disc_60fps_handspin_01.mp4`. The pattern is object, frame rate, what was done, trial number. Use only letters, numbers and underscores (no spaces).
- Videos made by `make_test_video.py` start with `synthetic_`. These are simulated, not real measurements, and results from them must never be presented as experimental results. `run_analysis.py` marks the plots and the printed summary of a video whose name starts with `synthetic` as simulated, so keep that start of the name for simulated videos and do not use it for real ones.
- Large video files are ignored by Git (see `.gitignore`). A small video that should be uploaded to GitHub as an example can be named `example_<something>.mp4`.
- Do not send phone videos through WhatsApp or similar apps before using them here. They are compressed again and the frame rate or resolution can change. Copy the original file with a USB cable or download the original from cloud storage.
