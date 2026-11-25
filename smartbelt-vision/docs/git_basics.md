# Git basics for this project

## One-time setup

```
git config --global user.name "Umair"
git config --global user.email "your email address"
```

Use the email of your GitHub account, or the private no-reply address that GitHub shows under Settings, Emails. Commits show this email.

## First upload

```
git init
git add .
git commit -m "SmartBelt Vision"
git branch -M main
git remote add origin https://github.com/umairamir1305/smartbelt-vision.git
git push -u origin main
```

Create the empty repository on GitHub first, without a README, a .gitignore or a license, because the project has them.

## Every day

```
git status                      what changed
git add -A                      stage everything
git commit -m "message"         save a snapshot
git push                        send it to GitHub
git log --oneline               the history
```

`.gitignore` keeps out the virtual environment, the results in `outputs/` and the videos in `sample_data/`. The pictures in `assets/` stay in, because the README uses them.

## Commit messages

One short line that says what changed: "Fix label drawing for OpenCV 5", "Add real recording R3 results", "Update README pictures". Do one thing per commit when you can.

## A history by phase (optional)

If you want the history to show how the project grew, unpack the phase zips in order into an empty repository folder and commit after each:

1. Phase 1: project setup, video reading, simulated disc video
2. Phase 2: HSV marker tracking and tracking quality
3. Phase 3: angle, unwrapping and circle fit of the rotation centre
4. Phase 4: angular velocity and rpm
5. Phase 5: signal analysis: smoothing, trend, spectrum
6. Phase 6: two-pulley belt drive: speed ratio and speed loss
7. Phase 7: belt mark, slip and calculators
8. Phase 8: Streamlit app and HTML report
9. Phase 9 and 10: validation, README pictures, docs, demo video (the files of `smartbelt-vision-github.zip`)

A single commit is fine too.

## When the README pictures change

```
python tools/make_readme_images.py
python tools/app_screenshots.py     (needs Playwright)
python tools/record_demo_video.py   (needs Playwright and ffmpeg)
```

## Releases

When a version is ready: `git tag v1.0` and `git push --tags`. On GitHub, Releases, Draft a new release, choose the tag.
