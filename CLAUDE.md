# FunAutoScripter

Automatic funscript generator from video using classical optical flow (OpenCV Farneback).

Entry points: `funautoscripter.py` (CLI) and `funautoscripter_gui.py` (GUI). Both are thin wrappers — engine is in `src/`, GUI is in `src/gui/`.

```
src/
  __init__.py      # re-exports full public API (import funautoscripter as fl still works)
  flow.py          # step 1: ffmpeg decode, Farneback, cache
  roi.py           # step 3: region selection, axis
  signal.py        # steps 4-7: velocity, integration, normalisation, keyframes
  output.py        # write_funscript, plot
  debug.py         # --debug DIR videos
  chapters.py      # chapter management, auto-detection, folder helpers
  pipeline.py      # process(), build_parser(), main()
  gui/
    __init__.py
    app.py         # App class and main()
    config.py      # constants, PARAMS, THEMES, presets
    utils.py       # colour helpers, make_theme, paths, load_config
    frame_source.py  # FrameSource (cv2.VideoCapture wrapper)
```

## Project rules

- No AI. No neural networks, YOLO, PyTorch, ONNX or downloaded models. This is a hard requirement, not a preference.
- CPU only. Must run well without a GPU.
- Minimal dependencies: numpy, scipy, opencv, matplotlib (only for `--plot`) and system ffmpeg. GUI adds DearPyGui.
- The system is NixOS. Use `nix-shell` (see `shell.nix`), never global `pip install`.
- User-facing text (logs, help, new comments) must be in English.

## How to run

```
nix-shell
python funautoscripter.py video.mp4 --plot debug.png
```

`--plot debug.png` generates two files:
- `debug.png`: three signal graphs (velocity, filtered position, final funscript with keyframes), with scene cuts as red dashed lines. Used to evaluate result quality and verify cut detection.
- `debug.roi.png`: average motion map with the used regions highlighted in cyan. Used to choose `--region`.

## Deep debug (`--debug DIR`)

```
python funautoscripter.py video.mp4 --debug dbg
```

Writes to `dbg/` six videos (real image with overlays on top, sliding graphs below, current frame in the centre), plus two data files:
- `1_roi_decisao.mp4`: region candidates with score, previous region, switch threshold (`--switch`) and decision (switched or kept). Red line on graphs = switch.
- `2_magnitude.mp4`: magnitude per frame and the cell-by-cell window average (this is where candidates come from).
- `3_periodicidade.mp4`: periodicity of each candidate (colour) and autocorrelation curves with the chosen peak.
- `4_script.mp4`: the final funscript (bar and graph) over the normalised signal, with keyframes.
- `5_velocidade.mp4`: axis velocity (arrow on region) and funscript velocity against `--max-speed`.
- `6_integral.mp4`: raw, filtered and normalised integral, plus the raw spectrum with the bandpass response.
- `sinais.csv` (all signals per frame) and `decisoes.json` (each window, candidates, scores and decision).

Uses the flow cache, only decodes the video in colour again. Takes about 15 frames per second at 1280x630. Options: `--debug-height`, `--debug-window` and `--debug-videos`. With `--debug-videos roi,script` (or `1,4`) only those videos are generated and the others are not drawn at all: on a 24 s test video it took 48 s with all, 7.6 s with `roi` only, 9 s with `script` only and 1 s with `none` (only `sinais.csv` and `decisoes.json`). Accepted names: `roi`, `magnitude`, `periodicidade`, `script`, `velocidade`, `integral`. Text in the videos has no accents because the OpenCV font cannot render them.

The score of each window uses the magnitude normalised by the window's own maximum, so it is only comparable within the window. Hysteresis compares against the score saved when the current region was chosen, not the current score. Video 1 shows both.

## Folder and 3D/VR

```
python funautoscripter.py folder/                    # one funscript next to each video that doesn't have one yet
python funautoscripter.py folder/ --recursive --overwrite
python funautoscripter.py vr.mp4 --sbs              # side by side, left half only
python funautoscripter.py vr.mp4 --tb --eye right   # top and bottom, bottom half only
```

With a folder, a video that already has `<name>.funscript` next to it is skipped (`--overwrite` regenerates it). A video that fails, or produces no keyframes (static), does not stop the others or leave an empty script, and appears in the final summary. `-o` is not used with folders. With `--plot PNG` and `--debug DIR` each video gets its own name (`PNG_name.png`, `DIR/name/`).

`--sbs` and `--tb` are shortcuts for `--crop left|right|top|bottom` (`--eye` chooses the half), so the cache uses the same crop. Test: a side-by-side video where each half has a different motion. With `--sbs` the script matches the left eye (correlation 1.00), with `--sbs --eye right` the right (0.98), and without the flag it follows the other eye (0.00).

## Subtle motion (`--zoom`)

The analysis uses 192 px width. In a 1280 px video, a 1 px movement becomes 0.15 px in the analysis, close to Farneback noise. `--zoom X0:Y0:X1:Y1` (fractions of the frame, after `--crop`/`--sbs`/`--tb`) crops that zone in the original video before downscaling, so the 192 px now only cover the motion zone. `--region` becomes relative to the zone. The cache includes zoom, and old caches were recalculated once when `zoom` was added to the key.

Measured with a textured square oscillating vertically at 1.2 Hz in a 1280x720 video, low contrast (30 out of 255), noise and heavy compression (`crf 28`). Script correlation with real motion:

| Motion in video | 192 px | `--noise 0.01` | `--width 384` | `--width 768` | `--zoom` | `--zoom` + `--noise 0.01` |
|---|---|---|---|---|---|---|
| 2 px (0.3 px in analysis) | 0.86 | 0.91 | 0.92 | 0.89 | 0.92 | 0.92 |
| 1 px (0.15 px) | 0.29 | 0.46 | 0.55 | 0.61 | 0.73 | 0.77 |
| 0.5 px (0.075 px) | 0.07 | 0.10 | 0.00 | 0.04 | 0.13 | 0.17 |

Zoom (3.7x, zone chosen manually) beats increasing the width and costs the same as normal analysis, while `--width 768` costs about 16x more. Lowering `--noise` helps a little in all cases, but the default was not changed because reference results from the test videos have not been reviewed. Below about 0.1 px in analysis nothing worked. With strong texture and little noise (full contrast, `crf 10`) even 1 px gave 0.99 with just 192 px, so the limit depends heavily on video quality.

Limitation: the zone must be provided manually. Automatic region detection uses magnitude, which at 192 px is already noise for subtle motion, so it cannot be used to choose the zoom zone on its own.

## Graphical interface (`funautoscripter_gui.py`)

```
nix-shell
python funautoscripter_gui.py [video or folder]
```

Separate optional file, using DearPyGui. `funautoscripter.py` has no new dependencies. The GUI imports the engine (`import funautoscripter as fl`) and uses `fl.compute(a, dx, dy, mag)`, which returns `{v,p,pos,pts,windows,rois,grid}`, and `fl.build_parser()`. `extract_flow` and `load_or_extract` accept `progress(n, total)`.

The flow (step 1) runs once and is cached. Everything else recalculates in milliseconds, so parameters for steps 2 to 7 are sliders with immediate effect. Step 1 parameters (crop, `--zoom`, width, cell, fps) do not recalculate on their own: there is a single "Recalculate flow" button at the top, which turns orange with a warning below saying what changed, when the flow is stale. Crop and zoom immediately show a video preview (no overlays), to choose the zone before recalculating. While stale, signals continue using the old flow parameters (`worker_sig`). At the bottom there are five linked timeline rows (ROI, velocity, integral, script, script velocity) with a draggable cursor. At the top, the video with overlays (region, candidates, heatmap, magnitude, periodicity, axis) and the current window inspector with candidates and autocorrelation.

Video selection: in "region" mode, draw a rectangle and "Apply selection" fixes `--region`. In "zoom" mode, `--zoom` is applied and flow is recalculated. "Clear" returns to automatic. The rectangle is drawn over the image and handled manually (drag to create, inside to move, edges to resize), because DearPyGui's `drag_rect` did not appear on this plot. Shortcuts: space (play/pause), arrows (one frame). With a folder the video list appears (`[ok]` already has a script): clicking the name opens the video and the checkbox marks it for batch. "All", "None" and "No script" select in bulk, and "Process selected" generates scripts in queue with current parameters, using each video's cache if available. By default skips videos that already have a script ("redo" changes this) and does not use the open video's zoom or region ("apply current zoom and region to all" changes this). A video that fails or produces no keyframes does not stop the others and appears in the summary.

Chapters (optional, collapsed section in the parameters panel): for scenes that need different zooms in the same video. "New chapter at cursor" splits the chapter where the cursor is (at least 1 s on each side) and the new one inherits the zoom. Choosing zoom or region applies to the chapter where the cursor is, and the selection activates it. Each chapter has its own flow, region, cache (`<video>.funscript.chN.flowcache.npz`) and its own timeline result, and the cursor outside the active chapter shows the video with that chapter's zoom, without overlays. "Calculate flow for all chapters" generates missing ones in queue (cancellable) and "Save" merges all keyframes in time, with the speed limit applied at the joins too. The chapter list, with each one's zoom and region, is stored in `<video>.chapters.json` and is restored when reopening the video. In the engine, `--chapters auto|FILE|TEXT` uses the same file (`auto`) or a text `T@ZOOM[@REGION];...` with `-` for nothing, e.g. `--chapters "0@-;42.5@0.2:0.3:0.7:0.9"`, also in folders, and the GUI batch uses each video's `.chapters.json` if it exists. With chapters, `--plot` and `--debug` are not generated. Each chapter is analysed independently (like separate scenes), so normalisation restarts in each one and amplitude is not comparable across chapters.

Nothing is calculated automatically: opening a video only reads the cache if it exists and matches current settings, otherwise a warning appears and the "Recalculate flow" button turns orange. The "Cancel" button (or Esc) stops flow or batch mid-way, kills ffmpeg and leaves everything as it was (flow is only cached when complete, so a cancellation loses that video's work). In the engine, `fl.Cancelled` raised by `progress()` interrupts `extract_flow`.

Settings (parameters, crop, overlays, speed, last video or folder) are stored in `~/.config/funautoscripter/gui.json` and restored on open (without an argument the GUI reopens the last path). Region and zoom do not persist, they are per-video. `FUNAUTOSCRIPTER_CONFIG` changes the file, used in tests to avoid touching the real one.

0 and 100 indicators: in the video, velocity vectors (yellow) and funscript position vectors (magenta) have both ends marked, 100 as a filled circle and 0 as an empty circle, with text "vel: towards 100" or "vel: towards 0". The velocity arrow always points in the real direction of motion, and 100 is on the axis side, or the opposite when `--invert` is on. In the timelines, the "Velocity" and "Integral" axes state which end positive signal points to.

Presets (parameters panel, above chapters): selecting one from the menu loads it, "Save" stores current values with the typed name and "Delete" removes one of yours. Stored in `presets.json` next to `gui.json` (so `FUNAUTOSCRIPTER_CONFIG` also changes them). Only parameters that already recalculate are included (nothing from flow) and never "invert". On load, starts from defaults and applies the differences. Built-in presets in `BUILTIN_PRESETS`: Default, Light (smooth, fewer strokes, smaller amplitude and speed), Heavy (full stroke, open light strokes, more speed), Precise (stable region, low noise, fine detail) and Enhanced (high-frequency boost and renormalisation). Values are estimates, not tuned on real video or validated against synthetic tests.

Headless testing: `xvfb-run -a -s "-screen 0 1700x1050x24" python funautoscripter_gui.py video.mp4 --captura out.png --tempo 3`. DearPyGui requires a DISPLAY, without one it segfaults.

Limitations: never tested on a real screen or GPU, only under Xvfb. Long videos take time to compute flow and OpenCV seek is slow with long GOPs. Does not compare against a reference script.

Enhancement sliders (group "Enhancement", none recalculate flow, neutral defaults): `--smooth` (smooth the signal before scaling, in seconds), `--boost` (opens light strokes without touching extremes, exponent 1/boost around centre), `--gain` (amplitude around centre) and `--offset` (shifts the centre). Applied after step 6 (`enhance`) and before `--invert`. With default values the output is byte-for-byte identical to before.

High-frequency enhancement (its own group in the GUI, also no flow recalculation, all 0 by default): for small fast strokes hidden inside a larger slow oscillation. A second pass after normalisation (`post_enhance`): `--hp2` (Hz) cuts low frequencies from the already-scaled funscript, `--hfboost` boosts what is above `--hf-freq` and `--renorm` (seconds, from 0.5) renormalises and amplifies with a short window, with floor `--renorm-floor` (lower amplifies weak segments more). Synthetic test, slow oscillation of 40 plus 2 Hz strokes with amplitude 4 between 20 and 40 s: correlation of the script with the strokes went from 0.09 to 0.99 with `--hp2 0.8 --renorm 3 --renorm-floor 0.1`, and with the slow oscillation from 1.0 to 0.0. Warning: also amplifies noise in weak segments, so raise `--renorm-floor` if false strokes appear.

Interface notes from the last revision:

- Direction per chapter: each chapter has the `invert` flag (checkbox "invert this chapter", field in `.chapters.json`, fourth field `i` in `--chapters` text). Effective = global invert XOR chapter invert. Flow is automatic and consistent within the chapter; human review at the end is just whether to invert or not.
- Integral: axes use `range_fit`, so scale follows the visible window. The raw integral (no bandpass) has its own axis on the right, toggled by "show raw integral".
- Themes: Dark, Light, Night blue, High contrast and Amber, derived from background/text/highlight in `THEMES`. The choice is saved in `gui.json`.
- Dockable windows: each module is a DearPyGui window with docking (Windows menu to show/hide, "reset layout" clears the layout). Layout is stored in `layout.ini` next to `gui.json` (changes with `FUNAUTOSCRIPTER_CONFIG`). The ini is indexed by window label (`use_internal_label=False`). With `init_file` DPG ignores position and size given at creation, so on first run `apply_default_layout` positions windows after 4 frames.
- Timeline overview: strip with coloured velocity (blue slow to red fast), position envelope, chapter bands and red lines where direction inverts. Click or drag to navigate, click a band to activate that chapter. Other chapters use their flow caches and are only recalculated when the GUI has been idle for 0.8 s.

Automatic chapters (`detect_chapters`, on by default):

- `--chapters auto` (default) uses `<video>.chapters.json` if it exists, otherwise detects chapters from the whole video flow without recalculating it. `detect` always detects and rewrites the file (only if it was saved by a detection, marked with `"auto": true`, to never overwrite manual work), `none` disables. Works equally for the command line, folder (`run_folder`) and GUI batch, all of which call `process()`.
- Splits on hard cuts (spike of average motion against local median: `--cut-ratio`, `--cut-min`), soft transitions (`find_soft`: fades, dissolves, slowly changing scenes; compares the map of where motion occurs in the `--chap-soft-span` seconds before and after each instant, requires motion and stability on both sides, and `--chap-soft` 0 to 1 is the sensitivity), motion axis change (`--chap-angle`) and zone change (`--chap-shift`). `--chap-sens` (0 to 1, 0.5 = current settings) is the overall sensitivity: multiplies axis, zone, persistence, minimum size and cut thresholds by 2^(1-2s), and adds to soft transition sensitivity. Zone only catches large changes (more than 0.3 of the frame); soft transitions also catch small ones as long as the motion map changes. A change only counts if it lasts `--chap-persist` (8 s by default, so a pause with a distractor does not split the chapter) and chapters shorter than `--chap-min` are merged. A hard cut always counts (minimum 2 s). `--chap-by cuts,angle,zone` selects what is used.
- Each chapter receives its `axis_ref` in screen degrees (0 = right, 90 = down), calculated from the region it will use (`chapter_axis`, where each window votes by periodicity so a passing distractor has little weight). `--axis-ref` and the chapter's `axis_ref` field also accept an angle. In `--chapters` text it is the 5th field: `T@ZOOM@REGION@i@ANGLE`.
- Chapters inherit zoom and region from the video, so they do not need their own flow: `flow_for` slices the whole-video cache (`flow_available` reports whether it exists). Only a chapter with a different zoom calculates its own (`.chN.flowcache.npz`). The first frame of each slice is set to zero (cut flow is meaningless).
- `--roi-aspect` (width/height) gives the automatic region a different aspect ratio from the video, with the same area as `--roi` (`roi_size`). 0 inherits the video's aspect ratio.
- Limitation: a false detection resets the automatic region and hysteresis in the next chapter. In videos with periodic distractors, raise `--chap-persist` or use `--chapters none`.

GUI: "Suggest chapters" button (and automatic detection when opening a video without `.chapters.json`), "Automatic chapters" section with settings, "Chapter direction" (combo, angle and "Draw direction on video", drag the arrow), compass in the video corner showing the active axis-ref, and the overview with colour modes velocity, position, motion, periodicity, confidence, amplitude, rhythm and direction. Left and right arrows move one frame (`frame_step`; the main loop is called `step`, do not reuse the name).

GUI performance (do not regress): `probe` and `cache_valid` cache their result in memory (each ffprobe call took tens of ms and a chapter change called them per chapter on the main thread: 3 s with 30 chapters, now 2 to 9 ms). The overview of other chapters runs in a thread, one chapter at a time, only redoes what changed (`ov_memo`) and quits as soon as the user interacts, without counting as "busy" (`working`). When leaving a chapter, its series appears immediately in the overview. The axis-ref and angle sliders do not calculate on drag: there is an "Apply direction" button, and a "recalculate on drag" checkbox (off; sliders only mark pending changes and the "Apply changes" button calculates).

Flow calculation speed: `--jobs N` (0 = auto, up to 6) splits the video into chunks of at least 20 s, calculated in threads (OpenCV releases the GIL, so no processes or `spawn`). Each chunk except the first starts one frame early and discards that frame, and the result is identical to sequential processing (difference 0.0 in a 120 s test). With 2 cores, 172 fps became 300 fps. Cancelling stops all chunks and kills the ffmpeg processes. `--hwaccel` uses `-hwaccel auto` during decoding (falls back to CPU if unavailable) and `--opencl` uses `cv2.UMat` when OpenCV has a usable OpenCL GPU (falls back to CPU and says why if not); the OpenCL path was only tested on a CPU device (pocl), never on a real GPU, and produces flow differing by up to 0.4 px from the CPU. With OpenCL enabled, parallel chunks are disabled (there is only one GPU). `python funautoscripter.py --accel-test` (in the GUI, Help, Test acceleration) reports what OpenCV finds, what device it is and how each path performs; `fl.BACKEND` and the GUI status show what is being calculated and at how many fps. On NixOS, `nixpkgs.opencv4` already ships with OpenCL (loads `ocl-icd` at runtime), so `shell.nix` only adds `ocl-icd`, `clinfo` and `OCL_ICD_VENDORS`; GPU drivers come from `hardware.graphics.extraPackages` in the system config, not the shell. To go faster: `--fps 15` and `--width 128` (less detail).

Parallel batch: `--folder-jobs N` (in the GUI, "parallel videos in batch") processes N videos simultaneously with `process_many` (threads; returns `(status, detail)` per video and accepts cancellation). Cores are divided across videos (`cpu // N` chunks each, `FIXED_THREADS` prevents `_extract_parallel` from touching `cv2.setNumThreads`, which is global) and the `_FULL` memo is per-thread. Scripts come out byte-for-byte identical to queue mode. With all cores already busy (one video with `--jobs` auto) total time barely changes (2 cores: 24 s vs 22.6 s for 4 videos of 40 s each); the expected gain is for short videos and with OpenCL (2 or 3 videos on the same GPU), not yet measured on a real GPU. Resume: each video saves its flow cache when it finishes and videos that already have a script are skipped, so an interrupted batch restarts from where it left off (only losing the video that was in progress).

Apply settings without recalculating flow (button in the batch panel): `bulk_run(..., cache_only=True)` sets `b.cache_only` and `extract_flow` raises `fl.NoFlow` instead of calculating, so only marked videos that have a valid cache for the current width, cell and fps (and their chapters with their own zoom) are reprocessed. Also reprocesses those that already have a script; those without a cache are left untouched and appear in the summary ("no flow cache"). Test: 4 videos, 2 without cache, with `gain` and `smooth` changed: 2 scripts reprocessed in 0.1 s, the other 2 unchanged.

Interface: the parameter list marks in orange (and with `*`) what differs from the reference preset (`preset_ref`, the last one loaded or saved, Default by default) and lists which ones; "Reset to preset" returns to it. Each element has a tooltip (`TIPS` for parameters, `TIPS_UI` for the rest, applied in `apply_tips`). Shortcuts in `SHORTCUTS`/`key()` (ignored while typing in a text field), listed in Help, Keyboard shortcuts. When adding a parameter or button, add a tooltip.

## Pipeline (in order)

1. `extract_flow`: ffmpeg decodes to small greyscale (192 px, 30 fps), Farneback between consecutive frames, average per cell (12 px) for dx, dy and magnitude. Also stores the average frame difference (`diff`).
2. `detect_cuts`: a cut is a spike in average difference above `--cut-thresh` (absolute), or above `--cut-min` and `--cut-ratio` times the recent normal level. The relative test catches subtle cuts with similar brightness before and after, which the absolute test misses. Each scene is processed independently in everything that follows.
3. `choose_rois`: within each scene, every `--roi-window` seconds selects the cell block with the most motion, with hysteresis (`--switch`). Windows never cross a cut, the first window of a scene starts at the cut and hysteresis resets to zero. The region is decided using the motion of the entire window (processing is offline), not just what has passed. With `--region` the region is fixed.
4. `velocity`: in the region, projects motion onto the principal axis (PCA, or vertical with `--axis y`), weighted by magnitude squared. PCA only gives the axis line; the direction (which end is 100) is chosen with `--axis-ref` (default `chapter`: the reference direction is the dominant axis of the entire chapter, calculated by `chapter_axis`, oriented upward or, if horizontal, to the left; also available: `down`, `up`, `left`, `right`): each window makes positive the direction closest to that screen direction, so an error does not propagate. Only when the axis is nearly perpendicular (`--axis-margin`, 0.35) does the direction follow the previous window. `--axis-ref auto` is the old behaviour, continuity only, which on a synthetic test with the axis rotating window by window got 75.5% of windows correct vs 100% with `down`. The GUI marks on the velocity timeline windows where direction inverted relative to the previous one (`axis_flip` in `dbg`), and `axis_ref` is not included in presets. For horizontal motion use `--axis-ref left` or `right`. Velocity on cut frames is set to zero.
5. `to_position`: per scene, integrates velocity and applies bandpass (`--highpass`, `--lowpass`). Uses `padtype="constant"` in the filter, which keeps scene edges clean (the default padding left visible drift in the first and last seconds). Scenes shorter than 0.5 s are left flat.
6. `normalise`: per scene, scales to 0 to 100 using percentiles in a rolling window (`--norm-window`) that stops at scene boundaries, with a floor (`--floor`) to avoid amplifying noise during pauses.
7. `keyframes`: peaks and valleys with `find_peaks` (`--min-amp`, `--min-interval`), alternating, plus one keyframe on each side of each cut, with speed limit (`--max-speed`).

## Cache

Step 1 is the slow one. The result is stored in `<output>.flowcache.npz`. Changing parameters for steps 2 to 7, including cut detection, does not recalculate it. If you add a parameter that affects step 1, add it to `cache_key()`. If you change what is stored in the cache, increment `CACHE_VERSION`. Otherwise the cache silently becomes stale.

## Testing

There are no automated tests. To verify the algorithm manually, run the pipeline on a real video and inspect the `--plot` output and the `--debug DIR` videos. Reference correlations when the algorithm is working correctly:

- No cuts: 0.99 on large strokes, 0.96–0.97 on small fast ones.
- With cuts: per-scene correlations of 0.989, 0.982, 0.978 and 0.979. Must detect 3 cuts (at 9.3 s, 21.7 s and 30.4 s).

Any change to the algorithm should be verified against real video to ensure quality does not regress.

## Known limitations

- Never tested on real video, only synthetic. Default values, especially for cut detection (`--cut-min 8`), may need adjustment. The log reports how many cuts were found and where, and `debug.png` shows them.
- Gradual transitions (fades, dissolves) are not detected, only hard cuts. During a transition flow is poor, but it is brief.
- Cuts between very similar and dark scenes may fall below `--cut-min`. A false cut in a fast scene only resets the region and normalisation; the damage is small.
- Because normalisation is per scene, amplitude is not comparable across scenes (camera may be at different distances). Each scene uses its full range.
- When action stops within a scene, the automatic region follows whatever moves next.
- The manual region is one for the whole video. For different zones over time, run in segments with `--start` and `--duration` and merge.
- Within a scene, a sudden amplitude change takes a few seconds to be reflected in normalisation.
- Does not handle video rotation (metadata) or fisheye. For 3D/VR use `--sbs` or `--tb`. VR180 fisheye distortion remains uncorrected, and ffmpeg still decodes the full frame before cropping.

## Ideas for later

- Time-ranged regions, e.g. `--region 0-120@x0:y0:x1:y1`.
- Manual cut list (`--cuts 12.5,48.0`) for gradual transitions.
- Subtract estimated global (camera) motion computed from cells outside the region.
- Generate a video preview with region and position overlaid.
