# FunAutoScripter

Fast funscript generator for interactive experiences. Runs fully unattended, supports bulk processing, and requires no manual work per video. (At least that's the objective)

Built on classical optical flow (OpenCV Farneback). No AI, no neural networks, no GPU required.

> Proudly 100% vibe coded.

## Requirements

- Python 3 with numpy, scipy, opencv-python, matplotlib (only for `--plot`)
- ffmpeg (system)
- NixOS: use `nix-shell` (see `shell.nix`)

## Usage

```bash
nix-shell
python funautoscripter.py video.mp4
python funautoscripter.py video.mp4 --plot debug.png
python funautoscripter.py folder/
python funautoscripter.py folder/ --recursive --overwrite
```

## Features

- **No AI or neural networks** — pure classical computer vision, runs on any CPU
- **Optical flow analysis** — detects motion direction, speed and position using Farneback
- **Automatic ROI selection** — finds the region with the most relevant motion, switches with hysteresis
- **Scene cut detection** — handles scene changes; each scene is normalised independently
- **3D/VR support** — `--sbs` and `--tb` for side-by-side and top-bottom formats
- **Subtle motion** — `--zoom` crops a small region before downscaling to capture low-amplitude movements
- **Flow cache** — the slow step (optical flow extraction) is cached; parameter tuning reruns in milliseconds
- **Debug output** — `--plot` generates signal graphs with scene cuts; `--debug DIR` generates 6 annotated videos
- **Batch processing** — pass a folder to process multiple videos; skips existing scripts by default
- **Parallel processing** — `--jobs N` splits long videos into chunks processed in parallel threads
- **Chapter system** — split a video into chapters with different zoom/region settings via `--chapters`
- **GUI** — `funautoscripter_gui.py` (DearPyGui) with live parameter sliders, timeline, video preview and batch mode

## Key Parameters

| Parameter | Description |
|---|---|
| `--region X0:Y0:X1:Y1` | Fix the analysis region (fractions 0–1) |
| `--zoom X0:Y0:X1:Y1` | Crop before downscaling, for subtle motion |
| `--sbs` / `--tb` | Side-by-side or top-bottom VR input |
| `--plot FILE` | Save signal debug graphs |
| `--debug DIR` | Save 6 annotated debug videos |
| `--max-speed N` | Limit keyframe velocity (default 400 units/s) |
| `--jobs N` | Parallel flow extraction threads (0 = auto) |
| `--chapters auto\|FILE` | Chapter definitions for mixed-scene videos |

## Limitations

- **No fade/dissolve detection** — only hard cuts are detected; gradual transitions produce bad flow briefly
- **Single region per video** — the automatic ROI follows whatever moves when action stops; use `--region` or chapters to fix this
- **Amplitude not comparable across scenes** — each scene is normalised to its own range
- **Subtle motion has a hard floor** — below ~0.1 px at analysis resolution, nothing works regardless of settings
- **VR180 fisheye distortion** — not corrected
- **Video rotation metadata** — not handled

## Testing

```bash
# no cuts: vertical object, pause, fast small strokes, horizontal distractor
python tests/make_test.py test.mp4 truth.npy
python funautoscripter.py test.mp4 -o test.funscript
python tests/compare.py test.funscript

# with cuts: 4 scenes, low-contrast backgrounds
python tests/make_test_cuts.py cuts.mp4 cuts_truth.npy
python funautoscripter.py cuts.mp4 -o cuts.funscript
python tests/compare_cuts.py cuts.funscript cuts_truth.npy
```

Expected correlation with ground truth: ≥0.97 on large strokes, ≥0.96 on small fast strokes, ≥0.978 per scene with cuts.

## Installation

### From a release binary (Linux / Windows)

Download the latest binary from the [Releases](../../releases) page:

- `funautoscripter` — CLI tool (Linux)
- `funautoscripter-gui` — GUI (Linux)
- `funautoscripter.exe` / `funautoscripter-gui.exe` — Windows equivalents

On Linux, make the binary executable and run it:

```bash
chmod +x funautoscripter
./funautoscripter video.mp4
```

The binary includes all Python dependencies. You still need **ffmpeg** installed on your system (e.g. `apt install ffmpeg` or `nix-env -iA nixpkgs.ffmpeg`).

### From source (NixOS / nix-shell)

```bash
git clone <repo-url>
cd customFunscriptGen
nix-shell
python funautoscripter.py video.mp4
python funautoscripter_gui.py
```

`shell.nix` provides all dependencies (Python, numpy, scipy, opencv, matplotlib, DearPyGui, ffmpeg) in an isolated environment.

### From source (other Linux / macOS)

```bash
pip install numpy scipy opencv-python matplotlib dearpygui
# install ffmpeg via your system package manager
python funautoscripter.py video.mp4
```

## Building

Self-contained binaries are built with [PyInstaller](https://pyinstaller.org). Run from the project root:

```bash
pip install pyinstaller numpy scipy opencv-python matplotlib dearpygui

# CLI binary
pyinstaller --onefile --name funautoscripter funautoscripter.py

# GUI binary
pyinstaller --onefile --name funautoscripter-gui funautoscripter_gui.py
```

Output binaries appear in `dist/`. The `build/` directory and `*.spec` files are generated automatically and are excluded from version control.

Release builds for Linux and Windows are produced automatically by the GitHub Actions workflow (`.github/workflows/release.yml`) on every `v*` tag push.
