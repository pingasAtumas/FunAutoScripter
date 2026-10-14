"""GUI constants, presets and theme definitions."""
import json
import os

import funautoscripter as fl

from .utils import config_path, mix

CW, CH = 576, 324          # video canvas size in pixels
MAXC = 8                    # candidates shown in the inspector
PALETTE_RGB = [(c[2], c[1], c[0]) for c in fl.PALETTE]   # engine uses BGR

# (group, attribute, label, type, min, max or options, format, stage)
# stage "c" = recalculates immediately (fast), "f" = requires flow recalculation
PARAMS = [
    ("Treble Enhancement", "hp2", "cut bass (Hz)", "float", 0.0, 3.0, "%.2f", "c"),
    ("Treble Enhancement", "hfboost", "treble boost", "float", 0.0, 6.0, "%.1f", "c"),
    ("Treble Enhancement", "hf_freq", "treble from (Hz)", "float", 0.3, 5.0, "%.1f", "c"),
    ("Treble Enhancement", "renorm", "renormalise (s)", "float", 0.0, 20.0, "%.1f", "c"),
    ("Treble Enhancement", "renorm_floor", "renorm floor", "float", 0.02, 1.0, "%.2f", "c"),
    ("Enhancement", "smooth", "smooth (s)", "float", 0.0, 0.5, "%.2f", "c"),
    ("Enhancement", "boost", "boost light strokes", "float", 1.0, 4.0, "%.2f", "c"),
    ("Enhancement", "gain", "amplitude gain", "float", 0.5, 4.0, "%.2f", "c"),
    ("Enhancement", "offset", "shift centre", "float", -40.0, 40.0, "%.0f", "c"),
    ("Region", "roi", "region size", "float", 0.1, 1.0, "%.2f", "c"),
    ("Region", "roi_aspect", "aspect ratio (0 = video's)", "float", 0.0, 4.0, "%.2f", "c"),
    ("Region", "roi_window", "window (s)", "float", 0.5, 12.0, "%.1f", "c"),
    ("Region", "switch", "switch (--switch)", "float", 0.0, 3.0, "%.2f", "c"),
    ("Region", "period_weight", "periodicity weight", "float", 0.0, 1.0, "%.2f", "c"),
    ("Region", "candidates", "candidates (inspector)", "int", 3, 12, "%d", "c"),
    ("Region", "axis", "axis", "combo", None, ["auto", "y"], None, "c"),
    ("Region", "axis_ref", "100 points to", "combo", None, ["chapter", "down", "up", "left", "right", "auto"], None, "c"),
    ("Region", "axis_margin", "direction margin", "float", 0.0, 0.9, "%.2f", "c"),
    ("Region", "noise", "noise (px/frame)", "float", 0.0, 0.3, "%.3f", "c"),
    ("Signal", "highpass", "highpass (Hz)", "float", 0.02, 1.0, "%.2f", "c"),
    ("Signal", "lowpass", "lowpass (Hz)", "float", 1.0, 14.0, "%.1f", "c"),
    ("Signal", "norm_window", "scaling window (s)", "float", 2.0, 60.0, "%.0f", "c"),
    ("Signal", "low_pct", "low percentile", "float", 0.0, 20.0, "%.1f", "c"),
    ("Signal", "high_pct", "high percentile", "float", 80.0, 100.0, "%.1f", "c"),
    ("Signal", "floor", "floor", "float", 0.0, 1.0, "%.2f", "c"),
    ("Signal", "invert", "invert", "bool", None, None, None, "c"),
    ("Keyframes", "min_amp", "min amplitude", "float", 1.0, 50.0, "%.0f", "c"),
    ("Keyframes", "min_interval", "min interval (s)", "float", 0.03, 1.0, "%.2f", "c"),
    ("Keyframes", "max_speed", "max speed (u/s)", "float", 50.0, 1000.0, "%.0f", "c"),
    ("Auto Chapters", "chap_sens", "overall sensitivity", "float", 0.0, 1.0, "%.2f", "d"),
    ("Auto Chapters", "chap_by", "split by", "combo", None,
     ["cuts,angle,zone,soft", "cuts,angle,zone", "cuts,soft", "cuts", "soft", "angle,zone,soft", "angle,zone",
      "cuts,angle,soft", "cuts,zone,soft"], None, "d"),
    ("Auto Chapters", "chap_soft", "soft transitions", "float", 0.0, 1.0, "%.2f", "d"),
    ("Auto Chapters", "chap_soft_span", "time each side (s)", "float", 1.0, 6.0, "%.1f", "d"),
    ("Auto Chapters", "chap_angle", "axis change (°)", "float", 10.0, 80.0, "%.0f", "d"),
    ("Auto Chapters", "chap_shift", "zone change", "float", 0.1, 0.8, "%.2f", "d"),
    ("Auto Chapters", "chap_persist", "must last (s)", "float", 1.0, 20.0, "%.1f", "d"),
    ("Auto Chapters", "chap_min", "min chapter (s)", "float", 2.0, 90.0, "%.0f", "d"),
    ("Auto Chapters", "chap_win", "analysis window (s)", "float", 0.5, 4.0, "%.1f", "d"),
    ("Auto Chapters", "cut_ratio", "cut: times normal", "float", 2.0, 12.0, "%.1f", "d"),
    ("Auto Chapters", "cut_min", "cut: min motion", "float", 0.05, 3.0, "%.2f", "d"),
    ("Processing Speed", "jobs", "parallel chunks (0 = auto)", "int", 0, 16, "%d", "d"),
    ("Processing Speed", "folder_jobs", "parallel videos in batch", "int", 1, 8, "%d", "d"),
    ("Processing Speed", "hwaccel", "hardware decode", "bool", None, None, None, "d"),
    ("Processing Speed", "opencl", "OpenCL (experimental)", "bool", None, None, None, "d"),
    ("Flow (needs recalculation)", "width", "analysis width", "int", 96, 768, "%d", "f"),
    ("Flow (needs recalculation)", "cell", "cell (px)", "int", 6, 24, "%d", "f"),
    ("Flow (needs recalculation)", "fps", "analysis fps", "combo", None, ["15", "20", "24", "30", "60"], None, "f"),
]
TIPS = {
    "hp2": "Cuts the low frequencies from the scaled funscript. Removes slow oscillation and preserves fast strokes. 0 disables.",
    "hfboost": "Boosts content above the treble frequency. 1 doubles fast-stroke amplitude relative to slow strokes. 0 disables.",
    "hf_freq": "Frequency (Hz) above which treble boost applies.",
    "renorm": "Re-normalises and amplifies the funscript with a short window (seconds, from 0.5 up). Lifts small strokes. 0 disables.",
    "renorm_floor": "Like floor, but for the renormalisation step. Lower amplifies weak sections more (and noise too).",
    "smooth": "Smooths the signal before scaling to 0-100, in seconds. 0 disables.",
    "boost": "Opens up light strokes without affecting the extremes. Above 1 emphasises small movements.",
    "gain": "Amplitude around the centre after scaling. Above 1 increases the stroke range.",
    "offset": "Shifts the funscript centre, in 0-100 units.",
    "roi": "Automatic region size as a fraction of the frame. This is the area; the aspect ratio is set below.",
    "roi_aspect": "Width/height ratio of the automatic region. 1 is square, 2 is twice as wide, 0.5 twice as tall. 0 uses the video's ratio.",
    "roi_window": "How often (in seconds) the best region is re-evaluated.",
    "switch": "The new region must score this many times more than the current one to replace it. Higher means the region changes less.",
    "period_weight": "0 picks the region by motion alone, 1 by periodicity (rhythmic back-and-forth) alone.",
    "candidates": "Number of region candidates evaluated and shown per window. Higher = more thorough but slower region selection.",
    "axis": "auto follows the principal direction of motion. y uses only the vertical axis.",
    "axis_ref": "Which end of the axis is 100. chapter uses the chapter's dominant direction, down/up/left/right fix a screen direction, auto only follows the previous window. Each chapter can have its own.",
    "axis_margin": "If a window's axis is nearly perpendicular to the reference (cosine below this), the direction is ambiguous and follows the previous window.",
    "noise": "Optical flow noise floor, in pixels per frame. Values below this are ignored.",
    "highpass": "Removes slow drift from the integral, in Hz. Higher makes the signal flatter during slow sections.",
    "lowpass": "Removes fast jitter, in Hz. Lower gives smoother strokes.",
    "norm_window": "Window in seconds used to scale the signal to 0-100. Short adapts quickly, long is more stable.",
    "low_pct": "Signal percentile mapped to 0.",
    "high_pct": "Signal percentile mapped to 100.",
    "floor": "Minimum local range relative to the typical range. Prevents amplifying noise during pauses.",
    "invert": "Inverts the entire funscript. To invert only one chapter use the chapter checkbox (key I).",
    "min_amp": "Smallest stroke kept, in 0-100 units. Higher means fewer keyframes.",
    "min_interval": "Minimum time between keyframes, in seconds.",
    "max_speed": "Maximum funscript speed in units per second. Limits jumps.",
    "chap_sens": "Overall detection sensitivity. 0.5 uses settings as-is. Higher lowers all thresholds (more chapters), lower raises them.",
    "chap_by": "What separates chapters: cuts (hard cuts), soft (soft transitions), angle (motion axis rotates), zone (motion moves to another screen area).",
    "chap_soft": "Sensitivity for soft transitions (fades, dissolves, slowly changing scenes). Added to overall sensitivity.",
    "chap_soft_span": "Seconds on each side compared when detecting a soft transition. The transition must be shorter than this.",
    "chap_angle": "How many degrees the motion axis must rotate to count as a new chapter.",
    "chap_shift": "How far the motion must shift on screen (fraction of frame) to count as a new chapter.",
    "chap_persist": "How long an axis or zone change must persist. A short pause will not split a chapter.",
    "chap_min": "Minimum chapter length. Shorter chapters are merged with their neighbour (hard cuts always apply).",
    "chap_win": "Window in seconds used to measure the motion axis and zone.",
    "cut_ratio": "A cut is a peak in average motion above this times the local normal level.",
    "cut_min": "Minimum average motion in pixels for a peak to qualify as a cut.",
    "jobs": "How many video chunks are processed in parallel, one per core. 0 picks automatically (up to 6). 1 is sequential. Result is identical.",
    "folder_jobs": "Batch only: how many videos are processed at the same time. Cores are shared among videos, so total time barely changes, but short videos fill the processor better. With OpenCL, 2 or 3 videos can share GPU idle time. 1 is sequential. Each video saves its cache when done, so an interrupted batch resumes from where it stopped.",
    "hwaccel": "Decodes the video with hardware (GPU, NVDEC on NVIDIA), if available. Helps with 4K and VR where decoding is slower than flow computation. Falls back to CPU if unavailable.",
    "opencl": "Experimental: computes flow on the GPU via OpenCL if OpenCV finds one. The status bar shows what is in use. Help > Test acceleration shows whether it is worth enabling. Results may differ slightly from CPU path.",
    "width": "Flow analysis width in pixels. Wider gives more detail but takes longer. Requires flow recalculation.",
    "cell": "Grid cell size in pixels. Requires flow recalculation.",
    "fps": "Frames per second for analysis. Lower is faster but may miss fast strokes. Requires flow recalculation.",
}
TIPS_UI = {
    "b_open": "Choose a video to analyse.",
    "b_dir": "Choose a folder to process multiple videos in batch.",
    "btn_flow": "Recalculate optical flow for the active chapter.  Shortcut: R",
    "btn_save_top": "Save the funscript to disk.  Shortcut: Ctrl+S",
    "btn_cancel": "Stop current computation.  Shortcut: Esc",
    "btn_play": "Play or pause (space).",
    "p_speed": "Playback speed.",
    "follow": "The signal plots follow the cursor.",
    "follow_w": "Width in seconds of the visible signal window when following the cursor.",
    "ly_roi": "Green box: the region the motion is extracted from.",
    "ly_cand": "Region candidates and their scores. The yellow one is the previous region when it changed.",
    "ly_heat": "Heat map of the average motion in the current window.",
    "ly_mag": "Flow magnitude of the current frame.",
    "ly_per": "Periodicity of each candidate: how rhythmic the back-and-forth is.",
    "ly_axis": "Compass showing the axis-ref and velocity (yellow) and position (magenta) vectors, with the 0 and 100 ends marked.",
    "sel_mode": "region: fixes the analysis region. zoom: analyses only a sub-area (recalculates flow). direction: draw where 100 points.",
    "b_sel_apply": "Applies the drawn rectangle as region or zoom.",
    "b_sel_clear": "Clears the region, zoom or chapter direction, depending on the current mode.",
    "p_crop": "For 3D or VR videos, analyses only one half.",
    "p_vr_zoom": "Trims this percentage from each edge of the VR half to remove fisheye distortion. Stacks with per-chapter zoom. Triggers flow recalculation.",
    "p_live": "On: sliders recalculate while dragging. Off: sliders only change the value and calculation happens on Apply Changes (Enter).",
    "btn_apply": "Apply pending parameter changes.  Shortcut: Enter",
    "chap_inv": "Inverts only this chapter (key I). Saved to .chapters.json.",
    "chap_axmode": "Where 100 points in this chapter. \"angle\" uses the slider value. Only takes effect on Apply Direction.",
    "chap_ang": "Direction of 100 on screen in degrees: 0 = right, 90 = down, 180 = left, 270 = up. Only takes effect on Apply Direction.",
    "btn_apply_axis": "Applies the selected direction and angle to this chapter.",
    "b_axis_draw": "Draw the axis-ref direction on the video.  Shortcut: D",
    "p_preset": "Load a preset. Parameters that differ from it are highlighted in orange.",
    "preset_name": "Name under which to save the current parameters as your own preset.",
    "b_pre_save": "Saves the current parameters as a preset with this name.",
    "b_pre_del": "Deletes your selected preset.",
    "b_pre_reset": "Resets to the reference preset values.",
    "chap_list": "Video chapters. Click to activate one (also [ and ]).",
    "b_chap_new": "Splits the chapter at the current cursor position.  Shortcut: N",
    "b_chap_del": "Merges this chapter into the previous one.",
    "b_chap_del_all": "Removes all chapters and resets to a single chapter covering the whole video.",
    "b_chap_flow": "Computes the flow for all chapters that do not have it yet.",
    "btn_suggest": "Detects cuts and motion changes and splits the video into chapters automatically.  Shortcut: G",
    "out_path": "Where the funscript is saved.",
    "b_save": "Saves the funscript (Ctrl+S).",
    "b_all": "Marks all videos.",
    "b_none": "Unmarks all.",
    "b_missing": "Marks only videos that do not yet have a funscript.",
    "b_over": "Also redoes videos that already have a funscript.",
    "b_same": "Uses the current zoom and region for all videos in the batch.",
    "b_chap": "Detects chapters (and each chapter's axis-ref) for each video if it has no .chapters.json.",
    "btn_bulk": "Processes the marked videos one after another.",
    "btn_bulk_apply": "Redoes the funscript for marked videos using current settings, only from cached flow. Never computes flow: videos without valid cache for the current width, cell and fps are skipped and shown in the summary. Also redoes videos that already have a script.",
    "show_raw": "Shows the unfiltered integral on the right axis.",
    "ov_mode": "Which quantity colours the timeline. Hover over (?) to see each one.",
}
AUTO_HELP = ("Overall sensitivity: above 0.5 all thresholds go down (more chapters), below they go up.\n"
             "Split by: cuts = hard cuts, soft = soft transitions (fades, dissolves, slowly changing scenes), "
             "angle = motion axis rotates, zone = motion moves to another screen area.\n"
             "Axis change (°) and zone change: how much the axis or zone must shift. Must last (s): how long "
             "the change must persist (a short pause will not split a chapter). Min chapter: shorter ones are merged.\n"
             "Soft transitions: own sensitivity, and the time on each side compared (the transition must be "
             "shorter than this).\n"
             "Cut: times normal and min motion (px) decide what counts as a cut peak.\n"
             "Settings only take effect when clicking \"Suggest chapters\" (or in batch with --chapters detect).")
AXMODES = ["(from settings)", "chapter", "down", "up", "left", "right", "auto", "angle"]
CROPS = {"full video": None, "SBS, left half": "left", "SBS, right half": "right",
         "TB, top half": "top", "TB, bottom half": "bottom"}

OV_H = 96                    # height of the overview timeline
OV_BAND = 20                 # chapter band height within it
CH_COLORS = [(70, 120, 200), (60, 170, 130), (190, 110, 60), (150, 90, 190), (180, 160, 60), (190, 80, 120)]
# Dockable windows: (tag, title, open by default). The title is the key in the layout file,
# so it must not be changed (otherwise windows reset to their initial position).
WINDOWS = [("w_ctrl", "Control", True), ("w_video", "Video", True), ("w_vidctrl", "Video Controls", True),
           ("w_ov", "Overview", True), ("w_params", "Parameters", True), ("w_insp", "Inspector", True),
           ("w_tl", "Signals", True), ("w_folder", "Folder Videos", False), ("w_status", "Status", False)]

# Each theme starts from three colours (background, text, accent) and derives the rest from them.
# head = headings, muted = secondary text, warn = warnings, fg = lines that must contrast with the background.
THEMES = {
    "Dark": dict(bg=(30, 30, 33), text=(232, 232, 232), accent=(60, 130, 215), dark=True,
                 head=(120, 200, 255), muted=(150, 150, 150), warn=(255, 170, 40), fg=(255, 255, 255)),
    "Light": dict(bg=(238, 238, 241), text=(25, 25, 28), accent=(40, 110, 200), dark=False,
                  head=(20, 90, 165), muted=(105, 105, 110), warn=(190, 100, 0), fg=(0, 0, 0)),
    "Night Blue": dict(bg=(16, 22, 38), text=(220, 228, 245), accent=(80, 140, 255), dark=True,
                       head=(130, 190, 255), muted=(140, 155, 185), warn=(255, 190, 70), fg=(255, 255, 255)),
    "High Contrast": dict(bg=(0, 0, 0), text=(255, 255, 255), accent=(255, 210, 0), dark=True,
                          head=(0, 255, 255), muted=(205, 205, 205), warn=(255, 160, 0), fg=(255, 255, 255)),
    "Amber": dict(bg=(32, 26, 20), text=(240, 225, 200), accent=(220, 140, 40), dark=True,
                  head=(255, 190, 100), muted=(170, 150, 125), warn=(255, 120, 80), fg=(255, 245, 225)),
    "Forest": dict(bg=(18, 26, 20), text=(215, 235, 215), accent=(80, 185, 100), dark=True,
                   head=(130, 220, 150), muted=(130, 155, 130), warn=(255, 210, 60), fg=(210, 245, 210)),
    "Rose": dict(bg=(30, 20, 26), text=(240, 225, 235), accent=(210, 80, 130), dark=True,
                 head=(255, 150, 180), muted=(170, 130, 150), warn=(255, 190, 60), fg=(255, 235, 245)),
    "Slate": dict(bg=(22, 26, 34), text=(200, 210, 230), accent=(100, 150, 210), dark=True,
                  head=(160, 195, 245), muted=(130, 145, 170), warn=(255, 190, 60), fg=(230, 240, 255)),
    "Mocha": dict(bg=(28, 22, 18), text=(230, 218, 205), accent=(165, 100, 60), dark=True,
                  head=(215, 160, 110), muted=(155, 135, 115), warn=(255, 200, 80), fg=(245, 235, 220)),
    "Teal": dict(bg=(14, 28, 32), text=(210, 235, 238), accent=(45, 185, 185), dark=True,
                 head=(100, 220, 220), muted=(125, 160, 165), warn=(255, 200, 60), fg=(215, 245, 245)),
    "Purple": dict(bg=(22, 16, 36), text=(228, 218, 248), accent=(155, 100, 230), dark=True,
                   head=(195, 155, 255), muted=(145, 130, 175), warn=(255, 190, 70), fg=(240, 230, 255)),
    "Sepia": dict(bg=(42, 36, 28), text=(235, 222, 200), accent=(185, 135, 70), dark=True,
                  head=(230, 185, 120), muted=(165, 148, 120), warn=(255, 155, 60), fg=(250, 238, 215)),
    "Soft Light": dict(bg=(246, 244, 240), text=(40, 38, 35), accent=(100, 130, 180), dark=False,
                       head=(50, 100, 165), muted=(120, 115, 105), warn=(185, 90, 0), fg=(0, 0, 0)),
    "Solarized": dict(bg=(0, 43, 54), text=(131, 148, 150), accent=(38, 139, 210), dark=True,
                      head=(42, 161, 152), muted=(88, 110, 117), warn=(203, 75, 22), fg=(253, 246, 227)),
    "Dracula": dict(bg=(40, 42, 54), text=(248, 248, 242), accent=(189, 147, 249), dark=True,
                    head=(80, 250, 123), muted=(98, 114, 164), warn=(255, 184, 108), fg=(255, 255, 255)),
    "Gruvbox": dict(bg=(29, 32, 33), text=(235, 219, 178), accent=(215, 153, 33), dark=True,
                    head=(142, 192, 124), muted=(146, 131, 116), warn=(251, 73, 52), fg=(251, 241, 199)),
    "Nord": dict(bg=(46, 52, 64), text=(216, 222, 233), accent=(129, 161, 193), dark=True,
                 head=(136, 192, 208), muted=(76, 86, 106), warn=(235, 203, 139), fg=(236, 239, 244)),
}
DEFAULT_THEME = "Dark"

# Only parameters that recalculate immediately (stage "c"). Excluded: flow params (need recalculation)
# and "invert" (depends on the video, not the style). Each preset only lists what differs from
# the default; the rest reverts to the default value when loaded.
PRESET_ATTRS = [p[1] for p in PARAMS if p[7] == "c" and p[1] not in ("invert", "axis_ref")]
PARAM_BY_ATTR = {p[1]: p for p in PARAMS}

BUILTIN_PRESETS = {
    "Default": ("Default values.", {}),
    "Light": ("Smooth and calm motion: fewer strokes, softer, reduced amplitude and speed.", {
        "smooth": 0.06, "lowpass": 3.0, "gain": 0.8, "floor": 0.45, "min_amp": 20.0,
        "min_interval": 0.2, "max_speed": 300.0, "norm_window": 15.0}),
    "Strong": ("Large, pronounced strokes: full range, opens light strokes, accepts higher speed.", {
        "gain": 1.4, "boost": 1.3, "low_pct": 1.0, "high_pct": 99.0, "lowpass": 7.0,
        "min_amp": 8.0, "min_interval": 0.1, "max_speed": 650.0, "floor": 0.3}),
    "Precise": ("Faithfully follows real motion: stable region, low noise, fine detail preserved.", {
        "period_weight": 0.6, "roi_window": 2.0, "noise": 0.03, "lowpass": 6.0,
        "norm_window": 15.0, "floor": 0.3, "min_amp": 8.0, "min_interval": 0.08, "max_speed": 500.0}),
    "Enhanced": ("Brings out small, fast strokes (treble boost and renormalisation). "
                 "May amplify noise during pauses.", {
        "hp2": 0.6, "hfboost": 1.5, "hf_freq": 1.5, "renorm": 4.0, "renorm_floor": 0.25,
        "boost": 1.3, "gain": 1.1, "noise": 0.03, "min_amp": 8.0}),
}


def presets_path():
    """User presets file, stored next to gui.json."""
    return os.path.join(os.path.dirname(config_path()), "presets.json")


def clean_param(attr, v):
    """Valid value for the parameter (clamped to the slider range), or None if invalid."""
    _, _, _, kind, lo, hi, _, _ = PARAM_BY_ATTR[attr]
    try:
        if kind == "float":
            return min(max(float(v), lo), hi)
        if kind == "int":
            return int(v)
        if kind == "bool":
            return bool(v)
        return v if v in hi else None
    except (TypeError, ValueError):
        return None


def load_user_presets():
    try:
        with open(presets_path(), encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(d, dict):
        return {}
    out = {}
    for name, vals in d.items():
        if isinstance(vals, dict):
            ok = {k: clean_param(k, x) for k, x in vals.items() if k in PRESET_ATTRS}
            out[str(name)] = {k: x for k, x in ok.items() if x is not None}
    return out
