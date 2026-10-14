"""Top-level pipeline: process one video, process many, run_folder, CLI entry point."""
import argparse
import copy
import os
import subprocess
import sys
import threading

import cv2
import numpy as np

from .flow import log, load_or_extract, accel_report, Cancelled, NoFlow, BACKEND, FIXED_THREADS
from .signal import compute
from .output import write_funscript, plot
from .debug import render_debug, parse_debug_videos
from .chapters import (chapters_file, load_chapters, chapter_args, limit_speed, load_full,
                       _full_args, flow_available, flow_for, process_chapters, detect_chapters,
                       write_chapters_json, resolve_chapters, find_videos, VIDEO_EXTS)


def process(a, progress=None):
    """Generates the funscript for a video (and debug output and plot, if requested).

    Returns the number of keyframes written.
    """
    import time as _time
    _t0 = _time.perf_counter()
    flow = None

    def load_flow():
        nonlocal flow
        if flow is None:
            flow = load_or_extract(a, progress)
        return flow

    chs, flow = resolve_chapters(a, load_flow)
    if chs:
        if a.plot or a.debug:
            log("with chapters, --plot and --debug are not generated (use --chapters none to get them)")
        pts = process_chapters(a, chs, progress)
        n = write_funscript(a.output, pts)
        log(f"wrote {n} actions to {a.output}  (done in {_time.perf_counter() - _t0:.1f} s)")
        return n
    dx, dy, mag = load_flow()
    fps = a.fps
    dbg = [] if a.debug else None
    r = compute(a, dx, dy, mag, dbg)
    v, p, pos, pts = r["v"], r["p"], r["pos"], r["pts"]
    windows, rois, grid = r["windows"], r["rois"], r["grid"]
    n = write_funscript(a.output, pts)
    log(f"wrote {n} actions to {a.output}  (done in {_time.perf_counter() - _t0:.1f} s)")
    if a.plot:
        plot(a.plot, v, p, pos, pts, rois, grid, mag, fps, a)
    if a.debug:
        render_debug(a, a.debug, v, np.cumsum(v), p, pos, pts, windows, rois, grid, dbg, mag)
    return n


def _error_text(e):
    if isinstance(e, subprocess.CalledProcessError):
        return "ffprobe could not read the file (not a valid video?)"
    return str(e)


def process_many(items, workers, progress=None, stop=None, on_item=None):
    """Processes multiple videos at the same time in threads (OpenCV and ffmpeg release the GIL).

    items: list of Namespace, each with video and output already set. Returns, in the same order,
    (status, detail) where status is ok (detail = keyframes), empty (static, script not saved),
    failed (detail = reason), noflow (cache_only and flow missing) or cancelled. progress(i, n, total)
    reports progress for video i, stop() returns True to abort (or progress raises Cancelled),
    on_item(i, status, detail) is called when each video finishes.

    CPU cores are divided among videos: each computes flow with cpu // workers chunks. With OpenCL
    chunks stay as requested (one GPU; latency headroom allows 2 or 3 videos)."""
    import queue
    n = len(items)
    workers = max(1, min(int(workers or 1), n)) if n else 1
    cpu = os.cpu_count() or 1
    share = max(1, cpu // workers)
    jobs_in = []
    for b in items:
        b = copy.copy(b)
        if workers > 1 and not getattr(b, "opencl", False):
            j = int(getattr(b, "jobs", 0) or 0)
            b.jobs = share if j <= 0 else min(j, share)
        jobs_in.append(b)
    q = queue.Queue()
    for i in range(n):
        q.put(i)
    res = [None] * n
    lock = threading.Lock()

    def finish(i, st, det):
        res[i] = (st, det)
        if on_item:
            with lock:
                on_item(i, st, det)

    def run():
        while True:
            try:
                i = q.get_nowait()
            except queue.Empty:
                return
            b = jobs_in[i]
            if stop and stop():
                finish(i, "cancelled", "")
                continue
            try:
                k = process(b, (lambda c, t, i=i: progress(i, c, t)) if progress else None)
                if k == 0:
                    try:
                        os.remove(b.output)        # an empty script would be "associated" with the video
                    except OSError:
                        pass
                    finish(i, "empty", "")
                else:
                    finish(i, "ok", k)
            except Cancelled:
                finish(i, "cancelled", "")
            except NoFlow:
                finish(i, "noflow", "")
            except (Exception, SystemExit) as e:
                finish(i, "failed", _error_text(e))

    old = cv2.getNumThreads()
    if workers > 1:
        FIXED_THREADS["on"] = True
        cv2.setNumThreads(share)
    ths = [threading.Thread(target=run, daemon=True) for _ in range(workers)]
    try:
        for t in ths:
            t.start()
        for t in ths:
            t.join()
    finally:
        FIXED_THREADS["on"] = False
        cv2.setNumThreads(old)
    return [r or ("cancelled", "") for r in res]


def run_folder(a):
    """Process videos in a folder that do not yet have a .funscript alongside.

    A video that fails does not stop the others. Returns the exit code.
    With --folder-jobs N, N videos are processed simultaneously.
    """
    import time as _time
    _t0 = _time.perf_counter()
    if a.output:
        sys.exit("-o is not used with a folder: each script is saved next to its video")
    vids = find_videos(a.video, a.recursive)
    if not vids:
        sys.exit(f"no videos found in {a.video}")
    todo, skipped = [], []
    for v in vids:
        out = os.path.splitext(v)[0] + ".funscript"
        (skipped if os.path.exists(out) and not a.overwrite else todo).append((v, out))
    log(f"{len(vids)} videos in {a.video}: {len(todo)} without script, "
        f"{len(skipped)} already have script (skipped)")
    items = []
    for v, out in todo:
        b = argparse.Namespace(**vars(a))
        b.video, b.output = v, out
        # unique name per video, also with --recursive (different folders, same name)
        rel = os.path.relpath(os.path.splitext(v)[0], a.video).replace(os.sep, "_")
        if a.debug:
            b.debug = os.path.join(a.debug, rel)
        if a.plot:
            b.plot = os.path.splitext(a.plot)[0] + "_" + rel + ".png"
        items.append(b)
    workers = int(getattr(a, "folder_jobs", 1) or 1)
    done, empty, failed = [], [], []
    if workers > 1 and len(items) > 1:
        workers = min(workers, len(items))
        log(f"{workers} videos in parallel (scripts appear in completion order)")
        names = [os.path.basename(b.video) for b in items]

        def on_item(i, st, det):
            k = f"[{sum(r is not None for r in res_live) + 1}/{len(items)}]"
            res_live[i] = st
            msg = {"ok": f"{det} keyframes", "empty": "no keyframes (static video?), script not saved",
                   "failed": f"failed: {det}", "cancelled": "cancelled",
                   "noflow": "no cached flow"}[st]
            log(f"{k} {names[i]}: {msg}")
        res_live = [None] * len(items)
        out = process_many(items, workers, on_item=on_item)
        for b, (st, det) in zip(items, out):
            {"ok": done, "empty": empty}.get(st, failed).append(b.video)
    else:
        for i, b in enumerate(items, 1):
            log(f"[{i}/{len(items)}] {b.video}")
            try:
                n = process(b)
            except (Exception, SystemExit) as e:
                log(f"  failed: {_error_text(e)}")
                failed.append(b.video)
                continue
            if n == 0:
                # an empty script would be "associated" and the video would never be retried
                os.remove(b.output)
                log("  no keyframes (static video?), script not saved")
                empty.append(b.video)
            else:
                done.append(b.video)
    elapsed = _time.perf_counter() - _t0
    log(f"done in {elapsed:.1f} s: {len(done)} scripts created, {len(skipped)} skipped, "
        f"{len(empty)} no keyframes, {len(failed)} failed")
    for v in empty + failed:
        log(f"  no script: {v}")
    return 1 if failed else 0


def build_parser():
    ap = argparse.ArgumentParser(description="FunAutoScripter — generate a funscript from video using optical flow (no AI).")
    ap.add_argument("video", help="video file, or a folder: generates a script for each video that does not yet have a .funscript next to it")
    ap.add_argument("-o", "--output", help="output .funscript (default: next to the video; not allowed with a folder)")
    ap.add_argument("--plot", metavar="PNG", help="save a debug plot (with a folder, gets the video name appended: PNG_name.png)")

    g = ap.add_argument_group("input")
    g.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    g.add_argument("--duration", type=float, default=0.0, help="seconds to process (0 = all)")
    g.add_argument("--crop", help="left|right|top|bottom (VR halves) or W:H:X:Y in source pixels")
    g.add_argument("--sbs", action="store_true", help="side-by-side 3D/VR video: analyse one half only (left by default, see --eye)")
    g.add_argument("--tb", action="store_true", help="top-bottom 3D/VR video: analyse one half only (top by default, see --eye)")
    g.add_argument("--eye", choices=["left", "right"], default="left", help="half to analyse with --sbs or --tb (left = left or top, right = right or bottom)")
    g.add_argument("--zoom", metavar="X0:Y0:X1:Y1", help="analyse only this zone, as fractions of the frame (after --crop/--sbs/--tb), using the full analysis width. For subtle motion: see the .roi.png from --plot and set the zone where it occurs. --region becomes relative to this zone")
    g.add_argument("--vr-zoom", type=float, default=0.0, metavar="FRACTION", dest="vr_zoom", help="VR edge crop: fraction (0.0–0.5) to trim from each edge after --sbs/--tb, before --zoom. Removes fisheye distortion at the edges. E.g. 0.1 trims 10%% from each side.")
    g.add_argument("--fps", type=int, default=30, help="analysis frame rate")
    g.add_argument("--width", type=int, default=192, help="analysis width in pixels")
    g.add_argument("--cell", type=int, default=12, help="grid cell size in pixels")
    g.add_argument("--chapters", metavar="auto|detect|none|FILE|TEXT", default="auto", help="chapters: different zoom, region and axis-ref over time. auto (default) uses <video>.chapters.json if it exists (the GUI saves it) and otherwise detects them from the flow; detect always detects (rewrites the file if it was saved by a detection); none = no chapters. Or a .json, or a text string 'T@ZOOM[@REGION[@i[@AXIS_REF]]];...' with times in seconds and '-' for none, e.g. '0@-;42.5@0.2:0.3:0.8:0.9@i@90'")
    g.add_argument("--chap-by", default="cuts,angle,zone,soft", help="what separates chapters: cuts (hard cuts), soft (soft transitions: fades, dissolves, slow scene changes), angle (motion axis changes), zone (motion moves to a different screen area). Comma-separated list")
    g.add_argument("--chap-sens", type=float, default=0.5, help="overall chapter detection sensitivity, 0 to 1. 0.5 uses settings as-is, higher lowers all thresholds (more chapters), lower raises them")
    g.add_argument("--chap-soft", type=float, default=0.5, help="soft transition sensitivity, 0 to 1 (added to --chap-sens)")
    g.add_argument("--chap-soft-span", type=float, default=2.0, help="seconds on each side compared to find a soft transition (the transition must be shorter than this)")
    g.add_argument("--chap-angle", type=float, default=35.0, help="degrees: motion axis change that counts as a new chapter")
    g.add_argument("--chap-shift", type=float, default=0.30, help="fraction of frame: motion zone change that counts as a new chapter")
    g.add_argument("--chap-persist", type=float, default=8.0, help="seconds the change must persist to count")
    g.add_argument("--chap-min", type=float, default=10.0, help="seconds: minimum chapter length (shorter ones are merged)")
    g.add_argument("--chap-win", type=float, default=1.0, help="seconds: analysis window for detection")
    g.add_argument("--cut-ratio", type=float, default=4.0, help="a cut is a mean-motion spike above this times the normal level")
    g.add_argument("--cut-min", type=float, default=0.5, help="minimum mean motion (px) for a spike to count as a cut")
    g.add_argument("--no-cache", action="store_true", help="ignore the flow cache")
    g.add_argument("--folder-jobs", type=int, default=1, help="with a folder: how many videos to process at once (1 = sequential). Cores are divided among videos so total time does not change. Useful for short videos that do not fully use all cores, and with OpenCL (2 or 3 videos on the same GPU)")
    g.add_argument("--jobs", type=int, default=0, help="video chunks processed in parallel (0 = auto, up to 6, one per core; 1 = sequential). Each chunk is at least 20 s. Does not visibly change the result. With --opencl the default is 1; 2 to 4 may increase GPU throughput")
    g.add_argument("--hwaccel", nargs="?", const="auto", default=None, metavar="METHOD", help="decode the video with hardware acceleration (ffmpeg -hwaccel), if available. Without a value uses auto, or specify the method: cuda (NVIDIA), vaapi (Intel/AMD), etc. (see --accel-test). Very helpful for 4K and VR where decoding is slower than the flow")
    g.add_argument("--accel-test", action="store_true", help="acceleration diagnostics (OpenCL/GPU, CPU) and exit: what OpenCV finds, what device it is, and throughput")
    g.add_argument("--opencl", action="store_true", help="experimental: compute flow with OpenCL (GPU) if OpenCV supports it. May give slightly different values and, with images this small, is not always faster")

    g = ap.add_argument_group("region and motion")
    g.add_argument("--region", metavar="X0:Y0:X1:Y1", help="lock the region, as fractions of the frame (0 to 1), e.g. 0.5:0.3:0.9:0.9. See the .roi.png from --plot")
    g.add_argument("--roi", type=float, default=0.4, help="auto region size as a fraction of the frame")
    g.add_argument("--roi-aspect", type=float, default=0.0, help="width/height ratio of the automatic region (1 = square, 2 = twice as wide, 0.5 = twice as tall). 0 inherits the video ratio. Area equals --roi")
    g.add_argument("--roi-window", type=float, default=3.0, help="seconds between region updates")
    g.add_argument("--switch", type=float, default=0.6, help="new region must score this many times better to replace the current one")
    g.add_argument("--period-weight", type=float, default=0.5, help="0 = pick region by magnitude only, 1 = pick by periodicity only")
    g.add_argument("--candidates", type=int, default=6, metavar="N",
                   help="number of region candidates evaluated per window (default 6)")
    g.add_argument("--axis", choices=["auto", "y"], default="auto", help="auto = main motion direction, y = vertical only")
    g.add_argument("--axis-ref", default="chapter", metavar="chapter|down|up|left|right|auto|DEGREES", help="which end of the motion axis is position 100 (axis only gives the line, not the direction). chapter = the dominant direction of the chapter itself (or video if no chapters), consistent within it; if reversed, use the chapter's invert flag. down, up, left, right = fixed screen direction. auto = continuity only (errors propagate)")
    g.add_argument("--axis-margin", type=float, default=0.35, help="if a window's axis is nearly perpendicular to the reference (cosine below this) the direction is ambiguous and follows the previous window")
    g.add_argument("--noise", type=float, default=0.05, help="flow noise floor in px/frame")

    g = ap.add_argument_group("signal")
    g.add_argument("--highpass", type=float, default=0.15, help="Hz, removes drift")
    g.add_argument("--lowpass", type=float, default=5.0, help="Hz, removes jitter")
    g.add_argument("--norm-window", type=float, default=10.0, help="seconds used to scale to 0..100")
    g.add_argument("--low-pct", type=float, default=3.0)
    g.add_argument("--high-pct", type=float, default=97.0)
    g.add_argument("--floor", type=float, default=0.35, help="minimum local range vs the typical range (stops noise being amplified in pauses)")
    g.add_argument("--invert", action="store_true", help="flip up and down")
    g.add_argument("--hp2", type=float, default=0.0, help="Hz: cut low frequencies from the already-scaled funscript (0 = off). Removes slow oscillation and keeps fast strokes")
    g.add_argument("--hfboost", type=float, default=0.0, help="boost high frequencies above --hf-freq (0 = off, 1 = double the amplitude of fast strokes relative to slow ones)")
    g.add_argument("--hf-freq", type=float, default=1.5, help="Hz: where the --hfboost enhancement begins")
    g.add_argument("--renorm", type=float, default=0.0, help="seconds: re-normalise and amplify the funscript with a short window (0 = off, minimum 0.5). Brings up small strokes")
    g.add_argument("--renorm-floor", type=float, default=0.15, help="like --floor but for re-normalisation: lower values amplify quiet sections more")
    g.add_argument("--smooth", type=float, default=0.0, help="smooth the signal before scaling (seconds, 0 = off)")
    g.add_argument("--boost", type=float, default=1.0, help="enhance small movements: above 1 opens light strokes without touching the extremes")
    g.add_argument("--gain", type=float, default=1.0, help="amplitude around the centre after scaling (above 1 increases it)")
    g.add_argument("--offset", type=float, default=0.0, help="shift the centre, in 0-100 units")

    g = ap.add_argument_group("keyframes")
    g.add_argument("--min-amp", type=float, default=12.0, help="smallest stroke to keep (0-100 units)")
    g.add_argument("--min-interval", type=float, default=0.12, help="seconds between keyframes")
    g.add_argument("--max-speed", type=float, default=450.0, help="units per second")

    g = ap.add_argument_group("debug")
    g.add_argument("--debug", metavar="DIR", help="write to DIR videos explaining each decision (region, magnitude, periodicity, script, velocity, integral), plus signals.csv and decisions.json (with a folder, each video uses DIR/name)")
    g.add_argument("--debug-height", type=int, default=360, help="image height in debug videos")
    g.add_argument("--debug-window", type=float, default=15.0, help="seconds visible in sliding graphs")
    g.add_argument("--debug-videos", default="all", metavar="LIST", help="which videos to generate, comma-separated: roi, magnitude, periodicity, script, velocity, integral (or 1 to 6). all = all, none = only signals.csv and decisions.json")

    g = ap.add_argument_group("folder")
    g.add_argument("--recursive", action="store_true", help="if the argument is a folder, also search subfolders")
    g.add_argument("--overwrite", action="store_true", help="if the argument is a folder, also regenerate videos that already have a .funscript")

    return ap


def main():
    if "--accel-test" in sys.argv:
        a = build_parser().parse_args(["_"] + [x for x in sys.argv[1:] if x != "--accel-test" and x != "_"])
        print("\n".join(accel_report(a)))
        return
    a = build_parser().parse_args()
    if not os.path.exists(a.video):
        sys.exit(f"no such file: {a.video}")
    if a.debug:
        a.debug_sel = parse_debug_videos(a.debug_videos)
    if a.sbs or a.tb:
        if a.sbs and a.tb:
            sys.exit("--sbs and --tb cannot be combined, choose one")
        if a.crop:
            sys.exit("--sbs and --tb already select the crop, do not use --crop at the same time")
        if a.sbs:
            a.crop = "left" if a.eye == "left" else "right"
        else:
            a.crop = "top" if a.eye == "left" else "bottom"
        log(f"mode {'SBS' if a.sbs else 'TB'}: analysing the {a.crop} half of the video")
    if os.path.isdir(a.video):
        sys.exit(run_folder(a))
    if not a.output:
        a.output = os.path.splitext(a.video)[0] + ".funscript"
    process(a)
