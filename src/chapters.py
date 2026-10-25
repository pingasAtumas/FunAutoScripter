"""Chapter management: loading, detecting, processing chapters; VIDEO_EXTS/find_videos."""
import copy
import json
import os
import threading

import numpy as np

from .flow import log, probe, cache_path, cache_key, cache_valid, load_or_extract
from .roi import choose_rois, chapter_axis
from .signal import compute, build_a2, velocity, to_position, normalise, enhance, post_enhance, keyframes
from .output import write_funscript


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wmv", ".flv", ".ts", ".mpg", ".mpeg"}


def find_videos(folder, recursive):
    """Videos in a folder sorted by name, ignoring hidden files."""
    found = []
    if recursive:
        for root, dirs, files in os.walk(folder):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            found += [os.path.join(root, f) for f in sorted(files)]
    else:
        found = [os.path.join(folder, f) for f in sorted(os.listdir(folder))]
    return [p for p in found
            if os.path.isfile(p) and not os.path.basename(p).startswith(".")
            and os.path.splitext(p)[1].lower() in VIDEO_EXTS]


def chapters_file(video):
    return os.path.splitext(video)[0] + ".chapters.json"


def load_chapters(source, video=None):
    """List of chapters [{start, zoom, region}] from 'auto' (file next to the video),
    a .json, or a text string 'T@ZOOM[@REGION];T@ZOOM...' with '-' for none.
    Returns [] if there are no chapters."""
    if not source:
        return []
    if source == "auto":
        path = chapters_file(video) if video else None
        source = path if path and os.path.exists(path) else None
        if not source:
            return []
    if os.path.isfile(source):
        with open(source, encoding="utf-8") as f:
            raw = json.load(f).get("chapters", [])
    else:
        raw = []
        for part in source.split(";"):
            bits = part.split("@")
            if not bits[0].strip():
                continue
            get = lambda i: (bits[i].strip() if len(bits) > i and bits[i].strip() not in ("", "-") else None)
            raw.append({"start": float(bits[0]), "zoom": get(1), "region": get(2),
                        "invert": (get(3) or "").lower() in ("i", "inv", "1", "true"),
                        "axis_ref": get(4)})
    chs = sorted(({"start": float(c.get("start", 0)), "zoom": c.get("zoom") or None, "region": c.get("region") or None,
                   "invert": bool(c.get("invert")), "excluded": bool(c.get("excluded")),
                   "axis_ref": c.get("axis_ref") or None,
                   "axis": c.get("axis") or None, "axis_angle": c.get("axis_angle"),
                   "why": c.get("why") or None}
                  for c in raw), key=lambda c: c["start"])
    if chs:
        chs[0]["start"] = 0.0
    return chs


def chapter_args(a, chs, i):
    """Copy of arguments for chapter i (its own time, zoom, region and cache)."""
    b = copy.copy(a)
    c = chs[i]
    if len(chs) > 1:
        end = chs[i + 1]["start"] if i + 1 < len(chs) else None
        b.start = c["start"]
        b.duration = (end - c["start"]) if end is not None else 0.0
        b.cache_tag = f".ch{i}"
    else:
        b.cache_tag = ""
    b.zoom, b.region = c.get("zoom"), c.get("region")
    if c.get("axis_ref") is not None:      # the chapter's axis-ref (e.g. an angle) overrides the global
        b.axis_ref = str(c["axis_ref"])
    if c.get("axis") is not None:          # chapter's forced axis mode overrides the global
        b.axis = str(c["axis"])
        if c["axis"] == "angle" and c.get("axis_angle") is not None:
            b.axis_angle = float(c["axis_angle"])
    # chapter invert toggles the global (--invert), does not replace it
    b.invert = bool(a.invert) != bool(c.get("invert"))
    return b


def limit_speed(pts, a):
    out = []
    for t, p in pts:
        if out:
            t0, p0 = out[-1]
            maxd = a.max_speed * (t - t0) / 1000.0
            if abs(p - p0) > maxd:
                p = p0 + np.sign(p - p0) * maxd
        out.append((t, p))
    return out


_FULL = threading.local()       # each thread (batch video) stores its own: multiple chapters slice the same


def load_full(fa):
    """Flow for the whole video (keeps the last one in memory: multiple chapters slice the same)."""
    p = cache_path(fa)
    key = (p, os.path.getmtime(p), cache_key(fa, *probe(fa.video)[:2]))
    if getattr(_FULL, "key", None) != key:
        _FULL.val = load_or_extract(fa)
        _FULL.key = key
    return _FULL.val


def _full_args(b):
    fa = copy.copy(b)
    fa.start, fa.duration, fa.cache_tag = 0.0, 0.0, ""
    return fa


def flow_available(b):
    """Is there flow for these arguments without computing anything? Chapter's own or a slice of the full video."""
    if cache_valid(b):
        return True
    return bool((b.start or b.duration) and cache_valid(_full_args(b)))


def flow_for(b, progress=None, cached_only=False):
    """Flow for a chapter (b comes from chapter_args). Uses the chapter's own cache and, if absent,
    slices the whole-video cache with the same zoom (what automatic detection computes), and only
    computes as a last resort. cached_only returns None instead of computing."""
    if cache_valid(b):
        return load_or_extract(b, progress)
    if b.start or b.duration:
        fa = _full_args(b)
        if cache_valid(fa):
            dx, dy, mag = load_full(fa)
            T = len(mag)
            f0 = min(T, int(round(b.start * b.fps)))
            f1 = min(T, int(round((b.start + b.duration) * b.fps))) if b.duration else T
            dx, dy, mag = dx[f0:f1].copy(), dy[f0:f1].copy(), mag[f0:f1].copy()
            if f0 > 0 and len(mag):
                dx[0] = dy[0] = mag[0] = 0       # first flow is at the cut (between scenes), meaningless
            return dx, dy, mag
    if cached_only:
        return None
    return load_or_extract(b, progress)


def process_chapters(a, chs, progress=None):
    """Each chapter is processed independently (its own region, its own axis-ref) and keyframes are merged in time.

    A chapter does not need its own flow if the whole-video flow already exists with the same zoom
    (flow_for slices it). Only chapters with a different zoom compute their own.
    Returns pts when single-axis, (pts, pts2) when dual_axis."""
    pts, pts2 = [], []
    dual = getattr(a, "dual_axis", False)
    for i in range(len(chs)):
        if chs[i].get("excluded"):
            log(f"chapter {i + 1}/{len(chs)}: excluded from save")
            continue
        b = chapter_args(a, chs, i)
        log(f"chapter {i + 1}/{len(chs)}: from {b.start:.1f}s, zoom {b.zoom or 'none'}, "
            f"100 towards {b.axis_ref}")
        dx, dy, mag = flow_for(b, progress)
        if len(mag) < 2:
            continue
        r = compute(b, dx, dy, mag, a2=build_a2(b))
        end_ms = (chs[i + 1]["start"] * 1000.0) if i + 1 < len(chs) else float("inf")
        pts += [(t, p) for t, p in r["pts"] if t < end_ms]
        if dual and "pts2" in r:
            pts2 += [(t, p) for t, p in r["pts2"] if t < end_ms]
    pts.sort(key=lambda q: q[0])
    if dual:
        pts2.sort(key=lambda q: q[0])
        return limit_speed(pts, a), limit_speed(pts2, a)
    return limit_speed(pts, a)


def _angle_diff(t1, t2):
    """Difference between two axes (angles in radians, modulo pi), in degrees, 0 to 90."""
    d = abs(((t1 - t2) + np.pi / 2) % np.pi - np.pi / 2)
    return float(np.degrees(d))


def _window_features(dx, dy, mag, s, e, a):
    """Axis (angle modulo pi), axis strength, centre and energy of strong motion in a window. None if still."""
    w = mag[s:e].astype(np.float32) ** 2
    mx, my = dx[s:e].astype(np.float32), dy[s:e].astype(np.float32)
    W = w.sum(0)
    if W.max() <= 0:
        return None
    hot = W >= np.percentile(W, 90)
    sxx = float((w * mx * mx).sum(0)[hot].sum())
    syy = float((w * my * my).sum(0)[hot].sum())
    sxy = float((w * mx * my).sum(0)[hot].sum())
    tot = sxx + syy
    energy = float(np.sqrt(W[hot].mean() / max(1, e - s)))
    if energy < 4 * a.noise or tot <= 1e-9:
        return None
    lam = np.sqrt((sxx - syy) ** 2 + 4 * sxy ** 2)
    strength = lam / tot
    if strength < getattr(a, "chap_strength", 0.25):
        return None
    gy, gx = W.shape
    ys, xs = np.mgrid[0:gy, 0:gx]
    Wh = np.where(hot, W, 0.0)
    cx = float((Wh * xs).sum() / Wh.sum()) / gx
    cy = float((Wh * ys).sum() / Wh.sum()) / gy
    return 0.5 * np.arctan2(2 * sxy, sxx - syy), strength, cx, cy


def find_cuts(mag, fps, a):
    """Frames where there is a cut: flow between two unrelated scenes produces a spike in mean motion."""
    from scipy.ndimage import median_filter
    m = mag.reshape(len(mag), -1).astype(np.float32).mean(1)
    k = max(3, int(fps))
    med = median_filter(m, size=2 * k + 1, mode="nearest")
    cand = np.flatnonzero((m >= getattr(a, "cut_min", 0.5)) &
                          (m >= getattr(a, "cut_ratio", 4.0) * np.maximum(med, 1e-3)))
    cand = cand[cand >= 1]
    cuts, grp = [], []
    for i in cand:
        if grp and i - grp[-1] > max(2, fps // 3):
            cuts.append(grp[int(np.argmax(m[grp]))])
            grp = []
        grp.append(int(i))
    if grp:
        cuts.append(grp[int(np.argmax(m[grp]))])
    return cuts


def find_soft(mag, fps, a, sens):
    """Soft transitions (fades, dissolves, slow scene changes): where the motion map no longer resembles
    the one from the previous seconds, with both sides stable and showing motion.

    Compares the mean motion map (magnitude per cell) from chap_soft_span seconds before and
    after each moment. Sensitivity (0 to 1) lowers the difference threshold."""
    from scipy.ndimage import uniform_filter1d
    T = len(mag)
    wl = max(1, int(0.5 * fps))
    n = max(2, int(round(getattr(a, "chap_soft_span", 2.0) / 0.5)))
    K = T // wl
    if K < 2 * n + 1:
        return []
    E = mag[:K * wl].astype(np.float32).reshape(K, wl, -1).mean(1)          # (K, cells)
    thr = 0.9 - 0.7 * min(max(sens, 0.0), 1.0)
    act = E.mean(1) > 4 * a.noise

    def corr(u, v):
        u, v = u - u.mean(), v - v.mean()
        d = float(np.sqrt((u * u).sum() * (v * v).sum()))
        return float((u * v).sum() / d) if d > 1e-9 else 1.0

    h = n // 2
    D = np.zeros(K)
    for k in range(n, K - n + 1):
        if not (act[k - n:k].all() and act[k:k + n].all()):
            continue                                      # no motion on one side: not a scene change
        L, R = E[k - n:k], E[k:k + n]
        stab = min(corr(L[:h].mean(0), L[h:].mean(0)), corr(R[:h].mean(0), R[h:].mean(0)))
        if stab < 0.3:
            continue
        D[k] = 1.0 - corr(L.mean(0), R.mean(0))
    out = []
    for k in range(n, K - n + 1):
        if D[k] > thr and D[k] >= D[max(0, k - n):k + n].max() - 1e-9:
            out.append(k * wl)
    return out


def detect_chapters(dx, dy, mag, fps, a):
    """Automatic chapters: cuts, motion axis changes and zone changes.

    Each chapter gets its own axis-ref (an angle in screen degrees, computed from the region
    it will use) and inherits zoom and region from the video. Settings (a.chap_*):
    chap_by (cuts,angle,zone), chap_angle (degrees), chap_shift (fraction of frame),
    chap_persist (s), chap_min (s), chap_win (s), cut_ratio, cut_min."""
    T = len(mag)
    by = set(str(getattr(a, "chap_by", "cuts,angle,zone,soft")).split(","))
    # general sensitivity (0 to 1, 0.5 = settings as-is): above 0.5 all thresholds drop (more chapters), below they rise
    sens = min(max(getattr(a, "chap_sens", 0.5), 0.0), 1.0)
    fac = 2.0 ** (1.0 - 2.0 * sens)
    ang_t = getattr(a, "chap_angle", 35.0) * fac
    shift_t = getattr(a, "chap_shift", 0.30) * fac
    persist = getattr(a, "chap_persist", 8.0) * fac
    cmin = getattr(a, "chap_min", 10.0) * fac
    wl = max(2, int(getattr(a, "chap_win", 1.0) * fps))

    ca = copy.copy(a)
    ca.cut_ratio = 1.0 + (getattr(a, "cut_ratio", 4.0) - 1.0) * fac
    cuts = find_cuts(mag, fps, ca) if "cuts" in by else []
    marks = [(c, "cut") for c in cuts]
    if "soft" in by:
        soft_s = min(max(getattr(a, "chap_soft", 0.5) + sens - 0.5, 0.0), 1.0)
        marks += [(f, "soft") for f in find_soft(mag, fps, a, soft_s)
                  if all(abs(f - c) > 2.0 * getattr(a, "chap_soft_span", 2.0) * fps for c in cuts)]
    seg = [0] + cuts + [T]
    if "angle" in by or "zone" in by:
        for k in range(len(seg) - 1):
            ref = None                       # (c2, s2, cx, cy): axis as double angle, and centre
            run = []
            for s in range(seg[k], seg[k + 1], wl):
                e = min(seg[k + 1], s + wl)
                if e - s < wl // 2:
                    break
                f = _window_features(dx, dy, mag, s, e, a)
                if f is None:
                    continue
                th, st, cx, cy = f
                if ref is None:
                    ref = [np.cos(2 * th), np.sin(2 * th), cx, cy]
                    continue
                rth = 0.5 * np.arctan2(ref[1], ref[0])
                d_ang = _angle_diff(th, rth)
                d_ctr = float(np.hypot(cx - ref[2], cy - ref[3]))
                what = ("axis" if ("angle" in by and d_ang > ang_t) else
                        "zone" if ("zone" in by and d_ctr > shift_t) else None)
                if what is None:
                    run = []
                    al = 0.2                 # reference tracks slow drift
                    ref = [(1 - al) * ref[0] + al * np.cos(2 * th), (1 - al) * ref[1] + al * np.sin(2 * th),
                           (1 - al) * ref[2] + al * cx, (1 - al) * ref[3] + al * cy]
                    continue
                if run:
                    pth, pcx, pcy = run[0][2:5]
                    if _angle_diff(th, pth) > ang_t / 2 or np.hypot(cx - pcx, cy - pcy) > shift_t / 2:
                        run = []             # not a stable change, restart
                run.append((s, e, th, cx, cy, what))
                if (run[-1][1] - run[0][0]) / fps >= persist and len(run) >= 2:
                    marks.append((run[0][0], run[0][5]))
                    ref = [np.cos(2 * th), np.sin(2 * th), cx, cy]
                    run = []
    # chapters too short are merged: a cut wins over an axis/zone change
    marks.sort()
    kept = []
    for f, why in marks:
        need = 2.0 if why == "cut" else cmin       # a cut is always a cut; drift must persist
        if f / fps < need or (T - f) / fps < need:
            continue
        if kept and (f - kept[-1][0]) / fps < need:
            if why == "cut" and kept[-1][1] != "cut":
                kept[-1] = (f, why)
            continue
        kept.append((f, why))
    starts = [(0, "start")] + kept
    chs = []
    for i, (f, why) in enumerate(starts):
        f1 = starts[i + 1][0] if i + 1 < len(starts) else T
        c = {"start": round(f / fps, 2), "zoom": a.zoom, "region": a.region, "invert": False,
             "axis_ref": "chapter", "why": why}
        if f1 - f >= 5:
            sub = (dx[f:f1], dy[f:f1], mag[f:f1])
            try:
                wins, rois, rh, rw = choose_rois(*sub, [0, f1 - f], fps, a)
                u = chapter_axis(*sub, wins, rois, rh, rw, fps, a)
            except Exception:
                u = None
            if u is not None:
                c["axis_ref"] = str(int(round(np.degrees(np.arctan2(u[1], u[0])))) % 360)
        chs.append(c)
    return chs


def write_chapters_json(path, chs, auto=False):
    out = {"auto": bool(auto), "chapters": [{k: v for k, v in c.items() if v is not None} for c in chs]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


def resolve_chapters(a, load_flow):
    """Chapters to use: --chapters none, text or file, auto (<video>.chapters.json if it exists,
    otherwise detect), detect (always detect). load_flow() returns the full video flow.
    Returns (chapters, flow or None). The file is only written if it does not exist or was
    written by a detection (auto), so manually edited files are never overwritten."""
    src = getattr(a, "chapters", None) or "auto"
    if src == "none":
        return [], None
    path = chapters_file(a.video)
    if src == "auto" and os.path.exists(path):
        return load_chapters(path), None
    if src not in ("auto", "detect"):
        return load_chapters(src, a.video), None
    if a.start or a.duration:
        return [], None
    flow = load_flow()
    chs = detect_chapters(flow[0], flow[1], flow[2], a.fps, a)
    log("chapters detected: " + ", ".join(f"{c['start']:.1f}s ({c['why']}, 100 towards {c['axis_ref']})" for c in chs))
    if len(chs) > 1:
        mine = True
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    mine = bool(json.load(f).get("auto"))
            except Exception:
                mine = False
        if mine:
            write_chapters_json(path, chs, auto=True)
    return (chs if len(chs) > 1 else []), flow
