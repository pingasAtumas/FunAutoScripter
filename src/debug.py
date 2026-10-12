"""Deep debug: six annotated videos, signals CSV and decisions JSON."""
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
from scipy.signal import butter, sosfreqz

from .flow import log, decode_cmd
from .signal import velocity, to_position, normalise, enhance, post_enhance, keyframes


PALETTE = [(60, 180, 255), (80, 220, 80), (255, 160, 60), (200, 90, 255),
           (60, 230, 230), (255, 90, 160), (160, 160, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def put(img, text, org, scale=0.45, color=(235, 235, 235), th=1):
    # black outline same thickness, offset 1 px. With higher thickness some OpenCV
    # versions widen the text and the outline becomes a ghost.
    for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.putText(img, text, (org[0] + ox, org[1] + oy), FONT, scale, (0, 0, 0), th, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, th, cv2.LINE_AA)


def box(img, r, col, th):
    cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), col, th)


def bar(img, x, y, w, h, frac, color):
    frac = min(max(float(frac), 0.0), 1.0)
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), 1)
    cv2.rectangle(img, (x + 1, y + 1), (x + 1 + int((w - 2) * frac), y + h - 1), color, -1)


class VideoOut:
    """Writes BGR frames to an H.264 mp4 via ffmpeg."""

    def __init__(self, path, w, h, fps):
        self.w, self.h = w, h
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
             "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE)

    def write(self, img):
        assert img.shape[:2] == (self.h, self.w), (img.shape, self.w, self.h)
        self.proc.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def draw_plot(w, h, t, win, title, series, vlines=(), dots=None):
    """Sliding graph that follows the video. The current frame is centred.

    series: list of (data, colour, name, min, max). min None = auto-scale to visible range.
    dots: (indices, values, colour, min, max).
    """
    img = np.full((h, w, 3), 24, np.uint8)
    a0, a1 = t - win // 2, t + win // 2

    def X(i):
        return ((np.asarray(i, np.float64) - a0) / (a1 - a0) * (w - 1)).astype(np.int32)

    for i, col in vlines:
        if a0 <= i <= a1:
            x = int(X(i))
            cv2.line(img, (x, 0), (x, h - 1), col, 1)
    legend = []
    for data, col, label, lo, hi in series:
        i0, i1 = max(0, a0), min(len(data), a1)
        cur = float(data[min(max(t, 0), len(data) - 1)])
        legend.append((label, col, cur))
        if i1 - i0 < 2:
            continue
        seg = data[i0:i1]
        if lo is None:
            lo, hi = float(seg.min()), float(seg.max())
            pad = (hi - lo) * 0.1 + 1e-9
            lo, hi = lo - pad, hi + pad
        if lo < 0 < hi:
            yz = int((1 - (0 - lo) / (hi - lo)) * (h - 1))
            cv2.line(img, (0, yz), (w - 1, yz), (60, 60, 60), 1)
        py = ((1 - (seg - lo) / (hi - lo)) * (h - 1)).clip(0, h - 1).astype(np.int32)
        pts = np.stack([X(np.arange(i0, i1)), py], 1).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, col, 1, cv2.LINE_AA)
    if dots is not None:
        di, dv, dcol, dlo, dhi = dots
        for i, val in zip(di, dv):
            if a0 <= i <= a1:
                y = int((1 - (val - dlo) / (dhi - dlo)) * (h - 1))
                cv2.circle(img, (int(X(i)), min(max(y, 0), h - 1)), 3, dcol, -1)
    cv2.line(img, (w // 2, 0), (w // 2, h - 1), (255, 255, 255), 1)
    put(img, title, (8, 16), 0.45)
    y = 34
    for label, col, cur in legend:
        put(img, f"{label}: {cur:+.2f}", (8, y), 0.4, col)
        y += 15
    return img


DEBUG_VIDEOS = ["1_roi_decision", "2_magnitude", "3_periodicity", "4_script", "5_velocity", "6_integral"]
DEBUG_ALIASES = {"roi": 0, "decision": 0, "magnitude": 1, "mag": 1, "periodicity": 2, "per": 2,
                 "script": 3, "velocity": 4, "vel": 4, "integral": 5}


def parse_debug_videos(spec):
    """Converts the --debug-videos value into the list of videos to generate."""
    spec = (spec or "all").strip().lower()
    if spec in ("all",):
        return list(DEBUG_VIDEOS)
    if spec in ("none",):
        return []
    sel = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok.isdigit() and 1 <= int(tok) <= len(DEBUG_VIDEOS):
            i = int(tok) - 1
        elif tok in DEBUG_ALIASES:
            i = DEBUG_ALIASES[tok]
        elif tok in DEBUG_VIDEOS:
            i = DEBUG_VIDEOS.index(tok)
        else:
            sys.exit(f"--debug-videos: '{tok}' does not exist. Use roi, magnitude, periodicity, "
                     "script, velocity, integral (or 1 to 6), all or none")
        if DEBUG_VIDEOS[i] not in sel:
            sel.append(DEBUG_VIDEOS[i])
    return sel


def render_debug(a, outdir, v, raw, p, pos, pts, windows, rois, grid, dbg, mag):
    """Writes to outdir six videos, a CSV with signals and a JSON with decisions.

    1_roi_decision.mp4     candidates, scores, threshold and decision per window
    2_magnitude.mp4        per-frame magnitude and the window average, cell by cell
    3_periodicity.mp4      periodicity of each candidate and autocorrelation curves
    4_script.mp4           the final funscript over the normalised signal
    5_velocity.mp4         axis velocity and funscript velocity
    6_integral.mp4         raw, filtered and spectrum with the filter response
    """
    os.makedirs(outdir, exist_ok=True)
    fps = a.fps
    T, gy, gx = mag.shape
    rh, rw = grid
    cmd, W, H, _ = decode_cmd(a, color=True)
    DH = a.debug_height - a.debug_height % 2
    DW = int(round(W * DH / H / 2)) * 2
    PH = 270
    cwid, chei = DW / gx, DH / gy
    win = max(30, int(a.debug_window * fps))
    nw = len(windows)

    # ---- per-frame signals ----
    widx = np.zeros(T, int)
    for k, (s, e, _) in enumerate(windows):
        widx[s:e] = k
    mag32 = mag.astype(np.float32)
    mag_roi = np.array([mag32[t, rois[widx[t]][0]:rois[widx[t]][0] + rh,
                              rois[widx[t]][1]:rois[widx[t]][1] + rw].mean() for t in range(T)])
    mag_all = mag32.mean(axis=(1, 2))
    ch_mag = np.array([dbg[k]["chosen_info"]["mag"] for k in widx], float)
    ch_per = np.array([dbg[k]["chosen_info"]["period"] for k in widx], float)
    ch_score = np.array([dbg[k]["chosen_info"]["score"] for k in widx], float)
    best_arr = np.array([dbg[k]["best_score"] for k in widx], float)
    tms = a.start * 1000.0 + np.arange(T) / fps * 1000.0
    if pts:
        fs = np.interp(tms, [x for x, _ in pts], [y for _, y in pts])
        kf_i = np.round((np.array([x for x, _ in pts]) - a.start * 1000.0) / 1000.0 * fps).astype(int)
        kf_v = np.array([y for _, y in pts])
    else:
        fs, kf_i, kf_v = np.zeros(T), np.array([], int), np.array([])
    fs_speed = np.gradient(fs) * fps
    vm = float(np.percentile(np.abs(v), 99)) + 1e-9
    mag_max = float(np.percentile(mag32[::5], 99)) + 1e-9
    mhi = float(max(mag_roi.max(), mag_all.max())) * 1.05 + 1e-9

    # ---- data files ----
    cols = ["t_s", "velocity", "raw_integral", "filtered_position", "signal_0_100",
            "funscript", "funscript_velocity", "mag_roi", "mag_total", "mag_chosen",
            "periodicity", "score"]
    M = np.column_stack([a.start + np.arange(T) / fps, v, raw, p, pos, fs, fs_speed,
                         mag_roi, mag_all, ch_mag, ch_per, ch_score])
    np.savetxt(os.path.join(outdir, "signals.csv"), M, delimiter=",", header=",".join(cols),
               comments="", fmt="%.5f")

    def clean(d):
        o = {k: x for k, x in d.items() if k not in ("E", "Smag")}
        o["start_s"] = d["start"] / fps + a.start
        o["end_s"] = d["end"] / fps + a.start
        cands = []
        for c in d["cands"]:
            c2 = {k: x for k, x in c.items() if k != "ac"}
            lag = c["ac"]["lag"] if c.get("ac") else None
            c2["period_s"] = None if lag is None else lag / fps
            cands.append(c2)
        o["cands"] = cands
        return o

    with open(os.path.join(outdir, "decisions.json"), "w") as f:
        json.dump([clean(d) for d in dbg], f, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # ---- drawing utilities ----
    def rect(y0, x0):
        return (int(x0 * cwid), int(y0 * chei), int((x0 + rw) * cwid), int((y0 + rh) * chei))

    def heat(m, vmax):
        x = np.clip(m.astype(np.float32) / max(vmax, 1e-9), 0, 1)
        x = cv2.resize(x, (DW, DH), interpolation=cv2.INTER_NEAREST)
        return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    def jet(x):
        c = cv2.applyColorMap(np.array([[int(255 * min(max(x, 0.0), 1.0))]], np.uint8), cv2.COLORMAP_JET)
        return tuple(int(q) for q in c[0, 0])

    C = {}

    def cached(kind, k, fn):
        # keeps only the last window of each type so memory does not grow with the video
        if kind not in C or C[kind][0] != k:
            C[kind] = (k, fn(k))
        return C[kind][1]

    vl = [(s, (70, 70, 70)) for s, _, _ in windows]
    vl += [(windows[d["window"]][0], (60, 60, 255)) for d in dbg if d["switched"]]

    # ---- panels that only change per window ----
    def roi_panel(k):
        d = dbg[k]
        img = np.full((DH, DW, 3), 24, np.uint8)
        put(img, f"Window {k + 1}/{nw}   {d['start'] / fps + a.start:.1f}s to {d['end'] / fps + a.start:.1f}s",
            (10, 20), 0.55)
        if d["fixed"]:
            put(img, "Fixed region (--region): no automatic decision", (10, 50), 0.5)
            return img
        put(img, f"score = {a.period_weight:.2f} x periodicity + {1 - a.period_weight:.2f} x magnitude",
            (10, 44), 0.42, (200, 200, 200))
        put(img, "M = mean magnitude / window max   P = autocorrelation peak",
            (10, 62), 0.4, (170, 170, 170))
        y = 86
        for c in sorted(d["cands"], key=lambda c: -c["score"]):
            col = PALETTE[c["i"] % len(PALETTE)]
            cv2.rectangle(img, (10, y - 10), (20, y), col, -1)
            tag = "*" if (c["y"], c["x"]) == tuple(d["chosen"]) else " "
            put(img, f"{tag}y{c['y']:>2} x{c['x']:>2}", (26, y), 0.4)
            for j, (lab, key, bc) in enumerate((("M", "mag", (60, 160, 255)),
                                                ("P", "period", (255, 220, 60)),
                                                ("S", "score", (80, 220, 80)))):
                bx = 120 + j * 170
                put(img, lab, (bx, y), 0.4, bc)
                bar(img, bx + 14, y - 10, 80, 10, c[key], bc)
                put(img, f"{c[key]:.2f}", (bx + 100, y), 0.4)
            y += 20
        y += 12
        if d["prev"] is None:
            put(img, "No previous region: using the best candidate.", (10, y), 0.45)
            y += 20
        else:
            put(img, f"Previous region y{d['prev'][0]} x{d['prev'][1]}: saved score "
                     f"{d['prev_score']:.2f}, now {d['now']['score']:.2f}", (10, y), 0.45)
            y += 20
            put(img, f"Switch threshold = {a.switch:.2f} x {d['prev_score']:.2f} = {d['threshold']:.2f}",
                (10, y), 0.45)
            y += 20
        put(img, f"Best candidate y{d['best'][0]} x{d['best'][1]}: {d['best_score']:.2f}", (10, y), 0.45)
        y += 24
        if d["switched"] and d["prev"] is not None:
            put(img, "SWITCHED region", (10, y), 0.7, (60, 255, 60), 2)
        elif d["switched"]:
            put(img, "INITIAL CHOICE", (10, y), 0.7, (60, 255, 60), 2)
        elif d["best"] == d["prev"]:
            put(img, "KEPT (already the best)", (10, y), 0.7, (0, 165, 255), 2)
        else:
            put(img, "KEPT (did not pass threshold)", (10, y), 0.7, (0, 165, 255), 2)
        return img

    def mag_panel(k):
        d = dbg[k]
        E = d["E"]
        img = heat(E, float(E.max()))
        for yy in range(gy):
            for xx in range(gx):
                put(img, f"{E[yy, xx]:.2f}", (int(xx * cwid) + 3, int((yy + 0.5) * chei) + 4), 0.3)
        for c in d["cands"]:
            box(img, rect(c["y"], c["x"]), PALETTE[c["i"] % len(PALETTE)], 1)
        box(img, rect(*d["chosen"]), (0, 255, 0), 3)
        put(img, f"window {k + 1} average (candidates are derived from this)", (8, 18), 0.5)
        return img

    def ac_panel(k):
        d = dbg[k]
        img = np.full((DH, DW, 3), 24, np.uint8)
        put(img, f"Autocorrelation per candidate (window {k + 1})", (10, 20), 0.5)
        acs = [c for c in d["cands"] if c.get("ac") is not None]
        if not acs:
            put(img, "no candidates with signal" if not d["fixed"] else "fixed region: no candidates",
                (10, 50), 0.5)
            return img
        L = len(acs[0]["ac"]["ac"])
        x0, x1, y0, y1 = 50, DW - 15, 40, DH - 45

        def mx(lag):
            return int(x0 + (x1 - x0) * lag / max(1, L - 1))

        def my(val):
            return int(y1 - (y1 - y0) * (val + 1.0) / 2.0)

        lo, hi = acs[0]["ac"]["lo"], acs[0]["ac"]["hi"]
        cv2.rectangle(img, (mx(lo), y0), (mx(hi), y1), (60, 40, 40), -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (70, 70, 70), 1)
        cv2.line(img, (x0, my(0)), (x1, my(0)), (70, 70, 70), 1)
        for sec in range(0, int((L - 1) / fps) + 1):
            put(img, f"{sec}s", (mx(sec * fps) - 8, y1 + 18), 0.4, (170, 170, 170))
        for c in acs:
            col = PALETTE[c["i"] % len(PALETTE)]
            ac = c["ac"]["ac"]
            line = np.array([[mx(i), my(ac[i])] for i in range(L)], np.int32).reshape(-1, 1, 2)
            thick = 2 if (c["y"], c["x"]) == tuple(d["chosen"]) else 1
            cv2.polylines(img, [line], False, col, thick, cv2.LINE_AA)
            lag = c["ac"]["lag"]
            if lag is not None:
                cv2.circle(img, (mx(lag), my(ac[lag])), 4, col, -1)
        put(img, "red band = periods 0.2 s to 4 s; circle = peak (value = periodicity)",
            (10, DH - 10), 0.4, (170, 170, 170))
        return img

    # ---- spectrum (integral video) ----
    nyq = fps / 2.0
    lp = min(a.lowpass, nyq * 0.9)
    sos = butter(2, [a.highpass, lp], btype="band", fs=fps, output="sos")
    fgrid, hresp = sosfreqz(sos, worN=512, fs=fps)
    gain = np.abs(hresp) ** 2
    FMAX = 8.0

    def spec_panel(t):
        img = np.full((DH, DW, 3), 24, np.uint8)
        x0, x1, y0, y1 = 45, DW - 15, 40, DH - 45
        put(img, "Spectrum of raw integral (orange) and bandpass filter (white)", (10, 20), 0.5)
        cv2.rectangle(img, (x0, y0), (x1, y1), (70, 70, 70), 1)

        def fx(f_):
            return int(x0 + (x1 - x0) * min(f_, FMAX) / FMAX)

        def fy(g_):
            return int(y1 - (y1 - y0) * min(max(g_, 0.0), 1.0))

        for f_ in range(0, int(FMAX) + 1):
            cv2.line(img, (fx(f_), y1), (fx(f_), y1 + 4), (150, 150, 150), 1)
            put(img, f"{f_}Hz", (fx(f_) - 10, y1 + 18), 0.4, (170, 170, 170))
        m = fgrid <= FMAX
        gp = np.array([[fx(f_), fy(g_)] for f_, g_ in zip(fgrid[m], gain[m])], np.int32)
        cv2.polylines(img, [gp.reshape(-1, 1, 2)], False, (255, 255, 255), 1, cv2.LINE_AA)
        seg = raw[max(0, t - 64): t + 64]
        if len(seg) >= 32:
            seg = seg - seg.mean()
            sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=512))
            fr = np.fft.rfftfreq(512, 1.0 / fps)
            sp = sp / (sp.max() + 1e-12)
            mm = fr <= FMAX
            spp = np.array([[fx(f_), fy(s_)] for f_, s_ in zip(fr[mm], sp[mm])], np.int32)
            cv2.polylines(img, [spp.reshape(-1, 1, 2)], False, (60, 160, 255), 2, cv2.LINE_AA)
            band = (fr >= a.highpass) & (fr <= lp)
            if band.any():
                fd = float(fr[band][np.argmax(sp[band])])
                cv2.line(img, (fx(fd), y0), (fx(fd), y1), (80, 220, 80), 1)
                put(img, f"in-band peak: {fd:.2f} Hz (period {1 / fd:.2f} s)", (10, DH - 12),
                    0.45, (80, 220, 80))
        return img

    # ---- writers ----
    names = DEBUG_VIDEOS
    vids = {n: VideoOut(os.path.join(outdir, n + ".mp4"), 2 * DW, DH + PH, fps)
            for n in names if n in a.debug_sel}
    if not vids:
        log(f"data only, no videos: signals.csv and decisions.json in {outdir}")
        return

    def out(name, left, right, bottom):
        vids[name].write(np.vstack([np.hstack([left, right]), bottom]))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    fsz = W * H * 3
    t0 = time.time()
    n = 0
    log(f"generating debug videos in {outdir} ({DW * 2}x{DH + PH})")
    for t in range(T):
        buf = proc.stdout.read(fsz)
        if len(buf) < fsz:
            break
        small = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        base = cv2.resize(small, (DW, DH), interpolation=cv2.INTER_LINEAR)
        k = int(widx[t])
        d = dbg[k]
        roi = rect(*rois[k])
        stamp = f"t={t / fps + a.start:.2f}s  window {k + 1}/{nw}"

        # 1: region decision
        if names[0] in vids:
            f = base.copy()
            if not d["fixed"]:
                eh = cached("eh", k, lambda q: heat(dbg[q]["E"], float(dbg[q]["E"].max())))
                f = cv2.addWeighted(base, 0.65, eh, 0.35, 0)
                for c in d["cands"]:
                    r = rect(c["y"], c["x"])
                    col = PALETTE[c["i"] % len(PALETTE)]
                    box(f, r, col, 1)
                    put(f, f"{c['score']:.2f}", (r[0] + 3, r[1] + 13), 0.4, col)
                if d["prev"] is not None and tuple(d["prev"]) != tuple(d["chosen"]):
                    r = rect(*d["prev"])
                    box(f, r, (255, 255, 0), 2)
                    put(f, "previous", (r[0] + 3, r[3] - 5), 0.4, (255, 255, 0))
            box(f, roi, (0, 255, 0), 3)
            put(f, stamp, (8, 18), 0.5)
            pn = cached("rp", k, roi_panel).copy()
            bar(pn, 10, DH - 20, DW - 20, 8, (t - d["start"]) / max(1, d["end"] - d["start"]), (0, 255, 0))
            bot = draw_plot(2 * DW, PH, t, win, "Active region selection score (red line = switch)",
                            [(ch_mag, (60, 160, 255), "magnitude", 0, 1.05),
                             (ch_per, (255, 220, 60), "periodicity", 0, 1.05),
                             (ch_score, (80, 220, 80), "score", 0, 1.05),
                             (best_arr, (200, 200, 200), "best candidate", 0, 1.05)], vl)
            out(names[0], f, pn, bot)

        # 2: magnitude
        if names[1] in vids:
            f = cv2.addWeighted(base, 0.5, heat(mag32[t], mag_max), 0.5, 0)
            box(f, roi, (0, 255, 0), 3)
            put(f, stamp + "  this frame magnitude", (8, 18), 0.5)
            bot = draw_plot(2 * DW, PH, t, win, "Flow magnitude (px/frame)",
                            [(mag_all, (150, 150, 150), "full frame", 0, mhi),
                             (mag_roi, (60, 160, 255), "active region", 0, mhi)], vl)
            out(names[1], f, cached("mp", k, mag_panel), bot)

        # 3: periodicity
        if names[2] in vids:
            ov = base.copy()
            for c in d["cands"]:
                cv2.rectangle(ov, (rect(c["y"], c["x"])[0], rect(c["y"], c["x"])[1]),
                              (rect(c["y"], c["x"])[2], rect(c["y"], c["x"])[3]), jet(c["period"]), -1)
            f = cv2.addWeighted(base, 0.6, ov, 0.4, 0)
            for c in d["cands"]:
                r = rect(c["y"], c["x"])
                box(f, r, (255, 255, 255), 1)
                put(f, f"P {c['period']:.2f}", (r[0] + 3, r[1] + 13), 0.4)
            box(f, roi, (0, 255, 0), 3)
            put(f, stamp + "  colour = periodicity", (8, 18), 0.5)
            bot = draw_plot(2 * DW, PH, t, win, "Region signal and its periodicity",
                            [(v, (255, 160, 60), "velocity", -vm, vm),
                             (ch_per, (255, 220, 60), "periodicity", 0, 1.05),
                             (ch_mag, (60, 160, 255), "magnitude", 0, 1.05)], vl)
            out(names[2], f, cached("ap", k, ac_panel), bot)

        # 4: the script
        if names[3] in vids:
            f = base.copy()
            box(f, roi, (0, 255, 0), 3)
            put(f, stamp, (8, 18), 0.5)
            put(f, f"funscript {fs[t]:.0f}", (8, DH - 12), 0.7, (255, 255, 255), 2)
            r4 = np.full((DH, DW, 3), 24, np.uint8)
            gx0, gw, gy0, gh = DW // 2 - 40, 80, 30, DH - 60
            cv2.rectangle(r4, (gx0, gy0), (gx0 + gw, gy0 + gh), (90, 90, 90), 1)
            yv = gy0 + gh - int(gh * min(max(fs[t], 0), 100) / 100)
            cv2.rectangle(r4, (gx0 + 2, yv), (gx0 + gw - 2, gy0 + gh - 2), (80, 220, 80), -1)
            yr = gy0 + gh - int(gh * min(max(pos[t], 0), 100) / 100)
            cv2.line(r4, (gx0 - 20, yr), (gx0 + gw + 20, yr), (200, 200, 200), 2)
            put(r4, "100", (gx0 + gw + 25, gy0 + 5), 0.45, (170, 170, 170))
            put(r4, "0", (gx0 + gw + 25, gy0 + gh), 0.45, (170, 170, 170))
            put(r4, f"funscript (bar): {fs[t]:.0f}", (10, 20), 0.5, (80, 220, 80))
            put(r4, f"normalised signal (line): {pos[t]:.0f}", (10, 40), 0.5, (200, 200, 200))
            bot = draw_plot(2 * DW, PH, t, win, "Final script (white) over normalised signal (grey), dots = keyframes",
                            [(pos, (150, 150, 150), "signal", 0, 100), (fs, (255, 255, 255), "funscript", 0, 100)],
                            vl, (kf_i, kf_v, (80, 220, 80), 0, 100))
            out(names[3], f, r4, bot)

        # 5: velocity
        if names[4] in vids:
            f = base.copy()
            box(f, roi, (0, 255, 0), 3)
            ux, uy = d["axis"]
            cx, cy = (roi[0] + roi[2]) // 2, (roi[1] + roi[3]) // 2
            cv2.line(f, (int(cx - ux * 40), int(cy - uy * 40)), (int(cx + ux * 40), int(cy + uy * 40)), (255, 255, 255), 1)
            ln = 60 * float(np.clip(v[t] / vm, -1, 1))
            cv2.arrowedLine(f, (cx, cy), (int(cx + ux * ln), int(cy + uy * ln)), (0, 255, 255), 3, tipLength=0.3)
            put(f, stamp + "  arrow = axis velocity", (8, 18), 0.5)
            r5 = np.full((DH, DW, 3), 24, np.uint8)
            put(r5, f"axis velocity: {v[t]:+.3f} px/frame", (10, 30), 0.55)
            bar(r5, 10, 40, DW - 20, 12, abs(v[t]) / vm, (0, 255, 255))
            sp_frac = abs(fs_speed[t]) / a.max_speed
            put(r5, f"funscript velocity: {fs_speed[t]:+.0f} units/s (limit {a.max_speed:.0f})", (10, 85), 0.55)
            bar(r5, 10, 95, DW - 20, 12, sp_frac, (60, 60, 255) if sp_frac >= 0.98 else (80, 220, 80))
            put(r5, f"axis: ({ux:+.2f}, {uy:+.2f})  angle {np.degrees(np.arctan2(uy, ux)):.0f} degrees", (10, 140), 0.5)
            put(r5, f"region magnitude: {mag_roi[t]:.3f} px/frame", (10, 165), 0.5)
            h2 = PH // 2
            bot = np.vstack([
                draw_plot(2 * DW, h2, t, win, "Axis velocity in active region",
                          [(v, (0, 255, 255), "px/frame", -vm, vm)], vl),
                draw_plot(2 * DW, PH - h2, t, win, "Funscript velocity (red = limit)",
                          [(fs_speed, (80, 220, 80), "units/s", -1.3 * a.max_speed, 1.3 * a.max_speed),
                           (np.full(T, float(a.max_speed)), (60, 60, 255), "limit", -1.3 * a.max_speed, 1.3 * a.max_speed),
                           (np.full(T, -float(a.max_speed)), (60, 60, 255), "-limit", -1.3 * a.max_speed, 1.3 * a.max_speed)], vl)])
            out(names[4], f, r5, bot)

        # 6: integral
        if names[5] in vids:
            f = base.copy()
            box(f, roi, (0, 255, 0), 3)
            put(f, stamp, (8, 18), 0.5)
            h3 = PH // 3
            bot = np.vstack([
                draw_plot(2 * DW, h3, t, win, "1. raw integral of velocity (drift shows here)",
                          [(raw, (255, 160, 60), "raw", None, None)], vl),
                draw_plot(2 * DW, h3, t, win, f"2. after bandpass {a.highpass:.2f} to {lp:.1f} Hz",
                          [(p, (60, 160, 255), "filtered", None, None)], vl),
                draw_plot(2 * DW, PH - 2 * h3, t, win, "3. normalised 0 to 100",
                          [(pos, (80, 220, 80), "signal", 0, 100)], vl)])
            out(names[5], f, spec_panel(t), bot)

        n += 1
        if n % 300 == 0:
            el = time.time() - t0
            log(f"  {n}/{T} frames, {n / el:.0f} fps")
    proc.stdout.close()
    proc.wait()
    for vo in vids.values():
        vo.close()
    log(f"debug done: {n} frames in {time.time() - t0:.0f}s, files in {outdir}")
