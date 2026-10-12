"""Stages 2b-6: velocity, position, normalisation, enhancement, keyframes, and compute()."""
import copy

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt, find_peaks

from .flow import log
from .roi import choose_rois, chapter_axis, axis_vec


def velocity(dx, dy, mag, bounds, fps, a, dbg=None):
    T = len(mag)
    windows, rois, rh, rw = choose_rois(dx, dy, mag, bounds, fps, a, dbg)
    v = np.zeros(T)
    eps = (a.noise ** 2) * rh * rw
    prev_axis = None
    mode = getattr(a, "axis_ref", "chapter")
    cref = chapter_axis(dx, dy, mag, windows, rois, rh, rw, fps, a) if (mode == "chapter" and a.axis != "y") else None
    for wi, ((s, e, _), (y0, x0)) in enumerate(zip(windows, rois)):
        sl = (slice(s, e), slice(y0, y0 + rh), slice(x0, x0 + rw))
        mx = dx[sl].astype(np.float64)
        my = dy[sl].astype(np.float64)
        mg = mag[sl].astype(np.float64)
        w = mg ** 2
        if a.axis == "y":
            ux, uy = 0.0, 1.0
        else:
            sxx = (w * mx * mx).sum()
            syy = (w * my * my).sum()
            sxy = (w * mx * my).sum()
            _, vecs = np.linalg.eigh(np.array([[sxx, sxy], [sxy, syy]]))
            ux, uy = vecs[:, 1]
            # PCA gives only the axis; the direction is arbitrary. With --axis-ref the positive
            # direction (towards 100) is the one closest to that reference, window by window, so
            # an error in one window does not propagate. Only when the axis is nearly perpendicular
            # to the reference (|cos| < --axis-margin) is direction ambiguous and follows the
            # previous window. With --axis-ref auto it is just continuity (old behaviour).
            ref = cref if mode == "chapter" else axis_vec(mode)
            d = ux * ref[0] + uy * ref[1] if ref else 0.0
            if ref and abs(d) >= getattr(a, "axis_margin", 0.35):
                if d < 0:
                    ux, uy = -ux, -uy
            elif prev_axis is None:
                if (d < 0) if ref else (uy < 0 or (abs(uy) < 1e-6 and ux < 0)):
                    ux, uy = -ux, -uy
            elif ux * prev_axis[0] + uy * prev_axis[1] < 0:
                ux, uy = -ux, -uy
            flipped = prev_axis is not None and ux * prev_axis[0] + uy * prev_axis[1] < 0
            prev_axis = (ux, uy)
            if dbg is not None:
                dbg[wi]["axis_flip"] = bool(flipped)
                dbg[wi]["axis_ref"] = None if ref is None else (float(ref[0]), float(ref[1]))
        if dbg is not None:
            dbg[wi]["axis"] = (float(ux), float(uy))
        u = mx * ux + my * uy
        v[s:e] = (w * u).sum(axis=(1, 2)) / (w.sum(axis=(1, 2)) + eps)
    return v, windows, rois, (rh, rw)


def to_position(v, fps, a):
    """Integrates velocity and applies a band-pass filter."""
    nyq = fps / 2.0
    lp = min(a.lowpass, nyq * 0.9)
    sos = butter(2, [a.highpass, lp], btype="band", fs=fps, output="sos")
    if len(v) < max(16, int(0.5 * fps)):
        return np.zeros(len(v))
    return sosfiltfilt(sos, np.cumsum(v), padtype="constant")


def normalise(p, fps, a):
    """Scales to 0..100 with a rolling percentile window."""
    T = len(p)
    hop = max(1, int(fps * 0.5))
    half = max(1, int(a.norm_window * fps / 2))
    centers = np.arange(0, T, hop)
    lo = np.empty(len(centers))
    hi = np.empty(len(centers))
    for i, c in enumerate(centers):
        seg = p[max(0, c - half): min(T, c + half + 1)]
        lo[i], hi[i] = np.percentile(seg, [a.low_pct, a.high_pct])
    r = hi - lo
    r = r[r > 1e-9]
    floor = a.floor * (np.median(r) if len(r) else 0.0)
    ctr = (hi + lo) / 2
    rng = np.maximum(hi - lo, floor)
    idx = np.arange(T)
    lo_f = np.interp(idx, centers, ctr - rng / 2)
    hi_f = np.interp(idx, centers, ctr + rng / 2)
    return np.clip((p - lo_f) / np.maximum(hi_f - lo_f, 1e-9), 0, 1) * 100.0


def post_enhance(pos, fps, a):
    """High-frequency enhancement after scaling: cuts low frequencies, boosts high, re-normalises.

    Useful for small fast strokes hidden inside a larger slow oscillation. With the default
    values (all 0) it is a no-op.
    """
    use_hp, use_hf, use_rn = a.hp2 > 0, a.hfboost > 0, a.renorm >= 0.5
    if not (use_hp or use_hf or use_rn) or len(pos) < max(16, int(0.5 * fps)):
        return pos
    nyq = fps / 2.0
    x = np.asarray(pos, dtype=float) - 50.0
    if use_hp:
        sos = butter(2, min(a.hp2, nyq * 0.9), btype="highpass", fs=fps, output="sos")
        x = sosfiltfilt(sos, x, padtype="constant")
    if use_hf:
        sos = butter(2, min(a.hf_freq, nyq * 0.9), btype="lowpass", fs=fps, output="sos")
        low = sosfiltfilt(sos, x, padtype="constant")
        x = low + (1.0 + a.hfboost) * (x - low)
    if use_rn:
        b = copy.copy(a)
        b.norm_window, b.floor = a.renorm, a.renorm_floor
        return normalise(x, fps, b)
    return np.clip(x + 50.0, 0.0, 100.0)


def enhance(pos, a):
    """Enhancement after scaling to 0..100. With default values it is a no-op."""
    if a.boost == 1.0 and a.gain == 1.0 and a.offset == 0.0:
        return pos
    x = (pos - 50.0) / 50.0
    if a.boost != 1.0:
        x = np.sign(x) * np.abs(x) ** (1.0 / max(a.boost, 1e-3))
    return np.clip(50.0 + a.offset + x * 50.0 * a.gain, 0.0, 100.0)


def keyframes(pos, fps, a):
    dist = max(1, int(a.min_interval * fps))
    pk, _ = find_peaks(pos, prominence=a.min_amp, distance=dist)
    vl, _ = find_peaks(-pos, prominence=a.min_amp, distance=dist)
    ext = sorted([(int(i), 1) for i in pk] + [(int(i), -1) for i in vl])
    alt = []
    for i, t in ext:
        if alt and alt[-1][1] == t:
            j = alt[-1][0]
            if (t == 1 and pos[i] > pos[j]) or (t == -1 and pos[i] < pos[j]):
                alt[-1] = (i, t)
        else:
            alt.append((i, t))
    idx = [i for i, _ in alt]
    pts = [(a.start * 1000.0 + i / fps * 1000.0, float(pos[i])) for i in idx]
    if not pts:
        return []
    res = [pts[0]]
    for t, p in pts[1:]:
        t0, p0 = res[-1]
        maxd = a.max_speed * (t - t0) / 1000.0
        if abs(p - p0) > maxd:
            p = p0 + np.sign(p - p0) * maxd
        res.append((t, p))
    return res


def compute(a, dx, dy, mag, dbg=None):
    """Steps 2 to 7 (fast): from the computed flow to keyframes.

    Writes nothing, so it works for both the CLI and the GUI.
    """
    fps = a.fps
    bounds = [0, len(mag)]
    v, windows, rois, grid = velocity(dx, dy, mag, bounds, fps, a, dbg)
    p = to_position(v, fps, a)
    ps = gaussian_filter1d(p, a.smooth * fps) if a.smooth > 0 else p
    pos = enhance(post_enhance(normalise(ps, fps, a), fps, a), a)
    if a.invert:
        pos = 100 - pos
    pts = keyframes(pos, fps, a)
    return {"v": v, "p": p, "pos": pos, "pts": pts, "windows": windows, "rois": rois, "grid": grid}
