"""Stage 2a: region-of-interest selection and motion axis."""
import numpy as np
from scipy.signal import find_peaks, correlate

from .flow import log


def split_windows(bounds, wl):
    """Windows of about wl frames that never cross a scene boundary.

    Each tuple is (start, end, first_window_of_scene). A short leftover at the
    end of a scene is merged into the previous window.
    """
    out = []
    for k in range(len(bounds) - 1):
        s0, e0 = bounds[k], bounds[k + 1]
        first = len(out)
        for s in range(s0, e0, wl):
            out.append([s, min(e0, s + wl), s == s0])
        if len(out) - first > 1 and out[-1][1] - out[-1][0] < wl // 2:
            tail = out.pop()
            out[-1][1] = tail[1]
    return [tuple(w) for w in out]


def periodicity_score(dx, dy, mag, y0, x0, rh, rw, fps, a, ac_out=None):
    """Autocorrelation-based periodicity score for a region.

    Projects the motion in the region onto its dominant axis, then computes the
    normalised autocorrelation and returns the strength of the first peak in the
    funscript-relevant range (0.2 s to 4 s period).
    """
    sl = (slice(None), slice(y0, y0 + rh), slice(x0, x0 + rw))
    mx = dx[sl].astype(np.float64)
    my = dy[sl].astype(np.float64)
    mg = mag[sl].astype(np.float64)
    w = mg ** 2
    eps = (a.noise ** 2) * rh * rw
    denom = w.sum(axis=(1, 2)) + eps
    if a.axis == "y":
        u = (w * my).sum(axis=(1, 2)) / denom
    else:
        sxx = (w * mx * mx).sum()
        syy = (w * my * my).sum()
        sxy = (w * mx * my).sum()
        _, vecs = np.linalg.eigh(np.array([[sxx, sxy], [sxy, syy]]))
        ux, uy = vecs[:, 1]
        u = (w * (mx * ux + my * uy)).sum(axis=(1, 2)) / denom
    u = u - u.mean()
    std = u.std()
    if std < 1e-9:
        return 0.0
    u = u / std
    ac = correlate(u, u, mode="full")[len(u) - 1:]
    ac = ac / ac[0]
    # look for peaks between 0.2 s and 4 s
    lo = max(1, int(0.2 * fps))
    hi = min(len(ac) - 1, int(4.0 * fps))
    if hi <= lo:
        return 0.0
    peaks, props = find_peaks(ac[lo:hi + 1], height=0.0)
    lag = None
    if len(peaks):
        lag = lo + int(peaks[int(np.argmax(props["peak_heights"]))])
    if ac_out is not None:
        # for debug only: the full curve and the chosen peak position
        ac_out.append({"ac": ac, "lo": lo, "hi": hi, "lag": lag})
    if lag is None:
        return 0.0
    return float(props["peak_heights"].max())


def roi_size(gy, gx, frac, aspect=0.0):
    """Size (height, width) of the automatic region, in cells.

    aspect 0 inherits the video ratio (frac of height and frac of width). With aspect > 0
    (width/height, 1 = square, 2 = twice as wide) the region has the same area but that
    ratio, clamped to fit within the frame."""
    if not aspect or aspect <= 0:
        return max(2, int(round(gy * frac))), max(2, int(round(gx * frac)))
    area = gy * gx * frac * frac
    rh = np.sqrt(area / aspect)
    rw = rh * aspect
    if rw > gx:
        rw, rh = gx, area / gx
    if rh > gy:
        rh, rw = gy, area / gy
    return max(2, min(gy, int(round(rh)))), max(2, min(gx, int(round(rw))))


def choose_rois(dx, dy, mag, bounds, fps, a, dbg=None):
    """Picks the block of cells with the most oscillatory motion, per time window.

    Scores each candidate region by a combination of mean magnitude and
    periodicity (autocorrelation peak strength), so regions with rhythmic
    back-and-forth motion score higher than equally bright but random noise.
    """
    from scipy.signal import convolve2d
    T, gy, gx = mag.shape
    wl = max(1, int(a.roi_window * fps))
    windows = split_windows(bounds, wl)
    rois = []
    if a.region:
        # fixed region given by the user, as fractions of the frame
        x0, y0, x1, y1 = (float(v) for v in a.region.split(":"))
        cx0, cx1 = int(np.floor(x0 * gx)), int(np.ceil(x1 * gx))
        cy0, cy1 = int(np.floor(y0 * gy)), int(np.ceil(y1 * gy))
        cx0, cy0 = min(max(cx0, 0), gx - 1), min(max(cy0, 0), gy - 1)
        rw = max(1, min(cx1, gx) - cx0)
        rh = max(1, min(cy1, gy) - cy0)
        rois = [(cy0, cx0)] * len(windows)
        if dbg is not None:
            for k, (s, e, _) in enumerate(windows):
                dbg.append({"window": k, "start": s, "end": e, "fixed": True, "cands": [],
                            "E": mag[s:e].astype(np.float32).mean(axis=0),
                            "prev": None, "prev_score": 0.0, "threshold": 0.0, "now": None,
                            "best": (cy0, cx0), "best_score": 0.0, "switched": False,
                            "chosen": (cy0, cx0),
                            "chosen_info": {"mag": 0.0, "period": 0.0, "score": 0.0}})
        return windows, rois, rh, rw
    rh, rw = roi_size(gy, gx, a.roi, getattr(a, "roi_aspect", 0.0))
    kernel = np.ones((rh, rw))
    cur = None
    cur_score = 0.0
    for wi, (s, e, _) in enumerate(windows):
        E = mag[s:e].astype(np.float32).mean(axis=0)
        S_mag = convolve2d(E, kernel, mode="valid")
        # normalise magnitude map to [0, 1] so it mixes with the periodicity score
        S_mag_norm = S_mag / (S_mag.max() + 1e-9)
        # evaluate periodicity for the top-N magnitude candidates to avoid
        # computing autocorrelation for every cell in the grid
        n_cand = min(getattr(a, 'candidates', 6), S_mag_norm.size)
        flat = S_mag_norm.ravel()
        cand_idx = np.argpartition(flat, -n_cand)[-n_cand:]
        scores = {}
        info = []
        for fi in cand_idx:
            cy, cx = np.unravel_index(fi, S_mag_norm.shape)
            acs = [] if dbg is not None else None
            p = periodicity_score(dx[s:e], dy[s:e], mag[s:e], cy, cx, rh, rw, fps, a, acs)
            scores[(cy, cx)] = a.period_weight * p + (1.0 - a.period_weight) * float(S_mag_norm[cy, cx])
            if dbg is not None:
                info.append({"i": len(info), "y": int(cy), "x": int(cx),
                             "mag": float(S_mag_norm[cy, cx]), "period": float(p),
                             "score": float(scores[(cy, cx)]),
                             "ac": acs[0] if acs else None})
        best = max(scores, key=scores.__getitem__)
        best_score = scores[best]
        prev_cur, prev_score = cur, cur_score
        switched = cur is None or best_score > a.switch * cur_score
        if switched:
            cur = best
            cur_score = best_score
        rois.append((int(cur[0]), int(cur[1])))
        if dbg is not None:
            by_pos = {(c["y"], c["x"]): c for c in info}
            best_info = by_pos[(int(best[0]), int(best[1]))]
            now_info = None
            if prev_cur is not None:
                key = (int(prev_cur[0]), int(prev_cur[1]))
                now_info = by_pos.get(key)
                if now_info is None:
                    # the previous region is no longer among the top candidates: score it separately
                    acs = []
                    pp = periodicity_score(dx[s:e], dy[s:e], mag[s:e], key[0], key[1], rh, rw, fps, a, acs)
                    mm = float(S_mag_norm[key])
                    now_info = {"i": len(info), "y": key[0], "x": key[1], "mag": mm,
                                "period": float(pp),
                                "score": a.period_weight * float(pp) + (1.0 - a.period_weight) * mm,
                                "ac": acs[0] if acs else None, "extra": True}
                    info.append(now_info)
            chosen = best_info if switched else now_info

            def lean(c):
                return None if c is None else {k: x for k, x in c.items() if k != "ac"}

            dbg.append({"window": wi, "start": s, "end": e, "fixed": False, "cands": info,
                        "E": E, "prev": None if prev_cur is None else (int(prev_cur[0]), int(prev_cur[1])),
                        "prev_score": float(prev_score), "threshold": float(a.switch * prev_score),
                        "now": lean(now_info), "best": (int(best[0]), int(best[1])),
                        "best_score": float(best_score), "switched": bool(switched),
                        "chosen": (int(cur[0]), int(cur[1])), "chosen_info": lean(chosen)})
    return windows, rois, rh, rw


# Screen direction (x right, y down) that corresponds to position 100.
# "chapter" computes it from the chapter's own motion (see chapter_axis), "auto" is just
# continuity with the previous window (old behaviour). Neither has a fixed direction.
AXIS_REFS = {"chapter": None, "down": (0.0, 1.0), "up": (0.0, -1.0), "right": (1.0, 0.0),
             "left": (-1.0, 0.0), "auto": None}


def axis_vec(mode):
    """Reference vector for an axis-ref: a name from AXIS_REFS, or an angle in screen degrees
    (0 = right, 90 = down, 180 = left, 270 = up). None for chapter/auto/invalid."""
    if mode in AXIS_REFS:
        return AXIS_REFS[mode]
    try:
        r = np.radians(float(mode))
    except (TypeError, ValueError):
        return None
    return float(np.cos(r)), float(np.sin(r))


def chapter_axis(dx, dy, mag, windows, rois, rh, rw, fps=None, a=None):
    """Dominant axis of the whole chapter, with a stable conventional direction. None if no motion.

    Each window votes with its own axis (normalised covariance, so a strong window does not drown
    weak ones), weighted by periodicity: an object that merely passes or drifts (a distractor
    during a pause) is barely periodic and weighs little; rhythmic back-and-forth weighs a lot.
    The direction is whichever points upward, or leftward if the axis is nearly horizontal.
    It is just a stable convention: if the video goes the other way, the chapter's invert flag flips it.
    """
    sxx = syy = sxy = 0.0
    for (s, e, _), (y0, x0) in zip(windows, rois):
        sl = (slice(s, e), slice(y0, y0 + rh), slice(x0, x0 + rw))
        mx = dx[sl].astype(np.float64)
        my = dy[sl].astype(np.float64)
        w = mag[sl].astype(np.float64) ** 2
        cxx, cyy, cxy = (w * mx * mx).sum(), (w * my * my).sum(), (w * mx * my).sum()
        tr = cxx + cyy
        if tr < 1e-12:
            continue
        wt = 1.0
        if a is not None and fps:
            wt = max(float(periodicity_score(dx[s:e], dy[s:e], mag[s:e], y0, x0, rh, rw, fps, a)), 0.02) ** 2
        sxx += wt * cxx / tr
        syy += wt * cyy / tr
        sxy += wt * cxy / tr
    if sxx + syy < 1e-12:
        return None
    _, vecs = np.linalg.eigh(np.array([[sxx, sxy], [sxy, syy]]))
    ux, uy = float(vecs[0, 1]), float(vecs[1, 1])
    if (uy >= 0) if abs(uy) >= 0.3 else (ux >= 0):
        ux, uy = -ux, -uy
    return ux, uy
