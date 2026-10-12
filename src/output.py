"""Stage 7 output: write_funscript and debug plot."""
import json
import os

import numpy as np

from .flow import log


def write_funscript(path, pts):
    actions = []
    last_t = -1
    for t, p in pts:
        t = int(round(t))
        if t <= last_t:
            continue
        actions.append({"at": t, "pos": int(round(min(100, max(0, p))))})
        last_t = t
    doc = {"version": "1.0", "inverted": False, "range": 100,
           "metadata": {"creator": "FunAutoScripter", "description": "optical flow, no AI"},
           "actions": actions}
    with open(path, "w") as f:
        json.dump(doc, f)
    return len(actions)


def plot(path, v, p, pos, pts, rois, grid, mag, fps, a):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not installed, skipping --plot")
        return
    t = np.arange(len(pos)) / fps + a.start
    fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    ax[0].plot(t, v, lw=0.7)
    ax[0].set_ylabel("velocity (px/frame)")
    ax[1].plot(t, p, lw=0.7)
    ax[1].set_ylabel("position (band-passed)")
    ax[2].plot(t, pos, lw=0.7, color="0.6")
    ax[2].plot([x / 1000 for x, _ in pts], [y for _, y in pts], "o-", ms=3, lw=1)
    ax[2].set_ylabel("funscript 0-100")
    ax[2].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    log(f"plot saved to {path}")

    # where the motion is, and which regions were used
    from matplotlib.patches import Rectangle
    gy, gx = mag.shape[1:]
    rh, rw = grid
    fig, axr = plt.subplots(figsize=(8, 8 * gy / gx + 0.6))
    heat = mag.astype(np.float32).mean(axis=0)
    axr.imshow(heat, extent=[0, 1, 1, 0], cmap="magma", aspect="auto")
    for (y0, x0) in sorted(set(rois)):
        axr.add_patch(Rectangle((x0 / gx, y0 / gy), rw / gx, rh / gy,
                                fill=False, ec="cyan", lw=1.5))
    axr.set_title("average motion (bright = more), cyan = regions used")
    axr.set_xlabel("x (fraction of frame)")
    axr.set_ylabel("y (fraction of frame)")
    roi_path = os.path.splitext(path)[0] + ".roi.png"
    fig.tight_layout()
    fig.savefig(roi_path, dpi=110)
    plt.close(fig)
    log(f"region map saved to {roi_path}")
