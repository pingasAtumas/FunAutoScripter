#!/usr/bin/env python3
"""
Optimize funautoscripter parameters against a folder of reference funscripts.

Usage:
    python tune.py folder/                  # tune default parameter set
    python tune.py folder/ -n Tuned         # save as preset "Tuned"
    python tune.py folder/ --starts 5       # 5 random starts (Gauss-Newton each)
    python tune.py folder/ --params switch,period_weight,highpass,lowpass
    python tune.py folder/ --list-params    # show available parameters

The folder must contain video files next to their .funscript references with the
same base name (video.mp4 + video.funscript).

The best parameter set is saved as a user preset that the GUI can load directly.
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

# project root on sys.path
sys.path.insert(0, os.path.dirname(__file__))
import funautoscripter as fl
from src.chapters import VIDEO_EXTS
from src.gui.config import presets_path, load_user_presets, clean_param, PARAM_BY_ATTR, PRESET_ATTRS

# ---------------------------------------------------------------------------
# Tunable parameters: (default, lo, hi)
# Only parameters that do NOT require flow recalculation.
# ---------------------------------------------------------------------------
ALL_PARAMS = {
    # region stability
    "switch":            (0.6,   0.1,   3.0),
    "period_weight":     (0.5,   0.0,   1.0),
    "roi_dist_weight":   (0.0,   0.0,   2.0),
    "roi_calm":          (0.0,   0.0,   1.0),
    "roi_period_floor":  (0.0,   0.0,   1.0),
    "roi_sine_weight":   (0.0,   0.0,   1.0),
    "roi_score_min":     (0.0,   0.0,   1.0),
    "roi_chapter_weight":(0.0,   0.0,   2.0),
    # signal
    "highpass":          (0.1,   0.01,  2.0),
    "lowpass":           (3.0,   0.5,  10.0),
    "norm_window":       (5.0,   1.0,  30.0),
    "floor":             (0.1,   0.01,  1.0),
    "min_amp":           (5.0,   1.0,  30.0),
    "min_interval":      (0.1,   0.05,  0.5),
    "max_speed":         (400.0, 100.0, 1000.0),
    # enhancement
    "smooth":            (0.0,   0.0,   1.0),
    "boost":             (1.0,   0.5,   3.0),
    "gain":              (1.0,   0.5,   2.0),
    "offset":            (0.0,  -20.0, 20.0),
    # background
    "bg_subtract":       (0.0,   0.0,   1.0),
    # bandpass
    "bp_q":              (0.0,   0.0,  10.0),
}

DEFAULT_PARAMS = [
    "switch", "period_weight",
    "highpass", "lowpass",
    "norm_window", "floor",
    "min_amp", "min_interval", "max_speed",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_pairs(folder):
    """Return list of (video_path, funscript_path) for all matched pairs."""
    pairs = []
    for fname in sorted(os.listdir(folder)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in VIDEO_EXTS:
            continue
        base = os.path.splitext(fname)[0]
        ref = os.path.join(folder, base + ".funscript")
        if os.path.exists(ref):
            pairs.append((os.path.join(folder, fname), ref))
    return pairs


def load_reference(path):
    """Return sorted list of {at, pos} actions from a .funscript file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    actions = data.get("actions", [])
    return sorted(actions, key=lambda a: a["at"])


def actions_to_signal(actions, t_ms):
    """Interpolate funscript actions to a dense time grid (ms). Returns np.array."""
    if not actions:
        return np.full(len(t_ms), 50.0)
    times = np.array([a["at"] for a in actions], dtype=float)
    pos   = np.array([a["pos"] for a in actions], dtype=float)
    idx   = np.argsort(times)
    times, pos = times[idx], pos[idx]
    f = interp1d(times, pos, kind="linear", bounds_error=False,
                 fill_value=(pos[0], pos[-1]))
    return f(t_ms)


def correlation(pts, ref_actions):
    """Pearson correlation between generated keyframes and reference funscript."""
    if not pts or not ref_actions:
        return 0.0
    t_lo = max(pts[0][0],  ref_actions[0]["at"])
    t_hi = min(pts[-1][0], ref_actions[-1]["at"])
    if t_hi - t_lo < 500.0:
        return 0.0
    t = np.arange(t_lo, t_hi, 50.0)
    gen_actions = [{"at": t_ms, "pos": p} for t_ms, p in pts]
    gen = actions_to_signal(gen_actions, t)
    ref = actions_to_signal(ref_actions, t)
    std_g, std_r = gen.std(), ref.std()
    if std_g < 1e-6 or std_r < 1e-6:
        return 0.0
    return float(np.corrcoef(gen, ref)[0, 1])


def make_args(base_a, param_names, x_norm, param_specs):
    """Return a copy of base_a with tunable parameters set from normalised x."""
    a = copy.deepcopy(base_a)
    for name, v_norm in zip(param_names, x_norm):
        _, lo, hi = param_specs[name]
        setattr(a, name, float(lo + v_norm * (hi - lo)))
    return a


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def build_objective(videos, param_names, param_specs):
    """
    Returns a residuals function f(x_norm) -> residuals for least_squares.

    Each residual is (1 - corr_i) for video i, so minimising ||r||² maximises
    the sum of squared correlations.
    """
    call_count = [0]

    def residuals(x_norm):
        call_count[0] += 1
        r = []
        for base_a, (dx, dy, mag), ref_actions in videos:
            a = make_args(base_a, param_names, x_norm, param_specs)
            try:
                result = fl.compute(a, dx, dy, mag)
                corr = correlation(result["pts"], ref_actions)
            except Exception:
                corr = 0.0
            r.append(1.0 - corr)
        mean_corr = 1.0 - float(np.mean(r))
        sys.stderr.write(f"\r  eval {call_count[0]:>4}  mean corr {mean_corr:.4f}   ")
        sys.stderr.flush()
        return np.array(r)

    return residuals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default=".", help="folder containing video + .funscript pairs")
    ap.add_argument("-n", "--name", default="Tuned",
                    help="preset name to save (default: Tuned)")
    ap.add_argument("--params", default=",".join(DEFAULT_PARAMS),
                    help="comma-separated list of parameters to tune")
    ap.add_argument("--starts", type=int, default=3,
                    help="number of random starting points (Gauss-Newton from each)")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--list-params", action="store_true",
                    help="print available tunable parameters and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the best values without saving the preset")
    # parse_known_args so any extra flags (--sbs, --zoom, --width, --fps, etc.)
    # are forwarded transparently to the pipeline's own argument parser
    args, pipeline_extra = ap.parse_known_args()

    if args.list_params:
        print("Available tunable parameters:")
        for name, (default, lo, hi) in ALL_PARAMS.items():
            marker = "*" if name in DEFAULT_PARAMS else " "
            print(f"  {marker} {name:<22}  default={default:.3g}  range=[{lo}, {hi}]")
        print("\n* = included in default --params set")
        return

    param_names = [p.strip() for p in args.params.split(",") if p.strip()]
    unknown = [p for p in param_names if p not in ALL_PARAMS]
    if unknown:
        sys.exit(f"unknown parameters: {unknown}  (run --list-params to see available)")

    # find video+script pairs
    pairs = find_pairs(args.folder)
    if not pairs:
        sys.exit(f"no video+funscript pairs found in {args.folder}")
    print(f"found {len(pairs)} video+script pair(s):")
    for v, s in pairs:
        print(f"  {os.path.basename(v)}  +  {os.path.basename(s)}")

    # load flow (uses existing cache; extracts if missing)
    # pipeline_extra contains any flags forwarded from the command line
    # (e.g. --sbs, --tb, --zoom, --width, --fps) that affect flow extraction
    if pipeline_extra:
        print(f"pipeline args: {' '.join(pipeline_extra)}")
    fl_parser = fl.build_parser()
    videos = []
    for video_path, ref_path in pairs:
        base_a = fl_parser.parse_args([video_path] + pipeline_extra)
        # output is used only to derive the flow cache path (<output>.flowcache.npz).
        # fl.load_or_extract only writes the .flowcache.npz file — never the funscript.
        # fl.compute is pure (returns keyframes, writes nothing). The reference funscript
        # is opened read-only by load_reference() and is never modified by tune.py.
        base_a.output = os.path.splitext(video_path)[0] + ".funscript"
        print(f"\nloading flow for {os.path.basename(video_path)} ...")
        dx, dy, mag = fl.load_or_extract(base_a)
        ref_actions = load_reference(ref_path)
        videos.append((base_a, (dx, dy, mag), ref_actions))
        print(f"  flow: {mag.shape}   ref: {len(ref_actions)} keyframes")

    # run optimization
    rng = np.random.default_rng(args.seed)
    param_specs = {n: ALL_PARAMS[n] for n in param_names}

    # starting point from defaults (normalised to [0,1])
    x0_default = np.array([
        (ALL_PARAMS[n][0] - ALL_PARAMS[n][1]) / (ALL_PARAMS[n][2] - ALL_PARAMS[n][1])
        for n in param_names
    ])

    obj = build_objective(videos, param_names, param_specs)
    best_cost = np.inf
    best_x = None

    starts = [x0_default] + [rng.uniform(0.0, 1.0, len(param_names))
                              for _ in range(args.starts - 1)]

    for i, x0 in enumerate(starts):
        print(f"\n--- start {i + 1}/{args.starts} ---")
        x0 = np.clip(x0, 0.0, 1.0)
        try:
            result = least_squares(
                obj,
                x0,
                bounds=(0.0, 1.0),
                method="trf",          # Trust Region Reflective (Gauss-Newton / LM)
                jac="3-point",         # numerical Jacobian (finite differences)
                ftol=1e-4,
                xtol=1e-4,
                gtol=1e-4,
                max_nfev=200 * len(param_names),
                verbose=0,
            )
        except Exception as e:
            print(f"\n  optimizer error: {e}")
            continue
        sys.stderr.write("\n")
        cost = float(result.cost)
        corrs = 1.0 - obj(result.x)
        mean_corr = float(np.mean(corrs))
        print(f"  cost={cost:.6f}  mean corr={mean_corr:.4f}  "
              f"per-video: {' '.join(f'{c:.3f}' for c in corrs)}")
        if cost < best_cost:
            best_cost = cost
            best_x = result.x.copy()

    sys.stderr.write("\n")
    if best_x is None:
        sys.exit("optimization failed on all starts")

    # decode best parameters
    print(f"\n=== best result  (mean corr = {1.0 - best_cost / len(videos):.4f}) ===")
    preset_vals = {}
    for name, v_norm in zip(param_names, best_x):
        _, lo, hi = param_specs[name]
        val = float(lo + v_norm * (hi - lo))
        default = ALL_PARAMS[name][0]
        marker = "  " if abs(val - default) < 1e-6 else " *"
        print(f"  {marker}{name:<22} = {val:.4g}  (default {default:.4g})")
        if name in PRESET_ATTRS:
            cleaned = clean_param(name, val)
            if cleaned is not None:
                preset_vals[name] = cleaned

    if not args.dry_run:
        path = presets_path()
        user_presets = load_user_presets()
        user_presets[args.name] = preset_vals
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(user_presets, f, ensure_ascii=False, indent=1)
        print(f"\npreset '{args.name}' saved to {path}")
        print("open the GUI and select it from the Presets menu.")
    else:
        print(f"\n(dry run — not saved)")


if __name__ == "__main__":
    main()
