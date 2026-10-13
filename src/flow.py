"""Stage 1: ffmpeg decode + Farneback optical flow + cache."""
import copy
import json
import os
import subprocess
import threading
import sys
import time

import numpy as np
import cv2


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def _ff_bin(name):
    """Return path to ffmpeg/ffprobe: bundled binary when frozen by PyInstaller, else system PATH."""
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, name)
    return name


_PROBE = {}


def probe(path):
    """(width, height, duration) of the video. Caches result: the GUI calls this many times and
    each ffprobe costs tens of ms."""
    try:
        k = (path, os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        k = None
    if k is not None and k in _PROBE:
        return _PROBE[k]
    r = _probe(path)
    if k is not None:
        _PROBE[k] = r
    return r


def _probe(path):
    out = subprocess.check_output([
        _ff_bin("ffprobe"), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration",
        "-of", "json", path])
    j = json.loads(out)
    s = j["streams"][0]
    dur = float(j.get("format", {}).get("duration", 0) or 0)
    return int(s["width"]), int(s["height"]), dur


def crop_filter(spec, w, h):
    """Returns (ffmpeg crop filter or None, cropped width, cropped height)."""
    if not spec:
        return None, w, h
    if spec == "left":
        cw, ch, x, y = w // 2, h, 0, 0
    elif spec == "right":
        cw, ch, x, y = w // 2, h, w // 2, 0
    elif spec == "top":
        cw, ch, x, y = w, h // 2, 0, 0
    elif spec == "bottom":
        cw, ch, x, y = w, h // 2, 0, h // 2
    else:
        cw, ch, x, y = (int(v) for v in spec.split(":"))
    return f"crop={cw}:{ch}:{x}:{y}", cw, ch


def parse_zoom(spec):
    """X0:Y0:X1:Y1 as fractions (0 to 1) of the frame after --crop/--sbs/--tb."""
    try:
        x0, y0, x1, y1 = (float(v) for v in spec.split(":"))
    except ValueError:
        sys.exit("--zoom: use X0:Y0:X1:Y1 as fractions, e.g. 0.55:0.3:0.85:0.75")
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        sys.exit("--zoom: must be 0 <= X0 < X1 <= 1 and 0 <= Y0 < Y1 <= 1")
    return x0, y0, x1, y1


CACHE_VERSION = 2


def cache_key(a, sw, sh):
    # Only things that change the flow itself belong here. If you add a stage 1
    # parameter, add it to this list, otherwise the cache goes stale silently.
    return json.dumps([CACHE_VERSION, os.path.getsize(a.video), sw, sh, a.width,
                       a.cell, a.fps, a.crop, a.zoom, a.start, a.duration])


def decode_cmd(a, color=False):
    """ffmpeg command that decodes exactly the same frames as the flow.

    Returns (command, width, height, duration). With color=True frames come
    in BGR so the debug renderer can draw on top of the real image.
    """
    sw, sh, dur = probe(a.video)
    cf, cw, ch = crop_filter(a.crop, sw, sh)
    cell = a.cell
    pix = "bgr24" if color else "gray"
    chain = [cf] if cf else []
    if a.zoom:
        # the 192 analysis pixels cover only this zone instead of the full frame
        zx0, zy0, zx1, zy1 = parse_zoom(a.zoom)
        chain.append(f"crop=iw*{zx1 - zx0:.6f}:ih*{zy1 - zy0:.6f}:iw*{zx0:.6f}:ih*{zy0:.6f}")
        cw, ch = cw * (zx1 - zx0), ch * (zy1 - zy0)
    W = (a.width // cell) * cell
    H = max(cell, int(round(W * ch / cw / cell)) * cell)
    vf = chain + [f"fps={a.fps}", f"scale={W}:{H}:flags=area", f"format={pix}"]
    cmd = [_ff_bin("ffmpeg"), "-v", "error", "-nostdin"]
    hw = getattr(a, "hwaccel", None)
    if hw:
        cmd += ["-hwaccel", "auto" if hw is True else str(hw)]    # use hardware if available, fall back to software
    if a.start:
        cmd += ["-ss", str(a.start)]
    cmd += ["-i", a.video]
    if a.duration:
        cmd += ["-t", str(a.duration)]
    if getattr(a, "max_frames", 0):
        cmd += ["-frames:v", str(int(a.max_frames))]
    cmd += ["-an", "-sn", "-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", pix, "-"]
    return cmd, W, H, dur


class Cancelled(Exception):
    """progress() can raise this to abort flow extraction (the GUI uses it on the Cancel button)."""


class NoFlow(Exception):
    """With cache_only, the video would need to compute flow but no valid cache exists for these settings."""


def extract_flow(a, progress=None):
    if getattr(a, "cache_only", False):
        raise NoFlow("no cached flow for these settings")
    cmd, W, H, dur = decode_cmd(a)
    cell = a.cell
    gx, gy = W // cell, H // cell

    total = int((a.duration or max(dur - a.start, 0)) * a.fps)
    use_cl, why, dev = opencl_state(a)
    # With OpenCL, default is one chunk (one GPU). With explicit --jobs N, multiple chunks run on the
    # same GPU: Farneback issues many small kernels and the latency leaves headroom that this exploits.
    jobs = (max(1, min(int(getattr(a, "jobs", 0) or 1), 4)) if use_cl else pick_jobs(a, total))
    jobs = max(1, min(jobs, total // max(1, a.fps * 20))) if use_cl else jobs
    BACKEND["text"] = (f"OpenCL on {dev}" + (f", {jobs} chunks" if jobs > 1 else "") if use_cl else
                       f"CPU, {jobs} chunks in parallel" if jobs > 1 else "CPU, 1 thread")
    if why:
        BACKEND["text"] += f" (OpenCL requested but not used: {why})"
    log(f"decoding at {W}x{H}, {a.fps} fps, grid {gx}x{gy} (about {total} frames)")
    log("flow: " + BACKEND["text"] + (", hardware decoding requested" if getattr(a, "hwaccel", False) else ""))
    if jobs > 1:
        return _extract_parallel(a, jobs, total, gx, gy, progress, use_cl)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    t0 = time.time()
    try:
        return _extract_loop(a, proc, W, H, gx, gy, W * H, total, progress, t0, use_cl=use_cl)
    finally:
        if proc.poll() is None:
            proc.kill()          # cancelled or error: don't leave ffmpeg running
            proc.wait()


BACKEND = {"text": ""}          # what is computing the flow (shown by the GUI)
FIXED_THREADS = {"on": False}   # set by process_many: OpenCV thread count is global, must not be changed per video


def opencl_state(a):
    """(use OpenCL?, reason for not using, device description). Only tested when --opencl is set."""
    if not getattr(a, "opencl", False):
        return False, "", ""
    try:
        if not cv2.ocl.haveOpenCL():
            return False, "OpenCV found no OpenCL runtime, run --accel-test", ""
        cv2.ocl.setUseOpenCL(True)
        if not cv2.ocl.useOpenCL():
            return False, "OpenCV has no usable OpenCL GPU (missing driver?), run --accel-test", ""
        d = cv2.ocl.Device.getDefault()
        kind = {2: "CPU", 4: "GPU", 8: "accelerator"}.get(int(d.type()), "device")
        return True, "", f"{d.name()} ({kind})"
    except Exception as e:
        return False, f"error initialising OpenCL: {e}", ""


def accel_report(a=None):
    """Acceleration diagnostics: what OpenCV has, which devices it sees, and throughput per path.

    Returns lines of text (the CLI prints them, the GUI displays them)."""
    import shutil
    out = []
    P = out.append
    P(f"OpenCV {cv2.__version__}, OpenCV threads: {cv2.getNumThreads()}, CPU cores: {os.cpu_count()}")
    info = cv2.getBuildInformation()
    for i, line in enumerate(info.splitlines()):
        if line.strip().startswith("OpenCL:") or "Link libraries: Dynamic load" in line:
            P("build: " + line.strip())
    P(f"OpenCV finds OpenCL: {cv2.ocl.haveOpenCL()}")
    vend = os.environ.get("OCL_ICD_VENDORS", "")
    P(f"OCL_ICD_VENDORS = {vend or '(not set)'}")
    for d in [vend, "/run/opengl-driver/etc/OpenCL/vendors", "/etc/OpenCL/vendors"]:
        if d and os.path.isdir(d):
            P(f"  {d}: {', '.join(sorted(os.listdir(d))) or '(empty)'}")
    if shutil.which("clinfo"):
        try:
            r = subprocess.run(["clinfo", "-l"], capture_output=True, text=True, timeout=20)
            P("clinfo -l:\n  " + (r.stdout.strip().replace("\n", "\n  ") or "(no platforms)"))
        except Exception as e:
            P(f"clinfo failed: {e}")
    else:
        P("clinfo is not installed (in nix-shell it comes from shell.nix)")
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=20)
            P("nvidia-smi: " + (r.stdout.strip() or r.stderr.strip() or "(no output)"))
        except Exception as e:
            P(f"nvidia-smi failed: {e}")
    try:
        r = subprocess.run([_ff_bin("ffmpeg"), "-hide_banner", "-hwaccels"], capture_output=True, text=True, timeout=20)
        hw = [x.strip() for x in r.stdout.splitlines()[1:] if x.strip()]
        P("ffmpeg hardware decoding (--hwaccel): " + (", ".join(hw) or "none")
          + ("   (cuda = NVIDIA NVDEC)" if "cuda" in hw else ""))
    except Exception as e:
        P(f"ffmpeg -hwaccels failed: {e}")
    have = False
    try:
        have = cv2.ocl.haveOpenCL()
        if have:
            cv2.ocl.setUseOpenCL(True)
            if not cv2.ocl.useOpenCL():
                P("OpenCV loads OpenCL but no usable device found: by default it only uses GPUs. "
                  "Without an OpenCL GPU driver (see shell.nix and NixOS config) this is expected. "
                  "OPENCV_OPENCL_DEVICE=':CPU:' forces a CPU device, only for testing the code path.")
                have = False
        if have:
            d = cv2.ocl.Device.getDefault()
            kind = {2: "CPU", 4: "GPU", 8: "accelerator"}.get(int(d.type()), "?")
            P(f"default device: {d.name()} | {d.vendorName()} | type {kind} | "
              f"{d.maxComputeUnits()} units | {d.globalMemSize() // (1 << 20)} MB | driver {d.driverVersion()}")
            if kind != "GPU":
                P("WARNING: device is not a GPU, no benefit over CPU")
    except Exception as e:
        P(f"error reading OpenCL device: {e}")
    # benchmark: synthetic frames with motion, at the analysis width
    w = (getattr(a, "width", 192) // 12) * 12 if a is not None else 192
    h = max(12, int(round(w * 9 / 16 / 12)) * 12)
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.integers(0, 255, (h, w), dtype=np.uint8), (0, 0), 2)
    frames = [np.roll(base, k % 5, axis=1) for k in range(40)]
    args = (None, 0.5, 3, 15, 3, 5, 1.2, 0)

    def bench(fn, n=40):
        fn(0)
        t = time.time()
        for k in range(1, n):
            fn(k)
        return (n - 1) / (time.time() - t)

    f_cpu = lambda k: cv2.calcOpticalFlowFarneback(frames[k - 1], frames[k], *args)
    old = cv2.getNumThreads()
    cv2.setNumThreads(1)
    r1 = bench(f_cpu)
    cv2.setNumThreads(old)
    rn = bench(f_cpu)
    P(f"CPU, {w}x{h}: {r1:.0f} frames/s with 1 thread, {rn:.0f} with {old} OpenCV threads "
      f"(--jobs multiplies this further with parallel chunks)")
    if have:
        try:
            um = [cv2.UMat(f) for f in frames]
            t = time.time()
            cv2.calcOpticalFlowFarneback(um[0], um[1], *args).get()      # first call compiles kernels
            P(f"OpenCL: first call (kernel compilation, once only) {time.time() - t:.1f} s")
            f_cl = lambda k: cv2.calcOpticalFlowFarneback(um[k - 1], um[k], *args).get()
            rc = bench(f_cl)
            diff = float(np.abs(f_cl(5) - f_cpu(5)).max())
            P(f"OpenCL, {w}x{h}: {rc:.0f} frames/s (GPU including .get()), max diff vs CPU {diff:.3f} px")
            best = max(r1, rn)
            if rc > best * 1.15:
                P(f"RESULT: OpenCL is worth it, {rc / best:.1f}x the CPU on this test. Use --opencl.")
            else:
                P(f"RESULT: OpenCL not worth it here ({rc / best:.1f}x the CPU). Use --jobs on CPU.")
        except Exception as e:
            P(f"OpenCL failed in benchmark: {e}")
    else:
        P("RESULT: no OpenCL, --opencl always falls back to CPU. Use --jobs.")
    return out


def pick_jobs(a, total):
    """How many video chunks to compute in parallel (--jobs, 0 = auto).

    Each chunk is at least 20 s long, otherwise it's not worth the overhead."""
    j = int(getattr(a, "jobs", 0) or 0)
    if j <= 0:
        j = min(os.cpu_count() or 1, 6)
    return max(1, min(j, total // max(1, a.fps * 20)))


def _extract_parallel(a, jobs, total, gx, gy, progress, use_cl=False):
    """Same flow, split into time chunks computed in threads (OpenCV releases the GIL).

    Each chunk except the first starts one frame earlier and discards that frame, so the flow
    between the last frame of one chunk and the first of the next exists as in sequential mode."""
    Lf = -(-total // jobs)
    parts = []
    for i in range(jobs):
        f0 = i * Lf
        n = min(Lf, total - f0)
        if n <= 0:
            break
        pre = 1 if i else 0
        b = copy.copy(a)
        b.start = a.start + (f0 - pre) / a.fps
        b.duration = 0.0
        b.max_frames = n + pre
        parts.append((b, n + pre))
    res = [None] * len(parts)
    counts = [0] * len(parts)
    errors = []
    stop = threading.Event()

    def work(i):
        b, nexp = parts[i]
        cmd, W, H, _ = decode_cmd(b)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        try:
            res[i] = _extract_loop(b, proc, W, H, gx, gy, W * H, nexp,
                                   lambda n, t: counts.__setitem__(i, n), time.time(), quiet=True, stop=stop,
                                   use_cl=use_cl)
        except BaseException as e:           # includes SystemExit from a short chunk
            errors.append(e)
            stop.set()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    old = cv2.getNumThreads()
    if not FIXED_THREADS["on"]:
        cv2.setNumThreads(max(1, (os.cpu_count() or 1) // len(parts)))
    t0 = time.time()
    ths = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(len(parts))]
    try:
        for t in ths:
            t.start()
        while any(t.is_alive() for t in ths):
            time.sleep(0.1)
            if progress:
                try:
                    progress(sum(counts), total)
                except Cancelled:
                    stop.set()
                    raise
    finally:
        if any(t.is_alive() for t in ths):
            stop.set()                       # exited due to cancellation or error: stop other chunks
        for t in ths:
            t.join()
        if not FIXED_THREADS["on"]:
            cv2.setNumThreads(old)
    if errors:
        raise errors[0]
    dx = np.concatenate([r[0][1:] if i else r[0] for i, r in enumerate(res)])
    dy = np.concatenate([r[1][1:] if i else r[1] for i, r in enumerate(res)])
    mg = np.concatenate([r[2][1:] if i else r[2] for i, r in enumerate(res)])
    log(f"flow done: {len(mg)} frames in {time.time() - t0:.0f}s ({len(parts)} chunks)")
    return dx, dy, mg


def _extract_loop(a, proc, W, H, gx, gy, fsz, total, progress, t0, quiet=False, stop=None, use_cl=False):
    cell = a.cell
    prev = None
    if use_cl:
        cv2.ocl.setUseOpenCL(True)      # OpenCL state is per-thread
    DX, DY, MG = [], [], []
    n = 0
    while True:
        if stop is not None and stop.is_set():
            raise Cancelled()
        buf = proc.stdout.read(fsz)
        if len(buf) < fsz:
            break
        cur = np.frombuffer(buf, np.uint8).reshape(H, W)
        if prev is None:
            DX.append(np.zeros((gy, gx), np.float16))
            DY.append(np.zeros((gy, gx), np.float16))
            MG.append(np.zeros((gy, gx), np.float16))
        else:
            if use_cl:
                fl = cv2.calcOpticalFlowFarneback(cv2.UMat(prev), cv2.UMat(cur), None, 0.5, 3, 15, 3, 5, 1.2, 0).get()
            else:
                fl = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            cells = fl.reshape(gy, cell, gx, cell, 2).mean(axis=(1, 3))
            mag = np.sqrt((fl ** 2).sum(axis=2)).reshape(gy, cell, gx, cell).mean(axis=(1, 3))
            DX.append(cells[..., 0].astype(np.float16))
            DY.append(cells[..., 1].astype(np.float16))
            MG.append(mag.astype(np.float16))
        prev = cur
        n += 1
        if progress and n % 10 == 0:
            progress(n, total)
        if n % (a.fps * 30) == 0 and not quiet:
            el = time.time() - t0
            pct = f"{100 * n / total:.0f}%" if total else ""
            log(f"  {n} frames, {n / el:.0f} fps {pct}")
    proc.wait()
    if n < 10:
        sys.exit("could not decode enough frames, check the input file")
    if not quiet:
        log(f"flow done: {n} frames in {time.time() - t0:.0f}s")
    return np.array(DX), np.array(DY), np.array(MG)


def cache_path(a):
    """Flow cache file path. With chapters each has its own (a.cache_tag, e.g. .ch1)."""
    return a.output + getattr(a, "cache_tag", "") + ".flowcache.npz"


_VALID = {}


def cache_valid(a):
    """Is there a valid cache for these parameters? Only reads the file, computes nothing."""
    try:
        sw, sh, _ = probe(a.video)
        cache = cache_path(a)
        if not os.path.exists(cache) or a.no_cache:
            return False
        key = cache_key(a, sw, sh)
        st = os.stat(cache)
        mk = (cache, st.st_mtime, st.st_size, key)
        if mk not in _VALID:
            z = np.load(cache, allow_pickle=False)
            _VALID[mk] = str(z["key"]) == key and "dx" in z.files and "diff" not in z.files
        return _VALID[mk]
    except Exception:
        return False


def load_or_extract(a, progress=None):
    sw, sh, _ = probe(a.video)
    key = cache_key(a, sw, sh)
    cache = cache_path(a)
    if os.path.exists(cache) and not a.no_cache:
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key and "dx" in z.files and "diff" not in z.files:
            log(f"using cached flow ({cache})")
            return z["dx"], z["dy"], z["mag"]
    dx, dy, mag = extract_flow(a, progress)
    np.savez_compressed(cache, dx=dx, dy=dy, mag=mag, key=np.array(key))
    return dx, dy, mag
