"""Video frame access: ffmpeg pipe with producer thread.

ffmpeg decodes and scales in one pass (hardware-accelerated if available),
outputting small frames into a queue. The main thread never blocks on decode.
Seeking restarts the pipe at the target position.
"""
import threading
import queue
import subprocess
import numpy as np
import cv2

PREVIEW_W = 1280
SEEK_PREBUFFER = 2.0   # fast-seek to (t - 2s), then accurate-seek 2s forward
QUEUE_SIZE = 8


class FrameSource:
    def __init__(self, path):
        self._path = path
        cap = cv2.VideoCapture(path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        full_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        self.dur = (n / self.fps) if n > 0 else 1e9
        cap.release()
        if full_w > PREVIEW_W:
            self.sw = PREVIEW_W
            self.sh = max(2, int(full_h * PREVIEW_W / full_w) & ~1)
        else:
            self.sw, self.sh = full_w, full_h
        self._frame_bytes = self.sw * self.sh * 3
        self._queue = queue.Queue(maxsize=QUEUE_SIZE)
        self._stop = threading.Event()
        self._gen = 0          # incremented on each restart to discard stale frames
        self._loading = False
        self.last = (-1, None)

    # ------------------------------------------------------------------
    def _open_pipe(self, idx):
        t = idx / self.fps
        prebuf = min(t, SEEK_PREBUFFER)
        cmd = [
            "ffmpeg",
            "-ss", f"{t - prebuf:.6f}",    # fast seek
            "-i", self._path,
            "-ss", f"{prebuf:.6f}",         # accurate seek from keyframe
            "-vf", f"scale={self.sw}:{self.sh}",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-v", "quiet", "pipe:1",
        ]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)

    def _read_frame(self, proc):
        try:
            raw = proc.stdout.read(self._frame_bytes)
        except Exception:
            return None
        if len(raw) != self._frame_bytes:
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.sh, self.sw, 3).copy()

    def _kill(self, proc):
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass

    def _start_pipe(self, idx):
        """Signal the current producer to stop and start a new one at idx."""
        self._stop.set()
        self._gen += 1
        gen = self._gen
        # Drain stale frames
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._stop.clear()
        self._loading = True
        proc = self._open_pipe(idx)
        threading.Thread(target=self._producer, args=(proc, idx, gen),
                         daemon=True).start()

    def _producer(self, proc, start_idx, gen):
        n = start_idx
        first = True
        try:
            while not self._stop.is_set():
                frame = self._read_frame(proc)
                if frame is None:
                    break
                if first:
                    self._loading = False
                    first = False
                while not self._stop.is_set():
                    try:
                        self._queue.put((gen, n, frame), timeout=0.05)
                        n += 1
                        break
                    except queue.Full:
                        pass
        finally:
            self._kill(proc)

    # ------------------------------------------------------------------
    @property
    def loading(self):
        return self._loading

    def get(self, t):
        idx = max(0, int(round(t * self.fps)))

        if self.last[0] == idx:
            return self.last[1]

        last_idx = self.last[0]
        needs_restart = (not self._loading and self._gen == 0) or \
                        (not self._loading and
                         (idx < last_idx or idx - last_idx > 60))
        if needs_restart:
            self._start_pipe(idx)
            return self.last[1]

        cur_gen = self._gen
        while True:
            try:
                g, qidx, frame = self._queue.get_nowait()
                if g != cur_gen:
                    continue           # stale frame from previous pipe
                if qidx >= idx:
                    self.last = (qidx, frame)
                    return frame
                # qidx < idx: behind, discard and keep reading
            except queue.Empty:
                return self.last[1]

    def close(self):
        self._stop.set()
        # Producer threads are daemon; they will exit and kill their proc
