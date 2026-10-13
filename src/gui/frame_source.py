"""Video frame access: ffmpeg for seeks, OpenCV for sequential reads."""
import threading
import subprocess
import numpy as np
import cv2


class FrameSource:
    """Access to the original video frames.

    Seeks use ffmpeg fast keyframe seek so the requested frame appears
    immediately without decoding the entire GOP. Sequential reads (playback,
    small forward steps) use OpenCV and are unaffected.
    """

    def __init__(self, path):
        self._path = path
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.sw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.sh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        self.dur = (n / self.fps) if n > 0 else 1e9
        self._frame_bytes = self.sw * self.sh * 3
        self.next = 0
        self.last = (-1, None)
        self._seek_wanted = -1
        self._seek_result = None   # (idx, frame, cap2)

    def _do_seek(self, idx):
        """Background: ffmpeg fast seek for the frame; OpenCV cap for subsequent sequential reads."""
        if self._seek_wanted != idx:
            return
        t = idx / self.fps
        frame = None
        try:
            r = subprocess.run(
                ["ffmpeg",
                 "-ss", f"{t:.6f}",
                 "-hwaccel", "none",               # force CPU decode, no GPU activity
                 "-i", self._path,
                 "-vframes", "1",
                 "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{self.sw}x{self.sh}",     # force exact size (handles rotation metadata)
                 "-autorotate", "0",
                 "-v", "quiet", "pipe:1"],
                capture_output=True, timeout=10,
            )
            if r.returncode == 0 and len(r.stdout) >= self._frame_bytes:
                frame = np.frombuffer(r.stdout[:self._frame_bytes], dtype=np.uint8).reshape(self.sh, self.sw, 3).copy()
        except Exception:
            pass

        if self._seek_wanted != idx:
            return

        # Reposition a fresh OpenCV cap for subsequent sequential reads.
        # This is the slow part, but it happens after the frame is already displayed.
        cap2 = cv2.VideoCapture(self._path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
        if frame is None:
            # ffmpeg failed; fall back to OpenCV for the frame itself
            ok, frame = cap2.read()
            if not ok:
                cap2.release()
                return
            cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)  # reset: read() advanced position

        if self._seek_wanted == idx:
            self._seek_result = (idx, frame, cap2)
        else:
            cap2.release()

    def get(self, t):
        idx = max(0, int(round(t * self.fps)))

        # Pick up a completed background seek.
        r = self._seek_result
        if r is not None and r[0] == idx:
            self._seek_result = None
            _, f, cap2 = r
            self.cap.release()
            self.cap = cap2       # positioned at idx; sequential reads start here
            self.next = idx
            self.last = (idx, f)
            return f

        if self.last[0] == idx:
            return self.last[1]

        needs_seek = idx < self.next or idx - self.next > 12
        if needs_seek:
            if self._seek_wanted != idx:
                self._seek_wanted = idx
                threading.Thread(target=self._do_seek, args=(idx,), daemon=True).start()
            return self.last[1]

        # Sequential read — fast, no seeking.
        while self.next < idx:
            if not self.cap.grab():
                break
            self.next += 1
        ok, f = self.cap.read()
        self.next += 1
        if ok:
            self.last = (idx, f)
        return self.last[1]

    def close(self):
        self._seek_wanted = -1
        self.cap.release()
        r = self._seek_result
        if r:
            r[2].release()
        self._seek_result = None
