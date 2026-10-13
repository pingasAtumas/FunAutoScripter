"""Video frame access: sequential during playback, async seek on large jumps."""
import threading
import cv2

MAX_PREVIEW_W = 1280   # preview is downscaled to at most this width


class FrameSource:
    """Access to the original video frames.

    Sequential reads (playback, small forward steps) happen on the calling thread
    and are fast. Large seeks (chapter transitions, timeline scrubbing) are handed
    off to a background thread so the UI never freezes. The last known frame is
    returned immediately; the display updates on the next render tick once the
    background thread delivers the result.

    Frames are downscaled to MAX_PREVIEW_W on read for performance.
    sw/sh reflect the downscaled dimensions.
    """

    def __init__(self, path):
        self._path = path
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        full_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        full_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        self.dur = (n / self.fps) if n > 0 else 1e9
        if full_w > MAX_PREVIEW_W:
            scale = MAX_PREVIEW_W / full_w
            self.sw = MAX_PREVIEW_W
            self.sh = max(1, int(full_h * scale))
        else:
            self.sw = full_w
            self.sh = full_h
        self.next = 0
        self.last = (-1, None)
        self._seek_wanted = -1
        self._seek_result = None  # (idx, frame, cap2)

    def _scale(self, frame):
        if frame is None:
            return None
        if frame.shape[1] != self.sw or frame.shape[0] != self.sh:
            return cv2.resize(frame, (self.sw, self.sh), interpolation=cv2.INTER_AREA)
        return frame

    @property
    def loading(self):
        # True until get() picks up the result and resets _seek_wanted
        return self._seek_wanted != -1

    # ------------------------------------------------------------------
    def _do_seek(self, idx):
        """Background thread: open a fresh cap, seek, decode one frame."""
        cap2 = cv2.VideoCapture(self._path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap2.read()
        if ok and self._seek_wanted == idx:
            self._seek_result = (idx, self._scale(f), cap2)
        else:
            cap2.release()

    # ------------------------------------------------------------------
    def get(self, t):
        idx = max(0, int(round(t * self.fps)))

        # Pick up a completed background seek.
        r = self._seek_result
        if r is not None and r[0] == idx:
            self._seek_result = None
            self._seek_wanted = -1   # clear flag — seek is done
            _, f, cap2 = r
            self.cap.release()
            self.cap = cap2       # already positioned just past idx
            self.next = idx + 1
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

        # Sequential read — fast, no blocking.
        while self.next < idx:
            if not self.cap.grab():
                break
            self.next += 1
        ok, f = self.cap.read()
        self.next += 1
        if ok:
            self.last = (idx, self._scale(f))
        return self.last[1]

    def close(self):
        self._seek_wanted = -1
        self.cap.release()
        r = self._seek_result
        if r:
            r[2].release()
        self._seek_result = None
