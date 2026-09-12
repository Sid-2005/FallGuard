"""
video_clip_recorder.py – Saves a video clip around each fall event.

Strategy: rolling buffer of the last N seconds of frames.
When a fall is confirmed, flush the buffer to an MP4 file,
then continue recording for N more seconds (post-fall).

This gives you:
  - Pre-fall footage (what happened before)
  - The fall itself
  - Post-fall footage (person on floor)
"""

import cv2
import threading
import time
import os
import logging
from collections import deque
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)


class VideoClipRecorder:
    """
    Maintains a rolling frame buffer and saves clips on demand.

    Usage:
        recorder = VideoClipRecorder(output_dir="static/clips", pre_seconds=5, post_seconds=8)
        recorder.push_frame(bgr_frame)          # call every frame
        clip_path = recorder.save_clip("Ward1") # call when fall confirmed
    """

    def __init__(
        self,
        output_dir:   str   = "static/clips",
        pre_seconds:  int   = 5,
        post_seconds: int   = 8,
        fps:          float = 20.0,
        max_width:    int   = 640,
    ):
        self._output_dir  = output_dir
        self._pre_secs    = pre_seconds
        self._post_secs   = post_seconds
        self._fps         = fps
        self._max_width   = max_width

        # Rolling buffer: stores (timestamp, frame) tuples
        buf_size = int((pre_seconds + post_seconds + 2) * fps)
        self._buffer: deque = deque(maxlen=buf_size)
        self._lock   = threading.Lock()

        # Post-fall capture state
        self._capturing      = False
        self._capture_start  = 0.0
        self._capture_label  = ""
        self._pending_clips  = []   # list of clip paths being recorded

        os.makedirs(output_dir, exist_ok=True)
        log.info("VideoClipRecorder ready → %s", output_dir)

    # ── Public API ────────────────────────────────────────────

    def push_frame(self, frame):
        """Call every frame with the annotated BGR frame."""
        with self._lock:
            self._buffer.append((time.time(), frame.copy()))

    def save_clip(self, label: str = "") -> Optional[str]:
        """
        Trigger a clip save. Returns the output file path.
        Saves pre-fall buffer + continues capturing post-fall frames.
        """
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe  = label.replace(" ", "_").replace("/", "-") or "fall"
        fname = f"clip_{safe}_{ts}.mp4"
        fpath = os.path.join(self._output_dir, fname)

        # Snapshot current buffer (pre-fall frames)
        with self._lock:
            pre_frames = list(self._buffer)

        # Save in background thread
        thread = threading.Thread(
            target=self._write_clip,
            args=(fpath, pre_frames, label),
            daemon=True,
            name="FG-ClipWriter",
        )
        thread.start()
        log.info("Clip save triggered → %s", fpath)
        return fpath

    # ── Internal ──────────────────────────────────────────────

    def _write_clip(self, fpath: str, pre_frames: list, label: str):
        """Write pre-fall frames then wait for post-fall frames."""
        try:
            if not pre_frames:
                log.warning("No frames in buffer — clip skipped")
                return

            # Get frame dimensions from first frame
            sample = pre_frames[0][1]
            h, w   = sample.shape[:2]

            # Resize for output
            if w > self._max_width:
                scale = self._max_width / w
                w     = self._max_width
                h     = int(h * scale)

            # Try multiple codecs for compatibility
            out = None
            for codec, ext in [('mp4v', '.mp4'), ('XVID', '.avi')]:
                try:
                    actual_path = fpath if fpath.endswith(ext) else \
                                  fpath.replace('.mp4', ext)
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out    = cv2.VideoWriter(actual_path, fourcc,
                                            self._fps, (w, h))
                    if out.isOpened():
                        fpath = actual_path
                        break
                    out.release()
                    out = None
                except Exception:
                    continue

            if out is None:
                log.error("Could not open VideoWriter for %s", fpath)
                return

            # Write pre-fall frames
            for _, frame in pre_frames:
                resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)
                out.write(resized)

            # Continue writing post-fall frames for post_seconds
            deadline = time.time() + self._post_secs
            last_written_ts = pre_frames[-1][0] if pre_frames else 0

            while time.time() < deadline:
                # Get new frames added since we started
                with self._lock:
                    new_frames = [
                        (ts, f) for ts, f in self._buffer
                        if ts > last_written_ts
                    ]

                for ts, frame in new_frames:
                    resized = cv2.resize(frame, (w, h),
                                         interpolation=cv2.INTER_NEAREST)
                    out.write(resized)
                    last_written_ts = ts

                time.sleep(0.05)

            out.release()
            size_mb = os.path.getsize(fpath) / 1_000_000
            log.info("Clip saved: %s (%.1f MB)", fpath, size_mb)

        except Exception as e:
            log.error("Clip write error: %s", e, exc_info=True)
            try:
                if out:
                    out.release()
            except Exception:
                pass
