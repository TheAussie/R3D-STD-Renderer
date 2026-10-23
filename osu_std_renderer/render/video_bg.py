"""Background-video decode — §4.10 LoadVideos on the R3D `load_video`
preset key. This is a PORT of the house implementation, mania v2's
video_bg.py (OsuManiaRenderer_v2/osu_mania_renderer_v2/video_bg.py) —
owner-directed: mania's is the reference; the only adaptations are the
plain `ffmpeg` prefix (no _ffmpeg_prefix() indirection here) and stderr
prints instead of the mania logger.

osu! maps can ship a background video (the ``Video,<start>,"file"`` event,
beatmap.video/video_offset). When enabled we play it behind gameplay,
decoding through ffmpeg — already a hard dependency for output encoding —
scaling+cropping each frame to *cover* the canvas (the same framing as the
static background) and resampling to the render fps, then reading raw RGBA
frames over a pipe.

A small reader thread pulls frames into a bounded queue so ffmpeg decodes a
few frames ahead while the GPU draws the previous one. The render loop
calls :meth:`frame_for` once per output frame with that frame's MAP time;
the decoder advances 1:1 at 1× (both sides run at the same fps after the
``fps`` filter). Under rate mods the map clock advances `speed`× per output
frame, so the loop drains proportionally more source frames — the video
fast-forwards exactly like stable under DT.

Returns ``None`` before the video's start offset, when decode failed, or
once it ends — the caller then falls back to the static background image
(matching osu!, which drops back to the bg image when the video finishes).
Every failure path is fail-soft: a missing/corrupt video must never kill a
render.
"""
from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path

from osu_std_renderer.security import ffmpeg_file_input_args

# Frames buffered ahead of the render loop. Enough to overlap decode with
# the GPU draw without hoarding memory — a 4K RGBA frame is ~33 MB, so 4
# frames is ~130 MB worst case at 2160p, a few MB at 720p. (mania value.)
_QUEUE_DEPTH = 4


class VideoBackground:
    """Streaming RGBA decode of a beatmap background video, frame-indexed
    by map time. One per render; call :meth:`close` when the render ends."""

    def __init__(self, path: Path, *, width: int, height: int, fps: int,
                 start_ms: int) -> None:
        self.width = width
        self.height = height
        self.fps = max(1, fps)
        self.start_ms = start_ms
        self.frame_size = width * height * 4
        self._q: queue.Queue[bytes | None] = queue.Queue(maxsize=_QUEUE_DEPTH)
        self._eof = False
        self._failed = False
        self._cur_idx = -1            # index of the frame currently held
        self._cur_bytes: bytes | None = None
        self._frames_seen = 0

        # cover = fill the canvas preserving aspect, crop the overflow (the
        # static-bg path does the identical thing via cover_size). fps=
        # resamples the source to the render rate so frame N here == render
        # frame N once the video has started (at 1× rate).
        vf = (f"scale={width}:{height}:force_original_aspect_ratio=increase,"
              f"crop={width}:{height},fps={self.fps},format=rgba")

        try:
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
                   *ffmpeg_file_input_args(path), "-an", "-sn", "-vf", vf,
                   "-f", "rawvideo", "-pix_fmt", "rgba", "-"]

            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, bufsize=0)

        except (OSError, ValueError) as e:
            print(f"WARNING: background video unavailable ({e}) — static "
                  "background", file=sys.stderr)
            self._proc = None
            self._failed = True
            return
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    # ── decode thread ──
    def _reader(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        stdout = self._proc.stdout
        fs = self.frame_size
        try:
            while True:
                buf = self._read_exact(stdout, fs)
                if buf is None:
                    break
                self._q.put(buf)      # blocks when full → backpressures ffmpeg
        finally:
            self._q.put(None)         # EOF sentinel

    @staticmethod
    def _read_exact(stream, n: int) -> bytes | None:
        """Read exactly n bytes (a full frame) or None at EOF/short read."""
        chunks: list[bytes] = []
        got = 0
        while got < n:
            chunk = stream.read(n - got)
            if not chunk:
                return None
            chunks.append(chunk)
            got += len(chunk)
        return b"".join(chunks)

    # ── render-loop API ──
    def active_at(self, t_ms: float) -> bool:
        """True once the video has started (and decode hasn't failed)."""
        return (not self._failed) and t_ms >= self.start_ms

    def frame_for(self, t_ms: float) -> bytes | None:
        """RGBA bytes for the output frame at map time ``t_ms`` (top-left
        origin, width*height*4), or ``None`` before the video starts, on
        failure, before the first frame is available, or once the video
        has ended (the caller then shows the static background image)."""
        if self._failed or t_ms < self.start_ms:
            return None
        target = int(round((t_ms - self.start_ms) * self.fps / 1000.0))
        while self._cur_idx < target and not self._eof:
            try:
                # Small timeout so a stalled decode can't wedge the render
                # loop forever; on timeout we just reuse the current frame.
                buf = self._q.get(timeout=30.0)
            except queue.Empty:
                print(f"WARNING: video decode stall at frame "
                      f"{self._cur_idx}", file=sys.stderr)
                break
            if buf is None:
                self._eof = True
                break
            self._cur_bytes = buf
            self._cur_idx += 1
            self._frames_seen += 1
        # Video has run out of frames and the song has moved past its end →
        # let the static background take over rather than freezing on the
        # last frame. (A transient decode stall keeps the current frame.)
        if self._eof and self._cur_idx < target:
            return None
        return self._cur_bytes

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                self._proc.kill()
        except Exception:  # noqa: BLE001
            pass
        # Drain so a reader blocked on a full queue can finish and exit.
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        if self._frames_seen == 0 and not self._failed:
            print("WARNING: video decode produced no frames",
                  file=sys.stderr)
