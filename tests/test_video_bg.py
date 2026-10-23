"""render/video_bg.py — the mania v2 port: streaming ffmpeg decode,
frame indexing by map time, EOF → None (static bg takes over), fail-soft
spawn. Uses a tiny ffmpeg-generated clip (ffmpeg is a hard dependency of
the renderer, so the test environment always has it)."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from osu_std_renderer.render.video_bg import VideoBackground

W, H, FPS = 64, 48, 30


def _make_clip(tmp: Path, seconds: float = 1.0) -> Path:
    """A testsrc clip whose frames CHANGE over time (moving pattern)."""
    out = tmp / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=size={W}x{H}:rate={FPS}",
         "-t", f"{seconds}", "-pix_fmt", "yuv420p", str(out)],
        check=True)
    return out


def test_decode_frames_by_map_time():
    with tempfile.TemporaryDirectory() as td:
        clip = _make_clip(Path(td))
        v = VideoBackground(clip, width=W, height=H, fps=FPS, start_ms=500)
        try:
            # before the [Events] offset → inactive, no frame
            assert not v.active_at(499.0)
            assert v.frame_for(0.0) is None
            # at/after the offset → RGBA frames of the right size
            assert v.active_at(500.0)
            f0 = v.frame_for(500.0)
            assert f0 is not None and len(f0) == W * H * 4
            # later frames differ (testsrc animates)
            f1 = v.frame_for(1000.0)
            assert f1 is not None and f1 != f0
            # past the clip's end → None (static bg takes over)
            assert v.frame_for(500.0 + 5000.0) is None
        finally:
            v.close()


def test_missing_file_fails_soft():
    v = VideoBackground(Path("/nonexistent/video.mp4"),
                        width=W, height=H, fps=FPS, start_ms=0)
    try:
        # ffmpeg spawns but produces nothing → EOF → None, never a raise
        assert v.frame_for(0.0) is None
        assert v.frame_for(1000.0) is None
    finally:
        v.close()
    v.close()      # close is idempotent


def test_negative_offset_video_already_playing():
    """Maps with Video,-320: the video starts BEFORE audio t=0 — at map
    t=0 the decoder is already 320 ms in."""
    with tempfile.TemporaryDirectory() as td:
        clip = _make_clip(Path(td))
        v = VideoBackground(clip, width=W, height=H, fps=FPS,
                            start_ms=-320)
        try:
            assert v.active_at(0.0)
            f = v.frame_for(0.0)
            assert f is not None and len(f) == W * H * 4
            # the decoder consumed ~10 frames (320 ms at 30 fps) already
            assert v._cur_idx >= 8
        finally:
            v.close()


def test_unsupported_video_extension_fails_soft():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "video.bin"
        path.write_bytes(b"unsupported media")

        v = VideoBackground(
            path, width=W, height=H, fps=FPS, start_ms=0
        )
        try:
            assert not v.active_at(0)
            assert v.frame_for(0) is None
        finally:
            v.close()