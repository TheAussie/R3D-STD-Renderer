"""ffmpeg raw-RGB pipe — RENDER_PLAN.md §5.6 semantics on our stack; adapted
from the mania v2 encoder (OsuManiaRenderer_v2/osu_mania_renderer_v2/
encode.py: encoder probing order, rawvideo stdin input shape) simplified to
the synchronous single-process model the catch/taiko engines use.

Differences from the reference (§5.6) — deliberate:
  * ONE ffmpeg process, video on stdin + audio as a FILE input, instead of
    danser's two processes + mux step. Our audio is premixed offline
    (record/audio.py — NO BASS), so there is nothing to stream in
    lockstep; ffmpeg muxes in the same invocation.
  * Pixel path is rgb24 (renderer reads back RGB); the GPU RGB→YUV shader
    + PBO pool (§5.6 readback) is a later perf phase — mania v2's
    gpu/readback.py already proves it on this stack.
  * loudnorm (single-pass, I=-18:TP=-1.5, the 2026-07-31 audio directive)
    is applied to the audio filter chain like every in-house engine.
"""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from ..render import perf
from ..security import ffmpeg_file_input_args

LOUDNORM = "loudnorm=I=-18:TP=-1.5:LRA=11"

# ---- x264 knobs behind env hooks (parity row 16) ----------------------------
# taiko spells these R3D_X264_* and catch R3D_CATCH_X264_*; the un-prefixed form
# is the useful one for a fleet, so std takes that.
#
# DEFAULT CHANGED 2026-10-06: "veryfast" at crf 20 (was "faster" at crf 16).
# This is a deliberate change to what a node WITHOUT a hardware encoder ships;
# hardware-encoder nodes use the bitrate ladder below and are not affected.
# Measured on std's own frames (lossless reference, 4637 frames at 1080p60):
#   faster   crf 16   43.1 ms of encoder CPU per frame   9.58 Mbit/s   PSNR 47.3
#   veryfast crf 20   24.4 ms                            6.06 Mbit/s   PSNR 43.8
# and as a whole render of that replay with the inline preview: 40.6 -> 30.8 s,
# master 94 -> 60 MB. taiko already ships veryfast at crf 20.
# The old output is one env away: R3D_X264_PRESET=faster R3D_X264_CRF=16.
_X264_PRESET = os.environ.get("R3D_X264_PRESET", "veryfast")
_X264_CRF = os.environ.get("R3D_X264_CRF")       # None = use the caller's crf
_X264_PARAMS = os.environ.get("R3D_X264_PARAMS", "")
# Default libx264 (the no-HW-encoder fallback) to leave >=2 logical cores free,
# so a fallback encode on a contributor's own machine cannot spawn threads on
# every core at normal priority and freeze their desktop (the rel/Stella
# semi-crash). R3D_X264_THREADS overrides. Matches catch/taiko/mania; harmless
# on dedicated render boxes, which encode on the GPU anyway.
_X264_THREADS = os.environ.get(
    "R3D_X264_THREADS") or str(max(2, (os.cpu_count() or 4) - 2))


def _x264_extra() -> list:
    out = []
    if _X264_PARAMS:
        out += ["-x264-params", _X264_PARAMS]
    if _X264_THREADS:
        out += ["-threads", _X264_THREADS]
    return out


class EncoderError(RuntimeError):
    pass


def _encoder_usable(ffmpeg: str, encoder: str, device: str | None) -> tuple[bool, str]:
    """Open the driver and encode one RGB frame; a compiled encoder can still fail."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    if encoder == "h264_vaapi":
        cmd += ["-vaapi_device", device or "/dev/dri/renderD128"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "128x128", "-r", "1",
            "-i", "pipe:0", "-frames:v", "1", "-an", "-vf",
            "format=nv12,hwupload" if encoder == "h264_vaapi" else "format=yuv420p",
            "-c:v", encoder, "-f", "null", "-"]
    try:
        result = subprocess.run(cmd, input=bytes(128 * 128 * 3),
                                capture_output=True, timeout=5, check=False)
    except subprocess.TimeoutExpired:
        return False, "encoder startup probe timed out"
    except OSError as exc:
        return False, str(exc)
    if result.returncode == 0:
        return True, ""
    reason = result.stderr.decode(errors="replace").strip().splitlines()
    return False, reason[0][:500] if reason else f"ffmpeg exit code {result.returncode}"


def probe_encoder(encoder: str = "auto", device: str | None = None) -> str:
    """Resolve auto to a working encoder: NVENC, VAAPI, then libx264.

    Explicit choices stay explicit. Hardware must successfully encode a frame
    on this machine, using the same VAAPI device as the actual render.
    """
    if encoder != "auto":
        return encoder
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise EncoderError("ffmpeg not found on PATH")
    try:
        result = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                                capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise EncoderError(f"Unable to query ffmpeg encoders: {exc}") from exc
    if result.returncode != 0:
        raise EncoderError("Unable to query ffmpeg encoders: " + result.stderr[-1024:])
    available = {parts[1] for line in result.stdout.splitlines()
                 if len(parts := line.split()) >= 2}
    if device is None and Path("/dev/dri/renderD128").exists():
        device = "/dev/dri/renderD128"
    candidates = ["h264_nvenc"]
    if device is not None:
        candidates.append("h264_vaapi")
    candidates.append("libx264")
    failures = []
    for candidate in candidates:
        if candidate not in available:
            continue
        usable, reason = _encoder_usable(ffmpeg, candidate, device)
        if usable:
            print(f"encoder: auto selected {candidate}", file=sys.stderr, flush=True)
            return candidate
        failures.append(f"{candidate}: {reason}")
        print(f"encoder: {candidate} unavailable: {reason}", file=sys.stderr, flush=True)
    reason = "; ".join(failures) or "ffmpeg reports none of the supported encoders"
    raise EncoderError("No usable H.264 encoder: " + reason)


def nvenc_target_bps(w: int, h: int, fps: float) -> int:
    """Resolution-scaled NVENC bitrate ladder (R3D cross-engine policy, 2026-07).

    Replaces the flat per-engine bitrate: scale a 4 Mbps 720p30 reference
    by pixel rate with a perceptual exponent (0.70 -- deliberately NOT
    linear), clamped to [2.5, 16] Mbps.  Anchors: 720p30=4.0M,
    720p60=6.5M, 1080p30=7.1M, 1080p60=11.5M, 1440p60/1080p120+=16M cap.
    Callers pair the target with maxrate=1.5x / bufsize=2x for NVENC VBR.
    Same formula in all four engines (catch/taiko/std/mania v2).
    """
    ref = 1280.0 * 720.0 * 30.0
    target = 4_000_000.0 * ((float(w) * float(h) * float(fps)) / ref) ** 0.70
    return int(min(16_000_000.0, max(2_500_000.0, target)))


def preview_video_bps(total_dur_s: "float | None") -> int:
    """Video bitrate of the lean preview embed. Mirrors the contributor
    client's makeEmbedVariant (and the bot's _transcode_embed_unbounded):
    ~1.4 Mbps, lowered on long maps so the file stays <= ~24 MiB, floor 500k.
    Same formula as the catch engine's inline preview."""
    vbps = 1_400_000
    if total_dur_s and total_dur_s > 0:
        vbps = int(24 * 1024 * 1024 * 8 / total_dur_s) - 128_000
        vbps = max(500_000, min(1_400_000, vbps))
    return vbps


# ---- inline preview on the Mac's media engine --------------------------------
# The preview is a 720p30 side output of the SAME ffmpeg process as the master.
# On a Mac the CPU is the encoder, so from two renders at once the preview's
# own x264 competes with the masters': on an M1 Max, encoding the preview with
# VideoToolbox instead (the media engine is otherwise idle, the master stays on
# x264) measured +16 to +23% aggregate throughput at 2-4 concurrent std renders
# at 720p and 1080p, +8% for one 1080p render, and nothing for one 720p render.
# The master is byte-for-byte the same file either way.
#
#   R3D_PREVIEW_VT=0   the CPU preview, as before
#   R3D_PREVIEW_VT=1   ask for the media engine on any platform
#   unset              the media engine on a Mac (perf.FAST_DEFAULT, so
#                      R3D_STD_STOCK=1 also means the CPU preview)
#
# Asking is not getting: the local ffmpeg must first open a hardware session
# (`_vt_probe`); where it cannot, the preview stays on libx264. One failed
# output kills the whole ffmpeg process and with it the render, so when an
# ffmpeg that carried a hardware preview dies naming videotoolbox, the media
# engine is switched off for a day (`note_preview_failure`) and the next render
# on this node is on the CPU preview without anyone touching a setting.
#
# Settings (chosen on lossless captures of all four modes, see the PR): the
# same target bitrate as the x264 preview, in the encoder's CONSTANT-RATE mode
# with B-frames, behind a half-second LEAD-IN.
#
# Why each: in its default mode, and worse under a peak cap, this encoder
# makes a small first keyframe and then starves the rest of the first second
# (worst frame VMAF 28 with x264's 1.25x cap). Constant-rate mode with B-frames
# cures that at 1400 kbit/s, but a long replay gets a lower preview bitrate
# from the size budget (507 kbit/s at five minutes) and there the first
# keyframe is too small again: first half-second VMAF 42 against 66 for the
# rest. The lead-in is the cure that holds at every bitrate: the first frame
# is repeated for _VT_LEAD_S seconds in front of the video, so the rate
# control has banked about what x264 spends on its first keyframe by the time
# the real video starts; a keyframe is forced on the first real frame, and two
# bitstream filters drop the lead-in's packets and shift the rest back, so the
# file starts on that keyframe at time 0 with exactly the frames it had before
# (42 -> 73 on that replay, with nothing taken from the seconds after).
# Constant-rate mode also makes the file size exact, which is what
# preview_video_bps budgets for. Frames are scaled on the CPU exactly as
# before, so the encoder is the only thing that changes.
_VT_RATE = 1.0
_VT_BFRAMES = 2
_VT_LEAD_S = 0.5


def _vt_lead_frames(pfps: int) -> int:
    return max(1, int(round(pfps * _VT_LEAD_S)))


def vt_lead_in_filter(pfps: int) -> str:
    """Tail of the preview branch's filter chain for the hardware preview: the
    first frame repeated in front (see the block comment above)."""
    return f"tpad=start={_vt_lead_frames(pfps)}:start_mode=clone"


def _vt_codec_args(bps: int, pfps: int = 30) -> "list[str]":
    """The hardware preview's codec arguments. The probe uses these too, so a
    machine that cannot do one of them (constant-rate mode needs macOS 13; the
    packet filters need ffmpeg 5.1) fails the probe and keeps the CPU preview
    instead of failing a render."""
    n = _vt_lead_frames(pfps)
    cut = n / float(pfps)
    return ["-c:v", "h264_videotoolbox", "-allow_sw", "0", "-realtime", "0",
            "-profile:v", "high", "-pix_fmt", "yuv420p", "-b:v", str(int(bps)),
            "-constant_bit_rate", "1", "-g", str(int(pfps)),
            "-bf", str(_VT_BFRAMES),
            # the cut must land on a keyframe whatever cadence the encoder
            # keeps (with B-frames it is 29 frames, not 30): force one on the
            # first real frame, by frame NUMBER (a time would round)
            "-force_key_frames", f"expr:eq(n,{n})",
            "-bsf:v", (f"noise=drop=lt(pts*tb\\,{cut - 0.001:.6f}),"
                       f"setts=pts=PTS-{n}/{int(pfps)}/TB:dts=DTS-{n}/{int(pfps)}/TB")]


# The x264 preview's preset, as a knob like R3D_X264_PRESET for the master
# (default unchanged). Measurement hook for nodes with no media engine.
_PREVIEW_X264_PRESET = os.environ.get("R3D_PREVIEW_X264_PRESET") or "veryfast"
_VT_OFF_FOR_S = 24 * 3600
_vt_decision: "bool | None" = None


def _r3d_cache_dir() -> str:
    import tempfile
    return (os.path.expanduser("~/Library/Caches/r3d") if sys.platform == "darwin"
            else os.path.join(tempfile.gettempdir(), "r3d-cache"))


def _vt_probe(ignore_off: bool = False) -> bool:
    """Can the ffmpeg on THIS machine open a VideoToolbox H.264 session right
    now? A half-second gray clip through it; a yes is remembered per ffmpeg
    binary (path, size, mtime), a no is asked again next time. Any trouble
    counts as no: a probe must never fail a render."""
    import hashlib
    import time
    try:
        exe = shutil.which("ffmpeg")
        if not exe:
            return False
        base = _r3d_cache_dir()
        if not ignore_off:
            try:
                with open(os.path.join(base, "vt-preview-off")) as fh:
                    if time.time() < float(fh.read().strip() or 0):
                        return False
            except (OSError, ValueError):
                pass
        st = os.stat(os.path.realpath(exe))
        key = hashlib.sha1(f"{os.path.realpath(exe)}|{st.st_size}|{st.st_mtime_ns}|v3"
                           .encode()).hexdigest()[:16]
        mark = os.path.join(base, f"vt-preview-probe-{key}")
        if os.path.exists(mark):
            return True
        p = subprocess.run(
            [exe, "-v", "error", "-f", "lavfi", "-i",
             "color=c=gray:s=320x180:r=30:d=1", "-vf", vt_lead_in_filter(30)]
            + _vt_codec_args(300_000) + ["-f", "null", "-"],
            capture_output=True, timeout=20)
        if p.returncode != 0:
            return False
        try:
            os.makedirs(base, exist_ok=True)
            with open(mark, "w") as fh:
                fh.write("1")
        except OSError:
            pass
        return True
    except Exception:  # noqa: BLE001
        return False


def preview_on_media_engine() -> bool:
    """Should this render's inline preview be encoded by VideoToolbox? Decided
    once per process (see the block comment above for the rule)."""
    global _vt_decision
    if _vt_decision is None:
        asked = os.environ.get("R3D_PREVIEW_VT") is not None
        want = perf.envflag("R3D_PREVIEW_VT") if asked else perf.FAST_DEFAULT
        # an explicit =1 overrides the day-long switch-off, never the probe
        _vt_decision = bool(want) and _vt_probe(ignore_off=asked)
    return _vt_decision


def note_preview_failure(cmd: "list[str]", err: bytes) -> bool:
    """ffmpeg exited non-zero. If it carried a hardware preview and its last
    words name videotoolbox, switch the media engine off for a day on this
    node. Returns whether it did."""
    import time
    try:
        if "h264_videotoolbox" not in cmd:
            return False
        low = (err or b"").lower()
        if b"videotoolbox" not in low and b"vtenc" not in low:
            return False
        base = _r3d_cache_dir()
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, "vt-preview-off"), "w") as fh:
            fh.write(str(int(time.time()) + _VT_OFF_FOR_S))
        return True
    except Exception:  # noqa: BLE001
        return False


def _preview_video_args(vbps: int, hw: bool, pfps: int = 30) -> "list[str]":
    """Video codec arguments of the inline preview output."""
    if hw:
        return _vt_codec_args(vbps * _VT_RATE, pfps)
    return ["-c:v", "libx264", "-preset", _PREVIEW_X264_PRESET, "-pix_fmt", "yuv420p",
            "-b:v", str(vbps), "-maxrate", str(int(vbps * 1.25)),
            "-bufsize", str(vbps * 2), "-g", "30",
            "-threads", str(max(2, min(4, (os.cpu_count() or 4) - 2)))]


def _null_sink_cmd() -> list[str]:
    """HARNESS ONLY (R3D_STD_NULL_SINK=1): swap ffmpeg for `cat` so the renderer
    runs with the encoder's cost REMOVED. The gap against a normal run is the
    encoder's share of wall, which is the budget every encoder-side optimisation
    (GPU-YUV, socketpair, mapped readback, preset) is competing over. Produces no
    usable output -- measurement only."""
    return ["cat"]


def compact_wanted(total_dur_s, w, h, fps, default_factor) -> bool:
    """Whether to write the inline Discord copy for this render.

    R3D_COMPACT_INLINE=1 asks for it. The copy costs a second software encode
    for the whole render, and it is only used when the master is too big to be
    the Discord file itself, so R3D_COMPACT_IF_OVER_BYTES=<n> limits it to
    renders whose master is EXPECTED to exceed n bytes: duration x bitrate,
    with the bitrate taken from R3D_COMPACT_EXPECT_BPS (what this node's
    masters of this kind have actually averaged, supplied by the client) or,
    lacking that, the encoder ladder times `default_factor`. Without a limit,
    or without a duration, the copy is always written."""
    if os.environ.get("R3D_COMPACT_INLINE") != "1":
        return False
    try:
        limit = int(os.environ.get("R3D_COMPACT_IF_OVER_BYTES", "0") or 0)
    except ValueError:
        limit = 0
    if limit <= 0 or not total_dur_s or total_dur_s <= 0:
        return True
    try:
        bps = float(os.environ.get("R3D_COMPACT_EXPECT_BPS", "0") or 0)
    except ValueError:
        bps = 0.0
    if bps <= 0:
        bps = nvenc_target_bps(int(w), int(h), float(fps)) * default_factor
    return total_dur_s * bps / 8.0 > 0.9 * limit


def compact_plan(total_dur_s: "float | None") -> "tuple[int, int, int, int]":
    """(scale_h, maxrate_bps, audio_bps, fps) for the inline Discord copy
    (`-embed-sm.mp4`). Same plan as the contributor client's compactPlan and
    the bot's _compact_plan at the 56 MiB node budget: 1080p60 on short plays,
    720p60 on longer ones, 720p30 only when the budget is genuinely too small."""
    budget_bits = 56 * 1024 * 1024 * 8
    dur = float(total_dur_s or 0.0)
    if dur <= 1:
        return 1080, 8_000_000, 192_000, 60
    total_rate = int(budget_bits / dur) or 1
    pref = 192_000 if dur <= 240 else (128_000 if dur <= 600 else 96_000)
    audio = min(pref, max(32_000, total_rate // 4))
    maxrate = max(32_000, min(8_000_000, total_rate - audio))
    if maxrate >= 3_000_000:
        return 1080, maxrate, audio, 60
    if maxrate >= 500_000:
        return 720, maxrate, audio, 60
    return 720, maxrate, audio, 30


def _preview_sink_args(preview_path, faststart: bool = True) -> list:
    """Output arguments for the inline preview.

    Default: one faststart mp4 at ``preview_path`` (unchanged).

    LIVE PREVIEW (R3D_PREVIEW_LIVE=1, default OFF): the same encode is written
    as 2 s self-contained fMP4 segments plus a growing playlist in
    ``<out stem>.live/`` (init.mp4, seg_00000.m4s ..., live.m3u8), so the
    contributor client can upload the preview WHILE the render runs and the
    site can play it before the render is done. No ``.embed.mp4`` is written in
    this mode; the client stitches one from the segments (a stream copy). A
    segment is renamed into place only when it is complete, and is listed in
    the playlist only after that."""
    if os.environ.get("R3D_PREVIEW_LIVE") != "1":
        return (["-movflags", "+faststart"] if faststart else []) + [str(preview_path)]
    live_dir = str(preview_path)[:-len(".embed.mp4")] + ".live"
    os.makedirs(live_dir, exist_ok=True)
    for _old in os.listdir(live_dir):       # a retry must not show stale segments
        try:
            os.remove(os.path.join(live_dir, _old))
        except OSError:
            pass
    return ["-f", "hls", "-hls_time", "2", "-hls_segment_type", "fmp4",
            "-hls_playlist_type", "event",
            "-hls_flags", "independent_segments+temp_file",
            "-hls_fmp4_init_filename", "init.mp4",
            "-hls_segment_filename", os.path.join(live_dir, "seg_%05d.m4s"),
            os.path.join(live_dir, "live.m3u8")]


def build_ffmpeg_cmd(*, encoder: str, resolution: tuple[int, int], fps: int,
                     output_path: Path, audio_path: Path | None = None,
                     audio_offset_ms: int = 0, video_bitrate: int | None = None,
                     crf: int = 20, audio_bitrate: str = "192k",
                     loudnorm: bool = True, extra_vf: str = "",
                     encoder_device: str | None = None,
                     preview_path: Path | None = None,
                     total_dur_s: float | None = None,
                     pix_fmt: str = "rgb24",
                     stream_master: bool = False,
                     compact_path: Path | None = None,
                     preview_hw: bool = False,
                     faststart: bool = True) -> list[str]:
    """rawvideo rgb24 on stdin → encoder → faststart mp4 (§5.6 shape).

    `preview_path` (INLINE PREVIEW, R3D_PREVIEW_INLINE=1 in the CLI; default
    None) makes the SAME ffmpeg process also write a lean 720p30 libx264
    preview embed as a second output. With it None the command is built
    exactly as before. `total_dur_s` (video length, if known) only sizes the
    preview's bitrate. `preview_hw` (the caller passes
    `preview_on_media_engine()`) encodes that preview with VideoToolbox
    instead of libx264; the master's arguments are the same either way.
    `faststart=False` (record/audio_late.py only) leaves the index at the end
    of every output: those files are temporary and are rewritten once, when
    the audio is joined to them."""
    w, h = resolution
    is_vaapi = encoder == "h264_vaapi"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if is_vaapi:
        # VAAPI needs the DRM render node initialised before the encoder and the
        # frames uploaded to a GPU surface (format=nv12,hwupload below). Without
        # this ffmpeg cannot open h264_vaapi -> dies at startup -> BrokenPipe on
        # the first frame write. std was NVENC-only until AMD contributors ran it.
        cmd += ["-vaapi_device", encoder_device or "/dev/dri/renderD128"]
    # pix_fmt yuv420p = the renderer converted on the GPU (R3D_STD_GPU_YUV) and
    # swscale has nothing left to do; vflip below is format-agnostic.
    cmd += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "pipe:0"]
    if audio_path is not None:
        if audio_offset_ms:
            cmd += ["-itsoffset", f"{audio_offset_ms / 1000.0:.3f}"]
        cmd += ffmpeg_file_input_args(audio_path)
    # Frames arrive BOTTOM-UP on stdin and ffmpeg's vflip filter restores
    # them (an exact row reorder on rawvideo — zero pixel math; the mania
    # v2 prod encoder ships the same shape). This lets the writer hand
    # the GL readback buffer to the pipe zero-copy instead of paying a
    # ~6 MB negative-stride flip copy per frame — see FfmpegPipe._writer.
    _vf = "vflip" + ("," + extra_vf if extra_vf else "")
    # master-only tail of the video chain (VAAPI surface upload). Kept apart
    # from `_vf` so the inline-preview graph can apply it on the master
    # branch only; without a preview it is appended to `-vf` as before.
    _vm_tail = "format=nv12,hwupload" if is_vaapi else ""
    # video codec args are collected in `vc` (appended below) so the
    # two-output preview command can place them after its own -map.
    vc: list[str] = ["-c:v", encoder]
    if encoder == "libx264":
        if video_bitrate:
            _vb = int(video_bitrate)
            vc += ["-b:v", str(_vb), "-maxrate", str(int(_vb * 1.5)),
                    "-bufsize", str(_vb * 2), "-preset", _X264_PRESET,
                    "-profile:v", "high"] + _x264_extra()
        else:
            vc += ["-crf", _X264_CRF or str(crf), "-preset", _X264_PRESET,
                    "-profile:v", "high"] + _x264_extra()
    elif encoder in ("h264_nvenc", "hevc_nvenc"):
        # Resolution-scaled NVENC bitrate ladder (R3D cross-engine policy,
        # 2026-07): replaces the 2026-07-21 CQ scheme (cq=crf+4, -b:v 0,
        # maxrate (w*h)/150k) with the shared target-VBR ladder so all four
        # engines land on the same size/quality curve -- see nvenc_target_bps.
        _tgt = video_bitrate or nvenc_target_bps(w, h, fps)
        vc += ["-rc", "vbr", "-b:v", str(_tgt),
                "-maxrate", str(int(_tgt * 1.5)), "-bufsize", str(_tgt * 2),
                "-profile:v", "high"]
    elif is_vaapi:
        _vb = video_bitrate or nvenc_target_bps(w, h, fps)
        vc += ["-b:v", str(_vb), "-maxrate", str(int(_vb * 1.5)),
                "-bufsize", str(_vb * 2)]
    elif video_bitrate:
        vc += ["-b:v", str(video_bitrate)]
    if not is_vaapi:
        # VAAPI output pixel format is set by the hwupload filtergraph (nv12 on a
        # GPU surface); forcing -pix_fmt yuv420p here conflicts with the encoder.
        vc += ["-pix_fmt", "yuv420p"]
    af: list[str] = []
    acodec: list[str] = []
    if audio_path is not None:
        # LOUDNORM DUCK FIX (#17): when the music was pre-normalised upstream
        # (loudnorm=False), do NOT loudnorm the mixed song+hits (that ducked the
        # song under hits) -- apply only a clamp-only true-peak limiter to catch
        # summed peaks without ducking.
        af = ([LOUDNORM] if loudnorm
              else ["alimiter=limit=0.95:level=disabled:attack=1:release=20"])
        # STREAMABLE MASTER (R3D_STREAM_MASTER=1 in the CLI; default False):
        # the loudness pass the contributor client would run on the finished
        # file (loudnorm on the MIXED output) is applied here instead, so
        # nothing has to rewrite the master after the render.
        if stream_master and LOUDNORM not in af:
            af = af + [LOUDNORM]
        acodec = ["-c:a", "aac", "-b:a", audio_bitrate, "-ar", "48000",
                  "-shortest"]
    _mfast = ([] if (stream_master or not faststart)
              else ["-movflags", "+faststart"])
    if preview_path is None:
        cmd += ["-vf", _vf + ("," + _vm_tail if _vm_tail else "")]
        cmd += vc
        if audio_path is not None:
            if af:
                cmd += ["-af", ",".join(af)]
            cmd += acodec
        # stream_master: no +faststart, which rewrites the whole file at close;
        # the master is then written front to back and the site moves the moov.
        cmd += _mfast + [str(output_path)]
        if os.environ.get("R3D_STD_NULL_SINK") == "1":
            return _null_sink_cmd()
        return cmd

    # TWO OUTPUTS FROM ONE PROCESS (inline preview). The frame pipe is read
    # once. The master's own video chain (`vflip` + extra_vf — frames arrive
    # BOTTOM-UP) runs BEFORE `split`, so both outputs are the right way up and
    # the master sees the same frames as with `-vf`; `split` then hands them
    # to the master encoder (unchanged settings; the VAAPI upload stays on
    # the master branch only) and to a 720p30 libx264 preview. Audio: the
    # master's own `-af` chain stays on the shared branch, then `asplit`; the
    # preview branch gets the loudness pass the contributor client would
    # otherwise apply before cutting its embed, so the preview needs no
    # post-processing at all.
    pfps = min(30, int(round(float(fps))))
    # hardware preview only: the first frame repeated in front, cut off again
    # after encoding (see _vt_codec_args); "" leaves the graph as it was
    if preview_hw and audio_path is not None and not (total_dur_s and total_dur_s > 0):
        preview_hw = False       # no known length to end the audio at: stay on x264
    _lead = ("," + vt_lead_in_filter(pfps)) if preview_hw else ""
    if compact_path is not None:
        # INLINE DISCORD COPY (R3D_COMPACT_INLINE=1 in the CLI): a third branch
        # encoded to the compact plan, so nothing is left to encode after the
        # render. Never upscaled past the master, never above its frame rate.
        c_h, c_max, c_abps, c_fps = compact_plan(total_dur_s)
        c_h = min(c_h, int(h))
        c_fps = min(c_fps, int(round(float(fps))))
        graph = [f"[0:v]{_vf},split=3[vm0][vp0][vc0];[vm0]{_vm_tail or 'null'}[vm];"
                 f"[vp0]fps={pfps},scale=-2:720{_lead}[vp];"
                 f"[vc0]fps={c_fps},scale=-2:{c_h}:flags=bilinear[vc]"]
    else:
        graph = [f"[0:v]{_vf},split=2[vm0][vp0];[vm0]{_vm_tail or 'null'}[vm];"
                 f"[vp0]fps={pfps},scale=-2:720{_lead}[vp]"]
    if audio_path is not None:
        # the audio file is input 1 (input 0 is the rawvideo pipe).
        # `aformat=sample_rates=48000` PINS the shared branch to the master's
        # own output rate (its `-ar 48000`; the mixed wav is 48 kHz too, so
        # this converts nothing). Without it the preview's loudnorm — which
        # runs at 192 kHz internally — wins format negotiation back through
        # `asplit`, the master's limiter then runs at 192 kHz and the master
        # audio is resampled 48k→192k→48k: measurably NOT the bytes the
        # `-af` path produces (ffmpeg 8.1). Pinned, the 192 kHz conversion
        # sits on the preview branch only and the master is byte-identical.
        graph.append(f"[1:a]{','.join(af) or 'anull'},"
                     f"aformat=sample_rates=48000[aout]")
        if stream_master:
            # the shared branch already carries the loudness pass (af above):
            # master, preview (and Discord copy) get the same normalised audio
            graph.append("[aout]asplit=3[am][ap][ac]" if compact_path is not None
                         else "[aout]asplit=2[am][ap]")
        elif compact_path is not None:
            # the Discord copy is cut from the FINAL (normalised) audio
            graph.append(f"[aout]asplit=2[am][ap0];[ap0]{LOUDNORM},asplit=2[ap][ac]")
        else:
            graph.append(f"[aout]asplit=2[am][ap0];[ap0]{LOUDNORM}[ap]")
    if _lead and audio_path is not None:
        # `-shortest` measures the preview's video BEFORE the lead-in is cut
        # off, so it cannot be used on this output (see where it is left out
        # below). The video's length is known here: end the preview's audio
        # there, which is what `-shortest` does for the x264 preview.
        graph = [g.replace("[ap]", "[ap_full]") for g in graph]
        graph.append(f"[ap_full]atrim=end={float(total_dur_s):.6f}[ap]")
    cmd += ["-filter_complex", ";".join(graph)]
    # output 1: the master, exactly as without the preview
    cmd += ["-map", "[vm]"] + (["-map", "[am]"] if audio_path is not None
                               else [])
    cmd += vc + acodec
    cmd += _mfast + [str(output_path)]
    # output 2: the preview. libx264 unless the caller proved a hardware
    # session opens here (`preview_hw`): a second NVENC/VAAPI session can fail
    # to open (session limits), and one failed output kills the whole process
    # and with it the render. On a Mac the master is on x264, so the preview's
    # is the only hardware session this process holds.
    vbps = preview_video_bps(total_dur_s)
    cmd += ["-map", "[vp]"] + (["-map", "[ap]"] if audio_path is not None
                               else [])
    cmd += _preview_video_args(vbps, preview_hw, pfps)
    if audio_path is not None:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-ar", "48000"]
        if not _lead:
            cmd += ["-shortest"]
        # with the lead-in `-shortest` is wrong in both directions: it compares
        # the streams BEFORE the lead-in is cut off, so it either lets extra
        # audio through or, once the audio is ended at the video's length (the
        # atrim above), cuts the last half second of VIDEO. The audio is ended
        # explicitly instead and the video ends when its frames do.
    cmd += _preview_sink_args(preview_path, faststart)
    if compact_path is not None:
        # output 3: the Discord copy. Same recipe as the node's own compact
        # encode (libx264 veryfast crf 21 + VBV at the plan's maxrate).
        cmd += ["-map", "[vc]"] + (["-map", "[ac]"] if audio_path is not None
                                   else [])
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-crf", "21", "-maxrate", str(c_max),
                "-bufsize", str(max(1, c_max // 2)), "-g", str(c_fps),
                "-threads", str(max(2, min(6, (os.cpu_count() or 4) // 2)))]
        if audio_path is not None:
            cmd += ["-c:a", "aac", "-ar", "48000", "-b:a", str(c_abps),
                    "-shortest"]
        cmd += _mfast + [str(compact_path)]
    if os.environ.get("R3D_STD_NULL_SINK") == "1":
        return _null_sink_cmd()
    return cmd


class _SockStdin:
    """File-like shim over the socketpair end.

    `sendall`, not `write`, is deliberate: a raw SocketIO.write may write
    PARTIALLY and return a short count, and the writer thread ignores write()'s
    return value -- that would silently truncate a frame."""

    __slots__ = ("_sk",)

    def __init__(self, sk):
        self._sk = sk

    def write(self, b):
        self._sk.sendall(b)
        return len(b)

    def flush(self):
        pass

    def fileno(self):
        return self._sk.fileno()

    def close(self):
        import socket as _sock
        try:
            self._sk.shutdown(_sock.SHUT_WR)
        except OSError:
            pass
        self._sk.close()


class FfmpegPipe:
    """Spawn ffmpeg, push raw frames, close. Mirrors mania v2's FfmpegPipe
    contract minus asyncio (the worker wraps the CLI in a subprocess
    already; in-process async buys nothing here).

    Frames are handed to a writer thread over a small bounded queue: the
    serialisation (`tobytes` — a negative-stride flip copy) and the
    blocking pipe write happen OFF the render thread, overlapping the next
    frame's draw. Order is FIFO so the byte stream ffmpeg sees (and the
    R3D_FRAME_MD5 hash, computed writer-side) is unchanged. The queue
    bounds memory (4 × ~2.7 MB frames) and provides natural backpressure
    when ffmpeg is the bottleneck; writer errors surface loudly on the
    next push() instead of deadlocking the producer."""

    _QUEUE_FRAMES = 4

    def __init__(self, cmd: list[str], recycle=None):
        # recycle: optional callable(frame) invoked writer-side once a
        # frame's bytes are in the pipe — the GL renderer's readback
        # buffer pool (SpriteRenderer.recycle_frame). Pure bookkeeping:
        # the byte stream ffmpeg sees is untouched.
        self.cmd = cmd
        self._recycle = recycle
        self.proc: subprocess.Popen | None = None
        self._q: "queue.Queue" = queue.Queue(maxsize=self._QUEUE_FRAMES)
        self._thread: threading.Thread | None = None
        self._werr: BaseException | None = None
        self._hash = None
        self._hash_frames = 0
        if perf.FRAME_MD5:
            import hashlib
            self._hash = hashlib.blake2b(digest_size=16)

    def __enter__(self) -> "FfmpegPipe":
        # macOS pushes every frame through the 64 KiB default pipe -- ~95 kernel
        # handoffs for ONE 6.22 MB rgb24 frame -- because F_SETPIPE_SZ is
        # Linux-only. A unix socketpair CAN be grown (SO_SNDBUF/SO_RCVBUF).
        # Catch measured pipe 256 fps -> socketpair 406 fps on this machine
        # against a 419 fps file-fed ceiling. Bytes on the wire are unchanged,
        # so output is byte-identical.
        if sys.platform == "darwin" and perf.envflag("R3D_MAC_SOCKET_PIPE",
                                                     perf.FAST_DEFAULT):
            import socket as _sock
            _par, _chi = _sock.socketpair(_sock.AF_UNIX, _sock.SOCK_STREAM)
            for _s, _opt in ((_par, _sock.SO_SNDBUF), (_chi, _sock.SO_RCVBUF)):
                try:
                    _s.setsockopt(_sock.SOL_SOCKET, _opt, 1 << 20)
                except OSError:
                    pass          # keep the default buffer; still correct
            self.proc = subprocess.Popen(
                self.cmd, stdin=_chi.fileno(), stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, bufsize=0)
            _chi.close()
            self.proc.stdin = _SockStdin(_par)     # type: ignore[assignment]
        else:
            self.proc = subprocess.Popen(
                self.cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                import fcntl
                # F_SETPIPE_SZ = 1031 (Linux); same intent as the socketpair
                fcntl.fcntl(self.proc.stdin.fileno(), 1031, 1 << 20)
            except (OSError, ImportError, AttributeError):
                pass              # not Linux, or not permitted -> default size
        self._thread = threading.Thread(target=self._writer,
                                        name="ffmpeg-writer", daemon=True)
        self._thread.start()
        return self

    def _writer(self) -> None:
        stdin = self.proc.stdin
        while True:
            frame = self._q.get()
            if frame is None:
                return
            if self._werr is not None:
                if self._recycle is not None and not hasattr(frame, "result"):
                    self._recycle(frame)
                continue          # drain (never write after an error)
            try:
                if hasattr(frame, "result"):
                    # a deferred frame (concurrent.futures.Future, e.g. the
                    # sub-1080p results outro's downscale pool). Resolved HERE,
                    # in queue order, so the stream stays FIFO.
                    frame = frame.result()
                if self._hash is not None:
                    # R3D_FRAME_MD5 hashes the TOP-DOWN semantic stream —
                    # the same bytes the pre-vflip pipe pushed, so digests
                    # stay comparable across the pipeline change.
                    self._hash.update(frame.tobytes())
                    self._hash_frames += 1
                # ffmpeg runs `vflip` (build_ffmpeg_cmd), so the pipe wants
                # the frame BOTTOM-UP. PBO frames are flipud views over a
                # contiguous readback buffer, so frame[::-1] recovers that
                # buffer C-contiguous → a zero-copy pipe write. Fresh
                # top-down frames (the SSAA results tail) fall back to the
                # same negative-stride flip copy the old path did.
                if frame.ndim == 1:
                    # planar yuv420p: already in GL (bottom-up) row order and
                    # contiguous, and ffmpeg's vflip reorders the rows. Reversing
                    # a FLAT planar buffer would reverse every byte and scramble
                    # the three planes into each other.
                    stdin.write(frame)
                else:
                    flipped = frame[::-1]
                    if flipped.flags["C_CONTIGUOUS"]:
                        stdin.write(flipped)
                    else:
                        stdin.write(flipped.tobytes())
                if self._recycle is not None:
                    self._recycle(frame)
            except BaseException as e:  # noqa: BLE001 - surfaced on push()
                self._werr = e

    def push(self, frame_rgb) -> None:
        assert self.proc is not None and self._thread is not None
        if self._werr is not None:
            raise EncoderError(f"ffmpeg writer failed: {self._werr!r}")
        with perf.T("encode_push"):
            self._q.put(frame_rgb)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.proc is None:
            return
        # The queue still holds frames the writer has not handed to ffmpeg, so
        # this join is real encode time, not teardown. It is invisible to any
        # frames/second figure because it happens after the last push.
        perf.mark("enc:queue_drain")
        if self._thread is not None:
            with perf.T("enc_writer_join"):
                self._q.put(None)
                self._thread.join()
        perf.mark("enc:writer_joined")
        if self._hash is not None:
            print(f"frame-stream-hash: {self._hash.hexdigest()} "
                  f"({self._hash_frames} frames)", file=sys.stderr, flush=True)
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        _, err = None, b""
        # Named, not left as a residual: this is ffmpeg's OWN remaining encode
        # time (lookahead flush + trailer), real wall cost that no frames/second
        # figure can show because it happens after the last frame is handed over.
        with perf.T("ffmpeg_finish"):
            try:
                err = self.proc.stderr.read() if self.proc.stderr else b""
            finally:
                code = self.proc.wait()
        perf.mark("enc:ffmpeg_reaped")
        hw_off = ""
        if code != 0 and note_preview_failure(self.cmd, err or b""):
            # whichever way the failure surfaces (here, or as a broken pipe on
            # a push), the NEXT render on this node must not repeat it
            hw_off = (" [the preview's hardware encoder is now off for 24 h on "
                      "this node; the next render uses the CPU preview]")
            # ffmpeg's own words too: when it dies at start-up the failure
            # reaches the caller as a broken pipe on a push, which says nothing
            print("preview: hardware encoder failed, switched off for 24 h; "
                  "ffmpeg said: " + (err or b"").decode(errors="replace").strip()[:400],
                  file=sys.stderr, flush=True)
        if exc_type is None and self._werr is not None \
                and not isinstance(self._werr, BrokenPipeError):
            raise EncoderError(f"ffmpeg writer failed: {self._werr!r}")
        if code != 0:
            message = f"ffmpeg exited {code}: {err.decode(errors='replace')[-2000:]}" + hw_off
            if exc_type is None or (isinstance(exc, EncoderError)
                                    and isinstance(self._werr, BrokenPipeError)):
                raise EncoderError(message) from exc
            # Preserve unrelated render failures while retaining ffmpeg's own diagnostic.
            print(message, file=sys.stderr, flush=True)
