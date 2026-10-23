"""Offline audio mixing — the NO-BASS design (locked decision).

The reference (§5.6) runs BASS as a non-playing 48 kHz float decode mixer
and streams 1 ms of PCM per sim tick into a second ffmpeg process. BASS is
proprietary (free for non-commercial use only — the one real license item
in the danser stack, APPROACH.md §1.1). This engine replaces it the same
way mania/catch/taiko already do in production:

  1. decode the map's music (any codec) → float32 stereo 48 kHz numpy via
     an ffmpeg subprocess (decode_to_pcm);
  2. mix hitsound samples into a track buffer at event times with numpy
     adds (mix_at) — sample volumes from timing points / object extras
     (§3.4 semantics), rate-mods handled by resampling the music once;
  3. write the mixed track as .wav (write_wav) and hand it to the single
     ffmpeg encode as a file input (record/encode.py), where loudnorm
     (I=-14:TP=-1.5) runs like every in-house engine.

Determinism is preserved: mixing happens in map-time before encoding, so
encoder speed can't shift audio (the property §5.2 guards).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from osu_std_renderer.security import ffmpeg_file_input_args

SAMPLE_RATE = 48000
CHANNELS = 2


class AudioError(RuntimeError):
    pass


# --- loudnorm PCM cache (cross-engine, box-local) ----------------------------
# The inline loudnorm normalisation in decode_to_pcm is the single largest
# audio setup cost and is deterministic in (source bytes, rate, pitch, params).
# Memoise its f32le PCM output on the fast local SSD so a repeat render of the
# same track skips the pass; using the SAME key recipe in every in-house engine
# means a track normalised by one mode is reused by another. Best-effort: any
# cache error falls back to a normal (uncached) decode. R3D_NO_LOUDNORM_CACHE=1
# disables the whole path; R3D_LOUDNORM_CACHE_DIR overrides the location.
_LOUDNORM_FILTER = "loudnorm=I=-18:TP=-1.5:LRA=11"
_DEFAULT_CACHE_DIR = "/data/r3d/loudnorm-cache"
_CACHE_EXT = "f32le"          # raw little-endian float32, 48 kHz stereo


def _loudnorm_cache_disabled() -> bool:
    return os.environ.get("R3D_NO_LOUDNORM_CACHE", "").strip().lower() \
        not in ("", "0", "false", "no", "off")


# Resolved ONCE per process: [Path] when a location is usable, [None] when none is.
_CACHE_DIR_RESOLVED: list = []


def _platform_cache_dir() -> Path:
    """Per-user fallback for boxes without the shared /data mount."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "r3d" / "loudnorm-cache"
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    return (Path(xdg) if xdg else Path.home() / ".cache") / "r3d" / "loudnorm-cache"


def _cache_dir_usable(d: Path) -> bool:
    """Can we actually create AND write here? mkdir alone is not enough -- a
    read-only mount passes mkdir(exist_ok=True) and fails at write time."""
    try:
        d.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(d), prefix=".probe-")
        os.close(fd)
        os.unlink(tmp)
        return True
    except (OSError, ValueError):
        return False


def _loudnorm_cache_dir() -> "Path | None":
    """Where to keep cached loudnorm PCM, or None if nowhere is writable.

    `/data/r3d/loudnorm-cache` stays the PREFERRED location -- that is what the
    production Linux boxes mount, and keeping it first means their behaviour is
    unchanged. The per-user fallback exists because a Mac render node cannot
    create `/data` without root, and the old code swallowed that OSError: the
    cache was silently off on EVERY render, costing **9.1% of total process
    wall** (14.91 s -> 13.55 s measured) while `fps` showed nothing, because the
    decode wait happens before the t0 that fps is measured from.

    An explicit R3D_LOUDNORM_CACHE_DIR is honoured as-is, with no fallback: if
    an operator names a directory, silently using a different one is worse than
    saying it does not work.
    """
    if _CACHE_DIR_RESOLVED:
        return _CACHE_DIR_RESOLVED[0]
    env = os.environ.get("R3D_LOUDNORM_CACHE_DIR", "").strip()
    cands = [Path(env)] if env else [Path(_DEFAULT_CACHE_DIR),
                                     _platform_cache_dir()]
    for i, d in enumerate(cands):
        if _cache_dir_usable(d):
            if i:
                print(f"loudnorm cache: {cands[0]} not writable, using {d}",
                      file=sys.stderr)
            _CACHE_DIR_RESOLVED.append(d)
            return d
    print("loudnorm cache: NO writable location (tried "
          + ", ".join(str(c) for c in cands)
          + ") - the loudnorm pass will re-run on every render",
          file=sys.stderr)
    _CACHE_DIR_RESOLVED.append(None)
    return None


def _loudnorm_cache_key(path: Path, rate: float, pitch: bool,
                        loudnorm_param: str) -> str:
    """Stable hash of everything that determines the loudnorm OUTPUT: sha256 of
    the SOURCE audio bytes + playback rate + pitch mode (NC resample vs DT/HT
    atempo) + the exact loudnorm param string. Same recipe in every engine so
    the cache is shared. Raises OSError if the source can't be read (the caller
    then decodes uncached)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    material = "\n".join((
        f"src={h.hexdigest()}",
        f"rate={float(rate)!r}",
        f"pitch={1 if pitch else 0}",
        f"param={loudnorm_param}",
    )).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _loudnorm_cache_load(cache_path: Path):
    """Return cached PCM (N,2 float32) or None on miss/corrupt/short read."""
    try:
        data = cache_path.read_bytes()
    except OSError:
        return None
    stride = CHANNELS * 4  # bytes per frame (float32 * channels)
    if not data or (len(data) % stride) != 0:
        return None  # empty or truncated/corrupt -> recompute
    try:
        return np.frombuffer(data, dtype=np.float32).reshape(-1, CHANNELS).copy()
    except ValueError:
        return None


def _loudnorm_cache_store(cache_path: Path, raw: bytes) -> None:
    """Atomically write raw f32le PCM bytes to the cache (temp + os.replace).
    Best-effort: never raises -- a cache failure must not fail a render."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(cache_path.parent),
                                   prefix=".tmp-", suffix="." + _CACHE_EXT)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(raw)
            os.replace(tmp, cache_path)          # atomic on the same fs
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except (OSError, ValueError):
        pass


def rate_audio_filter(rate: float, pitch: bool = False) -> str:
    """The ffmpeg ``-af`` filter string for a clock-rate change → "" at rate 1.

    Two modes, matching lazer's rate-mod audio (ModRateAdjust vs
    ModNightcore/ModDaycore):

      * ``pitch=False`` — DT/HT (ModRateAdjust, AdjustableProperty.Tempo):
        ``atempo={rate}`` — tempo change, pitch PRESERVED. Identical to the
        pre-custom-rate path, so a fixed 1.5/0.75 render is byte-identical.
      * ``pitch=True`` — NC/DC (ModNightcore/ModDaycore): the pitch is pinned
        to the mod's DEFAULT rate (the classic 1.5× nightcore / 0.75× daycore
        Frequency = ``freqAdjust = SpeedChange.Default``) via ``asetrate`` +
        resample, and ``atempo = rate / default`` (= lazer's ``tempoAdjust``)
        carries the remainder. Net speed = rate; the audio pitches. (NB: this
        matches CURRENT lazer, where NC pitch is FIXED at 1.5× regardless of
        the custom rate — not "pitch == rate".)

    atempo's stable range is 0.5–2.0; every custom-rate ``atempo`` here
    (DT/HT 0.5–2.0, NC tempo 0.673–1.333, DC tempo 0.667–1.32) stays inside
    it, so a single stage suffices."""
    if rate == 1.0:
        return ""
    if not pitch:
        return f"atempo={rate}"
    default = 1.5 if rate > 1.0 else 0.75          # NC vs DC classic pitch
    new_sr = int(round(SAMPLE_RATE * default))
    tempo = rate / default
    af = f"asetrate={new_sr},aresample={SAMPLE_RATE}"
    if abs(tempo - 1.0) > 1e-9:
        af += f",atempo={tempo}"
    return af


def decode_to_pcm(path: Path, *, rate: float = 1.0,
                  pitch: bool = False, loudnorm: bool = False) -> np.ndarray:
    """Decode any audio file → float32 stereo 48 kHz, shape (N, 2).

    `rate` != 1 applies the clock-rate change. `pitch=False` (DT/HT) is a
    pitch-preserving tempo change; `pitch=True` (NC/DC) shifts the pitch with
    the rate (nightcore/daycore). See :func:`rate_audio_filter`."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioError("ffmpeg not found on PATH")
    af = rate_audio_filter(rate, pitch)

    # LOUDNORM PCM CACHE: the loudnorm pass is deterministic in its input yet
    # reruns the full ffmpeg decode+normalise on every render of the same track.
    # When loudnorm is requested, memoise the decoded f32le PCM under a key over
    # everything that determines the OUTPUT (source bytes + rate + pitch mode +
    # the exact loudnorm param string). A hit returns byte-identical bytes and
    # skips the pass; a miss runs it exactly as before then writes atomically.
    cache_path = None
    if loudnorm and not _loudnorm_cache_disabled():
        try:
            _cdir = _loudnorm_cache_dir()
            if _cdir is not None:
                key = _loudnorm_cache_key(Path(path), rate, pitch,
                                          _LOUDNORM_FILTER)
                cache_path = _cdir / f"{key}.{_CACHE_EXT}"
        except OSError:
            cache_path = None  # can't hash the source -> just decode uncached
        if cache_path is not None:
            cached = _loudnorm_cache_load(cache_path)
            if cached is not None:
                return cached

    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", *ffmpeg_file_input_args(path)]
    if loudnorm:
        # LOUDNORM DUCK FIX (#17): normalise the MUSIC ALONE here (music-only,
        # no hit transients) so the encode does NOT loudnorm the song+hits mix
        # (that ducked the song ~4 dB under every hitsound). Hits are numpy-mixed
        # on top afterwards; the encode applies only a clamp-only peak limiter.
        af = (af + "," if af else "") + _LOUDNORM_FILTER
    if af:
        cmd += ["-af", af]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le",
            "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg decode failed: "
                         f"{proc.stderr.decode(errors='replace')[-500:]}")
    if cache_path is not None:
        _loudnorm_cache_store(cache_path, proc.stdout)
    pcm = np.frombuffer(proc.stdout, dtype=np.float32)
    return pcm.reshape(-1, CHANNELS).copy()


# --- WU/WD continuous rate ramp (ModTimeRamp) --------------------------------
# The constant-rate path (decode_to_pcm above) applies ONE atempo/asetrate to
# the whole track — impossible for Wind Up / Wind Down, whose rate ramps over
# the map. We warp the music PIECEWISE: partition the song into bands over which
# the rate is constant to within 0.01 (timewarp.rate_bands — the SAME 0.01
# granularity lazer itself quantizes SpeedChange to via Math.Round(rate, 2)),
# stretch each band at its average rate, and concatenate. Each band's output is
# pinned to the EXACT wall length the closed-form transform prescribes, so there
# is ZERO cumulative audio<->video drift at band boundaries.
#
#   * adjust_pitch=True  (WU/WD DEFAULT — pitch follows the rate): pure linear
#     RESAMPLE per band. Resampling a map-time slice to its wall-length both
#     changes tempo AND shifts pitch (the wind-up "chipmunk" / wind-down
#     "slowdown"). Exact, numpy-only, no ffmpeg, no drift.
#   * adjust_pitch=False (pitch preserved, tempo-only): a pitch-preserving
#     time stretch per band via ffmpeg `rubberband` (available on the box;
#     atempo fallback), then pinned to the exact band length.

def _resample_linear(src: np.ndarray, n: int) -> np.ndarray:
    """Linearly resample stereo PCM ``src`` (M,2) to exactly ``n`` frames."""
    if n <= 0:
        return np.zeros((0, CHANNELS), dtype=np.float32)
    if len(src) == 0:
        return np.zeros((n, CHANNELS), dtype=np.float32)
    if len(src) == 1:
        return np.repeat(src, n, axis=0).astype(np.float32)
    idx = np.linspace(0.0, len(src) - 1, n)
    lo = np.floor(idx).astype(np.int64)
    hi = np.minimum(lo + 1, len(src) - 1)
    frac = (idx - lo).astype(np.float32)[:, None]
    return (src[lo] * (1.0 - frac) + src[hi] * frac).astype(np.float32)


def _stretch_preserve_pitch(src: np.ndarray, tempo: float,
                            target_n: int) -> np.ndarray:
    """Pitch-preserving time stretch of ``src`` by ``tempo`` (>1 = faster),
    pinned to ``target_n`` frames. ffmpeg `rubberband` (or `atempo` fallback);
    a final micro-resample corrects any WSOLA length rounding so bands join
    sample-accurately."""
    if target_n <= 0 or len(src) == 0:
        return _resample_linear(src, target_n)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None and abs(tempo - 1.0) > 1e-6:
        filt = f"rubberband=tempo={tempo:.6f}"
        for af in (filt, f"atempo={tempo:.6f}"):
            cmd = [ffmpeg, "-hide_banner", "-loglevel", "error",
                   "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
                   "-i", "pipe:0", "-af", af,
                   "-f", "f32le", "-acodec", "pcm_f32le",
                   "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "pipe:1"]
            proc = subprocess.run(cmd, input=src.astype("<f4").tobytes(),
                                  capture_output=True, check=False)
            if proc.returncode == 0 and proc.stdout:
                out = np.frombuffer(proc.stdout,
                                    dtype=np.float32).reshape(-1, CHANNELS)
                return _resample_linear(out.copy(), target_n)
    # no ffmpeg / rate 1.0 / both filters failed -> plain resample (this DOES
    # shift pitch, but only reached as a last-resort fallback)
    return _resample_linear(src, target_n)


def warp_music_pcm(pcm: np.ndarray, warp, *, adjust_pitch: bool,
                   m_lo: float = 0.0, m_hi: float | None = None,
                   quant: float = 0.01) -> np.ndarray:
    """Warp decoded music ``pcm`` (native rate, map-time domain) into the
    WALL-time domain under a :class:`timewarp.TimeWarp`. Returns float32 (N,2)
    whose frame 0 corresponds to map time ``m_lo`` (== song time 0 for a map)
    and whose length is the wall duration ``to_wall(m_hi) - to_wall(m_lo)``.
    Lay it at the wall position of ``m_lo`` in the mixer."""
    n_src = len(pcm)
    if m_hi is None:
        m_hi = n_src / SAMPLE_RATE * 1000.0
    bands = warp.rate_bands(m_lo, m_hi, quant=quant)
    if not bands:
        return pcm.astype(np.float32, copy=False)
    out: list[np.ndarray] = []
    for (b0, b1) in bands:
        i0 = max(0, min(n_src, int(round(b0 / 1000.0 * SAMPLE_RATE))))
        i1 = max(0, min(n_src, int(round(b1 / 1000.0 * SAMPLE_RATE))))
        seg = pcm[i0:i1]
        wall_ms = warp.to_wall(b1) - warp.to_wall(b0)
        target_n = max(0, int(round(wall_ms / 1000.0 * SAMPLE_RATE)))
        if adjust_pitch:
            out.append(_resample_linear(seg, target_n))
        else:
            # tempo = source frames / target frames = the band's AVERAGE rate
            tempo = (len(seg) / target_n) if target_n > 0 else 1.0
            out.append(_stretch_preserve_pitch(seg, tempo, target_n))
    return (np.concatenate(out, axis=0) if out
            else np.zeros((0, CHANNELS), dtype=np.float32))


class AudioMixer:
    """A track buffer in map-time; numpy-add samples at millisecond offsets."""

    def __init__(self, duration_ms: float):
        n = int(duration_ms / 1000.0 * SAMPLE_RATE) + SAMPLE_RATE  # +1 s pad
        self.buf = np.zeros((n, CHANNELS), dtype=np.float32)

    def lay_music(self, pcm: np.ndarray, at_ms: float = 0.0,
                  volume: float = 1.0) -> None:
        self.mix_at(at_ms, pcm, volume=volume)

    def mix_at(self, time_ms: float, sample: np.ndarray,
               volume: float = 1.0) -> None:
        """Add `sample` (N,2 float32) into the track at time_ms.

        §3.4 volume floor: osu! floors sample volume at 0.08 — the caller
        passes the floored value; this mixer is policy-free."""
        start = int(time_ms / 1000.0 * SAMPLE_RATE)
        if start >= len(self.buf):
            return
        end = min(len(self.buf), start + len(sample))
        if end <= 0:
            return  # entirely before the window (a negative end would
            #         wrap the buf slice — the --start clip guard)
        if start < 0:
            sample = sample[-start:]
            start = 0
        self.buf[start:end] += sample[:end - start] * volume

    def silence_before(self, t_ms: float, fade_ms: float = 150.0) -> None:
        """§4.10 LeadInTime/SeizureWarning, audio side: the pre-roll is
        SILENT — zero everything before wall t_ms, with a short fade-in
        ending AT t_ms so a mid-track entry doesn't click. (For a
        map-start render the region is silent anyway — no-op.)"""
        if t_ms <= 0:
            return
        i1 = min(int(t_ms / 1000.0 * SAMPLE_RATE), len(self.buf))
        i0 = max(i1 - int(fade_ms / 1000.0 * SAMPLE_RATE), 0)
        self.buf[:i0] = 0.0
        if i1 > i0:
            ramp = np.linspace(0.0, 1.0, i1 - i0, endpoint=False,
                               dtype=np.float32)
            self.buf[i0:i1] *= ramp[:, None]

    def fade_out(self, t0_ms: float, t1_ms: float) -> None:
        """§4.10 FadeOutTime, audio side: linear gain 1→0 across
        [t0_ms, t1_ms) WALL time, silence after — the track fades with the
        video's fade-to-black (the results screen then sits on silence,
        matching 'fade out … before results'). No-op on a degenerate
        window."""
        if t1_ms <= t0_ms:
            return
        i0 = max(int(t0_ms / 1000.0 * SAMPLE_RATE), 0)
        i1 = min(int(t1_ms / 1000.0 * SAMPLE_RATE), len(self.buf))
        if i0 >= len(self.buf):
            return
        if i1 > i0:
            ramp = np.linspace(1.0, 0.0, i1 - i0, endpoint=False,
                               dtype=np.float32)
            self.buf[i0:i1] *= ramp[:, None]
        self.buf[i1:] = 0.0

    def write_wav(self, path: Path) -> Path:
        """Write float32 wav (ffmpeg reads it natively; no clipping — the
        encode step normalizes via loudnorm)."""
        path = Path(path)
        data = self.buf.astype("<f4").tobytes()
        with open(path, "wb") as fh:
            byte_rate = SAMPLE_RATE * CHANNELS * 4
            fh.write(b"RIFF")
            fh.write(struct.pack("<I", 36 + len(data)))
            fh.write(b"WAVEfmt ")
            fh.write(struct.pack("<IHHIIHH", 16, 3, CHANNELS, SAMPLE_RATE,
                                 byte_rate, CHANNELS * 4, 32))
            fh.write(b"data")
            fh.write(struct.pack("<I", len(data)))
            fh.write(data)
        return path
