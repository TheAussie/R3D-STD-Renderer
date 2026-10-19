"""AUDIO OFF THE START (R3D_STD_AUDIO_LATE=1 in the CLI; default OFF).

Stock: the render cannot draw its first frame until the audio is mixed,
because the one ffmpeg process that encodes the video also reads the mixed
wav. The mix waits for the song's loudness-normalised decode, which runs at
about 25 times realtime and is cached per node: 18 s for a 7.5 minute song the
node has not rendered before, about 1 s of mixing and writing for one it has.

With the flag the frame loop starts at once. The video is encoded to
temporary files with no audio; a worker thread runs the SAME mix and then the
SAME audio filters and encoders as their own ffmpeg process, during the
render; when the frames are done the audio is joined to each output with a
stream copy. Nothing about the picture or the sound is computed differently.

Left on the stock path (the CLI decides): the streamable master and the live
preview, whose bytes are uploaded while they are written, and a non-zero
audio offset.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

from .encode import LOUDNORM, compact_plan

_QUIET = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]


_AAC_FRAME = 1024


def video_end_sample(n_frames: int, src_fps: int, out_fps: "int | None" = None,
                     rate: int = 48000) -> int:
    """Where `-shortest` ends an output's audio in the stock single process.
    ffmpeg hands the AAC encoder whole 1024-sample frames and stops at the
    last one that ends inside the video (measured on ffmpeg 8.1: 1919 frames
    at 60 fps = 1,535,200 samples of video, 1,534,976 of audio encoded). The
    encoder must be given exactly those samples and no more: its last packet
    is computed from what follows, so even one extra frame changes it.
    ``out_fps`` is the output's own frame rate when an `fps=` filter reduced
    it (frames are then counted the way that filter counts them, ties up)."""
    if out_fps is None or out_fps == src_fps:
        n, f = int(n_frames), int(src_fps)
    else:
        n = (int(n_frames) * int(out_fps) + int(src_fps) // 2) // int(src_fps)
        f = int(out_fps)
    return (n * rate // f) // _AAC_FRAME * _AAC_FRAME


def build_audio_cmd(*, audio_path, master_out, audio_bitrate: str = "192k",
                    loudnorm: bool = True, preview_out=None, compact_out=None,
                    total_dur_s: "float | None" = None,
                    preview_lead: bool = False,
                    end_samples: "dict | None" = None) -> "list[str]":
    """The audio half of ``encode.build_ffmpeg_cmd`` as its own process: the
    same filters in the same order, the same encoders and rates, one .m4a per
    output. (tests/test_audio_late.py holds the two builders to each other.)

    ``end_samples`` ({"master"|"preview"|"compact": sample count}) ends each
    output's audio where `-shortest` would have ended it beside its video
    (see video_end_sample). There is no video in this process to do it."""
    end_samples = end_samples or {}

    def _end(key: str) -> str:
        n = end_samples.get(key)
        if not n:
            return ""
        # the preview and the Discord copy come out of loudnorm at 192 kHz;
        # count their samples at the rate they are encoded at
        pin = "" if key == "master" else "aformat=sample_rates=48000,"
        return f"{pin}atrim=end_sample={int(n)}"

    af = ([LOUDNORM] if loudnorm
          else ["alimiter=limit=0.95:level=disabled:attack=1:release=20"])
    acodec = ["-c:a", "aac", "-b:a", audio_bitrate, "-ar", "48000"]
    cmd = _QUIET + ["-i", str(audio_path), "-vn"]
    if preview_out is None:
        _af = af + ([_end("master")] if _end("master") else [])
        return cmd + ["-af", ",".join(_af)] + acodec + [str(master_out)]
    graph = [f"[0:a]{','.join(af) or 'anull'},"
             f"aformat=sample_rates=48000[aout]"]
    if compact_out is not None:
        graph.append(f"[aout]asplit=2[am][ap0];[ap0]{LOUDNORM},asplit=2[ap][ac]")
    else:
        graph.append(f"[aout]asplit=2[am][ap0];[ap0]{LOUDNORM}[ap]")
    if preview_lead:
        # the hardware preview's audio is ended at the video's length instead
        # of by `-shortest` (see build_ffmpeg_cmd)
        graph = [g.replace("[ap]", "[ap_full]") for g in graph]
        graph.append(f"[ap_full]atrim=end={float(total_dur_s):.6f}[ap]")
    for key, lab in (("master", "am"), ("preview", "ap"), ("compact", "ac")):
        if _end(key) and not (key == "preview" and preview_lead) \
                and (key != "compact" or compact_out is not None):
            graph = [g.replace(f"[{lab}]", f"[{lab}_all]") for g in graph]
            graph.append(f"[{lab}_all]{_end(key)}[{lab}]")
    cmd += ["-filter_complex", ";".join(graph)]
    cmd += ["-map", "[am]"] + acodec + [str(master_out)]
    cmd += ["-map", "[ap]", "-c:a", "aac", "-b:a", "128k", "-ar", "48000",
            str(preview_out)]
    if compact_out is not None:
        c_abps = compact_plan(total_dur_s)[2]
        cmd += ["-map", "[ac]", "-c:a", "aac", "-ar", "48000",
                "-b:a", str(c_abps), str(compact_out)]
    return cmd


def build_mux_cmd(video_path, audio_path, out_path, *,
                  shortest: bool = False) -> "list[str]":
    """Join one finished video to its finished audio: a stream copy, with the
    index at the front like every stock output. ``audio_path`` None = a silent
    render (the copy is still needed, for the index). No `-shortest`: the
    audio was already ended inside the video (build_audio_cmd), and here it
    would cut the video's last frames to the audio instead."""
    cmd = _QUIET + ["-i", str(video_path)]
    if audio_path is None:
        cmd += ["-map", "0:v:0", "-c", "copy"]
    else:
        cmd += ["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0",
                "-c", "copy"]
        if shortest:
            cmd += ["-shortest"]
    return cmd + ["-movflags", "+faststart", str(out_path)]


def _beside(path: Path, tag: str, ext: str) -> Path:
    return path.parent / f"{path.stem}.{tag}{ext}"


class LateAudio:
    """One render's audio, prepared beside the frame loop.

    ``mix`` is the CLI's own mix function (returns the wav's path, or None for
    a silent render). ``start()`` before the frame loop, ``finish()`` after the
    video encoder has exited, ``cleanup()`` in the ``finally``."""

    def __init__(self, *, output: Path, preview_path, compact_path, mix,
                 total_dur_s: "float | None", preview_lead: bool,
                 audio_bitrate: str = "192k", loudnorm: bool = False,
                 end_samples: "dict | None" = None):
        self.output = Path(output)
        self.preview_path = Path(preview_path) if preview_path else None
        self.compact_path = Path(compact_path) if compact_path else None
        self._mix = mix
        self._total_dur_s = total_dur_s
        self._preview_lead = bool(preview_lead and self.preview_path)
        self._audio_bitrate = audio_bitrate
        self._loudnorm = loudnorm
        self._end_samples = dict(end_samples or {})
        # what the video encoder writes instead of the real outputs
        self.video_master = _beside(self.output, "late-video", ".mp4")
        self.video_preview = (_beside(self.preview_path, "late-video", ".mp4")
                              if self.preview_path else None)
        self.video_compact = (_beside(self.compact_path, "late-video", ".mp4")
                              if self.compact_path else None)
        self._a_master = _beside(self.output, "late-audio", ".m4a")
        self._a_preview = (_beside(self.preview_path, "late-audio", ".m4a")
                           if self.preview_path else None)
        self._a_compact = (_beside(self.compact_path, "late-audio", ".m4a")
                           if self.compact_path else None)
        self._wav: "Path | None" = None
        self._err: "BaseException | None" = None
        self._proc: "subprocess.Popen | None" = None
        # cleanup() and the worker's spawn exclude each other: after a failed
        # render (which Metal answers by re-running the job in a fresh process)
        # no audio ffmpeg may be left writing these names
        self._lock = threading.Lock()
        self._cancelled = False
        self._thread = threading.Thread(target=self._run, name="std-late-audio",
                                        daemon=True)
        self._t0 = 0.0
        self.ready_s: "float | None" = None

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._thread.start()

    def _run(self) -> None:
        try:
            wav = self._mix(lambda *a, **k: None)   # no timing marks off-thread
            self._wav = wav
            if wav is not None:
                cmd = build_audio_cmd(
                    audio_path=wav, master_out=self._a_master,
                    audio_bitrate=self._audio_bitrate, loudnorm=self._loudnorm,
                    preview_out=self._a_preview, compact_out=self._a_compact,
                    total_dur_s=self._total_dur_s,
                    preview_lead=self._preview_lead,
                    end_samples=self._end_samples)
                with self._lock:
                    if self._cancelled:
                        try:
                            os.remove(wav)
                        except OSError:
                            pass
                        return
                    self._proc = subprocess.Popen(
                        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE)
                _out, err = self._proc.communicate()
                if self._proc.returncode != 0:
                    raise RuntimeError(
                        "late audio: ffmpeg exited %d: %s"
                        % (self._proc.returncode,
                           err.decode("utf-8", "replace")[-600:]))
            self.ready_s = time.monotonic() - self._t0
        except BaseException as e:  # noqa: BLE001 — re-raised by finish()
            self._err = e

    def finish(self) -> float:
        """Wait for the audio, join it to every output. Returns the seconds
        this took (the wait included). Raises what the worker raised."""
        t = time.monotonic()
        self._thread.join()
        if self._err is not None:
            raise self._err
        has = self._wav is not None
        jobs = [(self.video_master, self._a_master, self.output)]
        if self.preview_path is not None:
            jobs.append((self.video_preview, self._a_preview, self.preview_path))
        if self.compact_path is not None:
            jobs.append((self.video_compact, self._a_compact, self.compact_path))
        procs = [(out, subprocess.Popen(
            build_mux_cmd(v, a if has else None, out),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE)) for v, a, out in jobs]
        for out, p in procs:
            _o, err = p.communicate()
            if p.returncode != 0:
                raise RuntimeError("late audio: joining %s failed (%d): %s"
                                   % (out.name, p.returncode,
                                      err.decode("utf-8", "replace")[-600:]))
        return time.monotonic() - t

    def cleanup(self) -> None:
        with self._lock:
            self._cancelled = True
            if self._proc is not None and self._proc.poll() is None:
                self._proc.kill()
        for p in (self.video_master, self.video_preview, self.video_compact,
                  self._a_master, self._a_preview, self._a_compact, self._wav):
            if p is not None:
                try:
                    os.remove(p)
                except OSError:
                    pass
