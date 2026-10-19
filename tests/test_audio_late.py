"""Audio off the start (record/audio_late.py, R3D_STD_AUDIO_LATE).

What must hold:
  * with the flag off nothing is built differently (`faststart` defaults to
    what the command always was);
  * the separate audio process runs the same audio chain as the single
    process, so the two builders cannot drift apart;
  * each audio is ended where `-shortest` ends it beside its video, counted at
    the rate it is encoded at;
  * the join is a stream copy that never trims;
  * done for real with ffmpeg: video encoded alone, audio encoded alone and
    joined gives the master the same video packets as the single process, and
    the same audio packets up to the last frame.
"""
from __future__ import annotations

import hashlib
import math
import os
import shutil
import struct
import subprocess
import tempfile
import wave

from osu_std_renderer.record import audio_late, encode


def _cmd(**kw):
    base = dict(encoder="libx264", resolution=(1280, 720), fps=60,
                output_path="m.mp4", preview_path="m.embed.mp4",
                total_dur_s=60.0, pix_fmt="yuv420p", loudnorm=False)
    base.update(kw)
    return encode.build_ffmpeg_cmd(**base)


def _without_faststart(cmd):
    out, i = [], 0
    while i < len(cmd):
        if cmd[i] == "-movflags" and cmd[i + 1] == "+faststart":
            i += 2
            continue
        out.append(cmd[i])
        i += 1
    return out


def test_flag_off_builds_the_command_it_always_did():
    for extra in ({}, {"preview_path": None}, {"audio_path": "a.wav"},
                  {"compact_path": "m-embed-sm.mp4", "audio_path": "a.wav"},
                  {"preview_hw": True}, {"stream_master": True}):
        assert _cmd(**extra) == _cmd(faststart=True, **extra)
        # and faststart=False removes that one pair from every output, nothing else
        assert _cmd(faststart=False, **extra) == _without_faststart(_cmd(**extra))
    assert "+faststart" not in _cmd(faststart=False, compact_path="c.mp4")


def _audio_graph(cmd):
    """The audio statements of a -filter_complex, input label made neutral."""
    g = cmd[cmd.index("-filter_complex") + 1]
    return [s.replace("[1:a]", "[IN]").replace("[0:a]", "[IN]")
            for s in g.split(";") if "[v" not in s and "0:v" not in s]


def test_the_audio_process_runs_the_single_process_audio_chain():
    for extra, lead in (({}, False), ({"compact_path": "c.mp4"}, False),
                        ({"preview_hw": True}, True),
                        ({"preview_hw": True, "compact_path": "c.mp4"}, True)):
        one = _cmd(audio_path="a.wav", **extra)
        sep = audio_late.build_audio_cmd(
            audio_path="a.wav", master_out="m.m4a", loudnorm=False,
            preview_out="p.m4a",
            compact_out="c.m4a" if "compact_path" in extra else None,
            total_dur_s=60.0, preview_lead=lead)
        assert _audio_graph(sep) == _audio_graph(one), (extra, _audio_graph(sep), _audio_graph(one))
        # master: same audio encoder arguments; `-shortest` has nothing to compare here
        want = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
        j = sep.index("[am]")
        assert sep[j + 1:j + 7] == want
        assert any(one[k:k + 6] == want for k in range(len(one))), one
        assert "-shortest" not in sep
    # no preview: the plain -af chain
    one = _cmd(audio_path="a.wav", preview_path=None)
    sep = audio_late.build_audio_cmd(audio_path="a.wav", master_out="m.m4a",
                                     loudnorm=False)
    assert sep[sep.index("-af") + 1] == one[one.index("-af") + 1]


def test_where_each_audio_ends():
    # measured on ffmpeg 8.1: 1919 frames at 60 fps -> 1500 audio packets,
    # i.e. 1499 whole encoder frames; 28552 frames -> 22307 packets
    assert audio_late.video_end_sample(1919, 60) == 1499 * 1024
    assert audio_late.video_end_sample(28552, 60) == 22306 * 1024
    # the 30 fps preview of 1980 / 1919 source frames has 990 / 960 frames
    assert audio_late.video_end_sample(1980, 60, 30) == (990 * 1600) // 1024 * 1024
    assert audio_late.video_end_sample(1919, 60, 30) == (960 * 1600) // 1024 * 1024
    # never past the video
    for n in (1, 59, 60, 61, 1919, 28552):
        assert audio_late.video_end_sample(n, 60) <= n * 800


def test_ends_are_counted_at_the_rate_they_are_encoded_at():
    ends = {"master": 1024, "preview": 2048, "compact": 4096}
    g = audio_late.build_audio_cmd(
        audio_path="a.wav", master_out="m.m4a", loudnorm=False,
        preview_out="p.m4a", compact_out="c.m4a", total_dur_s=60.0,
        end_samples=ends)
    g = g[g.index("-filter_complex") + 1].split(";")
    assert "[am_all]atrim=end_sample=1024[am]" in g
    # after loudnorm the preview and the Discord copy are at 192 kHz
    assert "[ap_all]aformat=sample_rates=48000,atrim=end_sample=2048[ap]" in g
    assert "[ac_all]aformat=sample_rates=48000,atrim=end_sample=4096[ac]" in g
    # the hardware preview's audio is ended by time (as in the single process)
    g = audio_late.build_audio_cmd(
        audio_path="a.wav", master_out="m.m4a", loudnorm=False,
        preview_out="p.m4a", total_dur_s=60.0, preview_lead=True,
        end_samples=ends)
    g = g[g.index("-filter_complex") + 1]
    assert "[ap_full]atrim=end=60.000000[ap]" in g and "end_sample=2048" not in g
    # no preview at all
    c = audio_late.build_audio_cmd(audio_path="a.wav", master_out="m.m4a",
                                   loudnorm=False, end_samples=ends)
    assert c[c.index("-af") + 1].endswith(",atrim=end_sample=1024")


def test_the_join_is_a_copy_that_never_trims():
    c = audio_late.build_mux_cmd("v.mp4", "a.m4a", "out.mp4")
    assert c[-3:] == ["-movflags", "+faststart", "out.mp4"]
    assert c[c.index("-c") + 1] == "copy" and "-shortest" not in c
    assert c.count("-i") == 2 and "1:a:0" in c
    silent = audio_late.build_mux_cmd("v.mp4", None, "out.mp4")
    assert silent.count("-i") == 1 and "1:a:0" not in silent
    assert silent[-3:] == ["-movflags", "+faststart", "out.mp4"]


def test_temporary_names_sit_beside_the_outputs_and_are_removed():
    d = tempfile.mkdtemp(prefix="r3d-late-")
    try:
        out = os.path.join(d, "render_raw.mp4")
        la = audio_late.LateAudio(output=out, preview_path=os.path.join(d, "render_raw.embed.mp4"),
                                  compact_path=None, mix=lambda mark: None,
                                  total_dur_s=1.0, preview_lead=False)
        names = {la.video_master.name, la.video_preview.name}
        assert names == {"render_raw.late-video.mp4", "render_raw.embed.late-video.mp4"}
        assert la.video_compact is None
        for p in (la.video_master, la.video_preview):
            open(p, "wb").write(b"x")
        la.cleanup()
        assert os.listdir(d) == []
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_failed_mix_is_raised_on_the_render_thread():
    def boom(mark):
        raise ValueError("no samples")
    la = audio_late.LateAudio(output="x.mp4", preview_path=None, compact_path=None,
                              mix=boom, total_dur_s=1.0, preview_lead=False)
    la.start()
    try:
        la.finish()
    except ValueError as e:
        assert "no samples" in str(e)
    else:
        raise AssertionError("finish() swallowed the worker's error")


def test_no_audio_process_is_started_after_cleanup():
    # a render that fails while the song is still being mixed: cleanup() runs
    # in the `finally`, the worker comes out of the mix afterwards and must
    # not start ffmpeg on names a re-run of the job is about to write
    import threading
    d = tempfile.mkdtemp(prefix="r3d-late-")
    try:
        mixing, go = threading.Event(), threading.Event()
        wav = os.path.join(d, "mix.wav")

        def slow_mix(mark):
            open(wav, "wb").write(b"not audio")
            mixing.set()
            go.wait(5)
            return wav
        la = audio_late.LateAudio(output=os.path.join(d, "render_raw.mp4"),
                                  preview_path=None, compact_path=None,
                                  mix=slow_mix, total_dur_s=1.0, preview_lead=False)
        la.start()
        assert mixing.wait(5)
        la.cleanup()
        go.set()
        la._thread.join(5)
        assert not la._thread.is_alive()
        assert la._proc is None and la._err is None
        assert os.listdir(d) == []          # the mix it finished late is removed too
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _packets(ffmpeg, path, stream):
    out = subprocess.run([ffmpeg, "-v", "error", "-i", path, "-map", f"0:{stream}",
                          "-c", "copy", "-f", "framecrc", "-"],
                         capture_output=True, check=True).stdout.decode()
    return [l for l in out.splitlines() if l and not l.startswith("#")]


def test_real_split_encode_gives_the_master_the_single_process_packets():
    ff = shutil.which("ffmpeg")
    if not ff:
        return
    w, h, fps, n = 64, 64, 60, 127           # 127 frames: the audio must be cut
    d = tempfile.mkdtemp(prefix="r3d-late-")
    try:
        frames = b"".join(
            hashlib.sha256(struct.pack("<I", i)).digest() * (w * h * 3 // 32)
            for i in range(n))
        wav = os.path.join(d, "a.wav")
        with wave.open(wav, "wb") as f:
            f.setnchannels(2); f.setsampwidth(2); f.setframerate(48000)
            m = int(48000 * (n / fps + 0.25))      # longer than the video
            f.writeframes(b"".join(
                struct.pack("<hh", *(int(9000 * math.sin(i * 0.05)),) * 2)
                for i in range(m)))
        common = dict(encoder="libx264", resolution=(w, h), fps=fps, pix_fmt="rgb24",
                      loudnorm=False, crf=30)
        one = os.path.join(d, "one.mp4")
        r = subprocess.run(encode.build_ffmpeg_cmd(output_path=one, audio_path=wav, **common),
                           input=frames, capture_output=True)
        assert r.returncode == 0, r.stderr.decode(errors="replace")[-400:]
        # the same thing in three steps
        v = os.path.join(d, "v.mp4"); a = os.path.join(d, "a.m4a"); two = os.path.join(d, "two.mp4")
        r = subprocess.run(encode.build_ffmpeg_cmd(output_path=v, faststart=False, **common),
                           input=frames, capture_output=True)
        assert r.returncode == 0, r.stderr.decode(errors="replace")[-400:]
        ends = {"master": audio_late.video_end_sample(n, fps)}
        for c in (audio_late.build_audio_cmd(audio_path=wav, master_out=a, loudnorm=False,
                                             end_samples=ends),
                  audio_late.build_mux_cmd(v, a, two)):
            r = subprocess.run(c, capture_output=True)
            assert r.returncode == 0, r.stderr.decode(errors="replace")[-400:]
        assert _packets(ff, two, "v") == _packets(ff, one, "v")
        pa, pb = _packets(ff, one, "a"), _packets(ff, two, "a")
        # every ffmpeg must agree up to the tail; where `-shortest` cuts the
        # last frame is the one thing that has moved between versions
        assert pa[:-2] == pb[:len(pa) - 2] and abs(len(pa) - len(pb)) <= 2, (pa[-3:], pb[-3:])
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    for _n, _f in sorted(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f()
            print("ok  ", _n)
