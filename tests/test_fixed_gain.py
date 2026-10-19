"""Loudness by one fixed gain (record/audio.py, R3D_STD_FIXED_GAIN).

What must hold:
  * with the switch off nothing is built or decoded differently;
  * with it on, the gain stands exactly where `loudnorm` stood in every
    command (and nothing else in the command moves), in both the single
    process and the separate audio process;
  * the gain brings a track to the target, is one number for the whole track,
    and never lifts a sample past the ceiling;
  * the cache keeps the two kinds of normalised song apart;
  * done for real with ffmpeg: a decoded track lands on the target and is the
    plain decode times a constant.
"""
from __future__ import annotations

import math
import os
import shutil
import struct
import subprocess
import tempfile
import tracemalloc
from decimal import Decimal
from pathlib import Path

import numpy as np

from osu_std_renderer.record import audio, audio_late, encode

GAIN = "volume=-1.250000dB"


def _cmd(**kw):
    base = dict(encoder="libx264", resolution=(1280, 720), fps=60,
                output_path="m.mp4", preview_path="m.embed.mp4",
                total_dur_s=60.0, pix_fmt="yuv420p", loudnorm=False,
                audio_path="a.wav")
    base.update(kw)
    return encode.build_ffmpeg_cmd(**base)


CASES = ({}, {"preview_path": None}, {"compact_path": "c.mp4"},
         {"preview_hw": True}, {"preview_hw": True, "compact_path": "c.mp4"},
         {"stream_master": True}, {"stream_master": True, "compact_path": "c.mp4"},
         {"loudnorm": True}, {"loudnorm": True, "preview_path": None},
         {"audio_path": None})


def test_switch_off_builds_the_command_it_always_did():
    for extra in CASES:
        assert _cmd(**extra) == _cmd(mix_gain_db=None, **extra)
        assert not any("volume=" in a for a in _cmd(**extra))


def test_the_gain_stands_where_loudnorm_stood_and_nothing_else_moves():
    for extra in CASES:
        stock = _cmd(**extra)
        fixed = _cmd(mix_gain_db=-1.25, **extra)
        assert fixed == [a.replace(encode.LOUDNORM, GAIN) for a in stock], extra
        assert not any("loudnorm" in a for a in fixed), extra
    # the plain node set-up (limiter on the shared branch, preview normalised)
    g = _cmd(mix_gain_db=-1.25)
    g = g[g.index("-filter_complex") + 1]
    assert f"[aout]asplit=2[am][ap0];[ap0]{GAIN}[ap]" in g
    assert any("volume=3.500000dB" in a for a in _cmd(mix_gain_db=3.5))


def test_the_separate_audio_process_gets_the_same_gain():
    for extra in ({}, {"compact_out": "c.m4a"}, {"preview_lead": True},
                  {"preview_out": None}, {"loudnorm": True},
                  {"end_samples": {"master": 1024, "preview": 2048}}):
        kw = dict(audio_path="a.wav", master_out="m.m4a", loudnorm=False,
                  preview_out="p.m4a", total_dur_s=60.0)
        kw.update(extra)
        stock = audio_late.build_audio_cmd(**kw)
        fixed = audio_late.build_audio_cmd(mix_gain_db=-1.25, **kw)
        assert fixed == [a.replace(encode.LOUDNORM, GAIN) for a in stock], extra
        assert not any("loudnorm" in a for a in fixed), extra
        assert stock == audio_late.build_audio_cmd(mix_gain_db=None, **kw)


def test_gain_arithmetic():
    f = audio.fixed_gain_db
    # a loud master comes down to the target
    assert abs(f(-7.6, 1.0) - (-10.4)) < 1e-9
    # a quiet one goes up, all the way when there is room under the ceiling
    assert abs(f(-30.0, 0.1) - 12.0) < 1e-9
    # and only as far as the ceiling allows when there is not
    want = audio.PEAK_CEILING_DB - 20 * math.log10(0.9)
    assert abs(f(-30.0, 0.9) - want) < 1e-9 and f(-30.0, 0.9) < 12.0
    # float PCM above full scale: the cut already brings it under the ceiling
    assert abs(f(-7.6, 1.2) - (-10.4)) < 1e-9
    # silence (ebur128 reads -70) is left alone, and has no peak to hold it back
    assert f(None, 0.5) == 0.0 and f(-70.0, 0.5) == 0.0 and f(-18.0, 0.0) == 0.0


def test_gain_serialization_never_rounds_above_the_measured_gain():
    for gain in (-1.0544721057769548, 1.0544721057769548, -0.0000001,
                 0.0000001, -1.25, 3.5, -4.1, 0.0):
        applied = Decimal(encode.fixed_gain_filter(gain)[7:-2])
        error = Decimal.from_float(gain) - applied
        assert 0 <= error < Decimal("0.000001"), (gain, applied)


def _transient_wav(path, peak=1.6):
    """The real-audio QA spike fixture: quiet bed, isolated large transients."""
    n = 8 * audio.SAMPLE_RATE
    t = np.arange(n) / audio.SAMPLE_RATE
    x = .002 * np.sin(2 * np.pi * 440 * t)
    x[np.arange(audio.SAMPLE_RATE // 2, n, audio.SAMPLE_RATE)] = peak
    mixer = audio.AudioMixer(7000.)
    mixer.buf[:] = np.repeat(x[:, None], 2, axis=1).astype(np.float32)
    return mixer.write_wav(path)


def test_transient_pcm_peaks_stay_under_the_ceiling_in_both_builders():
    ff = shutil.which("ffmpeg")
    if not ff:
        return
    from tests.test_audio_late import _audio_graph

    with tempfile.TemporaryDirectory(prefix="r3d-gain-peak-") as d:
        d = Path(d)
        ceiling = 10 ** (audio.PEAK_CEILING_DB / 20)
        # Covers positive gain, attenuation, and the limiter's clamp.
        for peak in (.12, .6, .94, 1.6):
            source = _transient_wav(d / "spikes.wav", peak)
            gain = audio.mix_gain_db(source)
            assert gain is not None
            synchronous = _cmd(mix_gain_db=gain, compact_path="c.mp4")
            late = audio_late.build_audio_cmd(
                audio_path=source, master_out="m.m4a", loudnorm=False,
                preview_out="p.m4a", compact_out="c.m4a", total_dur_s=60.,
                mix_gain_db=gain)
            for name, command in (("sync", synchronous), ("late", late)):
                graph = ";".join(_audio_graph(command)).replace("[IN]", "[0:a]")
                # Execute the builders' audio graph as float PCM, before AAC.
                outputs = [d / f"{name}-{label}.wav" for label in ("am", "ap", "ac")]
                cmd = [ff, "-y", "-v", "error", "-i", str(source),
                       "-filter_complex", graph]
                for label, output in zip(("am", "ap", "ac"), outputs):
                    cmd += ["-map", f"[{label}]", "-c:a", "pcm_f32le", str(output)]
                subprocess.run(cmd, check=True, capture_output=True)
                for output in outputs[1:]:  # master retains its .95 mix limiter
                    pcm = audio.decode_to_pcm(output)
                    assert float(np.abs(pcm).max()) <= ceiling, (peak, name)
            if peak == 1.6:
                # Demonstrate the old two-decimal filter fails this fixture.
                old = subprocess.check_output([
                    ff, "-v", "error", "-i", str(source), "-af",
                    f"{audio.MIX_LIMITER},volume={gain:.2f}dB",
                    "-f", "f32le", "-acodec", "pcm_f32le", "-"])
                assert float(np.abs(np.frombuffer(old, dtype="<f4")).max()) > ceiling


def test_block_peak_matches_the_previous_whole_array_calculation():
    with tempfile.TemporaryDirectory(prefix="r3d-gain-scan-") as d:
        path = Path(d) / "pcm.wav"
        chunk = audio._PEAK_SCAN_SAMPLES
        values = np.random.default_rng(17).uniform(-.4, .4, 3 * chunk + 17).astype("<f4")
        for index, value in ((0, -1.6), (chunk - 1, 2.3), (chunk, -3.1),
                             (chunk + 1, 4.2), (len(values) - 1, -5.4)):
            data = values.copy()
            data[index] = value
            path.write_bytes(bytes(44) + data.tobytes())
            assert audio._wav_pcm_peak(path) == float(np.abs(data).max())
        for value in (0., float("inf"), float("nan")):
            data = np.zeros(chunk + 1, dtype="<f4")
            data[chunk - 1] = value
            path.write_bytes(bytes(44) + data.tobytes())
            expected = float(np.abs(data).max())
            got = audio._wav_pcm_peak(path)
            assert math.isnan(got) if math.isnan(expected) else got == expected
        path.write_bytes(bytes(47))
        try:
            audio._wav_pcm_peak(path)
        except ValueError:
            pass
        else:
            raise AssertionError("truncated float PCM was accepted")


def test_block_peak_temporary_memory_is_bounded():
    with tempfile.TemporaryDirectory(prefix="r3d-gain-memory-") as d:
        path = Path(d) / "pcm.wav"
        block = np.full(audio._PEAK_SCAN_SAMPLES, -.8, dtype="<f4").tobytes()
        with path.open("wb") as fh:
            fh.write(bytes(44))
            for _ in range(12):
                fh.write(block)
        tracemalloc.start()
        try:
            assert audio._wav_pcm_peak(path) == float(np.float32(.8))
            _, allocated = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert allocated < 2 * len(block), allocated


SUMMARY = """\
[Parsed_ebur128_1 @ 0x600001] Summary:

  Integrated loudness:
    I:          -7.6 LUFS
    Threshold: -17.8 LUFS

  Loudness range:
    LRA:         2.7 LU
    Threshold: -27.7 LUFS
    LRA low:    -9.2 LUFS
    LRA high:   -6.5 LUFS
"""


def test_the_summary_is_read():
    p = audio.parse_integrated_lufs
    assert p(SUMMARY) == -7.6
    assert p("") is None and p("Stream #0:0: Audio: mp3") is None
    assert p(SUMMARY.replace("-7.6", "-70.0")) == -70.0       # silence: a reading
    assert p(SUMMARY.replace("-7.6", "3.2")) == 3.2
    # "LRA low: -9.2 LUFS" and the thresholds are not the integrated value
    assert p(SUMMARY.replace("    I:          -7.6 LUFS\n", "")) is None


def test_the_cache_keeps_the_two_kinds_of_song_apart():
    d = tempfile.mkdtemp(prefix="r3d-gain-")
    try:
        src = Path(d) / "song.bin"
        src.write_bytes(b"not really audio")
        a = audio._loudnorm_cache_key(src, 1.0, False, audio._LOUDNORM_FILTER)
        b = audio._loudnorm_cache_key(src, 1.0, False, audio.FIXED_GAIN_PARAM)
        assert a != b
        assert "loudnorm" not in audio.FIXED_GAIN_PARAM
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _wav(path, seconds, level, rate=48000, channels=2):
    """A tone with a slow swell, so there is something for a gain rider to
    ride: float32 wav, as the mixer writes it."""
    n = int(seconds * rate)
    t = np.arange(n) / rate
    x = level * (0.35 + 0.65 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t))) \
        * np.sin(2 * np.pi * 440.0 * t)
    data = np.repeat(x[:, None], channels, axis=1).astype("<f4").tobytes()
    with open(path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt ")
        fh.write(struct.pack("<IHHIIHH", 16, 3, channels, rate,
                             rate * channels * 4, channels * 4, 32))
        fh.write(b"data" + struct.pack("<I", len(data)) + data)
    return path


def _measure(ffmpeg, path):
    r = subprocess.run([ffmpeg, "-hide_banner", "-nostats", "-loglevel", "info",
                        "-i", str(path), "-af", "ebur128=framelog=quiet",
                        "-f", "null", "-"], capture_output=True)
    return audio.parse_integrated_lufs(r.stderr.decode(errors="replace"))


def _with_cache(d):
    old = {k: os.environ.get(k) for k in ("R3D_LOUDNORM_CACHE_DIR",
                                          "R3D_NO_LOUDNORM_CACHE")}
    os.environ["R3D_LOUDNORM_CACHE_DIR"] = str(d)
    os.environ.pop("R3D_NO_LOUDNORM_CACHE", None)
    audio._CACHE_DIR_RESOLVED.clear()

    def restore():
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        audio._CACHE_DIR_RESOLVED.clear()
    return restore


def test_real_decode_lands_on_the_target_with_one_gain():
    ff = shutil.which("ffmpeg")
    if not ff:
        return
    d = tempfile.mkdtemp(prefix="r3d-gain-")
    restore = _with_cache(Path(d) / "cache")
    try:
        for level, name in ((0.9, "loud"), (0.02, "quiet")):
            src = _wav(Path(d) / f"{name}.wav", 8.0, level)
            plain = audio.decode_to_pcm(src)
            fixed = audio.decode_to_pcm(src, loudnorm=True, fixed_gain=True)
            assert fixed.shape == plain.shape and fixed.dtype == np.float32
            # one gain for the whole track: the plain decode times a constant
            k = float(np.abs(fixed).max() / np.abs(plain).max())
            assert np.allclose(fixed, plain * k, rtol=0, atol=2e-6), name
            assert (k < 1.0) == (name == "loud"), (name, k)
            # on the target (ebur128 prints one decimal: the gain is that fine)
            out = audio.AudioMixer(8000.0)
            out.buf[:len(fixed)] = fixed[:len(out.buf)]
            got = _measure(ff, out.write_wav(Path(d) / f"{name}.out.wav"))
            assert got is not None and abs(got - audio.TARGET_LUFS) <= 0.15, (name, got)
            assert float(np.abs(fixed).max()) <= 10 ** (audio.PEAK_CEILING_DB / 20) + 1e-6
            # the second decode comes from the cache, and it is the same samples
            again = audio.decode_to_pcm(src, loudnorm=True, fixed_gain=True)
            assert np.array_equal(again, fixed), name
        # the two kinds of normalised song have their own cache entries
        assert len(list((Path(d) / "cache").glob("*." + audio._CACHE_EXT))) == 2
        # and without `loudnorm` the switch does nothing at all
        src = Path(d) / "loud.wav"
        assert np.array_equal(audio.decode_to_pcm(src, fixed_gain=True),
                              audio.decode_to_pcm(src))
        # an ffmpeg whose summary cannot be read: the stock filter's samples,
        # and nothing is cached under the fixed-gain name
        real, audio.parse_integrated_lufs = audio.parse_integrated_lufs, lambda t: None
        try:
            src = _wav(Path(d) / "other.wav", 4.0, 0.5)
            got = audio.decode_to_pcm(src, loudnorm=True, fixed_gain=True)
        finally:
            audio.parse_integrated_lufs = real
        assert np.array_equal(got, audio.decode_to_pcm(src, loudnorm=True))
        assert len(list((Path(d) / "cache").glob("*." + audio._CACHE_EXT))) == 3
        # silence stays silence
        quiet = audio.decode_to_pcm(_wav(Path(d) / "silent.wav", 3.0, 0.0),
                                    loudnorm=True, fixed_gain=True)
        assert not quiet.any()
    finally:
        restore()
        shutil.rmtree(d, ignore_errors=True)


def _noise(path, seconds, level=0.25, rate=48000, seed=7):
    """Steady band-limited noise: nothing periodic, so two time-stretches of it
    only line up if they are the same stretch."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(seconds * rate))
    x = np.convolve(x, np.ones(8) / 8.0, mode="same") * level * 2.0
    data = np.repeat(x[:, None], 2, axis=1).astype("<f4").tobytes()
    with open(path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt ")
        fh.write(struct.pack("<IHHIIHH", 16, 3, 2, rate, rate * 8, 8, 32))
        fh.write(b"data" + struct.pack("<I", len(data)) + data)
    return Path(path)


def _as_aac(wav):
    """The same sound as an .m4a. Songs are mp3 or ogg, whose decoders hand
    ffmpeg PLANAR samples; that is the case in which the stock chain stretches
    at 192 kHz (a float wav does not trigger it). AAC decodes planar too, and
    every ffmpeg can encode it."""
    out = Path(str(wav)[:-4] + ".m4a")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i",
                    str(wav), "-c:a", "aac", "-b:a", "192k", str(out)], check=True)
    return out


def _corr0(a, b):
    """Correlation at lag 0 of two equally long signals."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


def test_a_speed_changed_song_is_stretched_exactly_as_stock():
    # where the chain is pinned to loudnorm's rate, and where it is not
    eb = "ebur128=framelog=quiet"
    assert audio.fixed_gain_chain("") == eb
    assert audio.fixed_gain_chain("atempo=1.5") == \
        "atempo=1.5,aformat=sample_rates=192000," + eb
    for rate, pitch in ((1.5, False), (0.75, False), (1.5, True), (1.3, True)):
        af = audio.rate_audio_filter(rate, pitch)
        want = af + (",aformat=sample_rates=192000" if "atempo" in af else "") + "," + eb
        assert audio.fixed_gain_chain(af) == want
    ff = shutil.which("ffmpeg")
    if not ff:
        return
    d = tempfile.mkdtemp(prefix="r3d-gain-")
    restore = _with_cache(Path(d) / "cache")
    try:
        src = _as_aac(_noise(Path(d) / "noise.wav", 6.0))
        for rate in (1.5, 0.75):
            stock = audio.decode_to_pcm(src, rate=rate, loudnorm=True)
            fixed = audio.decode_to_pcm(src, rate=rate, loudnorm=True, fixed_gain=True)
            # the same stretch: the same length to the sample, and lined up
            assert fixed.shape == stock.shape, (rate, fixed.shape, stock.shape)
            assert _corr0(fixed, stock) >= 0.97, (rate, _corr0(fixed, stock))
    finally:
        restore()
        shutil.rmtree(d, ignore_errors=True)


def test_corrected_time_stretch_does_not_reuse_older_fixed_gain_cache():
    if not shutil.which("ffmpeg"):
        return
    d = tempfile.mkdtemp(prefix="r3d-gain-")
    cache = Path(d) / "cache"
    restore = _with_cache(cache)
    try:
        src = _as_aac(_noise(Path(d) / "noise.wav", 6.0))
        for rate in (1.5, 0.75):
            stock = audio.decode_to_pcm(src, rate=rate, loudnorm=True)
            # The earlier PR revision cached its native-rate stretch under
            # this key. It is valid PCM, but must not bypass the 192 kHz fix.
            old_key = audio._loudnorm_cache_key(
                src, rate, False, audio.FIXED_GAIN_PARAM)
            old_path = cache / f"{old_key}.{audio._CACHE_EXT}"
            obsolete = np.zeros((7, 2), dtype="<f4").tobytes()
            old_path.write_bytes(obsolete)
            fixed = audio.decode_to_pcm(
                src, rate=rate, loudnorm=True, fixed_gain=True)
            assert fixed.shape == stock.shape, (rate, fixed.shape, stock.shape)
            assert _corr0(fixed, stock) >= 0.97
            assert old_path.read_bytes() == obsolete
            assert np.array_equal(fixed, audio.decode_to_pcm(
                src, rate=rate, loudnorm=True, fixed_gain=True))
        assert len(list(cache.glob("*." + audio._CACHE_EXT))) == 6
    finally:
        restore()
        shutil.rmtree(d, ignore_errors=True)


def test_real_mix_gain_is_the_distance_to_the_target():
    ff = shutil.which("ffmpeg")
    if not ff:
        return
    d = tempfile.mkdtemp(prefix="r3d-gain-")
    try:
        mix = _wav(Path(d) / "mix.wav", 8.0, 0.5)
        g = audio.mix_gain_db(mix)
        lim = subprocess.run([ff, "-hide_banner", "-nostats", "-loglevel", "info",
                              "-i", str(mix), "-af",
                              audio.MIX_LIMITER + ",ebur128=framelog=quiet",
                              "-f", "null", "-"], capture_output=True)
        lufs = audio.parse_integrated_lufs(lim.stderr.decode(errors="replace"))
        assert g is not None and abs(g - (audio.TARGET_LUFS - lufs)) < 1e-9
        # apply it the way the encode does and measure the result
        out = Path(d) / "out.wav"
        r = subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", "-i",
                            str(mix), "-af", f"{audio.MIX_LIMITER},{encode.fixed_gain_filter(g)}",
                            "-c:a", "pcm_f32le", str(out)], capture_output=True)
        assert r.returncode == 0, r.stderr.decode(errors="replace")[-300:]
        got = _measure(ff, out)
        assert abs(got - audio.TARGET_LUFS) <= 0.15, got
        # a file that is not audio cannot be measured: the caller keeps loudnorm
        bad = Path(d) / "bad.wav"
        bad.write_bytes(b"x" * 100)
        assert audio.mix_gain_db(bad) is None
        # silence: nothing to measure either
        assert audio.mix_gain_db(_wav(Path(d) / "silent.wav", 3.0, 0.0)) is None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_the_late_audio_worker_measures_the_mix_it_made():
    ff = shutil.which("ffmpeg")
    if not ff:
        return
    d = tempfile.mkdtemp(prefix="r3d-gain-")
    try:
        mix = Path(d) / "mix.wav"
        for fixed in (False, True):
            la = audio_late.LateAudio(
                output=Path(d) / "render_raw.mp4",
                preview_path=Path(d) / "render_raw.embed.mp4", compact_path=None,
                mix=lambda mark: _wav(mix, 4.0, 0.5), total_dur_s=4.0,
                preview_lead=False, fixed_gain=fixed)
            la.start()
            la._thread.join(60)
            assert la._err is None, la._err
            assert (la.mix_gain_db is not None) == fixed
            if fixed:
                assert abs(la.mix_gain_db - audio.mix_gain_db(_wav(mix, 4.0, 0.5))) < 1e-9
            la.cleanup()
        assert os.listdir(d) == []
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    for _n, _f in sorted(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f()
            print("ok  ", _n)
