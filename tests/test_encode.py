"""Working encoder selection and errors from a child that closes its frame pipe."""
from __future__ import annotations

import io
import subprocess
import sys
import time
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from osu_std_renderer.record import encode


def _nvenc_cmd(encoder="h264_nvenc", **kwargs):
    options = dict(encoder=encoder, resolution=(1280, 720), fps=60,
                   output_path="master.mp4")
    options.update(kwargs)
    with patch.dict("os.environ", {"R3D_STD_NULL_SINK": "0", "R3D_PREVIEW_LIVE": "0"}):
        return encode.build_ffmpeg_cmd(**options)


def _option(command, name):
    return command[command.index(name) + 1]


_NVENC_LADDER_ANCHORS = (
    (128, 128, 1, 2_500_000), (1280, 720, 30, 4_000_000),
    (1280, 720, 60, 6_498_019), (1920, 1080, 60, 11_463_276),
    (2560, 1440, 60, 16_000_000), (1920, 1080, 120, 16_000_000))


def test_h264_nvenc_auto_bitrate_uses_cq23_with_existing_ladder_caps():
    for w, h, fps, target in _NVENC_LADDER_ANCHORS:
        command = _nvenc_cmd("h264_nvenc", resolution=(w, h), fps=fps)
        assert encode.nvenc_target_bps(w, h, fps) == target
        assert _option(command, "-rc") == "vbr"
        assert _option(command, "-cq") == "23"
        assert _option(command, "-b:v") == "0"
        assert _option(command, "-maxrate") == str(int(target * 1.5))
        assert _option(command, "-bufsize") == str(target * 2)
        assert command.index("-c:v") < command.index("-cq") < command.index("master.mp4")


def test_hevc_nvenc_auto_bitrate_preserves_existing_target_vbr_ladder():
    for w, h, fps, target in _NVENC_LADDER_ANCHORS:
        command = _nvenc_cmd("hevc_nvenc", resolution=(w, h), fps=fps)
        assert _option(command, "-rc") == "vbr"
        assert "-cq" not in command
        assert _option(command, "-b:v") == str(target)
        assert _option(command, "-maxrate") == str(int(target * 1.5))
        assert _option(command, "-bufsize") == str(target * 2)


def test_nvenc_explicit_video_bitrate_keeps_its_vbr_target_contract():
    for encoder in ("h264_nvenc", "hevc_nvenc"):
        for bitrate in (1_000_000, 24_000_000, 42_000_000):
            command = _nvenc_cmd(encoder, video_bitrate=bitrate)
            assert _option(command, "-rc") == "vbr"
            assert _option(command, "-b:v") == str(bitrate)
            assert _option(command, "-maxrate") == str(int(bitrate * 1.5))
            assert _option(command, "-bufsize") == str(bitrate * 2)
            assert "-cq" not in command


def test_nvenc_default_quality_is_independent_of_x264_crf():
    for encoder in ("h264_nvenc", "hevc_nvenc"):
        assert _nvenc_cmd(encoder, crf=10) == _nvenc_cmd(encoder, crf=40)
        # Zero has always selected the automatic ladder, rather than a 0 bps override.
        assert _nvenc_cmd(encoder, video_bitrate=0) == _nvenc_cmd(encoder)


def test_nvenc_codec_policies_are_scoped_to_the_master_output():
    for encoder in ("h264_nvenc", "hevc_nvenc"):
        for stream in (False, True):
            for hardware_preview in (False, True):
                command = _nvenc_cmd(
                    encoder, audio_path="mix.wav", preview_path="preview.embed.mp4",
                    compact_path="compact.mp4", total_dur_s=180., loudnorm=False,
                    stream_master=stream, preview_hw=hardware_preview)
                master = command[:command.index("master.mp4")]
                side_outputs = command[command.index("master.mp4") + 1:]
                assert master.index("[vm]") < master.index("-c:v") < master.index("-rc")
                if encoder == "h264_nvenc":
                    assert _option(master, "-cq") == "23"
                    assert _option(master, "-b:v") == "0"
                else:
                    assert "-cq" not in master
                    assert _option(master, "-b:v") == "6498019"
                assert "-cq" not in side_outputs and "-rc" not in side_outputs
                assert command.count(encoder) == 1
                assert ("+faststart" in master) == (not stream)
                assert ("h264_videotoolbox" in side_outputs) == hardware_preview
                assert "libx264" in side_outputs


def test_nvenc_policy_does_not_leak_into_vaapi_or_x264_commands():
    for encoder in ("h264_vaapi", "libx264"):
        for bitrate in (None, 4_000_000):
            command = _nvenc_cmd(encoder, video_bitrate=bitrate,
                                 preview_path="preview.embed.mp4", compact_path="compact.mp4",
                                 total_dur_s=180.)
            assert "-cq" not in command and "-rc" not in command
            master = command[:command.index("master.mp4")]
            if encoder == "h264_vaapi":
                assert _option(master, "-vaapi_device") == "/dev/dri/renderD128"
                assert "format=nv12,hwupload" in _option(command, "-filter_complex")
                assert _option(master, "-b:v") == str(bitrate or 6_498_019)
            elif bitrate:
                assert _option(master, "-b:v") == str(bitrate)
            else:
                assert "-crf" in master and "-b:v" not in master


def _listing(*names):
    return SimpleNamespace(returncode=0, stderr="", stdout="\n".join(
        f" V....D {name} H.264 encoder" for name in names))


def _error(call, message):
    try:
        call()
    except encode.EncoderError as exc:
        assert message in str(exc), str(exc)
        return str(exc)
    raise AssertionError("Expected an encoder error")


def _auto(outcomes, device="/dev/dri/renderD128", device_exists=True):
    log = io.StringIO()
    with patch.object(encode.shutil, "which", return_value="/usr/bin/ffmpeg"), \
            patch.object(encode.subprocess, "run", return_value=_listing(
                "h264_nvenc", "h264_vaapi", "libx264")), \
            patch.object(encode.Path, "exists", return_value=device_exists), \
            patch.object(encode, "_encoder_usable", side_effect=outcomes) as probe, \
            redirect_stderr(log):
        result = encode.probe_encoder("auto", device)
        return result, probe.call_args_list, log.getvalue()


def test_auto_rejects_compiled_nvenc_without_driver_and_uses_vaapi():
    selected, calls, log = _auto([(False, "Cannot load libcuda.so.1"), (True, "")])
    assert selected == "h264_vaapi"
    assert [call.args[1] for call in calls] == ["h264_nvenc", "h264_vaapi"]
    assert calls[-1].args[2] == "/dev/dri/renderD128"
    assert "Cannot load libcuda.so.1" in log and "auto selected h264_vaapi" in log


def test_auto_falls_back_to_software_when_both_hardware_drivers_fail():
    selected, calls, _ = _auto([
        (False, "CUDA unavailable"), (False, "VAAPI unavailable"), (True, "")])
    assert selected == "libx264" and len(calls) == 3


def test_auto_keeps_working_nvenc_preference():
    selected, calls, _ = _auto([(True, "")])
    assert selected == "h264_nvenc" and len(calls) == 1


def test_auto_skips_vaapi_when_no_render_device_exists():
    selected, calls, _ = _auto([(False, "CUDA unavailable"), (True, "")],
                             device=None, device_exists=False)
    assert selected == "libx264"
    assert [call.args[1] for call in calls] == ["h264_nvenc", "libx264"]


def test_auto_uses_custom_device_even_when_default_device_is_absent():
    selected, calls, _ = _auto([(False, "CUDA unavailable"), (True, "")],
                             device="/dev/dri/renderD129", device_exists=False)
    assert selected == "h264_vaapi" and calls[-1].args[2] == "/dev/dri/renderD129"


def test_explicit_encoder_is_not_replaced_or_probed():
    with patch.object(encode.shutil, "which") as which, \
            patch.object(encode, "_encoder_usable") as probe:
        for name in ("h264_nvenc", "h264_vaapi", "libx264"):
            assert encode.probe_encoder(name) == name
        which.assert_not_called()
        probe.assert_not_called()


def test_missing_ffmpeg_is_reported_before_rendering():
    with patch.object(encode.shutil, "which", return_value=None):
        _error(encode.probe_encoder, "ffmpeg not found")


def test_auto_reports_all_encoder_failures():
    with patch.object(encode.shutil, "which", return_value="ffmpeg"), \
            patch.object(encode.Path, "exists", return_value=True), \
            patch.object(encode.subprocess, "run", return_value=_listing(
                "h264_nvenc", "h264_vaapi", "libx264")), \
            patch.object(encode, "_encoder_usable", side_effect=[
                (False, "CUDA unavailable"), (False, "VAAPI unavailable"),
                (False, "software unavailable")]), redirect_stderr(io.StringIO()):
        message = _error(encode.probe_encoder, "No usable H.264 encoder")
        assert all(reason in message for reason in (
            "CUDA unavailable", "VAAPI unavailable", "software unavailable"))


def test_encoder_query_failure_is_actionable():
    failures = [OSError("ffmpeg cannot start"), subprocess.TimeoutExpired("ffmpeg", 5)]
    for failure in failures:
        with patch.object(encode.shutil, "which", return_value="ffmpeg"), \
                patch.object(encode.subprocess, "run", side_effect=failure):
            _error(encode.probe_encoder, "Unable to query ffmpeg encoders")
    result = SimpleNamespace(returncode=1, stderr="ffmpeg setup failed", stdout="")
    with patch.object(encode.shutil, "which", return_value="ffmpeg"), \
            patch.object(encode.subprocess, "run", return_value=result):
        _error(encode.probe_encoder, "ffmpeg setup failed")


def test_encoder_probe_sends_a_frame_and_uploads_to_selected_vaapi_device():
    with patch.object(encode.subprocess, "run", return_value=SimpleNamespace(
            returncode=0, stderr=b"")) as run:
        assert encode._encoder_usable("ffmpeg", "h264_vaapi", "/dev/dri/renderD129") == (
            True, "")
    command = run.call_args.args[0]
    assert command[command.index("-vaapi_device") + 1] == "/dev/dri/renderD129"
    assert command[command.index("-vf") + 1] == "format=nv12,hwupload"
    assert len(run.call_args.kwargs["input"]) == 128 * 128 * 3
    assert run.call_args.kwargs["timeout"] == 5


def test_failed_encoder_probe_retains_driver_error_and_bounds_startup():
    result = SimpleNamespace(returncode=1, stderr=b"Cannot load libcuda.so.1\nOther details")
    with patch.object(encode.subprocess, "run", return_value=result):
        assert encode._encoder_usable("ffmpeg", "h264_nvenc", None) == (
            False, "Cannot load libcuda.so.1")
    for error in (OSError("driver unavailable"), subprocess.TimeoutExpired("ffmpeg", 5)):
        with patch.object(encode.subprocess, "run", side_effect=error):
            usable, reason = encode._encoder_usable("ffmpeg", "h264_nvenc", None)
            assert not usable and reason


def test_broken_frame_pipe_includes_child_error_and_exit_code():
    # A real child exits before reading a frame, reproducing missing hardware drivers.
    pipe = encode.FfmpegPipe([sys.executable, "-c",
                             "import sys; sys.stderr.write('encoder driver unavailable\\n'); "
                             "sys.exit(7)"])
    try:
        def render():
            with pipe:
                pipe.proc.wait(timeout=3)
                pipe.push(np.zeros((128, 128, 3), dtype=np.uint8))
                deadline = time.monotonic() + 3
                while pipe._werr is None and time.monotonic() < deadline:
                    time.sleep(0.005)
                assert isinstance(pipe._werr, BrokenPipeError)
                pipe.push(np.zeros((128, 128, 3), dtype=np.uint8))

        message = _error(render, "ffmpeg exited 7")
        assert "encoder driver unavailable" in message
        assert not pipe._thread.is_alive() and pipe.proc.poll() == 7
    finally:
        if pipe.proc is not None and pipe.proc.stderr is not None:
            pipe.proc.stderr.close()


def test_unrelated_render_exception_is_preserved_with_child_diagnostic():
    pipe = encode.FfmpegPipe([sys.executable, "-c",
                             "import sys; sys.stderr.write('encoder driver unavailable\\n'); "
                             "sys.exit(7)"])
    log = io.StringIO()
    try:
        try:
            with redirect_stderr(log), pipe:
                pipe.proc.wait(timeout=3)
                raise ValueError("scene draw failed")
        except ValueError as exc:
            assert str(exc) == "scene draw failed"
        else:
            raise AssertionError("The scene error must be preserved")
        assert "ffmpeg exited 7" in log.getvalue()
        assert "encoder driver unavailable" in log.getvalue()
    finally:
        if pipe.proc is not None and pipe.proc.stderr is not None:
            pipe.proc.stderr.close()
