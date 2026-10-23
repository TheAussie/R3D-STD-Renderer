import lzma
import tempfile
from pathlib import Path

from osu_std_renderer.security import bounded_lzma_decompress, ffmpeg_file_input_args


def test_bounded_lzma_decompress_accepts_small_payload():
    payload = b"frame-data" * 100
    assert bounded_lzma_decompress(lzma.compress(payload)) == payload


def test_bounded_lzma_decompress_rejects_bomb_before_materialising_it():
    compressed = lzma.compress(b"x" * (2 * 1024 * 1024))
    try:
        bounded_lzma_decompress(compressed, max_output=1024 * 1024)
    except ValueError as exc:
        assert "output limit" in str(exc)
    else:
        raise AssertionError("oversized LZMA stream was accepted")


def test_ffmpeg_file_input_args_forces_local_protocol_and_known_demuxer():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "audio.mp3"
        args = ffmpeg_file_input_args(path)
        assert args[:4] == ["-protocol_whitelist", "file", "-f", "mp3"]
        assert args[-2:] == ["-i", str(path)]

def test_bounded_lzma_decompress_rejects_concatenated_streams():
    first = lzma.compress(b"A" * 8, format=lzma.FORMAT_ALONE)
    second = lzma.compress(b"B" * 256, format=lzma.FORMAT_ALONE)

    payload = first + second

    try:
        bounded_lzma_decompress(payload, max_output=64)
    except ValueError as exc:
        assert "trailing compressed data" in str(exc)
    else:
        raise AssertionError(
            "Security regression: concatenated streams bypassed the limit"
        )

def test_ffmpeg_rejects_unknown_media_extension():
    try:
        ffmpeg_file_input_args(Path("/tmp/upload.bin"))
    except ValueError as exc:
        assert "Unsupported FFmpeg media extension" in str(exc)
    else:
        raise AssertionError(
            "Unknown media extension allowed FFmpeg auto-probing"
        )

def test_ffmpeg_supported_beatmap_video_extensions():
    for extension, demuxer in (
        (".avi", "avi"),
        (".flv", "flv"),
        (".mp4", "mov"),
    ):
        args = ffmpeg_file_input_args(
            Path(f"/tmp/video{extension}")
        )
        assert args[2:4] == ["-f", demuxer]
