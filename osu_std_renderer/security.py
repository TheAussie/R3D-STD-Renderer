"""Small, fail-closed guards for attacker-controlled renderer inputs."""
from __future__ import annotations

import lzma
import struct
from pathlib import Path


# Real replay frame streams are a few MiB.  This is deliberately generous while
# preventing a compact LZMA payload from materialising gigabytes in a worker.
MAX_REPLAY_DECOMPRESSED_BYTES = 32 * 1024 * 1024
MAX_REPLAY_FILE_BYTES = 64 * 1024 * 1024
_DECOMPRESS_CHUNK_BYTES = 1024 * 1024
_FFMPEG_DEMUXERS = {
    ".aac": "aac", ".flac": "flac", ".m4a": "mov", ".mkv": "matroska",
    ".mp3": "mp3", ".mp4": "mov", ".ogg": "ogg", ".opus": "ogg",
    ".wav": "wav", ".webm": "matroska",
    ".avi": "avi",
    ".flv": "flv",
}


def bounded_lzma_decompress(
    compressed: bytes,
    *,
    format: int = lzma.FORMAT_AUTO,
    max_output: int = MAX_REPLAY_DECOMPRESSED_BYTES,
) -> bytes:
    """Decode one LZMA stream without permitting unbounded output."""
    if max_output <= 0:
        raise ValueError("max_output must be positive")
    decoder = lzma.LZMADecompressor(format=format)
    chunks: list[bytes] = []
    total = 0
    source = compressed
    while True:
        limit = min(_DECOMPRESS_CHUNK_BYTES, max_output - total + 1)
        chunk = decoder.decompress(source, max_length=limit)
        source = b""
        total += len(chunk)
        if total > max_output:
            raise ValueError("LZMA payload exceeds replay output limit")
        chunks.append(chunk)
        if decoder.eof:
            if decoder.unused_data:
                raise ValueError("trailing compressed data after LZMA stream")
            return b"".join(chunks)
        if decoder.needs_input:
            raise lzma.LZMAError("truncated LZMA payload")


def _skip_osr_string(data: bytes, offset: int) -> int:
    if offset >= len(data):
        raise ValueError("truncated replay header")
    marker = data[offset]
    offset += 1
    if marker == 0x00:
        return offset
    if marker != 0x0B:
        raise ValueError("invalid replay string marker")
    length = shift = 0
    while True:
        if offset >= len(data) or shift > 63:
            raise ValueError("truncated replay string length")
        byte = data[offset]
        offset += 1
        length |= (byte & 0x7F) << shift
        if not byte & 0x80:
            break
        shift += 7
    if length > len(data) - offset:
        raise ValueError("truncated replay string")
    return offset + length


def validate_replay_payload(path: Path) -> None:
    """Preflight the legacy replay stream before ``osrparse`` decodes it.

    ``osrparse`` currently calls ``lzma.decompress`` without an output cap.  A
    successful bounded preflight makes its subsequent decode safe: it sees the
    exact same, already-proven-small byte stream.
    """
    path = Path(path)
    if path.stat().st_size > MAX_REPLAY_FILE_BYTES:
        raise ValueError("replay file exceeds input size limit")
    data = path.read_bytes()
    offset = 1 + 4
    for _ in range(3):
        offset = _skip_osr_string(data, offset)
    fixed_header = 2 * 6 + 4 + 2 + 1 + 4
    if offset + fixed_header > len(data):
        raise ValueError("truncated replay header")
    offset += fixed_header
    offset = _skip_osr_string(data, offset)
    if offset + 8 + 4 > len(data):
        raise ValueError("truncated replay header")
    offset += 8
    (replay_length,) = struct.unpack_from("<i", data, offset)
    offset += 4
    if replay_length < 0 or replay_length > len(data) - offset:
        raise ValueError("invalid replay payload length")
    bounded_lzma_decompress(data[offset:offset + replay_length])


def ffmpeg_file_input_args(path: Path) -> list[str]:
    """Allow only explicitly supported local media formats."""
    path = Path(path)
    demuxer = _FFMPEG_DEMUXERS.get(path.suffix.lower())

    if demuxer is None:
        raise ValueError(
            f"Unsupported FFmpeg media extension: {path.suffix or '(none)'}"
        )

    return [
        "-protocol_whitelist", "file",
        "-f", demuxer,
        "-i", str(path),
    ]
