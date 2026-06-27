"""Audio decoding service: WebM/Opus bytes -> 16 kHz mono float32 PCM."""

from __future__ import annotations

import subprocess

import numpy as np

_STDERR_TRUNCATE = 500


class AudioDecodeError(Exception):
    """Raised when audio input cannot be decoded to PCM."""


def decode_webm_opus_to_pcm(audio_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Decode WebM/Opus ``audio_bytes`` into a mono float32 PCM ndarray.

    The returned array is 1-D, ``float32``, normalized to ``[-1, 1]``, and
    resampled to ``sample_rate`` Hz. The output is suitable for direct
    consumption by Whisper.

    Args:
        audio_bytes: WebM container bytes with Opus codec (e.g. from a browser
            ``MediaRecorder``).
        sample_rate: Target sample rate in Hz. Defaults to 16000 for Whisper.

    Returns:
        ``np.ndarray`` with ``dtype=float32`` and ``ndim == 1``.

    Raises:
        AudioDecodeError: If ``audio_bytes`` is empty, ffmpeg exits non-zero,
            or ffmpeg produces no audio output.
        FileNotFoundError: If ffmpeg is not installed on the host.
    """
    if not audio_bytes:
        raise AudioDecodeError("Empty audio input")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        "pipe:0",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]

    result = subprocess.run(  # noqa: S603 - command is fully specified
        cmd,
        input=audio_bytes,
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        if len(stderr) > _STDERR_TRUNCATE:
            stderr = stderr[:_STDERR_TRUNCATE] + "..."
        raise AudioDecodeError(f"ffmpeg failed to decode audio: {stderr}")

    if not result.stdout:
        raise AudioDecodeError("ffmpeg produced no audio output")

    return np.frombuffer(result.stdout, dtype=np.float32).copy()
