import shutil
from pathlib import Path

import numpy as np
import pytest

from autobot_stt.services.audio_decoder import (
    AudioDecodeError,
    decode_webm_opus_to_pcm,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not installed",
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample.webm"


def test_decode_returns_float32_mono_array() -> None:
    pcm = decode_webm_opus_to_pcm(FIXTURE_PATH.read_bytes())
    assert pcm.dtype == np.float32
    assert pcm.ndim == 1
    assert len(pcm) > 0


def test_decode_sample_rate() -> None:
    pcm = decode_webm_opus_to_pcm(FIXTURE_PATH.read_bytes(), sample_rate=16000)
    # 0.5 s clip at 16 kHz -> ~8000 samples (tolerance for encoder padding)
    assert 7900 <= len(pcm) <= 8100


def test_decode_values_in_normalized_range() -> None:
    pcm = decode_webm_opus_to_pcm(FIXTURE_PATH.read_bytes())
    assert pcm.min() >= -1.0
    assert pcm.max() <= 1.0


def test_decode_custom_sample_rate() -> None:
    pcm_16k = decode_webm_opus_to_pcm(FIXTURE_PATH.read_bytes(), sample_rate=16000)
    pcm_8k = decode_webm_opus_to_pcm(FIXTURE_PATH.read_bytes(), sample_rate=8000)
    assert len(pcm_8k) == pytest.approx(len(pcm_16k) / 2, rel=0.05)


def test_garbage_input_raises_audio_decode_error() -> None:
    with pytest.raises(AudioDecodeError):
        decode_webm_opus_to_pcm(b"not webm at all")


def test_empty_bytes_raises_audio_decode_error() -> None:
    with pytest.raises(AudioDecodeError, match="Empty"):
        decode_webm_opus_to_pcm(b"")
