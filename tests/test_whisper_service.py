from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autobot_stt.config import Settings
from autobot_stt.services.whisper_service import (
    INITIAL_PROMPT_MAX_CHARS,
    WhisperService,
    build_initial_prompt,
)

# --- build_initial_prompt -------------------------------------------------


def test_build_initial_prompt_draft_only() -> None:
    prompt = build_initial_prompt("Hello world", [])
    assert prompt == "Hello world"


def test_build_initial_prompt_draft_and_messages() -> None:
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    prompt = build_initial_prompt("draft", history)
    assert prompt == "draft\nuser: hi\nassistant: hello"


def test_build_initial_prompt_caps_at_max_messages() -> None:
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
        {"role": "assistant", "content": "fourth"},
        {"role": "user", "content": "fifth"},
    ]
    prompt = build_initial_prompt("", history, max_messages=2)
    assert prompt == "assistant: fourth\nuser: fifth"


def test_build_initial_prompt_empty_returns_empty_string() -> None:
    assert build_initial_prompt("", []) == ""
    assert build_initial_prompt("   \n", []) == ""


def test_build_initial_prompt_strips_draft_whitespace() -> None:
    assert build_initial_prompt("  spaced  ", []) == "spaced"


def test_build_initial_prompt_skips_empty_message_content() -> None:
    history = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "   "},
        {"role": "user", "content": "real"},
    ]
    prompt = build_initial_prompt("", history)
    assert prompt == "user: real"


def test_build_initial_prompt_truncates_to_max_chars_keeping_tail() -> None:
    long_message = "x" * (INITIAL_PROMPT_MAX_CHARS + 200)
    history = [{"role": "user", "content": long_message}]
    prompt = build_initial_prompt("draft", history)
    assert len(prompt) == INITIAL_PROMPT_MAX_CHARS
    assert prompt.endswith("x")


# --- WhisperService.load --------------------------------------------------


@patch("autobot_stt.services.whisper_service.WhisperModel")
def test_load_constructs_whisper_model_with_settings(mock_model_cls: MagicMock) -> None:
    service = WhisperService(Settings(whisper_model="small", whisper_device="cuda"))
    service.load()
    mock_model_cls.assert_called_once_with(
        "small",
        device="cuda",
        compute_type="float16",
    )


@patch("autobot_stt.services.whisper_service.WhisperModel")
def test_load_is_idempotent(mock_model_cls: MagicMock) -> None:
    service = WhisperService(Settings())
    service.load()
    service.load()
    mock_model_cls.assert_called_once()


@patch("autobot_stt.services.whisper_service.WhisperModel")
def test_load_cpu_uses_int8_compute_type(mock_model_cls: MagicMock) -> None:
    service = WhisperService(Settings(whisper_device="cpu"))
    service.load()
    assert mock_model_cls.call_args.kwargs["compute_type"] == "int8"


@patch("autobot_stt.services.whisper_service.WhisperModel")
def test_close_releases_model_reference(mock_model_cls: MagicMock) -> None:
    service = WhisperService(Settings())
    service.load()
    assert service._model is not None
    service.close()
    assert service._model is None


# --- WhisperService.transcribe -------------------------------------------


@patch("autobot_stt.services.whisper_service.WhisperModel")
def test_transcribe_concatenates_segments(mock_model_cls: MagicMock) -> None:
    segment_a = MagicMock(text=" Hello")
    segment_b = MagicMock(text="world ")
    segment_c = MagicMock(text="   ")  # empty after strip -> skipped
    mock_model_cls.return_value.transcribe.return_value = (
        iter([segment_a, segment_b, segment_c]),
        MagicMock(),
    )

    service = WhisperService(Settings())
    service.load()
    result = service.transcribe(np.zeros(1600, dtype=np.float32))

    assert result == "Hello world"
    mock_model_cls.return_value.transcribe.assert_called_once()


@patch("autobot_stt.services.whisper_service.WhisperModel")
def test_transcribe_uses_beam_size_5_and_vad_filter(mock_model_cls: MagicMock) -> None:
    mock_model_cls.return_value.transcribe.return_value = (iter([]), MagicMock())

    service = WhisperService(Settings())
    service.load()
    service.transcribe(np.zeros(1600, dtype=np.float32))

    kwargs = mock_model_cls.return_value.transcribe.call_args.kwargs
    assert kwargs["beam_size"] == 5
    assert kwargs["vad_filter"] is True


@patch("autobot_stt.services.whisper_service.WhisperModel")
def test_transcribe_forwards_initial_prompt(mock_model_cls: MagicMock) -> None:
    mock_model_cls.return_value.transcribe.return_value = (iter([]), MagicMock())

    service = WhisperService(Settings())
    service.load()
    service.transcribe(np.zeros(1600, dtype=np.float32), initial_prompt="hi there")

    kwargs = mock_model_cls.return_value.transcribe.call_args.kwargs
    assert kwargs["initial_prompt"] == "hi there"


@patch("autobot_stt.services.whisper_service.WhisperModel")
def test_transcribe_omits_initial_prompt_when_blank(mock_model_cls: MagicMock) -> None:
    mock_model_cls.return_value.transcribe.return_value = (iter([]), MagicMock())

    service = WhisperService(Settings())
    service.load()
    service.transcribe(np.zeros(1600, dtype=np.float32), initial_prompt="   ")

    kwargs = mock_model_cls.return_value.transcribe.call_args.kwargs
    assert "initial_prompt" not in kwargs


@patch("autobot_stt.services.whisper_service.WhisperModel")
def test_transcribe_empty_pcm_returns_empty_string_without_model_call(
    mock_model_cls: MagicMock,
) -> None:
    mock_model_cls.return_value.transcribe.return_value = (iter([]), MagicMock())

    service = WhisperService(Settings())
    service.load()
    result = service.transcribe(np.array([], dtype=np.float32))

    assert result == ""
    mock_model_cls.return_value.transcribe.assert_not_called()


@patch("autobot_stt.services.whisper_service.WhisperModel")
def test_transcribe_no_segments_returns_empty_string(mock_model_cls: MagicMock) -> None:
    mock_model_cls.return_value.transcribe.return_value = (iter([]), MagicMock())

    service = WhisperService(Settings())
    service.load()
    result = service.transcribe(np.zeros(1600, dtype=np.float32))

    assert result == ""


def test_transcribe_multidimensional_audio_raises_value_error() -> None:
    service = WhisperService(Settings())
    service._model = MagicMock()  # avoid real WhisperModel load
    two_d = np.zeros((4, 4), dtype=np.float32)  # stays 2-D after squeeze
    with pytest.raises(ValueError, match="1-D"):
        service.transcribe(two_d)


@patch("autobot_stt.services.whisper_service.WhisperModel")
def test_transcribe_loads_model_if_not_already_loaded(mock_model_cls: MagicMock) -> None:
    mock_model_cls.return_value.transcribe.return_value = (iter([]), MagicMock())

    service = WhisperService(Settings())
    # load() not called explicitly — transcribe() should trigger it
    service.transcribe(np.zeros(1600, dtype=np.float32))

    mock_model_cls.assert_called_once()
