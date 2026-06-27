"""Whisper speech-to-text service backed by faster-whisper."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from faster_whisper import WhisperModel

from autobot_stt.config import Settings

INITIAL_PROMPT_MAX_CHARS = 800
DEFAULT_BEAM_SIZE = 5

# faster-whisper requires a compute_type matching the device backend. Deriving
# from `whisper_device` avoids a separate env var while keeping the recommended
# default for CPU (int8) and CUDA (float16) per the locked decisions.
_DEVICE_COMPUTE_TYPES: dict[str, str] = {
    "cpu": "int8",
    "cuda": "float16",
}


class ChatMessage(TypedDict):
    """Minimal chat message contract used to build Whisper's initial_prompt."""

    role: str
    content: str


def _compute_type_for_device(device: str) -> str:
    """Return the faster-whisper ``compute_type`` for a device string."""
    return _DEVICE_COMPUTE_TYPES.get(device, "default")


def build_initial_prompt(
    draft_text: str,
    chat_history: list[ChatMessage],
    max_messages: int = 3,
) -> str:
    """Combine ``draft_text`` with recent chat history into a Whisper initial_prompt.

    The result is capped at ``INITIAL_PROMPT_MAX_CHARS`` characters (~224 tokens
    heuristic). When truncation is required, the tail is preserved so the most
    recent context (draft + latest messages) is always retained — older context
    is dropped from the left.

    Args:
        draft_text: Current partial draft text the user is composing.
        chat_history: Chronologically ordered chat messages; the last
            ``max_messages`` entries are used.
        max_messages: Maximum number of trailing chat messages to include.

    Returns:
        Joined prompt string, or ``""`` if both ``draft_text`` and the trailing
        messages are empty.
    """
    parts: list[str] = []
    if draft_text.strip():
        parts.append(draft_text.strip())

    for message in chat_history[-max_messages:]:
        content = message["content"].strip()
        if not content:
            continue
        parts.append(f"{message['role']}: {content}")

    prompt = "\n".join(parts)
    if len(prompt) > INITIAL_PROMPT_MAX_CHARS:
        prompt = prompt[-INITIAL_PROMPT_MAX_CHARS:]
    return prompt


class WhisperService:
    """Loads a faster-whisper model once and transcribes PCM audio chunks."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: WhisperModel | None = None

    def load(self) -> None:
        """Construct the underlying ``WhisperModel``. Idempotent."""
        if self._model is not None:
            return
        compute_type = _compute_type_for_device(self._settings.whisper_device)
        self._model = WhisperModel(
            self._settings.whisper_model,
            device=self._settings.whisper_device,
            compute_type=compute_type,
        )

    def close(self) -> None:
        """Release the model reference. Safe to call when not loaded."""
        self._model = None

    def transcribe(
        self,
        pcm_audio: np.ndarray,
        initial_prompt: str | None = None,
    ) -> str:
        """Transcribe ``pcm_audio`` into concatenated segment text.

        Args:
            pcm_audio: 1-D ``float32`` array at 16 kHz mono (as produced by
                ``audio_decoder.decode_webm_opus_to_pcm``). Empty arrays return
                ``""`` immediately without invoking the model.
            initial_prompt: Optional context string from ``build_initial_prompt``.
                ``None`` or whitespace-only prompts are not forwarded.

        Returns:
            Segment texts joined with single spaces. Empty if no speech was
            detected.
        """
        if self._model is None:
            self.load()
        assert self._model is not None  # noqa: S101 - narrowed for type checkers

        audio = np.asarray(pcm_audio, dtype=np.float32).squeeze()
        if audio.ndim != 1:
            raise ValueError(
                f"pcm_audio must be 1-D after squeeze; got shape {audio.shape}"
            )
        if audio.size == 0:
            return ""

        transcribe_kwargs: dict[str, object] = {
            "beam_size": DEFAULT_BEAM_SIZE,
            "vad_filter": True,
        }
        if initial_prompt and initial_prompt.strip():
            transcribe_kwargs["initial_prompt"] = initial_prompt

        segments, _info = self._model.transcribe(audio, **transcribe_kwargs)
        return " ".join(
            segment.text.strip() for segment in segments if segment.text.strip()
        )
