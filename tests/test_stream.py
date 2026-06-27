"""Tests for the WebSocket streaming endpoint (subtask 6)."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from autobot_stt.config import get_settings
from autobot_stt.dependencies.store import get_session_store
from autobot_stt.main import app
from autobot_stt.routes.stream import STREAM_FLUSH_SAMPLES, STREAM_MIN_BYTES_FOR_DECODE
from autobot_stt.services.audio_decoder import AudioDecodeError
from autobot_stt.stores.memory import InMemorySessionStore

# WebM EBML magic header; pairs with a payload big enough to clear the
# STREAM_MIN_BYTES_FOR_DECODE threshold in one frame.
_WEBM_EBML_HEADER = b"\x1a\x45\xdf\xa3"


@pytest.fixture
def mock_whisper() -> MagicMock:
    whisper = MagicMock()
    whisper.transcribe.return_value = "hello world"
    return whisper


@pytest.fixture
def mock_pcm() -> np.ndarray:
    return np.zeros(STREAM_FLUSH_SAMPLES, dtype=np.float32)


@pytest.fixture
def stream_client(
    session_store: InMemorySessionStore,
    mock_whisper: MagicMock,
) -> TestClient:
    app.dependency_overrides[get_session_store] = lambda: session_store
    with patch("autobot_stt.services.whisper_service.WhisperModel"):
        with TestClient(app) as client:
            app.state.whisper_service = mock_whisper
            yield client
    app.dependency_overrides.clear()


@pytest.fixture
def api_key_env(monkeypatch: pytest.MonkeyPatch) -> str:
    """Configure STT_API_KEY and clear the settings cache; return the key."""
    monkeypatch.setenv("STT_API_KEY", "test-secret")
    get_settings.cache_clear()
    return "test-secret"


def _create_session(client: TestClient, token: str | None = None) -> str:
    headers = {"Authorization": f"Bearer {token}"} if token else None
    response = client.post("/v1/sessions", json={}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def _audio_chunk() -> bytes:
    return _WEBM_EBML_HEADER + b"\x00" * STREAM_MIN_BYTES_FOR_DECODE


async def _get_session(
    store: InMemorySessionStore, session_id: str
) -> object:
    session = await store.get(session_id)
    assert session is not None
    return session


def test_stream_connect_receives_ready(
    stream_client: TestClient,
    mock_whisper: MagicMock,
    mock_pcm: np.ndarray,
) -> None:
    session_id = _create_session(stream_client)
    with patch(
        "autobot_stt.routes.stream.decode_webm_opus_to_pcm", return_value=mock_pcm
    ):
        with stream_client.websocket_connect(f"/v1/sessions/{session_id}/stream") as ws:
            ready = ws.receive_json()
    assert ready == {"type": "ready", "session_id": session_id}
    mock_whisper.transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_stream_binary_triggers_partial_transcript(
    stream_client: TestClient,
    session_store: InMemorySessionStore,
    mock_whisper: MagicMock,
    mock_pcm: np.ndarray,
) -> None:
    session_id = _create_session(stream_client)

    with patch(
        "autobot_stt.routes.stream.decode_webm_opus_to_pcm", return_value=mock_pcm
    ):
        with stream_client.websocket_connect(f"/v1/sessions/{session_id}/stream") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready"

            ws.send_bytes(_audio_chunk())
            partial = ws.receive_json()

    assert partial["type"] == "partial_transcript"
    assert partial["text"] == "hello world"
    assert partial["is_final"] is False

    mock_whisper.transcribe.assert_called_once()
    stored = await _get_session(session_store, session_id)
    assert stored.raw_transcript == "hello world"
    assert stored.partial_transcripts == ["hello world"]


@pytest.mark.asyncio
async def test_stream_partial_transcript_is_cumulative(
    stream_client: TestClient,
    session_store: InMemorySessionStore,
    mock_whisper: MagicMock,
    mock_pcm: np.ndarray,
) -> None:
    session_id = _create_session(stream_client)
    mock_whisper.transcribe.side_effect = ["first chunk", "second chunk"]

    with patch(
        "autobot_stt.routes.stream.decode_webm_opus_to_pcm", return_value=mock_pcm
    ):
        with stream_client.websocket_connect(f"/v1/sessions/{session_id}/stream") as ws:
            ws.receive_json()

            ws.send_bytes(_audio_chunk())
            first = ws.receive_json()
            ws.send_bytes(_audio_chunk())
            second = ws.receive_json()

    assert first["text"] == "first chunk"
    assert second["text"] == "first chunk second chunk"

    stored = await _get_session(session_store, session_id)
    assert stored.raw_transcript == "first chunk second chunk"
    assert stored.partial_transcripts == ["first chunk", "second chunk"]


def test_stream_auth_failure_closes_4401(
    stream_client: TestClient,
    api_key_env: str,
) -> None:
    session_id = _create_session(stream_client, token=api_key_env)

    with pytest.raises(WebSocketDisconnect) as exc:
        with stream_client.websocket_connect(f"/v1/sessions/{session_id}/stream"):
            pass
    assert exc.value.code == 4401


def test_stream_auth_via_query_token(
    stream_client: TestClient,
    api_key_env: str,
) -> None:
    session_id = _create_session(stream_client, token=api_key_env)

    with stream_client.websocket_connect(
        f"/v1/sessions/{session_id}/stream?token={api_key_env}"
    ) as ws:
        ready = ws.receive_json()
    assert ready == {"type": "ready", "session_id": session_id}


def test_stream_auth_via_bearer_header(
    stream_client: TestClient,
    api_key_env: str,
) -> None:
    session_id = _create_session(stream_client, token=api_key_env)

    with stream_client.websocket_connect(
        f"/v1/sessions/{session_id}/stream",
        headers={"Authorization": f"Bearer {api_key_env}"},
    ) as ws:
        ready = ws.receive_json()
    assert ready == {"type": "ready", "session_id": session_id}


def test_stream_unknown_session_closes_4404(stream_client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect) as exc:
        with stream_client.websocket_connect(
            "/v1/sessions/00000000-0000-4000-8000-000000000000/stream"
        ):
            pass
    assert exc.value.code == 4404


def test_stream_decode_error_sends_error_event(
    stream_client: TestClient,
    mock_whisper: MagicMock,
) -> None:
    session_id = _create_session(stream_client)

    with patch(
        "autobot_stt.routes.stream.decode_webm_opus_to_pcm",
        side_effect=AudioDecodeError("bad bytes"),
    ):
        with patch("autobot_stt.routes.stream.STREAM_SILENCE_TIMEOUT_SECONDS", 0.01):
            with stream_client.websocket_connect(f"/v1/sessions/{session_id}/stream") as ws:
                ws.receive_json()
                ws.send_bytes(_audio_chunk())
                error_event = ws.receive_json()

    assert error_event["type"] == "error"
    assert error_event["message"] == "Failed to decode audio"
    mock_whisper.transcribe.assert_not_called()


def test_stream_ffmpeg_missing_sends_error_event(
    stream_client: TestClient,
    mock_whisper: MagicMock,
) -> None:
    session_id = _create_session(stream_client)

    with patch(
        "autobot_stt.routes.stream.decode_webm_opus_to_pcm",
        side_effect=FileNotFoundError("ffmpeg not found"),
    ):
        with stream_client.websocket_connect(f"/v1/sessions/{session_id}/stream") as ws:
            ws.receive_json()
            ws.send_bytes(_audio_chunk())
            error_event = ws.receive_json()

    assert error_event == {"type": "error", "message": "Audio decoder unavailable"}
    mock_whisper.transcribe.assert_not_called()


def test_stream_whisper_failure_sends_error_event(
    stream_client: TestClient,
    mock_whisper: MagicMock,
    mock_pcm: np.ndarray,
) -> None:
    session_id = _create_session(stream_client)
    mock_whisper.transcribe.side_effect = RuntimeError("model crashed")

    with patch(
        "autobot_stt.routes.stream.decode_webm_opus_to_pcm", return_value=mock_pcm
    ):
        with stream_client.websocket_connect(f"/v1/sessions/{session_id}/stream") as ws:
            ws.receive_json()
            ws.send_bytes(_audio_chunk())
            error_event = ws.receive_json()

    assert error_event == {"type": "error", "message": "Transcription failed"}


def test_stream_incomplete_webm_buffers_without_error(
    stream_client: TestClient,
    mock_whisper: MagicMock,
    mock_pcm: np.ndarray,
) -> None:
    """Partial MediaRecorder fragments that fail decode should buffer, not error."""
    session_id = _create_session(stream_client)
    decode_calls = 0

    def decode_side_effect(batch: bytes) -> np.ndarray:
        nonlocal decode_calls
        decode_calls += 1
        if decode_calls == 1:
            raise AudioDecodeError("incomplete webm fragment")
        return mock_pcm

    with patch(
        "autobot_stt.routes.stream.decode_webm_opus_to_pcm",
        side_effect=decode_side_effect,
    ):
        with stream_client.websocket_connect(f"/v1/sessions/{session_id}/stream") as ws:
            ws.receive_json()
            ws.send_bytes(_audio_chunk())
            ws.send_bytes(_audio_chunk())
            partial = ws.receive_json()

    assert partial["type"] == "partial_transcript"
    assert partial["text"] == "hello world"
    mock_whisper.transcribe.assert_called_once()


def test_stream_works_without_store_dependency_override(
    mock_whisper: MagicMock,
    mock_pcm: np.ndarray,
) -> None:
    """Regression: get_session_store must accept HTTPConnection, not only Request."""
    with patch("autobot_stt.services.whisper_service.WhisperModel"):
        with TestClient(app) as client:
            app.state.whisper_service = mock_whisper
            session_id = client.post("/v1/sessions", json={}).json()["session_id"]
            with patch(
                "autobot_stt.routes.stream.decode_webm_opus_to_pcm",
                return_value=mock_pcm,
            ):
                with client.websocket_connect(f"/v1/sessions/{session_id}/stream") as ws:
                    ready = ws.receive_json()
                    ws.send_bytes(_audio_chunk())
                    partial = ws.receive_json()

    assert ready["type"] == "ready"
    assert partial["type"] == "partial_transcript"


def test_stream_skips_auth_when_key_empty(
    stream_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STT_API_KEY", raising=False)
    get_settings.cache_clear()

    session_id = _create_session(stream_client)

    with stream_client.websocket_connect(f"/v1/sessions/{session_id}/stream") as ws:
        ready = ws.receive_json()
    assert ready == {"type": "ready", "session_id": session_id}


def test_stream_passes_initial_prompt_to_whisper(
    stream_client: TestClient,
    mock_whisper: MagicMock,
    mock_pcm: np.ndarray,
) -> None:
    response = stream_client.post(
        "/v1/sessions",
        json={
            "draft_text": "meeting notes",
            "chat_history": [{"role": "user", "content": "hi there"}],
        },
    )
    session_id = response.json()["session_id"]

    with patch(
        "autobot_stt.routes.stream.decode_webm_opus_to_pcm", return_value=mock_pcm
    ):
        with stream_client.websocket_connect(f"/v1/sessions/{session_id}/stream") as ws:
            ws.receive_json()
            ws.send_bytes(_audio_chunk())
            ws.receive_json()

    mock_whisper.transcribe.assert_called_once()
    args, _kwargs = mock_whisper.transcribe.call_args
    assert args[1] == "meeting notes\nuser: hi there"


def test_stream_ignores_text_frames(
    stream_client: TestClient,
    mock_whisper: MagicMock,
    mock_pcm: np.ndarray,
) -> None:
    """Non-bytes (text) frames must be skipped, not crash or transcribe."""
    session_id = _create_session(stream_client)

    with patch(
        "autobot_stt.routes.stream.decode_webm_opus_to_pcm", return_value=mock_pcm
    ):
        with stream_client.websocket_connect(f"/v1/sessions/{session_id}/stream") as ws:
            ws.receive_json()  # ready
            ws.send_text("ignore me")  # text frame -> message["bytes"] is None
            ws.send_bytes(_audio_chunk())  # this one drives transcription
            partial = ws.receive_json()

    assert partial["type"] == "partial_transcript"
    assert partial["text"] == "hello world"
    mock_whisper.transcribe.assert_called_once()


def test_stream_buffers_small_bytes_until_threshold_met(
    stream_client: TestClient,
    mock_whisper: MagicMock,
    mock_pcm: np.ndarray,
) -> None:
    """Frames under STREAM_MIN_BYTES_FOR_DECODE accumulate without decoding."""
    session_id = _create_session(stream_client)
    decode_sizes: list[int] = []

    def _track_decode(batch: bytes) -> np.ndarray:
        decode_sizes.append(len(batch))
        return mock_pcm

    small_chunk = b"\x00" * (STREAM_MIN_BYTES_FOR_DECODE // 2)

    with patch(
        "autobot_stt.routes.stream.decode_webm_opus_to_pcm", side_effect=_track_decode
    ):
        with stream_client.websocket_connect(f"/v1/sessions/{session_id}/stream") as ws:
            ws.receive_json()
            ws.send_bytes(small_chunk)  # under threshold -> non-forced flush returns
            # Second large frame pushes buffer over threshold and drives a real
            # flush, giving us a synchronization point via receive_json().
            ws.send_bytes(_audio_chunk())
            partial = ws.receive_json()

    # Only the combined-buffer flush decoded; the small-only flush was skipped.
    assert partial["type"] == "partial_transcript"
    assert decode_sizes == [STREAM_MIN_BYTES_FOR_DECODE // 2 + len(_audio_chunk())]
    mock_whisper.transcribe.assert_called_once()


def test_stream_small_pcm_defers_transcribe_until_threshold_met(
    stream_client: TestClient,
    mock_whisper: MagicMock,
    mock_pcm: np.ndarray,
) -> None:
    """PCM under STREAM_FLUSH_SAMPLES skips transcription on non-forced flushes."""
    session_id = _create_session(stream_client)
    small_pcm = np.zeros(STREAM_FLUSH_SAMPLES - 1, dtype=np.float32)
    decode_calls = 0

    def _decode_side_effect(_batch: bytes) -> np.ndarray:
        nonlocal decode_calls
        decode_calls += 1
        # First flush: small PCM under threshold -> non-forced flush skips.
        # Second flush: threshold-meeting PCM -> emits, syncs the test.
        return small_pcm if decode_calls == 1 else mock_pcm

    with patch(
        "autobot_stt.routes.stream.decode_webm_opus_to_pcm",
        side_effect=_decode_side_effect,
    ):
        with stream_client.websocket_connect(f"/v1/sessions/{session_id}/stream") as ws:
            ws.receive_json()
            ws.send_bytes(_audio_chunk())  # small PCM -> line 93 returns early
            ws.send_bytes(_audio_chunk())  # threshold PCM -> transcribe + emit
            partial = ws.receive_json()

    assert partial["type"] == "partial_transcript"
    # Without line 93 the first flush would have transcribed too.
    mock_whisper.transcribe.assert_called_once()


def test_stream_empty_pcm_skips_transcribe(
    stream_client: TestClient,
    mock_whisper: MagicMock,
    mock_pcm: np.ndarray,
) -> None:
    """Empty PCM reaching _transcribe_and_emit returns before calling whisper."""
    session_id = _create_session(stream_client)
    empty_pcm = np.zeros(0, dtype=np.float32)
    decode_calls = 0

    def _decode_side_effect(_batch: bytes) -> np.ndarray:
        nonlocal decode_calls
        decode_calls += 1
        return empty_pcm if decode_calls == 1 else mock_pcm

    # Patch the threshold to 0 so empty PCM passes the non-forced flush gate and
    # reaches _transcribe_and_emit where line 141 returns early.
    with (
        patch(
            "autobot_stt.routes.stream.decode_webm_opus_to_pcm",
            side_effect=_decode_side_effect,
        ),
        patch("autobot_stt.routes.stream.STREAM_FLUSH_SAMPLES", 0),
    ):
        with stream_client.websocket_connect(f"/v1/sessions/{session_id}/stream") as ws:
            ws.receive_json()
            ws.send_bytes(_audio_chunk())  # empty PCM -> _transcribe_and_emit skips
            ws.send_bytes(_audio_chunk())  # normal PCM -> transcribe + emit
            partial = ws.receive_json()

    assert partial["type"] == "partial_transcript"
    # Second flush is what called transcribe; first flush's empty PCM did not.
    mock_whisper.transcribe.assert_called_once()


@pytest.mark.asyncio
async def test_stream_stops_accumulating_after_session_deleted(
    stream_client: TestClient,
    session_store: InMemorySessionStore,
    mock_whisper: MagicMock,
    mock_pcm: np.ndarray,
) -> None:
    """Deleting a session mid-stream must not append further transcript text."""
    session_id = _create_session(stream_client)
    mock_whisper.transcribe.side_effect = ["first chunk", "second chunk"]

    with patch(
        "autobot_stt.routes.stream.decode_webm_opus_to_pcm", return_value=mock_pcm
    ):
        with stream_client.websocket_connect(f"/v1/sessions/{session_id}/stream") as ws:
            ws.receive_json()
            ws.send_bytes(_audio_chunk())
            first = ws.receive_json()
            assert first["text"] == "first chunk"

            await session_store.delete(session_id)
            ws.send_bytes(_audio_chunk())

    assert mock_whisper.transcribe.call_count == 1
    assert await session_store.get(session_id) is None


@pytest.mark.asyncio
async def test_stream_empty_transcript_skips_emit_and_accumulation(
    stream_client: TestClient,
    session_store: InMemorySessionStore,
    mock_whisper: MagicMock,
    mock_pcm: np.ndarray,
) -> None:
    """Whisper returning empty text skips emit and transcript accumulation."""
    session_id = _create_session(stream_client)
    mock_whisper.transcribe.side_effect = ["", "real text"]

    with patch(
        "autobot_stt.routes.stream.decode_webm_opus_to_pcm", return_value=mock_pcm
    ):
        with stream_client.websocket_connect(f"/v1/sessions/{session_id}/stream") as ws:
            ws.receive_json()
            ws.send_bytes(_audio_chunk())  # whisper returns "" -> no emit
            ws.send_bytes(_audio_chunk())  # whisper returns "real text" -> emit
            partial = ws.receive_json()

    assert partial["type"] == "partial_transcript"
    assert partial["text"] == "real text"
    assert mock_whisper.transcribe.call_count == 2
    stored = await _get_session(session_store, session_id)
    assert stored.raw_transcript == "real text"
    assert stored.partial_transcripts == ["real text"]


def test_stream_silence_timeout_on_empty_buffer_no_decode(
    stream_client: TestClient,
    mock_whisper: MagicMock,
) -> None:
    """Silence timeout firing on an empty buffer is a no-op (no decode, no error)."""
    session_id = _create_session(stream_client)

    with (
        patch("autobot_stt.routes.stream.decode_webm_opus_to_pcm") as decode_mock,
        patch("autobot_stt.routes.stream.STREAM_SILENCE_TIMEOUT_SECONDS", 0.01),
    ):
        with stream_client.websocket_connect(f"/v1/sessions/{session_id}/stream") as ws:
            ws.receive_json()  # ready
            # Allow at least one silence-timeout tick to fire on empty buffer.
            time.sleep(0.05)

    decode_mock.assert_not_called()
    mock_whisper.transcribe.assert_not_called()
