"""Tests for the WebSocket streaming endpoint (subtask 6)."""

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


def _create_session(client: TestClient, token: str | None = None) -> str:
    headers = {"Authorization": f"Bearer {token}"} if token else None
    response = client.post("/v1/sessions", json={}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def _audio_chunk() -> bytes:
    return b"\x1a\x45\xdf\xa3" + b"\x00" * STREAM_MIN_BYTES_FOR_DECODE


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


def test_stream_binary_triggers_partial_transcript(
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
    stored = session_store._sessions[session_id]
    assert stored.raw_transcript == "hello world"
    assert stored.partial_transcripts == ["hello world"]


def test_stream_partial_transcript_is_cumulative(
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

    stored = session_store._sessions[session_id]
    assert stored.raw_transcript == "first chunk second chunk"
    assert stored.partial_transcripts == ["first chunk", "second chunk"]


def test_stream_auth_failure_closes_4401(
    stream_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STT_API_KEY", "test-secret")
    get_settings.cache_clear()

    session_id = _create_session(stream_client, token="test-secret")

    with pytest.raises(WebSocketDisconnect) as exc:
        with stream_client.websocket_connect(f"/v1/sessions/{session_id}/stream"):
            pass
    assert exc.value.code == 4401


def test_stream_auth_via_query_token(
    stream_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STT_API_KEY", "test-secret")
    get_settings.cache_clear()

    session_id = _create_session(stream_client, token="test-secret")

    with stream_client.websocket_connect(
        f"/v1/sessions/{session_id}/stream?token=test-secret"
    ) as ws:
        ready = ws.receive_json()
    assert ready == {"type": "ready", "session_id": session_id}


def test_stream_auth_via_bearer_header(
    stream_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STT_API_KEY", "test-secret")
    get_settings.cache_clear()

    session_id = _create_session(stream_client, token="test-secret")

    with stream_client.websocket_connect(
        f"/v1/sessions/{session_id}/stream",
        headers={"Authorization": "Bearer test-secret"},
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
        with stream_client.websocket_connect(f"/v1/sessions/{session_id}/stream") as ws:
            ws.receive_json()
            ws.send_bytes(_audio_chunk())
            error_event = ws.receive_json()

    assert error_event["type"] == "error"
    assert error_event["message"] == "Failed to decode audio"
    mock_whisper.transcribe.assert_not_called()


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
