"""End-to-end lifecycle: create -> WebSocket stream -> finalize -> cleanup.

All external dependencies are mocked:
- Whisper model load + transcribe (no HF download, no GPU)
- ffmpeg audio decode (no system dep)
- OpenAI cleanup_transcript (no network)

This complements unit tests in test_sessions.py, test_stream.py, and
test_finalize.py by exercising the full path where streaming populates
``raw_transcript`` and finalize consumes it.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from autobot_stt.routes.stream import STREAM_MIN_BYTES_FOR_DECODE
from autobot_stt.stores.memory import InMemorySessionStore

pytestmark = pytest.mark.integration

# WebM EBML magic header + padding; only size matters since the decoder is mocked.
_WEBM_EBML_HEADER = b"\x1a\x45\xdf\xa3"


def _audio_chunk() -> bytes:
    return _WEBM_EBML_HEADER + b"\x00" * STREAM_MIN_BYTES_FOR_DECODE


_CREATE_BODY = {
    "draft_text": "meeting notes",
    "chat_history": [{"role": "user", "content": "discuss roadmap"}],
    "comments": [{"author": "alice", "body": "sounds good"}],
}


@pytest.mark.asyncio
async def test_full_session_lifecycle(
    integration_client: TestClient,
    session_store: InMemorySessionStore,
    mock_whisper: MagicMock,
    mock_pcm: np.ndarray,
    mock_cleanup: AsyncMock,
    api_key_env: str,
) -> None:
    headers = {"Authorization": f"Bearer {api_key_env}"}

    # Step 1 — create
    create_resp = integration_client.post(
        "/v1/sessions", json=_CREATE_BODY, headers=headers
    )
    assert create_resp.status_code == 201, create_resp.text
    session_id = create_resp.json()["session_id"]

    stored = await session_store.get(session_id)
    assert stored is not None
    assert stored.draft_text == "meeting notes"
    assert stored.chat_history[0].content == "discuss roadmap"
    assert stored.comments[0].author == "alice"
    assert stored.raw_transcript == ""

    # Step 2 — stream binary chunk -> partial transcript accumulates on session
    with patch(
        "autobot_stt.routes.stream.decode_webm_opus_to_pcm", return_value=mock_pcm
    ):
        with integration_client.websocket_connect(
            f"/v1/sessions/{session_id}/stream?token={api_key_env}"
        ) as ws:
            ready = ws.receive_json()
            assert ready == {"type": "ready", "session_id": session_id}

            ws.send_bytes(_audio_chunk())
            partial = ws.receive_json()

    assert partial["type"] == "partial_transcript"
    assert partial["text"] == "hello world"
    assert partial["is_final"] is False

    mock_whisper.transcribe.assert_called_once()
    streamed = await session_store.get(session_id)
    assert streamed is not None
    assert streamed.raw_transcript == "hello world"

    # Step 3 — finalize consumes the streamed transcript and deletes the session
    finalize_resp = integration_client.post(
        f"/v1/sessions/{session_id}/finalize", headers=headers
    )
    assert finalize_resp.status_code == 200, finalize_resp.text
    payload = finalize_resp.json()
    assert payload["text"] == "Hello world."
    assert payload["raw_transcript"] == "hello world"

    mock_cleanup.assert_awaited_once()
    session_arg = mock_cleanup.await_args.args[0]
    assert session_arg.raw_transcript == "hello world"
    assert session_arg.draft_text == "meeting notes"
    assert session_arg.chat_history[0].content == "discuss roadmap"
    assert session_arg.comments[0].author == "alice"

    # Step 4 — cleanup: session deleted by finalize; subsequent ops 404
    assert await session_store.get(session_id) is None

    second_finalize = integration_client.post(
        f"/v1/sessions/{session_id}/finalize", headers=headers
    )
    assert second_finalize.status_code == 404

    delete_resp = integration_client.delete(
        f"/v1/sessions/{session_id}", headers=headers
    )
    assert delete_resp.status_code == 404
