"""Integration-test fixtures: full session lifecycle with mocked externals.

Mocks Whisper (model + service), the audio decoder (no ffmpeg required), and
OpenAI cleanup (no network) so the lifecycle test runs fast and CPU-only.
"""

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from autobot_stt.config import get_settings
from autobot_stt.dependencies.store import get_session_store
from autobot_stt.main import app
from autobot_stt.routes.stream import STREAM_FLUSH_SAMPLES
from autobot_stt.stores.memory import InMemorySessionStore


@pytest.fixture
def session_store() -> InMemorySessionStore:
    return InMemorySessionStore()


@pytest.fixture
def mock_whisper() -> MagicMock:
    whisper = MagicMock()
    whisper.transcribe.return_value = "hello world"
    return whisper


@pytest.fixture
def mock_pcm() -> np.ndarray:
    return np.zeros(STREAM_FLUSH_SAMPLES, dtype=np.float32)


@pytest.fixture
def mock_cleanup() -> Iterator[AsyncMock]:
    """Patch OpenAI cleanup at the route import site; default returns cleaned text."""
    with patch(
        "autobot_stt.routes.sessions.cleanup_transcript",
        new_callable=AsyncMock,
    ) as m:
        m.return_value = "Hello world."
        yield m


@pytest.fixture
def api_key_env(monkeypatch: pytest.MonkeyPatch) -> str:
    """Configure STT_API_KEY + OPENAI_API_KEY and clear the settings cache."""
    monkeypatch.setenv("STT_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    return "test-secret"


@pytest.fixture
def integration_client(
    session_store: InMemorySessionStore,
    mock_whisper: MagicMock,
    api_key_env: str,
) -> Iterator[TestClient]:
    """TestClient (REST + WebSocket) sharing one mocked store + whisper service.

    Lifespan installs a real WhisperService backed by the patched WhisperModel;
    we then swap app.state.whisper_service for ``mock_whisper`` so transcribe
    calls never reach the model. The whisper_lock from lifespan is reused.
    """
    app.dependency_overrides[get_session_store] = lambda: session_store
    with patch("autobot_stt.services.whisper_service.WhisperModel"):
        with TestClient(app) as client:
            app.state.whisper_service = mock_whisper
            yield client
    app.dependency_overrides.clear()
