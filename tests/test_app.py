from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from autobot_stt.main import app, run
from autobot_stt.services.whisper_service import WhisperService


def test_openapi_schema_includes_health() -> None:
    schema = app.openapi()
    paths = schema["paths"]
    assert "/health" in paths
    assert "get" in paths["/health"]


def test_openapi_schema_includes_sessions_endpoints() -> None:
    schema = app.openapi()
    paths = schema["paths"]
    assert "/v1/sessions" in paths
    assert "post" in paths["/v1/sessions"]
    assert "/v1/sessions/{session_id}" in paths
    assert "delete" in paths["/v1/sessions/{session_id}"]
    assert "/v1/sessions/{session_id}/finalize" in paths
    assert "post" in paths["/v1/sessions/{session_id}/finalize"]


def test_run_invokes_uvicorn_with_expected_args() -> None:
    with patch("autobot_stt.main.uvicorn") as uvicorn_mock:
        run()
    uvicorn_mock.run.assert_called_once_with(
        "autobot_stt.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


@pytest.mark.asyncio
@patch("autobot_stt.services.whisper_service.WhisperModel")
async def test_lifespan_loads_whisper_service(mock_model_cls: MagicMock) -> None:
    async with app.router.lifespan_context(app):
        assert hasattr(app.state, "whisper_service")
        assert isinstance(app.state.whisper_service, WhisperService)
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get("/health")
            assert response.status_code == 200
    mock_model_cls.assert_called_once()


@pytest.mark.asyncio
@patch("autobot_stt.services.whisper_service.WhisperModel")
async def test_lifespan_releases_whisper_service_on_shutdown(
    mock_model_cls: MagicMock,
) -> None:
    async with app.router.lifespan_context(app):
        assert app.state.whisper_service._model is not None
    assert app.state.whisper_service._model is None
