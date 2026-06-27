from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from autobot_stt.config import get_settings
from autobot_stt.main import app, run
from autobot_stt.services.whisper_service import WhisperService


def test_openapi_advertises_expected_routes_and_schemas() -> None:
    """OpenAPI must expose health + session routes and their Pydantic schemas."""
    schema = app.openapi()
    paths = schema["paths"]
    assert "get" in paths["/health"]
    assert "post" in paths["/v1/sessions"]
    assert "delete" in paths["/v1/sessions/{session_id}"]
    assert "post" in paths["/v1/sessions/{session_id}/finalize"]

    schemas = schema["components"]["schemas"]
    assert {"CreateSessionRequest", "CreateSessionResponse", "FinalizeSessionResponse"} <= set(
        schemas
    )

    tag_names = {t["name"] for t in schema["tags"]}
    assert {"health", "sessions", "streaming"} <= tag_names


def test_openapi_session_routes_have_summaries_and_schema_refs() -> None:
    """Each session route must carry a summary, the right tag, and correct schema refs."""
    paths = app.openapi()["paths"]

    health = paths["/health"]["get"]
    assert health["summary"]
    assert "health" in health["tags"]

    create = paths["/v1/sessions"]["post"]
    assert create["summary"]
    assert "sessions" in create["tags"]
    assert (
        create["requestBody"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/CreateSessionRequest"
    )

    delete = paths["/v1/sessions/{session_id}"]["delete"]
    assert delete["summary"]
    assert "sessions" in delete["tags"]

    finalize = paths["/v1/sessions/{session_id}/finalize"]["post"]
    assert finalize["summary"]
    assert "sessions" in finalize["tags"]
    assert (
        finalize["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/FinalizeSessionResponse"
    )


def test_run_invokes_uvicorn_with_expected_args() -> None:
    with patch("autobot_stt.main.uvicorn") as uvicorn_mock:
        run()
    uvicorn_mock.run.assert_called_once_with(
        "autobot_stt.main:app",
        host="0.0.0.0",
        port=8000,
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


@pytest.mark.asyncio
@patch("autobot_stt.services.whisper_service.WhisperModel")
async def test_lifespan_warns_when_stt_api_key_unset(
    mock_model_cls: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("STT_API_KEY", raising=False)
    get_settings.cache_clear()
    with caplog.at_level("WARNING", logger="autobot_stt.main"):
        async with app.router.lifespan_context(app):
            pass
    assert any(
        "STT_API_KEY unset" in record.message and record.levelname == "WARNING"
        for record in caplog.records
    )


@pytest.mark.asyncio
@patch("autobot_stt.services.whisper_service.WhisperModel")
async def test_lifespan_no_warn_when_stt_api_key_set(
    mock_model_cls: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("STT_API_KEY", "configured-key")
    get_settings.cache_clear()
    with caplog.at_level("WARNING", logger="autobot_stt.main"):
        async with app.router.lifespan_context(app):
            pass
    assert not any("STT_API_KEY unset" in record.message for record in caplog.records)
