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


def test_openapi_components_include_session_models() -> None:
    schemas = app.openapi()["components"]["schemas"]
    assert "CreateSessionRequest" in schemas
    assert "CreateSessionResponse" in schemas
    assert "FinalizeSessionResponse" in schemas


def test_openapi_session_routes_have_summaries_and_tags() -> None:
    paths = app.openapi()["paths"]
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


def test_openapi_health_has_summary_and_health_tag() -> None:
    health = app.openapi()["paths"]["/health"]["get"]
    assert health["summary"]
    assert "health" in health["tags"]


def test_openapi_tag_metadata_includes_health_sessions_streaming() -> None:
    tag_names = {t["name"] for t in app.openapi()["tags"]}
    assert {"health", "sessions", "streaming"} <= tag_names


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
