from unittest.mock import patch

from autobot_stt.main import app, run


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
