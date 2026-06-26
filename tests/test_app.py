from unittest.mock import patch

from autobot_stt.main import app, run


def test_openapi_schema_includes_health() -> None:
    schema = app.openapi()
    paths = schema["paths"]
    assert "/health" in paths
    assert "get" in paths["/health"]


def test_run_invokes_uvicorn_with_expected_args() -> None:
    with patch("autobot_stt.main.uvicorn") as uvicorn_mock:
        run()
    uvicorn_mock.run.assert_called_once_with(
        "autobot_stt.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
