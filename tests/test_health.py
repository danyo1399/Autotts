import pytest
from httpx import ASGITransport, AsyncClient

from autobot_stt import __version__
from autobot_stt.main import app


@pytest.mark.anyio
async def test_health_returns_ok() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_health_returns_json_content_type() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")
    assert response.headers["content-type"] == "application/json"


def test_app_metadata() -> None:
    assert app.title == "autobot-stt"
    assert app.version == __version__
