import pytest

from autobot_stt.config import get_settings


@pytest.mark.asyncio
async def test_v1_requires_auth_when_key_set(client, auth_headers) -> None:
    response = await client.post("/v1/sessions", json={})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_v1_rejects_wrong_key(client, auth_headers) -> None:
    response = await client.post(
        "/v1/sessions", json={}, headers={"Authorization": "Bearer wrong-token"}
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_v1_accepts_correct_bearer(client, auth_headers) -> None:
    response = await client.post("/v1/sessions", json={}, headers=auth_headers)
    assert response.status_code == 201
    assert "session_id" in response.json()


@pytest.mark.asyncio
async def test_health_no_auth_when_key_set(client, auth_headers) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_v1_rejects_non_bearer_scheme(client, monkeypatch) -> None:
    monkeypatch.setenv("STT_API_KEY", "test-secret")
    get_settings.cache_clear()
    response = await client.post(
        "/v1/sessions",
        json={},
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_v1_skips_auth_when_key_empty(client, monkeypatch) -> None:
    monkeypatch.delenv("STT_API_KEY", raising=False)
    get_settings.cache_clear()
    response = await client.post("/v1/sessions", json={})
    assert response.status_code == 201
