from uuid import UUID

import pytest

from autobot_stt.stores.memory import InMemorySessionStore


@pytest.mark.asyncio
async def test_create_session_returns_uuid(client, session_store: InMemorySessionStore) -> None:
    response = await client.post("/v1/sessions", json={})
    assert response.status_code == 201
    payload = response.json()
    assert "session_id" in payload
    UUID(payload["session_id"])  # raises ValueError if not a valid UUID


@pytest.mark.asyncio
async def test_create_session_persists_data(client, session_store: InMemorySessionStore) -> None:
    body = {
        "draft_text": "hello world",
        "chat_history": [{"role": "user", "content": "hi"}],
        "comments": [{"author": "alice", "body": "looks good"}],
    }
    response = await client.post("/v1/sessions", json=body)
    assert response.status_code == 201
    session_id = response.json()["session_id"]

    stored = await session_store.get(session_id)
    assert stored is not None
    assert stored.draft_text == "hello world"
    assert stored.chat_history[0].role == "user"
    assert stored.chat_history[0].content == "hi"
    assert stored.comments[0].author == "alice"
    assert stored.comments[0].body == "looks good"


@pytest.mark.asyncio
async def test_create_session_defaults(client, session_store: InMemorySessionStore) -> None:
    response = await client.post("/v1/sessions", json={})
    session_id = response.json()["session_id"]

    stored = await session_store.get(session_id)
    assert stored is not None
    assert stored.draft_text == ""
    assert stored.chat_history == []
    assert stored.comments == []
    assert stored.raw_transcript == ""
    assert stored.partial_transcripts == []
    assert stored.created_at is not None


@pytest.mark.asyncio
async def test_delete_session_returns_204(client, session_store: InMemorySessionStore) -> None:
    create_response = await client.post("/v1/sessions", json={})
    session_id = create_response.json()["session_id"]

    delete_response = await client.delete(f"/v1/sessions/{session_id}")
    assert delete_response.status_code == 204
    assert delete_response.content == b""

    assert await session_store.get(session_id) is None


@pytest.mark.asyncio
async def test_delete_session_then_delete_again_returns_404(
    client, session_store: InMemorySessionStore
) -> None:
    create_response = await client.post("/v1/sessions", json={})
    session_id = create_response.json()["session_id"]

    first_delete = await client.delete(f"/v1/sessions/{session_id}")
    assert first_delete.status_code == 204

    second_delete = await client.delete(f"/v1/sessions/{session_id}")
    assert second_delete.status_code == 404


@pytest.mark.asyncio
async def test_delete_unknown_session_returns_404(client) -> None:
    response = await client.delete("/v1/sessions/00000000-0000-4000-8000-000000000000")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_create_invalid_chat_role_returns_422(client) -> None:
    body = {"chat_history": [{"role": "system", "content": "hi"}]}
    response = await client.post("/v1/sessions", json=body)
    assert response.status_code == 422
