from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autobot_stt.config import get_settings
from autobot_stt.stores.memory import InMemorySessionStore


def _set_openai_key(monkeypatch: pytest.MonkeyPatch, value: str = "sk-test") -> None:
    monkeypatch.setenv("OPENAI_API_KEY", value)
    get_settings.cache_clear()


async def _create_session_with_transcript(
    client,
    session_store: InMemorySessionStore,
    transcript: str,
    *,
    auth_headers: dict | None = None,
    draft_text: str | None = None,
    chat_history: list | None = None,
    comments: list | None = None,
) -> str:
    body: dict = {}
    if draft_text is not None:
        body["draft_text"] = draft_text
    if chat_history is not None:
        body["chat_history"] = chat_history
    if comments is not None:
        body["comments"] = comments
    resp = await client.post("/v1/sessions", json=body, headers=auth_headers)
    assert resp.status_code == 201
    session_id = resp.json()["session_id"]
    stored = await session_store.get(session_id)
    assert stored is not None
    stored.raw_transcript = transcript
    return session_id


@pytest.mark.asyncio
async def test_finalize_returns_cleaned_text(
    client, session_store, auth_headers, monkeypatch
) -> None:
    _set_openai_key(monkeypatch)
    session_id = await _create_session_with_transcript(
        client, session_store, "hello world", auth_headers=auth_headers
    )

    with patch(
        "autobot_stt.routes.sessions.cleanup_transcript",
        new_callable=AsyncMock,
    ) as mock_cleanup:
        mock_cleanup.return_value = "cleaned text"
        response = await client.post(
            f"/v1/sessions/{session_id}/finalize", headers=auth_headers
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "cleaned text"
    assert payload["raw_transcript"] == "hello world"
    mock_cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_finalize_empty_transcript_returns_400(
    client, session_store, auth_headers, monkeypatch
) -> None:
    _set_openai_key(monkeypatch)
    session_id = await _create_session_with_transcript(
        client, session_store, "   ", auth_headers=auth_headers
    )

    with patch(
        "autobot_stt.routes.sessions.cleanup_transcript",
        new_callable=AsyncMock,
    ) as mock_cleanup:
        response = await client.post(
            f"/v1/sessions/{session_id}/finalize", headers=auth_headers
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Session has no transcript"
    mock_cleanup.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalize_requires_auth(
    client, session_store, auth_headers, monkeypatch
) -> None:
    _set_openai_key(monkeypatch)
    session_id = await _create_session_with_transcript(
        client, session_store, "hello", auth_headers=auth_headers
    )

    response = await client.post(f"/v1/sessions/{session_id}/finalize")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_finalize_unknown_session_returns_404(
    client, auth_headers, monkeypatch
) -> None:
    _set_openai_key(monkeypatch)
    response = await client.post(
        "/v1/sessions/00000000-0000-4000-8000-000000000000/finalize",
        headers=auth_headers,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_finalize_missing_openai_key_returns_503(
    client, session_store, auth_headers, monkeypatch
) -> None:
    _set_openai_key(monkeypatch, value="")
    session_id = await _create_session_with_transcript(
        client, session_store, "hello", auth_headers=auth_headers
    )

    with patch(
        "autobot_stt.routes.sessions.cleanup_transcript",
        new_callable=AsyncMock,
    ) as mock_cleanup:
        response = await client.post(
            f"/v1/sessions/{session_id}/finalize", headers=auth_headers
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "OpenAI API key not configured"
    mock_cleanup.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalize_deletes_session(
    client, session_store, auth_headers, monkeypatch
) -> None:
    _set_openai_key(monkeypatch)
    session_id = await _create_session_with_transcript(
        client, session_store, "hello", auth_headers=auth_headers
    )

    with patch(
        "autobot_stt.routes.sessions.cleanup_transcript",
        new_callable=AsyncMock,
    ) as mock_cleanup:
        mock_cleanup.return_value = "cleaned"
        response = await client.post(
            f"/v1/sessions/{session_id}/finalize", headers=auth_headers
        )

    assert response.status_code == 200
    assert await session_store.get(session_id) is None
    mock_cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_finalize_mock_receives_context(
    client, session_store, auth_headers, monkeypatch
) -> None:
    _set_openai_key(monkeypatch)
    session_id = await _create_session_with_transcript(
        client,
        session_store,
        "hello whisper",
        auth_headers=auth_headers,
        draft_text="draft line",
        chat_history=[{"role": "user", "content": "hi"}],
        comments=[{"author": "alice", "body": "looks good"}],
    )

    with patch(
        "autobot_stt.routes.sessions.cleanup_transcript",
        new_callable=AsyncMock,
    ) as mock_cleanup:
        mock_cleanup.return_value = "cleaned"
        await client.post(f"/v1/sessions/{session_id}/finalize", headers=auth_headers)

    mock_cleanup.assert_awaited_once()
    call_args = mock_cleanup.await_args
    session_arg = call_args.args[0]
    assert session_arg.raw_transcript == "hello whisper"
    assert session_arg.draft_text == "draft line"
    assert session_arg.chat_history[0].content == "hi"
    assert session_arg.comments[0].author == "alice"
    assert call_args.kwargs["api_key"] == "sk-test"


@pytest.mark.asyncio
async def test_cleanup_transcript_calls_openai_with_expected_payload(monkeypatch) -> None:
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from autobot_stt.models.session import ChatMessage, Comment, Session
    from autobot_stt.services import llm_cleanup

    session = Session(
        id="s1",
        draft_text="draft line",
        chat_history=[ChatMessage(role="user", content="hi")],
        comments=[Comment(author="alice", body="looks good")],
        created_at=datetime.now(UTC),
        raw_transcript="hello whisper",
    )

    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="corrected output"))]
    )
    create_mock = AsyncMock(return_value=fake_response)
    client_mock = MagicMock()
    client_mock.chat.completions.create = create_mock

    with patch.object(llm_cleanup, "AsyncOpenAI", return_value=client_mock):
        result = await llm_cleanup.cleanup_transcript(session, api_key="sk-test")

    assert result == "corrected output"
    create_mock.assert_awaited_once()
    call_kwargs = create_mock.await_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    user_content = messages[1]["content"]
    assert "draft line" in user_content
    assert "user: hi" in user_content
    assert "alice: looks good" in user_content
    assert "hello whisper" in user_content
