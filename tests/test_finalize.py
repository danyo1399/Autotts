from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autobot_stt.config import get_settings
from autobot_stt.models.session import ChatMessage, Comment, Session
from autobot_stt.services import llm_cleanup
from autobot_stt.stores.memory import InMemorySessionStore


def _set_openai_key(monkeypatch: pytest.MonkeyPatch, value: str = "sk-test") -> None:
    monkeypatch.setenv("OPENAI_API_KEY", value)
    get_settings.cache_clear()


async def _create_session_with_transcript(
    client,
    session_store: InMemorySessionStore,
    transcript: str,
    *,
    auth_headers: dict[str, str],
    draft_text: str = "",
    chat_history: list[dict] | None = None,
    comments: list[dict] | None = None,
) -> str:
    body: dict = {"draft_text": draft_text}
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


@pytest.fixture
def mock_cleanup() -> AsyncMock:
    with patch(
        "autobot_stt.routes.sessions.cleanup_transcript",
        new_callable=AsyncMock,
    ) as m:
        yield m


@pytest.mark.asyncio
async def test_finalize_returns_cleaned_text_passes_context_and_deletes_session(
    client, session_store, auth_headers, monkeypatch, mock_cleanup
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
    mock_cleanup.return_value = "cleaned text"

    response = await client.post(
        f"/v1/sessions/{session_id}/finalize", headers=auth_headers
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "cleaned text"
    assert payload["raw_transcript"] == "hello whisper"

    mock_cleanup.assert_awaited_once()
    call_args = mock_cleanup.await_args
    session_arg = call_args.args[0]
    assert session_arg.raw_transcript == "hello whisper"
    assert session_arg.draft_text == "draft line"
    assert session_arg.chat_history[0].content == "hi"
    assert session_arg.comments[0].author == "alice"
    assert call_args.kwargs["api_key"] == "sk-test"

    assert await session_store.get(session_id) is None


@pytest.mark.asyncio
async def test_finalize_empty_transcript_returns_400(
    client, session_store, auth_headers, monkeypatch, mock_cleanup
) -> None:
    _set_openai_key(monkeypatch)
    session_id = await _create_session_with_transcript(
        client, session_store, "   ", auth_headers=auth_headers
    )

    response = await client.post(
        f"/v1/sessions/{session_id}/finalize", headers=auth_headers
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Session has no transcript"
    mock_cleanup.assert_not_awaited()


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
    client, session_store, auth_headers, monkeypatch, mock_cleanup
) -> None:
    _set_openai_key(monkeypatch, value="")
    session_id = await _create_session_with_transcript(
        client, session_store, "hello", auth_headers=auth_headers
    )

    response = await client.post(
        f"/v1/sessions/{session_id}/finalize", headers=auth_headers
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "OpenAI API key not configured"
    mock_cleanup.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalize_returns_empty_text_when_cleanup_yields_empty(
    client, session_store, auth_headers, monkeypatch, mock_cleanup
) -> None:
    _set_openai_key(monkeypatch)
    session_id = await _create_session_with_transcript(
        client, session_store, "raw only", auth_headers=auth_headers
    )
    mock_cleanup.return_value = ""

    response = await client.post(
        f"/v1/sessions/{session_id}/finalize", headers=auth_headers
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == ""
    assert payload["raw_transcript"] == "raw only"
    assert await session_store.get(session_id) is None


def _build_session() -> Session:
    return Session(
        id="s1",
        draft_text="draft line",
        chat_history=[ChatMessage(role="user", content="hi")],
        comments=[Comment(author="alice", body="looks good")],
        created_at=datetime.now(UTC),
        raw_transcript="hello whisper",
    )


def _mock_openai_client(content: str | None) -> MagicMock:
    create_mock = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )
    )
    client_mock = MagicMock()
    client_mock.chat.completions.create = create_mock
    client_mock.__aenter__ = AsyncMock(return_value=client_mock)
    client_mock.__aexit__ = AsyncMock(return_value=None)
    return client_mock


@pytest.mark.asyncio
async def test_cleanup_transcript_calls_openai_with_expected_payload() -> None:
    client_mock = _mock_openai_client(content="corrected output")

    with patch.object(llm_cleanup, "AsyncOpenAI", return_value=client_mock):
        result = await llm_cleanup.cleanup_transcript(
            _build_session(), api_key="sk-test"
        )

    assert result == "corrected output"
    create_mock = client_mock.chat.completions.create
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


@pytest.mark.asyncio
async def test_cleanup_transcript_strips_response_whitespace() -> None:
    client_mock = _mock_openai_client(content="  corrected output  \n")

    with patch.object(llm_cleanup, "AsyncOpenAI", return_value=client_mock):
        result = await llm_cleanup.cleanup_transcript(
            _build_session(), api_key="sk-test"
        )

    assert result == "corrected output"


@pytest.mark.asyncio
async def test_cleanup_transcript_empty_choices_returns_empty_string() -> None:
    create_mock = AsyncMock(return_value=SimpleNamespace(choices=[]))
    client_mock = MagicMock()
    client_mock.chat.completions.create = create_mock
    client_mock.__aenter__ = AsyncMock(return_value=client_mock)
    client_mock.__aexit__ = AsyncMock(return_value=None)

    with patch.object(llm_cleanup, "AsyncOpenAI", return_value=client_mock):
        result = await llm_cleanup.cleanup_transcript(
            _build_session(), api_key="sk-test"
        )

    assert result == ""


@pytest.mark.asyncio
async def test_cleanup_transcript_none_message_content_returns_empty_string() -> None:
    client_mock = _mock_openai_client(content=None)

    with patch.object(llm_cleanup, "AsyncOpenAI", return_value=client_mock):
        result = await llm_cleanup.cleanup_transcript(
            _build_session(), api_key="sk-test"
        )

    assert result == ""


@pytest.mark.asyncio
async def test_cleanup_transcript_propagates_openai_error() -> None:
    from openai import OpenAIError

    client_mock = MagicMock()
    client_mock.chat.completions.create = AsyncMock(
        side_effect=OpenAIError("upstream failure")
    )
    client_mock.__aenter__ = AsyncMock(return_value=client_mock)
    client_mock.__aexit__ = AsyncMock(return_value=None)

    with patch.object(llm_cleanup, "AsyncOpenAI", return_value=client_mock):
        with pytest.raises(OpenAIError):
            await llm_cleanup.cleanup_transcript(_build_session(), api_key="sk-test")


@pytest.mark.asyncio
async def test_cleanup_transcript_passes_api_key_to_openai_client() -> None:
    client_mock = _mock_openai_client(content="ok")

    with patch.object(llm_cleanup, "AsyncOpenAI", return_value=client_mock) as ctor:
        await llm_cleanup.cleanup_transcript(_build_session(), api_key="sk-secret")

    ctor.assert_called_once_with(api_key="sk-secret")


@pytest.mark.asyncio
async def test_cleanup_transcript_renders_empty_context_as_placeholders() -> None:
    empty_session = Session(
        id="s2",
        draft_text="   ",
        chat_history=[],
        comments=[],
        created_at=datetime.now(UTC),
        raw_transcript="raw only",
    )
    client_mock = _mock_openai_client(content="ok")

    with patch.object(llm_cleanup, "AsyncOpenAI", return_value=client_mock):
        await llm_cleanup.cleanup_transcript(empty_session, api_key="sk-test")

    user_content = client_mock.chat.completions.create.await_args.kwargs["messages"][
        1
    ]["content"]
    assert "(empty)" in user_content
    assert "(none)" in user_content
    assert user_content.count("(none)") == 2
    assert "raw only" in user_content
