from datetime import UTC, datetime

import pytest

from autobot_stt.models.session import Session
from autobot_stt.stores.memory import InMemorySessionStore


def _session(session_id: str = "s1") -> Session:
    return Session(
        id=session_id,
        draft_text="",
        chat_history=[],
        comments=[],
        created_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_create_then_get_returns_same_session() -> None:
    store = InMemorySessionStore()
    session = _session()

    await store.create(session)
    assert await store.get("s1") is session


@pytest.mark.asyncio
async def test_get_returns_none_for_unknown_id() -> None:
    store = InMemorySessionStore()
    assert await store.get("missing") is None


@pytest.mark.asyncio
async def test_delete_removes_session_and_returns_true() -> None:
    store = InMemorySessionStore()
    await store.create(_session())

    assert await store.delete("s1") is True
    assert await store.get("s1") is None


@pytest.mark.asyncio
async def test_delete_returns_false_for_unknown_id() -> None:
    store = InMemorySessionStore()
    assert await store.delete("missing") is False
