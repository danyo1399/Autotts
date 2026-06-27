from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from autobot_stt.config import get_settings
from autobot_stt.dependencies.store import get_session_store
from autobot_stt.main import app
from autobot_stt.stores.memory import InMemorySessionStore


@pytest.fixture
def session_store() -> InMemorySessionStore:
    return InMemorySessionStore()


@pytest.fixture
async def client(session_store: InMemorySessionStore) -> AsyncIterator[AsyncClient]:
    app.dependency_overrides[get_session_store] = lambda: session_store
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    monkeypatch.setenv("STT_API_KEY", "test-secret")
    get_settings.cache_clear()
    return {"Authorization": "Bearer test-secret"}


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
