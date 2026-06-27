import asyncio

from autobot_stt.models.session import Session


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create(self, session: Session) -> None:
        async with self._lock:
            self._sessions[session.id] = session

    async def get(self, session_id: str) -> Session | None:
        async with self._lock:
            return self._sessions.get(session_id)

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            return self._sessions.pop(session_id, None) is not None
