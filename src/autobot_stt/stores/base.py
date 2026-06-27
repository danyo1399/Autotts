from typing import Protocol

from autobot_stt.models.session import Session


class SessionStore(Protocol):
    async def create(self, session: Session) -> None: ...

    async def get(self, session_id: str) -> Session | None: ...

    async def delete(self, session_id: str) -> bool: ...
