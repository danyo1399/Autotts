import asyncio

from starlette.requests import HTTPConnection

from autobot_stt.services.whisper_service import WhisperService
from autobot_stt.stores.base import SessionStore


def get_session_store(conn: HTTPConnection) -> SessionStore:
    return conn.app.state.session_store


def get_whisper_service(conn: HTTPConnection) -> WhisperService:
    service: WhisperService | None = getattr(conn.app.state, "whisper_service", None)
    if service is None:
        raise RuntimeError("Whisper service is not initialized on app.state")
    return service


def get_whisper_lock(conn: HTTPConnection) -> asyncio.Lock:
    lock: asyncio.Lock | None = getattr(conn.app.state, "whisper_lock", None)
    if lock is None:
        raise RuntimeError("Whisper lock is not initialized on app.state")
    return lock
