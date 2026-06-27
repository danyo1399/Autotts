import asyncio

from fastapi import Request, WebSocket

from autobot_stt.services.whisper_service import WhisperService
from autobot_stt.stores.base import SessionStore


def get_session_store(request: Request) -> SessionStore:
    return request.app.state.session_store


def get_whisper_service(websocket: WebSocket) -> WhisperService:
    service: WhisperService | None = getattr(websocket.app.state, "whisper_service", None)
    if service is None:
        raise RuntimeError("Whisper service is not initialized on app.state")
    return service


def get_whisper_lock(websocket: WebSocket) -> asyncio.Lock:
    lock: asyncio.Lock | None = getattr(websocket.app.state, "whisper_lock", None)
    if lock is None:
        raise RuntimeError("Whisper lock is not initialized on app.state")
    return lock
