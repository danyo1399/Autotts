from fastapi import Request

from autobot_stt.stores.base import SessionStore


def get_session_store(request: Request) -> SessionStore:
    return request.app.state.session_store
