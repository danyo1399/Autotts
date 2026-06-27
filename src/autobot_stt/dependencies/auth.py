from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.websockets import WebSocket

from autobot_stt.config import Settings, get_settings

_bearer = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.stt_api_key:
        return
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Unauthorized")
    if credentials.credentials != settings.stt_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _extract_ws_token(websocket: WebSocket, token: str | None) -> str | None:
    """Return the API key candidate from the query param or Authorization header."""
    if token:
        return token
    auth = websocket.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[len("bearer "):].strip()
    return None


def check_ws_api_key(
    websocket: WebSocket,
    token: str | None,
    settings: Settings,
) -> bool:
    """Return True if WS auth passes (or is disabled when the configured key is empty)."""
    if not settings.stt_api_key:
        return True
    provided = _extract_ws_token(websocket, token)
    return provided == settings.stt_api_key
