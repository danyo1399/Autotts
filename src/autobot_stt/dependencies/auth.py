import hmac

from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.websockets import WebSocket

from autobot_stt.config import Settings, get_settings

_bearer = HTTPBearer(auto_error=False)


def _keys_match(provided: str | None, expected: str) -> bool:
    """Constant-time comparison so attackers cannot time-side-channel the key.

    Returns False when ``provided`` is None — compare_digest requires both
    arguments to be the same type, so a None credential must short-circuit.
    """
    if provided is None:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.stt_api_key:
        return
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not _keys_match(credentials.credentials, settings.stt_api_key):
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
    return _keys_match(provided, settings.stt_api_key)
