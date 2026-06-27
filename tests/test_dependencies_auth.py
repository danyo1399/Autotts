"""Unit tests for the WebSocket auth helpers in dependencies/auth.py."""

from unittest.mock import MagicMock

from autobot_stt.config import Settings
from autobot_stt.dependencies.auth import _extract_ws_token, check_ws_api_key


def _websocket(headers: dict[str, str] | None = None) -> MagicMock:
    """Build a minimal WebSocket stand-in exposing ``headers``."""
    ws = MagicMock()
    ws.headers = headers or {}
    return ws


# --- _extract_ws_token ----------------------------------------------------


def test_extract_token_prefers_query_param_over_header() -> None:
    ws = _websocket({"authorization": "Bearer from-header"})
    assert _extract_ws_token(ws, "from-query") == "from-query"


def test_extract_token_reads_bearer_header_when_no_query_param() -> None:
    ws = _websocket({"authorization": "Bearer abc123"})
    assert _extract_ws_token(ws, None) == "abc123"


def test_extract_token_strips_whitespace_around_bearer_credential() -> None:
    ws = _websocket({"authorization": "Bearer   padded   "})
    assert _extract_ws_token(ws, None) == "padded"


def test_extract_token_accepts_lowercase_bearer_scheme() -> None:
    ws = _websocket({"authorization": "bearer mixed-case"})
    assert _extract_ws_token(ws, None) == "mixed-case"


def test_extract_token_returns_none_for_non_bearer_scheme() -> None:
    ws = _websocket({"authorization": "Basic dXNlcjpwYXNz"})
    assert _extract_ws_token(ws, None) is None


def test_extract_token_returns_none_when_no_auth_present() -> None:
    ws = _websocket({})
    assert _extract_ws_token(ws, None) is None


# --- check_ws_api_key -----------------------------------------------------


def test_check_ws_api_key_bypasses_when_setting_empty() -> None:
    settings = Settings(stt_api_key="")
    ws = _websocket({})
    assert check_ws_api_key(ws, None, settings) is True


def test_check_ws_api_key_accepts_matching_query_token() -> None:
    settings = Settings(stt_api_key="secret")
    ws = _websocket({})
    assert check_ws_api_key(ws, "secret", settings) is True


def test_check_ws_api_key_accepts_matching_bearer_header() -> None:
    settings = Settings(stt_api_key="secret")
    ws = _websocket({"authorization": "Bearer secret"})
    assert check_ws_api_key(ws, None, settings) is True


def test_check_ws_api_key_rejects_mismatched_token() -> None:
    settings = Settings(stt_api_key="secret")
    ws = _websocket({})
    assert check_ws_api_key(ws, "wrong", settings) is False


def test_check_ws_api_key_rejects_when_no_credential_provided() -> None:
    settings = Settings(stt_api_key="secret")
    ws = _websocket({})
    assert check_ws_api_key(ws, None, settings) is False
