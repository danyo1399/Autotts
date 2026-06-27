"""Unit tests for the dependency-injection accessors in dependencies/store.py."""

import asyncio
from unittest.mock import MagicMock

import pytest
from starlette.applications import Starlette

from autobot_stt.dependencies.store import (
    get_session_store,
    get_whisper_lock,
    get_whisper_service,
)


def _connection(app: Starlette) -> MagicMock:
    """Minimal stand-in for an HTTPConnection exposing ``app.state``."""
    conn = MagicMock()
    conn.app = app
    return conn


def test_get_session_store_returns_store_from_app_state() -> None:
    app = Starlette()
    expected = MagicMock()
    app.state.session_store = expected

    assert get_session_store(_connection(app)) is expected


def test_get_whisper_service_returns_service_when_initialized() -> None:
    app = Starlette()
    service = MagicMock()
    app.state.whisper_service = service

    assert get_whisper_service(_connection(app)) is service


def test_get_whisper_service_raises_when_missing() -> None:
    app = Starlette()
    # No ``whisper_service`` attribute set on state.
    with pytest.raises(RuntimeError, match="Whisper service"):
        get_whisper_service(_connection(app))


def test_get_whisper_lock_returns_lock_when_initialized() -> None:
    app = Starlette()
    lock = asyncio.Lock()
    app.state.whisper_lock = lock

    assert get_whisper_lock(_connection(app)) is lock


def test_get_whisper_lock_raises_when_missing() -> None:
    app = Starlette()
    with pytest.raises(RuntimeError, match="Whisper lock"):
        get_whisper_lock(_connection(app))
