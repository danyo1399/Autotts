import pytest
from pydantic import ValidationError

from autobot_stt.config import Settings, get_settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.stt_api_key == ""
    assert settings.openai_api_key == ""
    assert settings.whisper_model == "base"
    assert settings.whisper_device == "cpu"
    assert settings.log_level == "info"


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    assert first is second


def test_settings_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHISPER_MODEL", "small")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    settings = Settings()
    assert settings.whisper_model == "small"
    assert settings.log_level == "debug"


def test_log_level_rejects_invalid_value() -> None:
    with pytest.raises(ValidationError):
        Settings(log_level="invalid")  # type: ignore[arg-type]
