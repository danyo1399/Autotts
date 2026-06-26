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
