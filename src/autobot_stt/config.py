from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["debug", "info", "warning", "error", "critical"]
WhisperDevice = Literal["cpu", "cuda"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    stt_api_key: str = ""
    openai_api_key: str = ""
    whisper_model: str = "base"
    whisper_device: WhisperDevice = "cpu"
    log_level: LogLevel = "info"


@lru_cache
def get_settings() -> Settings:
    return Settings()
