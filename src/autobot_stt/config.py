from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    stt_api_key: str = ""
    openai_api_key: str = ""
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    log_level: str = "info"


def get_settings() -> Settings:
    return Settings()
