from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="ADVISOR_", extra="ignore")

    database_url: str = "sqlite:///./data/advisor.db"
    market_data_url: str = "http://127.0.0.1:8201"
    intelligence_url: str = "http://127.0.0.1:8202"
    research_url: str = "http://127.0.0.1:8203"

    # Chat: Claude when an Anthropic key is set, otherwise Groq.
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5-5"
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"


@lru_cache
def get_settings() -> Settings:
    return Settings()
