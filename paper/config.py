from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="PAPER_", extra="ignore")

    database_url: str = "sqlite:///./data/paper.db"
    research_url: str = "http://127.0.0.1:8203"
    intelligence_url: str = "http://127.0.0.1:8202"
    enable_scheduler: bool = True
    max_portfolios_per_user: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
