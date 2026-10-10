from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="MARKET_DATA_", extra="ignore")

    database_url: str = "sqlite:///./data/market_data.db"
    # Market data cache lifetime. Yahoo data is delayed anyway; 15 min keeps us
    # well under rate limits while staying fresh enough for daily-bar analysis.
    market_cache_seconds: int = 900
    # Keeps featured markets (India, US) and recently viewed ones refreshed in the background.
    enable_refresher: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
