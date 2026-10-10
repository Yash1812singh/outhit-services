from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="INTELLIGENCE_", extra="ignore")

    database_url: str = "sqlite:///./data/intelligence.db"
    # market-data supplies the company universe, the quant model and technical signals.
    market_data_url: str = "http://127.0.0.1:8201"

    tracked_markets: str = "in,us"
    tracked_per_market: int = 40
    price_interval_seconds: int = 60
    news_interval_seconds: int = 600
    outlook_interval_seconds: int = 60
    enable_scheduler: bool = True

    # Optional: LLM news scoring through Groq (falls back to the trained sentiment model).
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-20b"

    @property
    def tracked_market_list(self) -> list[str]:
        return [m.strip() for m in self.tracked_markets.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
