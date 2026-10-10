from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RESEARCH_", extra="ignore")

    database_url: str = "sqlite:///./data/research.db"
    market_data_url: str = "http://127.0.0.1:8201"
    intelligence_url: str = "http://127.0.0.1:8202"
    tracked_markets: str = "in,us"
    enable_scheduler: bool = True
    # SEC asks automated clients to identify themselves: "Company/App contact@email"
    sec_user_agent: str = "Outhit research app admin@outhit.local"

    @property
    def tracked_market_list(self) -> list[str]:
        return [m.strip() for m in self.tracked_markets.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
