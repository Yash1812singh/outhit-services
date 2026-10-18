from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="GATEWAY_", extra="ignore")

    database_url: str = "sqlite:///./data/auth.db"
    jwt_secret: str = "dev-only-insecure-secret-set-GATEWAY_JWT_SECRET"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7
    # OAuth 2.0 Web client ID from Google Cloud Console; empty disables Google sign-in.
    google_client_id: str = ""
    cors_origins: str = "http://localhost:3000,http://localhost:3050"

    market_data_url: str = "http://127.0.0.1:8201"
    intelligence_url: str = "http://127.0.0.1:8202"
    research_url: str = "http://127.0.0.1:8203"
    advisor_url: str = "http://127.0.0.1:8204"
    paper_url: str = "http://127.0.0.1:8205"
    broker_url: str = "http://127.0.0.1:8206"

    # Requests per minute, per client.
    rate_limit_default: int = 600
    rate_limit_auth: int = 10
    rate_limit_chat: int = 12
    rate_limit_orders: int = 30

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
