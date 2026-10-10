from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="BROKER_", extra="ignore")

    database_url: str = "sqlite:///./data/broker.db"
    # Fernet key that encrypts API secrets and access tokens at rest. Generate one with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    encryption_key: str = ""
    intelligence_url: str = "http://127.0.0.1:8202"
    # Must equal the Redirect URL configured in your Upstox app.
    redirect_uri: str = "http://localhost:3050/trade/broker/callback"
    upstox_web_url: str = "https://pro.upstox.com/"
    upstox_funds_help_url: str = "https://upstox.com/help-center/250454/"
    # Per-user limits (INR): defaults, and the highest a user may set them.
    default_max_order_value: float = 50_000
    default_max_daily_value: float = 200_000
    cap_max_order_value: float = 500_000
    cap_max_daily_value: float = 2_000_000
    preview_ttl_seconds: int = 120


@lru_cache
def get_settings() -> Settings:
    return Settings()
