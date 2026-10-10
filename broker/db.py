from collections.abc import Iterator
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from broker.config import get_settings

settings = get_settings()
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def now() -> datetime:
    return datetime.now(timezone.utc)


def aware(dt: datetime | None) -> datetime | None:
    return dt.replace(tzinfo=timezone.utc) if dt is not None and dt.tzinfo is None else dt


class Base(DeclarativeBase):
    pass


class Connection(Base):
    """One user's link to their own Upstox account. Secrets are stored encrypted."""

    __tablename__ = "connections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    broker: Mapped[str] = mapped_column(String(20), default="upstox")
    mode: Mapped[str] = mapped_column(String(10))  # sandbox | live
    api_key_enc: Mapped[str | None] = mapped_column(Text, nullable=True)
    api_secret_enc: Mapped[str | None] = mapped_column(Text, nullable=True)
    access_token_enc: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    oauth_state: Mapped[str | None] = mapped_column(String(64), nullable=True)
    oauth_state_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    profile: Mapped[dict] = mapped_column(JSON, default=dict)
    live_trading_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    live_enabled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    paused: Mapped[bool] = mapped_column(Boolean, default=False)
    max_order_value: Mapped[float] = mapped_column(Float)
    max_daily_value: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)


class Order(Base):
    """Every order Outhit sends (or was asked to preview), for the audit trail and daily limits."""

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    mode: Mapped[str] = mapped_column(String(10))
    nonce: Mapped[str] = mapped_column(String(64), unique=True)
    status: Mapped[str] = mapped_column(String(12))  # preview | placed | rejected | error | expired | cancelled
    broker_order_ids: Mapped[list] = mapped_column(JSON, default=list)
    symbol: Mapped[str] = mapped_column(String(30))
    name: Mapped[str] = mapped_column(String(200))
    instrument_key: Mapped[str] = mapped_column(String(40))
    side: Mapped[str] = mapped_column(String(4))  # BUY | SELL
    qty: Mapped[int] = mapped_column(Integer)
    order_type: Mapped[str] = mapped_column(String(6))  # MARKET LIMIT SL SL-M
    product: Mapped[str] = mapped_column(String(2))  # D (delivery) | I (intraday)
    price: Mapped[float] = mapped_column(Float, default=0)
    trigger_price: Mapped[float] = mapped_column(Float, default=0)
    ref_price: Mapped[float] = mapped_column(Float)
    est_value: Mapped[float] = mapped_column(Float)
    message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
