from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Index, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from research.config import get_settings

settings = get_settings()
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def now() -> datetime:
    return datetime.now(timezone.utc)


def aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class Base(DeclarativeBase):
    pass


class RecommendationLog(Base):
    """Every pick the recommendation engine makes, kept to measure its real-world record."""

    __tablename__ = "recommendation_log"
    __table_args__ = (Index("ix_rec_unique", "day", "market", "horizon", "symbol", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    day: Mapped[str] = mapped_column(String(10))
    market: Mapped[str] = mapped_column(String(4))
    horizon: Mapped[str] = mapped_column(String(16))
    symbol: Mapped[str] = mapped_column(String(32))
    side: Mapped[int] = mapped_column(Integer, default=1)  # 1 = buy idea, -1 = avoid/short idea
    rank: Mapped[int] = mapped_column(Integer)
    price: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float)
    made_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    outcome: Mapped[dict | None] = mapped_column(JSON, default=None)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
