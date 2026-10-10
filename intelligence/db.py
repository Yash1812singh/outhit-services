from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Index, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from intelligence.config import get_settings

settings = get_settings()
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def now() -> datetime:
    return datetime.now(timezone.utc)


def aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class Base(DeclarativeBase):
    pass


class Company(Base):
    """A company the AI tracks (from the main API's universe, or added on demand)."""

    __tablename__ = "companies"

    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    market: Mapped[str] = mapped_column(String(4), index=True)
    sector: Mapped[str | None] = mapped_column(String(64), default=None)
    currency: Mapped[str | None] = mapped_column(String(8), default=None)
    analyst_rating: Mapped[str | None] = mapped_column(String(32), default=None)
    governance: Mapped[dict | None] = mapped_column(JSON, default=None)
    pinned: Mapped[bool] = mapped_column(default=False)  # part of the always-tracked set
    last_requested: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    news_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)


class NewsItem(Base):
    __tablename__ = "news_items"
    __table_args__ = (Index("ix_news_symbol_published", "symbol", "published_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32))
    key: Mapped[str] = mapped_column(String(200), index=True)  # normalised title, for de-duplication
    title: Mapped[str] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(120), default=None)
    url: Mapped[str] = mapped_column(Text)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    sentiment: Mapped[float] = mapped_column(Float)  # P(bullish) - P(bearish), -1..1
    label: Mapped[str] = mapped_column(String(10))
    confidence: Mapped[float] = mapped_column(Float)
    events: Mapped[list] = mapped_column(JSON, default=list)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    llm_score: Mapped[float | None] = mapped_column(Float, default=None)  # LLM's read of the headline, -1..1
    llm_scored: Mapped[bool | None] = mapped_column(default=None)


class Prediction(Base):
    """One AI outlook per symbol per day, kept forever so the AI's live track record can be measured."""

    __tablename__ = "predictions"
    __table_args__ = (Index("ix_pred_symbol_day", "symbol", "day", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32))
    market: Mapped[str] = mapped_column(String(4))
    day: Mapped[str] = mapped_column(String(10))
    made_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    price: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float)
    label: Mapped[str] = mapped_column(String(20))
    components: Mapped[dict] = mapped_column(JSON)
    outcome: Mapped[dict | None] = mapped_column(JSON, default=None)  # realised returns, filled in later


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


def _add_missing_columns() -> None:
    """Tiny migration step: add columns introduced after a table was first created (SQLite)."""
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name not in have:
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {col.type.compile(engine.dialect)}'))
