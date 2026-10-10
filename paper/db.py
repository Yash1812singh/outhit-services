from collections.abc import Iterator
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from paper.config import get_settings

settings = get_settings()
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def now() -> datetime:
    return datetime.now(timezone.utc)


def aware(dt: datetime | None) -> datetime | None:
    return dt.replace(tzinfo=timezone.utc) if dt is not None and dt.tzinfo is None else dt


class Base(DeclarativeBase):
    pass


class Portfolio(Base):
    """A virtual account. Manual ones take your orders; copy ones follow a strategy automatically."""

    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    name: Mapped[str] = mapped_column(String(100))
    market: Mapped[str] = mapped_column(String(4))
    currency: Mapped[str] = mapped_column(String(4))
    initial_cash: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    strategy_key: Mapped[str | None] = mapped_column(String(40), nullable=True)
    strategy_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    strategy_params: Mapped[dict] = mapped_column(JSON, default=dict)
    rebalance: Mapped[str | None] = mapped_column(String(2), nullable=True)  # W M Q A F
    next_rebalance_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_rebalance_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_signature: Mapped[str | None] = mapped_column(String(40), nullable=True)
    status_message: Mapped[str | None] = mapped_column(String(300), nullable=True)
    benchmark_symbol: Mapped[str] = mapped_column(String(20))
    benchmark_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    copying: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)

    positions: Mapped[list["Position"]] = relationship(cascade="all, delete-orphan", lazy="selectin")


class Position(Base):
    __tablename__ = "positions"
    __table_args__ = (UniqueConstraint("portfolio_id", "symbol"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), index=True)
    symbol: Mapped[str] = mapped_column(String(30))
    name: Mapped[str] = mapped_column(String(200))
    qty: Mapped[float] = mapped_column(Float)
    avg_cost: Mapped[float] = mapped_column(Float)


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), index=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    symbol: Mapped[str] = mapped_column(String(30))
    name: Mapped[str] = mapped_column(String(200))
    side: Mapped[str] = mapped_column(String(4))  # buy | sell
    qty: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    fees: Mapped[float] = mapped_column(Float)
    realized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str] = mapped_column(String(200))
    market_open: Mapped[bool] = mapped_column(Boolean, default=True)


class Snapshot(Base):
    """End-of-day value, for the equity curve."""

    __tablename__ = "snapshots"
    __table_args__ = (UniqueConstraint("portfolio_id", "day"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), index=True)
    day: Mapped[datetime] = mapped_column(Date)
    equity: Mapped[float] = mapped_column(Float)
    benchmark: Mapped[float | None] = mapped_column(Float, nullable=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
