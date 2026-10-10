from datetime import datetime

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from market_data.db import Base, _now


class StockQuote(Base):
    """Latest screener snapshot for one listed company."""

    __tablename__ = "stock_quotes"

    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    market: Mapped[str] = mapped_column(String(4), index=True)
    sector: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(255))
    exchange: Mapped[str] = mapped_column(String(16))
    currency: Mapped[str] = mapped_column(String(8))
    market_cap: Mapped[float | None] = mapped_column(default=None, index=True)
    data: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class MarketRefresh(Base):
    """When each market's universe was last pulled, so the refresher and the UI know how fresh it is."""

    __tablename__ = "market_refreshes"

    market: Mapped[str] = mapped_column(String(4), primary_key=True)
    refreshed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    companies: Mapped[int] = mapped_column(default=0)
    error: Mapped[str | None] = mapped_column(String(500), default=None)


class FundamentalsCache(Base):
    """Detailed per-company fundamentals (ROE, debt, margins...), refreshed daily."""

    __tablename__ = "fundamentals_cache"

    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    data: Mapped[dict] = mapped_column(JSON)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
