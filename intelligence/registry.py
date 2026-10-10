"""Which companies the AI tracks: the largest in each tracked market, plus any symbol a user opens."""

import logging
import re

import yfinance as yf
from sqlalchemy import select

from intelligence import backend, live
from intelligence.config import get_settings
from intelligence.db import Company, SessionLocal, now

log = logging.getLogger(__name__)

SUFFIX_MARKET = {
    "NS": "in", "BO": "in", "L": "gb", "T": "jp", "DE": "de", "F": "de", "PA": "fr", "HK": "hk", "TO": "ca", "V": "ca",
    "AX": "au", "KS": "kr", "KQ": "kr", "TW": "tw", "TWO": "tw", "SS": "cn", "SZ": "cn", "SA": "br", "MX": "mx",
    "SW": "ch", "AS": "nl", "MC": "es", "MI": "it", "ST": "se", "CO": "dk", "HE": "fi", "BR": "be", "VI": "at",
    "IR": "ie", "LS": "pt", "AT": "gr", "SI": "sg", "JK": "id", "KL": "my", "NZ": "nz", "TA": "il", "JO": "za",
    "BA": "ar", "IS": "tr", "BK": "th", "SR": "sa", "OL": "no", "WA": "pl", "SN": "cl", "CA": "eg", "QA": "qa",
    "KW": "kw", "PR": "cz", "BD": "hu", "RO": "ro", "IC": "is", "TL": "ee", "VS": "lt", "RG": "lv",
}


def market_of(symbol: str) -> str:
    m = re.search(r"\.([A-Z]+)$", symbol.upper())
    return SUFFIX_MARKET.get(m.group(1), "us") if m else "us"


def _upsert(row: dict, market: str, pinned: bool) -> None:
    with SessionLocal() as db:
        c = db.get(Company, row["symbol"])
        if c is None:
            c = Company(symbol=row["symbol"], name=row["name"], market=market)
            db.add(c)
        c.name = row.get("name") or c.name
        c.sector = row.get("sector") or c.sector
        c.currency = row.get("currency") or c.currency
        c.analyst_rating = row.get("analyst_rating") or c.analyst_rating
        c.pinned = c.pinned or pinned
        db.commit()


def bootstrap() -> int:
    """Pin the largest companies of each tracked market. Safe to call repeatedly."""
    s = get_settings()
    n = 0
    for market in s.tracked_market_list:
        for row in backend.top_companies(market, s.tracked_per_market):
            _upsert(row, market, pinned=True)
            n += 1
    live.pin(pinned_symbols())
    return n


def pinned_symbols() -> list[str]:
    with SessionLocal() as db:
        return list(db.scalars(select(Company.symbol).where(Company.pinned)).all())


def active_symbols(hours: int = 2) -> list[str]:
    """Pinned companies plus anything opened in the last few hours."""
    from datetime import timedelta

    with SessionLocal() as db:
        recent = db.scalars(select(Company.symbol).where(Company.last_requested >= now() - timedelta(hours=hours))).all()
    return sorted(set(pinned_symbols()) | set(recent))


def ensure(symbol: str) -> Company | None:
    """Start tracking a symbol on demand (e.g. when someone opens its AI page)."""
    symbol = symbol.upper()
    with SessionLocal() as db:
        c = db.get(Company, symbol)
        if c:
            c.last_requested = now()
            db.commit()
            return c
    market = market_of(symbol)
    row = backend.company(market, symbol)
    if row is None:
        try:
            info = yf.Ticker(symbol).info or {}
        except Exception:
            info = {}
        name = info.get("longName") or info.get("shortName")
        if not name:
            return None
        row = {"symbol": symbol, "name": name, "sector": info.get("sector"), "currency": info.get("currency")}
    _upsert(row, market, pinned=False)
    with SessionLocal() as db:
        c = db.get(Company, symbol)
        c.last_requested = now()
        db.commit()
        return c


def by_market(market: str) -> list[Company]:
    with SessionLocal() as db:
        return list(db.scalars(select(Company).where(Company.market == market, Company.pinned)).all())


def track(symbols: list[str]) -> int:
    """Pin recommended stocks so news, reputation, outlooks and live prices are collected for them too."""
    n = 0
    for sym in symbols:
        try:
            c = ensure(sym)
        except Exception as exc:
            log.info("could not track %s: %s", sym, exc)
            continue
        if c is None:
            continue
        with SessionLocal() as db:
            row = db.get(Company, sym)
            if row and not row.pinned:
                row.pinned = True
                db.commit()
                n += 1
    live.pin(pinned_symbols())
    return n
