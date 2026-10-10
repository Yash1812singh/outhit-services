"""Company universe per country, pulled from Yahoo's equity screener.

For each market we fetch the largest companies in each of the 11 sectors
(the screener has no sector field in its output, so querying sector by sector
is how we label them), store the snapshot in the database, and keep it fresh
with a background thread. Prices are delayed by the exchange's usual 15-20
minutes; Yahoo reports the delay per quote.
"""

import logging
import threading
import time
from datetime import datetime, timedelta, timezone

import yfinance as yf
from sqlalchemy import delete, select
from yfinance import EquityQuery as Q

from market_data.db import SessionLocal
from market_data.markets import MARKETS, MARKETS_BY_CODE, SECTORS, Market
from market_data.models import MarketRefresh, StockQuote

log = logging.getLogger(__name__)

PAGE = 250
REFRESH_EVERY = timedelta(minutes=15)
KEEP_WARM_FOR = timedelta(hours=2)  # markets someone looked at recently stay refreshed

# Screener field -> our field. Percent fields from Yahoo are already in percent units.
FIELDS = {
    "regularMarketPrice": "price",
    "regularMarketChangePercent": "change_pct",
    "regularMarketVolume": "volume",
    "averageDailyVolume3Month": "avg_volume_3m",
    "marketCap": "market_cap",
    "trailingPE": "pe",
    "forwardPE": "forward_pe",
    "priceToBook": "pb",
    "epsTrailingTwelveMonths": "eps_ttm",
    "epsForward": "eps_forward",
    "dividendYield": "dividend_yield_pct",
    "fiftyTwoWeekChangePercent": "change_52w_pct",
    "fiftyTwoWeekHigh": "high_52w",
    "fiftyTwoWeekLow": "low_52w",
    "fiftyDayAverage": "sma50",
    "twoHundredDayAverage": "sma200",
    "averageAnalystRating": "analyst_rating",
    "marketState": "market_state",
    "exchangeDataDelayedBy": "delay_minutes",
    "regularMarketTime": "quote_time",
    "earningsTimestamp": "earnings_time",
}

_last_access: dict[str, datetime] = {}
_locks: dict[str, threading.Lock] = {m.code: threading.Lock() for m in MARKETS}


class UnknownMarket(Exception):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _query(market: Market, sector: str | None) -> Q:
    parts = [Q("eq", ["region", market.code])]
    if len(market.exchanges) == 1:
        parts.append(Q("eq", ["exchange", market.exchanges[0]]))
    elif market.exchanges:
        parts.append(Q("is-in", ["exchange", *market.exchanges]))
    if sector:
        parts.append(Q("eq", ["sector", sector]))
    return Q("and", parts)


def _fetch(market: Market, sector: str | None, limit: int) -> list[dict]:
    out: list[dict] = []
    offset = 0
    while len(out) < limit:
        res = yf.screen(_query(market, sector), sortField="intradaymarketcap", sortAsc=False, size=min(PAGE, limit - len(out)), offset=offset)
        quotes = res.get("quotes", []) if res else []
        out.extend(quotes)
        offset += len(quotes)
        if not quotes or offset >= (res.get("total") or 0):
            break
    return out


def _row(q: dict, market: Market, sector: str) -> StockQuote | None:
    price = q.get("regularMarketPrice")
    if not q.get("symbol") or not price:
        return None
    data = {ours: q.get(theirs) for theirs, ours in FIELDS.items()}
    return StockQuote(
        symbol=q["symbol"],
        market=market.code,
        sector=sector,
        name=q.get("longName") or q.get("shortName") or q["symbol"],
        exchange=q.get("exchange") or "",
        currency=q.get("currency") or market.currency,
        market_cap=q.get("marketCap"),
        data=data,
        updated_at=_now(),
    )


def refresh_market(code: str) -> int:
    """Pull the market's universe and replace the stored snapshot. Returns the company count."""
    market = MARKETS_BY_CODE[code]
    with _locks[code]:
        rows: dict[str, StockQuote] = {}
        error = None
        try:
            for sector in SECTORS:
                for q in _fetch(market, sector, market.per_sector):
                    if (row := _row(q, market, sector)) and row.symbol not in rows:
                        rows[row.symbol] = row
                time.sleep(0.15)  # be polite to the free endpoint
            if not rows:  # some small markets have no sector tags in Yahoo's data
                for q in _fetch(market, None, market.per_sector * 3):
                    if (row := _row(q, market, "Unclassified")) and row.symbol not in rows:
                        rows[row.symbol] = row
        except Exception as exc:
            log.warning("Screener refresh failed for %s: %s", code, exc)
            error = str(exc)[:500]

        with SessionLocal() as db:
            if rows:
                db.execute(delete(StockQuote).where(StockQuote.market == code))
                db.add_all(rows.values())
            db.merge(MarketRefresh(market=code, refreshed_at=_now(), companies=len(rows) or _count(db, code), error=error))
            db.commit()
        log.info("Refreshed %s: %d companies", code, len(rows))
        return len(rows)


def _count(db, code: str) -> int:
    return len(db.scalars(select(StockQuote.symbol).where(StockQuote.market == code)).all())


def status(code: str) -> MarketRefresh | None:
    with SessionLocal() as db:
        return db.get(MarketRefresh, code)


def ensure_market(code: str) -> MarketRefresh:
    """Make sure a market has data, loading it synchronously the first time."""
    if code not in MARKETS_BY_CODE:
        raise UnknownMarket(f"Unknown market '{code}'")
    _last_access[code] = _now()
    st = status(code)
    if st is None or st.companies == 0:
        refresh_market(code)
        st = status(code)
    return st


def load(code: str, sector: str | None = None) -> list[StockQuote]:
    ensure_market(code)
    with SessionLocal() as db:
        stmt = select(StockQuote).where(StockQuote.market == code)
        if sector:
            stmt = stmt.where(StockQuote.sector == sector)
        return list(db.scalars(stmt.order_by(StockQuote.market_cap.desc().nulls_last())).all())


def get_quote(symbol: str) -> StockQuote | None:
    with SessionLocal() as db:
        return db.get(StockQuote, symbol)


# ---------------------------------------------------------------------------
# Background refresher
# ---------------------------------------------------------------------------

_stop = threading.Event()


def _due(code: str) -> bool:
    st = status(code)
    return st is None or _now() - _aware(st.refreshed_at) > REFRESH_EVERY


def _loop() -> None:
    while not _stop.is_set():
        for m in MARKETS:
            if _stop.is_set():
                break
            recently_used = _now() - _last_access.get(m.code, datetime.min.replace(tzinfo=timezone.utc)) < KEEP_WARM_FOR
            if (m.featured or recently_used) and _due(m.code):
                try:
                    refresh_market(m.code)
                except Exception as exc:  # never let the thread die
                    log.warning("Refresher error for %s: %s", m.code, exc)
        _stop.wait(60)


def start_refresher() -> None:
    _stop.clear()
    threading.Thread(target=_loop, name="screener-refresher", daemon=True).start()


def stop_refresher() -> None:
    _stop.set()
