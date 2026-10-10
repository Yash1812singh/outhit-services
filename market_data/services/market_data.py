"""Price history from Yahoo Finance, with an in-process TTL cache.

Everything downstream works on daily closes, so one batch download per cache
window serves every endpoint.
"""

import logging
import threading
import time

import pandas as pd
import yfinance as yf

from market_data.config import get_settings

log = logging.getLogger(__name__)

HISTORY_PERIOD = "5y"  # enough for a 200-day trend filter plus a multi-year backtest

_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_lock = threading.Lock()


class MarketDataError(Exception):
    pass


def _fresh(symbol: str) -> pd.DataFrame | None:
    entry = _cache.get(symbol)
    if entry and time.time() - entry[0] < get_settings().market_cache_seconds:
        return entry[1]
    return None


def _download(symbols: list[str]) -> dict[str, pd.DataFrame]:
    raw = yf.download(
        symbols,
        period=HISTORY_PERIOD,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            frame = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        frame = frame[["Close", "Volume"]].dropna(subset=["Close"])
        if len(frame) >= 30:
            frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
            out[sym] = frame
    return out


def get_history(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Daily Close/Volume frames keyed by symbol. Symbols with no data are omitted."""
    symbols = list(dict.fromkeys(s.upper() for s in symbols))
    with _lock:
        result = {s: df for s in symbols if (df := _fresh(s)) is not None}
        missing = [s for s in symbols if s not in result]
        if missing:
            try:
                fetched = _download(missing)
            except Exception as exc:  # yfinance raises a wide variety of network errors
                log.warning("Yahoo download failed for %s: %s", missing, exc)
                fetched = {}
            now = time.time()
            for sym, df in fetched.items():
                _cache[sym] = (now, df)
            result.update(fetched)
    return result


def get_one(symbol: str) -> pd.DataFrame:
    data = get_history([symbol])
    if symbol.upper() not in data:
        raise MarketDataError(f"No price data found for '{symbol}'. Check the symbol (e.g. AAPL, RELIANCE.NS, BTC-USD).")
    return data[symbol.upper()]


def search(query: str, limit: int = 8) -> list[dict]:
    try:
        quotes = yf.Search(query, max_results=limit, news_count=0).quotes
    except Exception as exc:
        log.warning("Yahoo search failed for %r: %s", query, exc)
        return []
    return [
        {
            "symbol": q.get("symbol"),
            "name": q.get("shortname") or q.get("longname") or q.get("symbol"),
            "exchange": q.get("exchDisp") or q.get("exchange"),
            "type": q.get("quoteType"),
        }
        for q in quotes
        if q.get("symbol")
    ]
