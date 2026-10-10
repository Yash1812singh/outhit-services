"""Shared market-data layer for the research engines: stock universes and price panels,
cached on disk so restarts don't re-download five years of history."""

import hashlib
import logging
import pickle
import threading
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from research import backend

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def cached(key: str, ttl: int, fn):
    """Disk cache for expensive results (pickle). Concurrent callers wait for the first."""
    path = CACHE_DIR / f"{hashlib.md5(key.encode()).hexdigest()}.pkl"
    with _lock(key):
        if path.exists() and time.time() - path.stat().st_mtime < ttl:
            try:
                return pickle.loads(path.read_bytes())
            except Exception:
                path.unlink(missing_ok=True)
        value = fn()
        path.write_bytes(pickle.dumps(value))
        return value


def universe(market: str, n: int = 300) -> pd.DataFrame:
    """The n largest companies in a market (from the main API), with screener metrics."""

    def load():
        rows, page = [], 1
        while len(rows) < n:
            data = backend._get(f"/api/stocks/{market}/companies?sort=market_cap&desc=true&page_size=200&page={page}", ttl=3600, timeout=120)
            batch = data.get("rows", []) if isinstance(data, dict) else []
            rows.extend(batch)
            if len(batch) < 200:
                break
            page += 1
        return pd.DataFrame(rows[:n])

    return cached(f"universe:{market}:{n}", 6 * 3600, load)


def _download(symbols: list[str], **kw) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(symbols), 100):
        chunk = symbols[i : i + 100]
        try:
            raw = yf.download(chunk, group_by="ticker", auto_adjust=True, progress=False, threads=True, **kw)
        except Exception as exc:
            log.warning("download failed: %s", exc)
            continue
        for s in chunk:
            try:
                f = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            f = f.dropna(subset=["Close"])
            if len(f):
                out[s] = f
    return out


def daily(symbols: list[str], period: str = "5y") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide (date x symbol) adjusted closes and volumes."""
    key = f"daily:{period}:{','.join(sorted(symbols))}"

    def load():
        frames = _download(symbols, period=period, interval="1d")
        frames = {s: f for s, f in frames.items() if len(f) > 260}
        closes = pd.DataFrame({s: f["Close"] for s, f in frames.items()}).sort_index()
        volumes = pd.DataFrame({s: f["Volume"] for s, f in frames.items()}).reindex(closes.index)
        closes.index = pd.DatetimeIndex(closes.index).tz_localize(None)
        volumes.index = closes.index
        keep = closes.notna().mean(axis=1) >= 0.5  # days most of the market traded
        return closes[keep].ffill(limit=3), volumes[keep]

    return cached(key, 6 * 3600, load)


def intraday(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """~60 sessions of 5-minute OHLCV bars per symbol."""
    key = f"intraday:{','.join(sorted(symbols))}"

    def load():
        frames = _download(symbols, period="60d", interval="5m", prepost=False)
        return {s: f[f["Volume"] > 0] for s, f in frames.items() if len(f) > 300}

    return cached(key, 600, load)
