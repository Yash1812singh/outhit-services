"""Minute-by-minute prices.

Every `price_interval_seconds` we pull 1-minute bars for every tracked symbol
(the always-tracked companies plus anything a browser is currently showing)
and keep the latest quote and today's intraday path in memory. Browsers get
updates pushed over Server-Sent Events (see main.py).

Exchange data on Yahoo is delayed (typically 15 min for stocks, near real time
for crypto and FX). `delayed_minutes` in each quote says how old the last bar is.
"""

import logging
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

SUBSCRIPTION_TTL = 600  # a symbol stays live for 10 min after a browser last asked for it
BATCH = 100

_quotes: dict[str, dict] = {}
_subscribed: dict[str, float] = {}
_pinned: set[str] = set()
_lock = threading.Lock()
version = 0  # bumps after every poll; stream clients compare against it


def pin(symbols: list[str]) -> None:
    with _lock:
        _pinned.update(symbols)


def subscribe(symbols: list[str]) -> None:
    t = time.time()
    with _lock:
        for s in symbols:
            _subscribed[s] = t


def tracked() -> list[str]:
    cutoff = time.time() - SUBSCRIPTION_TTL
    with _lock:
        live = {s for s, t in _subscribed.items() if t >= cutoff}
        return sorted(_pinned | live)


def get(symbols: list[str]) -> dict[str, dict]:
    with _lock:
        return {s: _quotes[s] for s in symbols if s in _quotes}


def _quote(symbol: str, f: pd.DataFrame) -> dict | None:
    f = f.dropna(subset=["Close"])
    if f.empty:
        return None
    # Split sessions by trading gaps, not calendar dates: a batch that mixes exchanges comes
    # back in one shared timezone, where a US session can straddle midnight.
    gaps = f.index.to_series().diff() > pd.Timedelta(hours=2)
    if gaps.any():
        start = gaps[gaps].index[-1]
    else:  # 24/7 markets (crypto): use the UTC calendar day
        utc_days = f.index.tz_convert("UTC").normalize()
        start = f.index[utc_days == utc_days[-1]][0]
    today = f[f.index >= start]
    prev = f[f.index < start]
    last = float(today["Close"].iloc[-1])
    prev_close = float(prev["Close"].iloc[-1]) if not prev.empty else float(today["Open"].iloc[0])
    vol = today["Volume"].fillna(0)
    tp = (today["High"] + today["Low"] + today["Close"]) / 3
    vwap = float((tp * vol).sum() / vol.sum()) if vol.sum() > 0 else None
    ts = today.index[-1]
    age_min = (datetime.now(timezone.utc) - ts.tz_convert("UTC").to_pydatetime()).total_seconds() / 60
    path = today["Close"].iloc[:: max(1, len(today) // 120)]  # ~120 points for the chart
    return {
        "symbol": symbol,
        "price": last,
        "prev_close": prev_close,
        "change": last - prev_close,
        "change_pct": (last / prev_close - 1) * 100 if prev_close else None,
        "open": float(today["Open"].iloc[0]),
        "day_high": float(today["High"].max()),
        "day_low": float(today["Low"].min()),
        "volume": float(vol.sum()),
        "vwap": vwap,
        "last_bar": ts.isoformat(),
        "delayed_minutes": round(age_min),
        "trading": age_min < 30,  # a bar within the last half hour means the market is (probably) open
        "intraday": [[t.isoformat(), round(float(v), 6)] for t, v in path.items()],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def poll(symbols: list[str] | None = None) -> int:
    global version
    symbols = symbols if symbols is not None else tracked()
    updated = 0
    for i in range(0, len(symbols), BATCH):
        chunk = symbols[i : i + BATCH]
        try:
            raw = yf.download(chunk, period="2d", interval="1m", group_by="ticker", auto_adjust=False, progress=False, threads=True)
        except Exception as exc:
            log.warning("live price download failed: %s", exc)
            continue
        for s in chunk:
            try:
                f = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            q = _quote(s, f)
            if q is None:
                continue
            with _lock:
                old = _quotes.get(s)
                q["tick"] = 0 if old is None else int(np.sign(q["price"] - old["price"]))
                _quotes[s] = q
            updated += 1
    with _lock:
        version += 1
    return updated
