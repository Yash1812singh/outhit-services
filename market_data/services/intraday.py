"""Intraday scanner: opening-range breakout (ORB) with VWAP and relative-volume filters.

For the most-traded stocks in a market we pull 5-minute bars for the last ~60
sessions, look for today's setups, and backtest the exact same rule on every
earlier session (after trading costs), so the user sees how this style of
trading has actually done recently, not just today's signals.

Intraday trading is where most retail traders lose money (SEBI found ~7 in 10
Indian intraday traders made a loss in FY23). The API returns that context
with every response.
"""

import logging
import threading
import time

import numpy as np
import pandas as pd
import yfinance as yf

from market_data.services import ranking

log = logging.getLogger(__name__)

OPENING_RANGE_BARS = 6  # 30 minutes of 5-minute bars
MIN_RVOL = 1.2
COST_ROUND_TRIP = 0.001  # 0.05% per side: brokerage + taxes + slippage, roughly
UNIVERSE = 40
CACHE_TTL = 300

_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()

WARNING = (
    "Intraday trading is high-risk: SEBI's study found about 7 in 10 individual intraday traders in India lost money. "
    "Data is delayed, so prices may have moved. Never trade money you can't afford to lose; always use a stop-loss."
)


def _download(symbols: list[str]) -> dict[str, pd.DataFrame]:
    raw = yf.download(symbols, period="60d", interval="5m", group_by="ticker", auto_adjust=False, progress=False, threads=True)
    out = {}
    for s in symbols:
        try:
            f = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        f = f[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        f = f[f["Volume"] > 0]
        if len(f) > 200:
            out[s] = f
    return out


def _sessions(f: pd.DataFrame) -> list[pd.DataFrame]:
    return [g for _, g in f.groupby(f.index.date) if len(g) > OPENING_RANGE_BARS + 2]


def _vwap(s: pd.DataFrame) -> pd.Series:
    tp = (s["High"] + s["Low"] + s["Close"]) / 3
    return (tp * s["Volume"]).cumsum() / s["Volume"].cumsum()


def _simulate(s: pd.DataFrame) -> float | None:
    """One ORB trade per session: first close beyond the opening range on the VWAP side,
    stop at the other side of the range, target 2R, otherwise exit at the close."""
    orng = s.iloc[:OPENING_RANGE_BARS]
    hi, lo = orng["High"].max(), orng["Low"].min()
    vwap = _vwap(s)
    rest = s.iloc[OPENING_RANGE_BARS:]
    for i, (ts, bar) in enumerate(rest.iterrows()):
        side = 1 if (bar["Close"] > hi and bar["Close"] > vwap[ts]) else -1 if (bar["Close"] < lo and bar["Close"] < vwap[ts]) else 0
        if side == 0:
            continue
        entry = bar["Close"]
        stop = lo if side == 1 else hi
        risk = abs(entry - stop)
        if risk <= 0:
            return None
        target = entry + side * 2 * risk
        for _, nb in rest.iloc[i + 1 :].iterrows():
            if (side == 1 and nb["Low"] <= stop) or (side == -1 and nb["High"] >= stop):
                return side * (stop - entry) / entry - COST_ROUND_TRIP
            if (side == 1 and nb["High"] >= target) or (side == -1 and nb["Low"] <= target):
                return side * (target - entry) / entry - COST_ROUND_TRIP
        return side * (rest["Close"].iloc[-1] - entry) / entry - COST_ROUND_TRIP
    return None


def _today(sessions: list[pd.DataFrame]) -> dict | None:
    s = sessions[-1]
    prev = sessions[-2] if len(sessions) > 1 else None
    orng = s.iloc[:OPENING_RANGE_BARS]
    hi, lo = float(orng["High"].max()), float(orng["Low"].min())
    vwap = float(_vwap(s).iloc[-1])
    last = float(s["Close"].iloc[-1])

    # Relative volume: today's cumulative volume vs the average at the same point in the last 20 sessions.
    k = len(s)
    history = [h["Volume"].iloc[:k].sum() for h in sessions[-21:-1] if len(h) >= k]
    rvol = float(s["Volume"].sum() / np.mean(history)) if history else None

    day_ranges = [(h["High"].max() - h["Low"].min()) / h["Close"].iloc[-1] for h in sessions[-21:-1]]
    avg_range = float(np.mean(day_ranges)) if day_ranges else None

    setup, side = "No setup", 0
    if len(s) > OPENING_RANGE_BARS and rvol and rvol >= MIN_RVOL:
        if last > hi and last > vwap:
            setup, side = "Long: opening-range breakout", 1
        elif last < lo and last < vwap:
            setup, side = "Short: opening-range breakdown", -1
    if side == 0 and len(s) > OPENING_RANGE_BARS:
        if last > hi:
            setup = "Above range, but weak volume (no trade)"
        elif last < lo:
            setup = "Below range, but weak volume (no trade)"
        elif last > vwap and (hi - last) / last < 0.003:
            setup = "Watch: near breakout"
        elif last < vwap and (last - lo) / last < 0.003:
            setup = "Watch: near breakdown"

    stop = target = None
    if side:
        stop = lo if side == 1 else hi
        target = last + side * 2 * abs(last - stop)

    strength = 0.0
    if rvol and avg_range:
        strength = 50 * min(rvol, 3) / 3 + 50 * min(abs(last - vwap) / last / avg_range, 1)

    return {
        "session_date": str(s.index[-1].date()),
        "last_bar_time": s.index[-1].isoformat(),
        "price": last,
        "open": float(s["Open"].iloc[0]),
        "gap_pct": None if prev is None else float(s["Open"].iloc[0] / prev["Close"].iloc[-1] - 1) * 100,
        "change_from_open_pct": float(last / s["Open"].iloc[0] - 1) * 100,
        "vwap": vwap,
        "opening_range": [lo, hi],
        "relative_volume": rvol,
        "avg_daily_range_pct": None if avg_range is None else avg_range * 100,
        "setup": setup,
        "side": side,
        "entry": last if side else None,
        "stop_loss": stop,
        "target": target,
        "strength": round(strength, 1),
        "sparkline": [round(float(x), 4) for x in s["Close"].tolist()],
    }


def scan(code: str) -> dict:
    with _lock:
        hit = _cache.get(code)
        if hit and time.time() - hit[0] < CACHE_TTL:
            return hit[1]

    df, st = ranking.market_frame(code)
    liquid = df[df["liquid"]].sort_values("turnover", ascending=False).head(UNIVERSE)
    meta = {r["symbol"]: r for _, r in liquid.iterrows()}
    bars = _download(list(meta))

    rows, all_trades = [], []
    for sym, f in bars.items():
        sessions = _sessions(f)
        if len(sessions) < 10:
            continue
        today = _today(sessions)
        trades = [t for t in (_simulate(s) for s in sessions[:-1]) if t is not None]
        all_trades.extend(trades)
        r = meta[sym]
        rows.append({
            "symbol": sym,
            "name": r["name"],
            "sector": r["sector"],
            "currency": r["currency"],
            **today,
            "backtest": {
                "trades": len(trades),
                "win_rate": float(np.mean([t > 0 for t in trades])) if trades else None,
                "avg_return_pct": float(np.mean(trades) * 100) if trades else None,
            },
        })

    active = sorted([x for x in rows if x["side"] != 0], key=lambda x: x["strength"], reverse=True)
    watch = sorted([x for x in rows if x["side"] == 0 and x["setup"].startswith("Watch")], key=lambda x: x["strength"], reverse=True)
    movers = sorted(rows, key=lambda x: abs(x["change_from_open_pct"]), reverse=True)[:10]

    t = np.array(all_trades)
    wins, losses = t[t > 0], t[t <= 0]
    summary = {
        "sessions": int(max((x["backtest"]["trades"] for x in rows), default=0)),
        "trades": int(len(t)),
        "win_rate": float((t > 0).mean()) if len(t) else None,
        "avg_return_pct": float(t.mean() * 100) if len(t) else None,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else None,
        "cost_round_trip_pct": COST_ROUND_TRIP * 100,
    }
    if summary["avg_return_pct"] is not None:
        summary["verdict"] = (
            "This strategy made money on average over the test period after costs, but with many losing trades along the way."
            if summary["avg_return_pct"] > 0
            else "This strategy LOST money on average over the test period after costs. Treat today's setups with extra caution."
        )

    result = {
        "market": code,
        "market_state": next((r.get("market_state") for r in meta.values() if r.get("market_state")), None),
        "as_of": st.refreshed_at.isoformat() if st else None,
        "universe_size": len(rows),
        "method": "Opening range = first 30 minutes. Long when a 5-min bar closes above the range high AND above VWAP with relative volume >= 1.2 (short = mirror image). Stop at the other side of the range, target 2x the risk, otherwise exit at the close.",
        "warning": WARNING,
        "backtest": summary,
        "setups": active,
        "watchlist": watch[:10],
        "movers": movers,
    }
    with _lock:
        _cache[code] = (time.time(), result)
    return result
