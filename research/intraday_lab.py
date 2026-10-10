"""Intraday strategy lab: five well-known day-trading strategies compete on ~60
sessions of 5-minute bars for a market's most-traded stocks, after costs.

  orb           Opening-range breakout: break of the first 30 minutes' range, on the
                VWAP side, with above-normal volume. Stop at the other side; 2R target.
  vwap_revert   VWAP mean reversion: after the first hour, fade a stretched move
                away from VWAP when RSI is extreme; target VWAP.
  gap_and_go    A gap of >1% that holds in the first 15 minutes; ride it.
  gap_fade      A gap of >1.5% that fails within 30 minutes; trade back toward
                yesterday's close.
  late_momentum Intraday momentum (Gao, Han, Li & Zhou 2018): the first half hour's
                return predicts the last half hour's; trade the last 30 minutes.

The strategy is chosen on the FIRST half of the sessions and judged on the
SECOND half. If even the chosen strategy loses money after costs there, the
verdict is "no edge": the most useful intraday advice is often not to trade.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research import data, jobs

log = logging.getLogger(__name__)

COSTS = {"in": 0.001, "us": 0.0005}  # round-trip: brokerage, taxes, slippage (approx.)
UNIVERSE = 40
TTL = 600
OR_BARS = 6

STRATEGIES = {
    "orb": "Opening-range breakout",
    "vwap_revert": "VWAP mean reversion",
    "gap_and_go": "Gap and go",
    "gap_fade": "Gap fade",
    "late_momentum": "Last-half-hour momentum",
}


def _vwap(s: pd.DataFrame) -> pd.Series:
    tp = (s["High"] + s["Low"] + s["Close"]) / 3
    return (tp * s["Volume"]).cumsum() / s["Volume"].cumsum()


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _walk(s: pd.DataFrame, i: int, side: int, entry: float, stop: float, target: float | None) -> tuple[float, str, int | None]:
    """Follow bars after index i until stop, target or the close. Returns (exit price, reason, exit index or None if still open)."""
    for j in range(i + 1, len(s)):
        b = s.iloc[j]
        if (side == 1 and b["Low"] <= stop) or (side == -1 and b["High"] >= stop):
            return stop, "stop", j
        if target is not None and ((side == 1 and b["High"] >= target) or (side == -1 and b["Low"] <= target)):
            return target, "target", j
    return float(s["Close"].iloc[-1]), "close", None


def _trade(name: str, s: pd.DataFrame, prev_close: float, rel_vol: float | None) -> dict | None:
    """Run one strategy on one session. Returns the trade (possibly still open) or None."""
    if len(s) < OR_BARS + 3:
        return None
    o = float(s["Open"].iloc[0])
    gap = o / prev_close - 1
    vwap = _vwap(s)

    def make(i, side, stop, target, why):
        entry = float(s["Close"].iloc[i])
        risk = abs(entry - stop)
        if risk <= 0:
            return None
        if target is None and name not in ("late_momentum",):
            target = entry + side * 2 * risk
        exit_px, reason, j = _walk(s, i, side, entry, stop, target)
        return {"strategy": name, "side": side, "entry_index": i, "entry_time": s.index[i].isoformat(), "entry": entry, "stop": float(stop),
                "target": None if target is None else float(target), "exit": exit_px, "exit_reason": reason, "closed": j is not None,
                "gross": side * (exit_px - entry) / entry, "why": why}

    if name == "orb":
        hi, lo = s["High"].iloc[:OR_BARS].max(), s["Low"].iloc[:OR_BARS].min()
        if rel_vol is not None and rel_vol < 1.2:
            return None
        for i in range(OR_BARS, len(s) - 1):
            c = s["Close"].iloc[i]
            if c > hi and c > vwap.iloc[i]:
                return make(i, 1, lo, None, f"Broke above the opening range ({hi:.2f}) on {rel_vol or 0:.1f}x volume")
            if c < lo and c < vwap.iloc[i]:
                return make(i, -1, hi, None, f"Broke below the opening range ({lo:.2f}) on {rel_vol or 0:.1f}x volume")
    elif name == "vwap_revert":
        dev = s["Close"] / vwap - 1
        rsi = _rsi(s["Close"])
        for i in range(12, len(s) - 6):
            thr = max(0.004, 1.5 * dev.iloc[: i + 1].std())
            d = dev.iloc[i]
            if d < -thr and rsi.iloc[i] < 30:
                e = s["Close"].iloc[i]
                return make(i, 1, e * (1 - thr), float(vwap.iloc[i]), f"Stretched {d:.1%} below VWAP with RSI {rsi.iloc[i]:.0f}")
            if d > thr and rsi.iloc[i] > 70:
                e = s["Close"].iloc[i]
                return make(i, -1, e * (1 + thr), float(vwap.iloc[i]), f"Stretched {d:+.1%} above VWAP with RSI {rsi.iloc[i]:.0f}")
    elif name == "gap_and_go":
        if abs(gap) > 0.01:
            i = 2
            first = s.iloc[: i + 1]
            side = 1 if gap > 0 else -1
            if side * (first["Close"].iloc[-1] - o) > 0:
                stop = first["Low"].min() if side == 1 else first["High"].max()
                return make(i, side, stop, None, f"Gapped {gap:+.1%} and held it for 15 minutes")
    elif name == "gap_fade":
        if abs(gap) > 0.015:
            i = OR_BARS - 1
            side = -1 if gap > 0 else 1
            c = s["Close"].iloc[i]
            if side * (c - o) > 0:  # gap already failing
                stop = s["High"].iloc[: i + 1].max() if side == -1 else s["Low"].iloc[: i + 1].min()
                return make(i, side, stop, prev_close, f"Gapped {gap:+.1%} but faded in the first 30 minutes")
    elif name == "late_momentum":
        first = s["Close"].iloc[OR_BARS - 1] / prev_close - 1
        i = len(s) - 7
        if abs(first) > 0.003 and i > OR_BARS:
            side = 1 if first > 0 else -1
            e = s["Close"].iloc[i]
            return make(i, side, e * (1 - side * 0.01), None, f"First half hour {first:+.1%}; trading the last 30 minutes in that direction")
    return None


def _sessions(f: pd.DataFrame) -> list[pd.DataFrame]:
    gaps = f.index.to_series().diff() > pd.Timedelta(hours=2)
    ids = gaps.cumsum()
    return [g for _, g in f.groupby(ids.to_numpy()) if len(g) > OR_BARS + 3]


def _stats(rets: list[float]) -> dict:
    r = np.array(rets)
    if not len(r):
        return {"trades": 0, "win_rate": None, "avg_return": None, "total_return": None, "profit_factor": None}
    wins, losses = r[r > 0], r[r <= 0]
    return {
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "avg_return": float(r.mean()),
        "total_return": float(r.sum()),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else None,
    }


def compute(market: str, progress=lambda m: None) -> dict:
    uni = data.universe(market, 300)
    symbols = list(uni["symbol"].head(UNIVERSE))
    meta = uni.set_index("symbol")
    progress(f"Downloading 60 days of 5-minute bars for {len(symbols)} stocks…")
    bars = data.intraday(symbols)
    cost = COSTS.get(market, 0.001)

    trades: dict[str, list] = {k: [] for k in STRATEGIES}
    today_signals = []
    session_dates: set = set()
    progress("Backtesting five strategies…")
    for sym, f in bars.items():
        sessions = _sessions(f)
        if len(sessions) < 20:
            continue
        for k, s in enumerate(sessions):
            if k == 0:
                continue
            prev = float(sessions[k - 1]["Close"].iloc[-1])
            n = len(s)
            hist = [h["Volume"].iloc[:min(n, OR_BARS)].sum() for h in sessions[max(0, k - 20) : k] if len(h) >= OR_BARS]
            rv = float(s["Volume"].iloc[:OR_BARS].sum() / np.mean(hist)) if hist else None
            day = s.index[0].date()
            session_dates.add(day)
            last = k == len(sessions) - 1
            for name in STRATEGIES:
                t = _trade(name, s, prev, rv)
                if not t:
                    continue
                t["net"] = t["gross"] - cost
                if last:
                    today_signals.append({"symbol": sym, "name": meta.at[sym, "name"] if sym in meta.index else sym,
                                          "currency": meta.at[sym, "currency"] if sym in meta.index else None, "session": str(day), **t})
                else:
                    trades[name].append((day, t["net"]))

    days = sorted(session_dates)
    if len(days) < 20:
        raise RuntimeError("Not enough intraday history.")
    split = days[len(days) // 2]
    table = []
    for name, ts in trades.items():
        sel = [r for d, r in ts if d < split]
        hold = [r for d, r in ts if d >= split]
        table.append({"strategy": name, "label": STRATEGIES[name], "selection": _stats(sel), "holdout": _stats(hold), "all": _stats(sel + hold)})

    eligible = [t for t in table if t["selection"]["trades"] >= 20]
    chosen = max(eligible, key=lambda t: t["selection"]["avg_return"]) if eligible else None
    hold = chosen["holdout"] if chosen else {}
    if chosen and chosen["selection"]["avg_return"] > 0 and (hold.get("avg_return") or -1) > 0.0002 and (hold.get("profit_factor") or 0) > 1.1:
        level = "edge"
        text = f"{chosen['label']} made money after costs both when it was chosen and on the later sessions it wasn't chosen on. The edge per trade is small; size positions so a stop-loss costs at most ~1% of your capital."
    elif chosen and chosen["selection"]["avg_return"] > 0 and (hold.get("avg_return") or -1) > 0:
        level = "marginal"
        text = f"{chosen['label']} roughly broke even after costs on the later sessions. That is not enough to be worth the risk; not trading is the evidence-based choice."
    elif chosen and chosen["selection"]["avg_return"] > 0:
        level = "none"
        text = f"{chosen['label']} looked best on the earlier sessions but lost money after costs on the later ones. There is no reliable intraday edge right now: not trading is the evidence-based choice."
    else:
        level = "none"
        text = "No strategy made money after costs on the earlier sessions. There is no reliable intraday edge right now: not trading is the evidence-based choice."

    last_bar = max((f.index[-1] for f in bars.values()), default=None)
    live = last_bar is not None and (datetime.now(timezone.utc) - last_bar.tz_convert("UTC").to_pydatetime()).total_seconds() < 40 * 60
    chosen_name = chosen["strategy"] if chosen else None
    signals = sorted(today_signals, key=lambda t: (t["strategy"] != chosen_name, not (not t["closed"] and live), t["entry_time"]))
    return {
        "status": "ready",
        "market": market,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "sessions": len(days),
        "stocks": len(bars),
        "cost_round_trip": cost,
        "selection_period": [str(days[0]), str(split)],
        "holdout_period": [str(split), str(days[-1])],
        "strategies": table,
        "chosen": chosen_name,
        "verdict_level": level,
        "verdict": text,
        "market_open": live,
        "today": signals[:40],
        "warning": "Intraday trading is high-risk; SEBI found ~7 in 10 individual intraday traders in India lost money. Prices are delayed ~15 minutes. Always use a stop-loss and never risk more than you can afford to lose.",
    }


def get(market: str) -> dict:
    return jobs.run(f"intraday:{market}", lambda p: compute(market, p), TTL)
