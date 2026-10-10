"""Market regime, for momentum-crash protection and exposure guidance.

Momentum's worst losses come right after market crashes, when beaten-down
stocks rebound sharply (Daniel & Moskowitz 2016), and its risk is predictable
from its own recent volatility (Barroso & Santa-Clara 2015). We use two simple,
transparent indicators on the market index:

  trend        index above / below its 200-day average
  volatility   last month's realised volatility vs. its one-year average

  risk-on   uptrend and normal volatility   full weight on momentum; full equity exposure
  caution   downtrend OR volatility spike   momentum weight x0.6; exposure 75%
  risk-off  downtrend AND volatility spike  momentum weight x0.3; exposure 50%
"""

import math

import numpy as np

from research import backend, data

FALLBACK_INDEX = {"in": "^NSEI", "us": "^GSPC"}
SETTINGS = {
    "risk-on": {"momentum_scale": 1.0, "equity_exposure": 1.0},
    "caution": {"momentum_scale": 0.6, "equity_exposure": 0.75},
    "risk-off": {"momentum_scale": 0.3, "equity_exposure": 0.5},
}


def _index_symbol(market: str) -> str | None:
    rows = backend._get("/api/stocks/markets", ttl=24 * 3600) or []
    for m in rows:
        if m.get("code") == market and m.get("index"):
            return m["index"]
    return FALLBACK_INDEX.get(market)


def detect(market: str) -> dict:
    sym = _index_symbol(market)
    if not sym:
        return {"regime": "unknown", **SETTINGS["caution"], "explanation": "No market index available for this country."}
    closes, _ = data.daily([sym], period="3y")
    if sym not in closes:
        return {"regime": "unknown", **SETTINGS["caution"], "explanation": "Index data unavailable."}
    c = closes[sym].dropna()
    sma200 = c.rolling(200).mean().iloc[-1]
    r = np.log(c).diff()
    vol_1m = float(r.iloc[-21:].std() * math.sqrt(252))
    vol_1y = float(r.iloc[-252:].std() * math.sqrt(252))
    up = bool(c.iloc[-1] > sma200)
    spike = vol_1m > 1.3 * vol_1y
    regime = "risk-on" if up and not spike else "risk-off" if (not up and spike) else "caution"
    text = {
        "risk-on": "The market is in an uptrend with normal volatility. Momentum-style picks get full weight.",
        "caution": "The market is below its 200-day average or volatility has jumped. Momentum crashes cluster in these periods, so momentum gets less weight and quality more, and new money is better invested gradually.",
        "risk-off": "The market is in a downtrend with a volatility spike: historically the most dangerous time for momentum strategies and leveraged trades. Momentum weight is cut sharply; keep more in cash, short-term debt or gold, and avoid selling options.",
    }[regime]
    return {
        "regime": regime, "index": sym, "price": float(c.iloc[-1]), "sma200": float(sma200), "above_200d": up,
        "distance_200d": float(c.iloc[-1] / sma200 - 1), "vol_1m": vol_1m, "vol_1y": vol_1y,
        "return_6m": float(c.iloc[-1] / c.iloc[-127] - 1) if len(c) > 127 else None,
        **SETTINGS[regime], "explanation": text,
    }
