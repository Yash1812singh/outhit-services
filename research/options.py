"""F&O (futures & options) analysis and strategy selection.

Data: NSE option chains for Indian indices and F&O stocks; Yahoo Finance option
chains for US stocks and ETFs.

For each underlying we compute:
- ATM implied volatility (IV) vs. 20-day historical volatility (HV): are options
  cheap or expensive right now?
- Put-call ratio by open interest, max pain, and the strikes with the largest
  put OI (support) and call OI (resistance).
- The market's expected move to expiry: spot x IV x sqrt(days / 365).

Strategy selection: for the view (bullish / bearish / neutral) we build several
defined-risk candidates from the live chain (debit and credit spreads at
different strikes and widths; iron condors when neutral) and score each on
expected profit per unit of maximum loss, under Outhit's volatility forecast
(a blend of implied and recent realised volatility) plus a small drift from the
directional signal. Pricing a trade at the market's own implied volatility
would make every option look fair; the edge of selling expensive options (or
buying cheap ones) only shows up when realised volatility is forecast
separately. If no candidate has positive expected value, the answer is "no trade".
Every strategy has capped loss. Naked option selling is never suggested.
"""

import logging
import math
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd
import yfinance as yf

from research import data

log = logging.getLogger(__name__)

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain",
}
NSE_INDICES = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "NIFTY_FIN_SERVICE.NS", "MIDCPNIFTY": "NIFTY_MID_SELECT.NS"}
RATE = {"in": 0.065, "us": 0.043}
MIN_DAYS = 5
MIN_EV_PER_RISK = 0.05  # expected profit must be at least 5% of the maximum loss to be worth the model risk


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------

def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(spot: float, strike: float, t: float, r: float, vol: float, kind: str) -> float:
    if t <= 0 or vol <= 0:
        return max(0.0, (spot - strike) if kind == "call" else (strike - spot))
    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    if kind == "call":
        return spot * _ncdf(d1) - strike * math.exp(-r * t) * _ncdf(d2)
    return strike * math.exp(-r * t) * _ncdf(-d2) - spot * _ncdf(-d1)


def implied_vol(price: float, spot: float, strike: float, t: float, r: float, kind: str) -> float | None:
    if not price or price <= 0 or t <= 0:
        return None
    intrinsic = max(0.0, (spot - strike) if kind == "call" else (strike - spot))
    if price <= intrinsic:
        return None
    lo, hi = 0.01, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs_price(spot, strike, t, r, mid, kind) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------

def _nse(path: str) -> dict:
    with httpx.Client(headers=NSE_HEADERS, timeout=20, follow_redirects=True) as c:
        r = c.get(f"https://www.nseindia.com/api/{path}")
        r.raise_for_status()
        return r.json()


def nse_fo_stocks() -> list[str]:
    return data.cached("nse:fo-list", 24 * 3600, lambda: _nse("master-quote"))


def _mid(bid, ask, last) -> float | None:
    bid, ask, last = (float(x) if x else 0.0 for x in (bid, ask, last))
    if bid > 0 and ask > 0 and ask >= bid:
        return (bid + ask) / 2
    return last or None


def _expiry_days(exp: datetime) -> float:
    return (exp - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds() / 86400


def chain_india(symbol: str) -> tuple[float, datetime, pd.DataFrame]:
    def load():
        info = _nse(f"option-chain-contract-info?symbol={symbol}")
        exps = [datetime.strptime(e, "%d-%b-%Y").replace(hour=15, minute=30) for e in info["expiryDates"]]
        exp = next((e for e in exps if _expiry_days(e) >= MIN_DAYS), exps[0])
        kind = "Indices" if symbol in NSE_INDICES else "Equity"
        d = _nse(f"option-chain-v3?type={kind}&symbol={symbol}&expiry={exp.strftime('%d-%b-%Y')}")
        rows = []
        for row in d["records"]["data"]:
            ce, pe = row.get("CE") or {}, row.get("PE") or {}
            rows.append({
                "strike": float(row["strikePrice"]),
                "call_price": _mid(ce.get("buyPrice1"), ce.get("sellPrice1"), ce.get("lastPrice")),
                "put_price": _mid(pe.get("buyPrice1"), pe.get("sellPrice1"), pe.get("lastPrice")),
                "call_iv": (ce.get("impliedVolatility") or 0) / 100 or None,
                "put_iv": (pe.get("impliedVolatility") or 0) / 100 or None,
                "call_oi": float(ce.get("openInterest") or 0),
                "put_oi": float(pe.get("openInterest") or 0),
                "call_volume": float(ce.get("totalTradedVolume") or 0),
                "put_volume": float(pe.get("totalTradedVolume") or 0),
            })
        return float(d["records"]["underlyingValue"]), exp, pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)

    return data.cached(f"nse-chain:{symbol}", 600, load)


def chain_us(symbol: str) -> tuple[float, datetime, pd.DataFrame]:
    def load():
        t = yf.Ticker(symbol)
        exps = [datetime.strptime(e, "%Y-%m-%d").replace(hour=20) for e in t.options]
        exp = next((e for e in exps if _expiry_days(e) >= MIN_DAYS), None)
        if exp is None:
            raise RuntimeError(f"No listed options for {symbol}")
        ch = t.option_chain(exp.strftime("%Y-%m-%d"))
        spot = float(t.history(period="5d")["Close"].iloc[-1])

        def side(df, p):
            return pd.DataFrame({"strike": df["strike"], f"{p}_price": [_mid(b, a, l) for b, a, l in zip(df["bid"], df["ask"], df["lastPrice"])],
                                 f"{p}_iv": df["impliedVolatility"].where(df["impliedVolatility"] > 0.01), f"{p}_oi": df["openInterest"].fillna(0),
                                 f"{p}_volume": df["volume"].fillna(0)})

        merged = side(ch.calls, "call").merge(side(ch.puts, "put"), on="strike", how="outer")
        return spot, exp, merged.sort_values("strike").reset_index(drop=True)

    return data.cached(f"us-chain:{symbol}", 600, load)


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def _hv(symbol: str) -> float | None:
    closes, _ = data.daily([symbol], period="2y")
    if symbol not in closes:
        return None
    r = np.log(closes[symbol].dropna()).diff().dropna()
    return float(r.iloc[-20:].std() * math.sqrt(252)) if len(r) > 25 else None


def analytics(market: str, symbol: str, yahoo_symbol: str) -> dict:
    spot, exp, ch = chain_india(symbol) if market == "in" else chain_us(symbol)
    days = max(_expiry_days(exp), 0.5)
    t = days / 365
    r = RATE[market]
    ch = ch[(ch["strike"] > spot * 0.6) & (ch["strike"] < spot * 1.4)].copy()
    for col in ("call_oi", "put_oi", "call_volume", "put_volume"):
        ch[col] = ch[col].fillna(0)  # strikes listed on one side only
    # Fill missing IVs from prices so thinly-quoted strikes still count.
    for kind in ("call", "put"):
        ch[f"{kind}_iv"] = [iv if iv and iv > 0 else implied_vol(p, spot, k, t, r, kind) for iv, p, k in zip(ch[f"{kind}_iv"], ch[f"{kind}_price"], ch["strike"])]
    atm_i = (ch["strike"] - spot).abs().idxmin()
    near = ch.loc[(ch["strike"] - spot).abs().sort_values().index[:4]]
    atm_iv = float(pd.concat([near["call_iv"], near["put_iv"]]).dropna().median()) if len(near) else None
    hv = _hv(yahoo_symbol)
    pcr = float(ch["put_oi"].sum() / ch["call_oi"].sum()) if ch["call_oi"].sum() > 0 else None

    # Max pain: the expiry price at which option buyers collectively lose the most.
    strikes = ch["strike"].to_numpy()
    pain = [(np.maximum(0, s - strikes) * ch["call_oi"].to_numpy()).sum() + (np.maximum(0, strikes - s) * ch["put_oi"].to_numpy()).sum() for s in strikes]
    max_pain = float(strikes[int(np.argmin(pain))]) if ch["call_oi"].sum() + ch["put_oi"].sum() > 0 else None
    below, above = ch[ch["strike"] <= spot], ch[ch["strike"] >= spot]
    support = float(below.loc[below["put_oi"].idxmax(), "strike"]) if len(below) and below["put_oi"].max() > 0 else None
    resistance = float(above.loc[above["call_oi"].idxmax(), "strike"]) if len(above) and above["call_oi"].max() > 0 else None
    em = spot * atm_iv * math.sqrt(t) if atm_iv else None
    return {
        "symbol": symbol, "spot": spot, "expiry": exp.strftime("%Y-%m-%d"), "days_to_expiry": round(days, 1),
        "atm_strike": float(ch.loc[atm_i, "strike"]), "atm_iv": atm_iv, "hv_20": hv,
        "iv_hv_ratio": (atm_iv / hv) if atm_iv and hv else None,
        "pcr_oi": pcr, "max_pain": max_pain, "oi_support": support, "oi_resistance": resistance,
        "expected_move": em, "expected_range": [spot - em, spot + em] if em else None,
        "_chain": ch, "_t": t, "_r": r,
    }


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _nearest(ch: pd.DataFrame, target: float, kind: str) -> pd.Series | None:
    c = ch[ch[f"{kind}_price"].notna() & (ch[f"{kind}_price"] > 0)]
    if c.empty:
        return None
    return c.loc[(c["strike"] - target).abs().idxmin()]


def _payoff(legs: list[dict], prices: np.ndarray) -> np.ndarray:
    total = np.zeros_like(prices)
    for leg in legs:
        intrinsic = np.maximum(0, prices - leg["strike"]) if leg["type"] == "call" else np.maximum(0, leg["strike"] - prices)
        sign = 1 if leg["action"] == "buy" else -1
        total += sign * (intrinsic - leg["price"])
    return total


def _evaluate(legs: list[dict], spot: float, iv: float, t: float, forecast_vol: float | None = None, drift: float = 0.0) -> dict:
    """Payoff at expiry. Probabilities are computed twice: under the market's implied
    volatility (what option prices assume) and under Outhit's forecast (recent realised
    volatility plus a small directional drift). Positive expected value under the forecast
    is what makes a trade worth taking."""
    grid = np.linspace(spot * 0.4, spot * 1.6, 6001)
    pay = _payoff(legs, grid)

    def weights(vol: float, mu: float) -> np.ndarray:
        sig = max(vol, 0.01) * math.sqrt(t)
        z = (np.log(grid / spot) - (math.log1p(mu) - 0.5 * sig * sig)) / sig
        pdf = np.exp(-0.5 * z * z) / (grid * sig * math.sqrt(2 * math.pi))
        return pdf / pdf.sum()

    w_mkt = weights(iv, 0.0)
    w_fc = weights(forecast_vol or iv, drift)
    breakevens = [float(grid[i]) for i in range(1, len(grid)) if np.sign(pay[i]) != np.sign(pay[i - 1]) and pay[i] != 0]
    net = sum((1 if l["action"] == "sell" else -1) * l["price"] for l in legs)
    max_profit, max_loss = float(pay.max()), float(pay.min())
    ev = float((w_fc * pay).sum())
    return {
        "net_premium": float(net),  # + credit received, - debit paid (per unit)
        "max_profit": max_profit,
        "max_loss": max_loss,
        "reward_to_risk": float(max_profit / -max_loss) if max_loss < 0 else None,
        "breakevens": [round(b, 2) for b in breakevens],
        "prob_profit": float(w_fc[pay > 0].sum()),
        "prob_profit_market": float(w_mkt[pay > 0].sum()),
        "expected_value": ev,
        "ev_per_risk": float(ev / -max_loss) if max_loss < 0 else None,
    }


STRATEGY_TEXT = {
    "Bull call spread": "Buy a call near the price and sell a higher one to cut the cost. Profits if the price rises.",
    "Bull put spread": "Sell a put below the price and buy a lower one to cap the loss. Profits if the price stays above the short strike.",
    "Bear put spread": "Buy a put near the price and sell a lower one to cut the cost. Profits if the price falls.",
    "Bear call spread": "Sell a call above the price and buy a higher one to cap the loss. Profits if the price stays below the short strike.",
    "Iron condor": "Sell a put spread below and a call spread above the price. Profits if the price stays in the range.",
}


def choose_strategy(a: dict, direction: str, drift: float = 0.0, allow_credit: bool = True) -> dict:
    """Build candidate defined-risk strategies that fit the view, score each on expected
    profit per unit of risk under Outhit's volatility forecast, and pick the best one.
    If no candidate has positive expected value, recommend no trade."""
    ch, spot, t = a["_chain"], a["spot"], a["_t"]
    iv, em = a["atm_iv"], a["expected_move"]
    if not iv or not em:
        return {"name": "No trade", "reason": "Not enough option prices to judge volatility."}
    hv = a.get("hv_20")
    forecast_vol = 0.5 * iv + 0.5 * hv if hv else iv  # realised vol tends to revert toward implied; blend the two
    ratio = a["iv_hv_ratio"] or 1.0
    regime = "expensive" if ratio > 1.15 else "cheap" if ratio <= 1.0 else "fair"

    def leg(action, kind, target):
        row = _nearest(ch, target, kind)
        return None if row is None else {"action": action, "type": kind, "strike": float(row["strike"]), "price": float(row[f"{kind}_price"])}

    cands: list[tuple[str, list]] = []
    for k in (0.5, 1.0):
        for wd in (0.5, 1.0):
            width = max(em * wd, spot * 0.01)
            if direction == "bullish":
                cands.append(("Bull call spread", [leg("buy", "call", spot), leg("sell", "call", spot + k * em)]))
                cands.append(("Bull put spread", [leg("sell", "put", spot - k * em + em * 0.25), leg("buy", "put", spot - k * em + em * 0.25 - width)]))
            elif direction == "bearish":
                cands.append(("Bear put spread", [leg("buy", "put", spot), leg("sell", "put", spot - k * em)]))
                cands.append(("Bear call spread", [leg("sell", "call", spot + k * em - em * 0.25), leg("buy", "call", spot + k * em - em * 0.25 + width)]))
            else:
                cands.append(("Iron condor", [leg("sell", "put", spot - k * em), leg("buy", "put", spot - k * em - width),
                                              leg("sell", "call", spot + k * em), leg("buy", "call", spot + k * em + width)]))
    scored = []
    seen = set()
    if not allow_credit:  # risk-off regime: don't sell options into a volatility spike
        cands = [(n, legs) for n, legs in cands if n in ("Bull call spread", "Bear put spread")]
    for name, legs in cands:
        if any(l is None for l in legs):
            continue
        key = (name, tuple((l["type"], l["strike"], l["action"]) for l in legs))
        if key in seen or len({(l["type"], l["strike"]) for l in legs}) < len(legs):
            continue
        seen.add(key)
        ev = _evaluate(legs, spot, iv, t, forecast_vol, drift)
        if ev["max_loss"] >= 0 or ev["max_profit"] <= 0:
            continue
        scored.append((ev["ev_per_risk"], name, legs, ev))
    if not scored:
        return {"name": "No trade", "reason": "The option chain is too thin at the needed strikes.", "volatility_regime": regime}
    best = max(scored, key=lambda x: x[0])
    _, name, legs, ev = best
    if (ev["ev_per_risk"] or 0) < MIN_EV_PER_RISK:
        return {"name": "No trade", "volatility_regime": regime, "candidates_checked": len(scored),
                "reason": f"None of the {len(scored)} {direction} strategies checked has a meaningful positive expected value at Outhit's volatility forecast ({forecast_vol:.0%} a year vs {iv:.0%} implied); the best was {ev['ev_per_risk']:+.2f} per unit of risk, too thin to beat model error. Options here are priced fairly or against you."}
    why = f"{STRATEGY_TEXT[name]} Chosen from {len(scored)} candidates as the best expected profit per unit of risk. Options are {regime} (implied {iv:.0%} vs recent {hv:.0%})." if hv else f"{STRATEGY_TEXT[name]} Chosen from {len(scored)} candidates."
    return {"name": name, "legs": legs, "reason": why, "volatility_regime": regime, "forecast_vol": forecast_vol, **ev}


def public(a: dict) -> dict:
    return {k: v for k, v in a.items() if not k.startswith("_")}
