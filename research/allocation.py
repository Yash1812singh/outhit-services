"""ETFs, indices, sectors, commodities and crypto: trend-following and dual momentum.

For whole asset classes the best-documented approach is not stock-picking but
time-series momentum / trend-following (Faber 2007; Moskowitz, Ooi & Pedersen
2012) and dual momentum (Antonacci 2014):

  Trend filter   Hold the asset while its price is above its 200-day average,
                 otherwise sit in cash. Checked daily, acted on the next day.
  Dual momentum  Within each group, rank assets by 12-month return (skipping the
                 latest month) and pick the strongest ones whose return is also
                 positive (absolute momentum).

Both rules are fixed in advance (nothing is fitted), and each is backtested over
up to 10 years against simply buying and holding. Their main benefit is
historically smaller crashes, not higher returns.
"""

import logging
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research import data, jobs

log = logging.getLogger(__name__)
TTL = 6 * 3600

GROUPS = {
    "India ETFs": {
        "NIFTYBEES.NS": "Nifty 50 ETF", "JUNIORBEES.NS": "Nifty Next 50 ETF", "BANKBEES.NS": "Bank Nifty ETF", "ITBEES.NS": "Nifty IT ETF",
        "PSUBNKBEES.NS": "PSU Bank ETF", "PHARMABEES.NS": "Nifty Pharma ETF", "CPSEETF.NS": "CPSE ETF", "MON100.NS": "Nasdaq 100 ETF (INR)",
        "GOLDBEES.NS": "Gold ETF", "SILVERBEES.NS": "Silver ETF",
    },
    "India sectors": {
        "^NSEBANK": "Bank", "^CNXIT": "IT", "^CNXPHARMA": "Pharma", "^CNXAUTO": "Auto", "^CNXFMCG": "FMCG",
        "^CNXMETAL": "Metal", "^CNXREALTY": "Realty", "^CNXENERGY": "Energy", "^CNXINFRA": "Infrastructure", "^CNXPSUBANK": "PSU Bank",
    },
    "US ETFs": {
        "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "US small caps", "EFA": "Developed markets ex-US", "EEM": "Emerging markets",
        "VNQ": "US real estate", "TLT": "20+ year Treasuries", "IEF": "7-10 year Treasuries", "LQD": "Investment-grade bonds", "HYG": "High-yield bonds",
        "GLD": "Gold", "SLV": "Silver", "DBC": "Commodities basket",
    },
    "US sectors": {
        "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health care", "XLI": "Industrials", "XLY": "Consumer discretionary",
        "XLP": "Consumer staples", "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real estate", "XLC": "Communication services", "SMH": "Semiconductors",
    },
    "World indices": {
        "^NSEI": "Nifty 50", "^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^FTSE": "FTSE 100", "^GDAXI": "DAX", "^N225": "Nikkei 225",
        "^HSI": "Hang Seng", "000001.SS": "Shanghai", "^BVSP": "Bovespa", "^AXJO": "ASX 200",
    },
    "Commodities": {"GC=F": "Gold", "SI=F": "Silver", "CL=F": "Crude oil (WTI)", "NG=F": "Natural gas", "HG=F": "Copper"},
    "Crypto": {
        "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana", "BNB-USD": "BNB", "XRP-USD": "XRP", "ADA-USD": "Cardano",
        "DOGE-USD": "Dogecoin", "AVAX-USD": "Avalanche", "LINK-USD": "Chainlink", "DOT-USD": "Polkadot", "LTC-USD": "Litecoin",
        "TRX-USD": "Tron", "BCH-USD": "Bitcoin Cash", "XLM-USD": "Stellar", "ATOM-USD": "Cosmos",
    },
}
PICKS_PER_GROUP = 3


def _backtest(close: pd.Series, ppy: int) -> dict:
    r = close.pct_change().fillna(0)
    sma = close.rolling(200, min_periods=200).mean()
    invested = (close > sma).shift(1).fillna(False).astype(float)  # act the day after the signal
    strat = r * invested

    def summary(x: pd.Series) -> dict:
        eq = (1 + x).cumprod()
        years = len(x) / ppy
        cagr = float(eq.iloc[-1] ** (1 / years) - 1) if years > 0.5 else None
        vol = float(x.std() * math.sqrt(ppy))
        return {"cagr": cagr, "volatility": vol, "sharpe": (cagr / vol) if cagr is not None and vol else None,
                "max_drawdown": float((eq / eq.cummax() - 1).min())}

    valid = sma.notna()
    return {
        "years": round(valid.sum() / ppy, 1),
        "trend_rule": summary(strat[valid]),
        "buy_and_hold": summary(r[valid]),
        "time_invested": float(invested[valid].mean()),
        "switches_per_year": float(invested[valid].diff().abs().sum() / max(0.5, valid.sum() / ppy)),
    }


def compute(progress=lambda m: None) -> dict:
    symbols = [s for g in GROUPS.values() for s in g]
    progress(f"Downloading up to 10 years of prices for {len(symbols)} assets…")
    closes, _ = data.daily(symbols, period="10y")
    out = {"status": "ready", "computed_at": datetime.now(timezone.utc).isoformat(), "groups": []}
    for gname, members in GROUPS.items():
        rows = []
        for sym, label in members.items():
            if sym not in closes:
                continue
            c = closes[sym].dropna()
            if len(c) < 260:
                continue
            ppy = 365 if sym.endswith("-USD") else 252
            sma200 = c.rolling(200).mean().iloc[-1]
            mom = c.iloc[-22] / c.iloc[-min(len(c), 253)] - 1 if len(c) > 253 else None
            rows.append({
                "symbol": sym, "name": label, "price": float(c.iloc[-1]),
                "above_200d": bool(c.iloc[-1] > sma200), "distance_200d": float(c.iloc[-1] / sma200 - 1),
                "momentum_12_1": None if mom is None else float(mom),
                "return_1m": float(c.iloc[-1] / c.iloc[-22] - 1), "return_1y": float(c.iloc[-1] / c.iloc[-min(len(c), 253)] - 1),
                "signal": "Invest" if c.iloc[-1] > sma200 else "Stay in cash",
                "backtest": _backtest(c, ppy),
            })
        ranked = sorted([r for r in rows if r["momentum_12_1"] is not None], key=lambda r: r["momentum_12_1"], reverse=True)
        picks = [r["symbol"] for r in ranked if r["momentum_12_1"] > 0 and r["above_200d"]][:PICKS_PER_GROUP]
        for i, r in enumerate(ranked):
            r["momentum_rank"] = i + 1
        bt = [r["backtest"] for r in rows if r["backtest"]["trend_rule"]["cagr"] is not None]
        out["groups"].append({
            "name": gname,
            "assets": sorted(rows, key=lambda r: r.get("momentum_rank", 99)),
            "dual_momentum_picks": picks,
            "summary": {
                "assets_in_uptrend": sum(r["above_200d"] for r in rows), "assets": len(rows),
                "avg_drawdown_trend": float(np.mean([b["trend_rule"]["max_drawdown"] for b in bt])) if bt else None,
                "avg_drawdown_hold": float(np.mean([b["buy_and_hold"]["max_drawdown"] for b in bt])) if bt else None,
                "avg_cagr_trend": float(np.mean([b["trend_rule"]["cagr"] for b in bt])) if bt else None,
                "avg_cagr_hold": float(np.mean([b["buy_and_hold"]["cagr"] for b in bt])) if bt else None,
            },
        })
    return out


def get() -> dict:
    return jobs.run("allocation", compute, TTL)
