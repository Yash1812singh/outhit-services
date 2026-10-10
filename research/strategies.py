"""Strategy library: rule sets of famous investors, classic allocation portfolios, and a custom builder.

Every strategy is a function (context, signal date, params) -> target weights (a Series of symbol -> weight,
summing to at most 1; the rest is cash). It may only look at data up to the signal date, so the same
function drives the backtest, today's holdings, and copy-trading in the paper portfolios.

The stock rules are simplified, published descriptions of each investor's approach, built from the data
we have point-in-time: prices, volumes, and annual fundamentals known 90 days after each fiscal year end.
They are "in the style of", not the investors' actual portfolios. Their real holdings are in gurus.py.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from research import data, fundamentals_history, methods, tournament

log = logging.getLogger(__name__)

BENCHMARK = {"in": ("NIFTYBEES.NS", "Nifty 50 (NIFTYBEES ETF)"), "us": ("SPY", "S&P 500 (SPY ETF)")}
CURRENCY = {"in": "INR", "us": "USD"}
RISK_FREE = {"in": 0.065, "us": 0.04}  # rough average cash yield, for Sharpe ratios
DEFAULT_COST_BPS = {"in": 20, "us": 10}  # per side: brokerage, taxes, spread
CONTEXT_TTL = 6 * 3600
FREQ_LABEL = {"W": "Weekly", "M": "Monthly", "Q": "Quarterly", "A": "Yearly", "F": "When a new 13F is filed"}


@dataclass
class Strategy:
    key: str
    name: str
    person: str
    kind: str  # stocks | allocation | custom | guru
    style: str
    summary: str
    rules: list[str]
    rebalance: str  # W | M | Q | A | F
    select: Callable | None = None
    markets: tuple[str, ...] = ("in", "us")
    source: str = ""
    needs: tuple[str, ...] = ()  # fundamental panels required
    top_n: int = 20
    show: list[tuple[str, str, str]] = field(default_factory=list)  # (feature, label, format) for holdings
    assets: dict[str, dict[str, str]] = field(default_factory=dict)  # allocation: market -> {symbol: name}

    def meta(self) -> dict:
        return {
            "key": self.key, "name": self.name, "person": self.person, "kind": self.kind, "style": self.style,
            "summary": self.summary, "rules": self.rules, "rebalance": self.rebalance,
            "rebalance_label": FREQ_LABEL[self.rebalance], "markets": list(self.markets), "source": self.source,
            "holdings": self.top_n if self.kind in ("stocks", "custom") else None,
        }


# ---------------------------------------------------------------------------
# Context: everything a strategy may look at
# ---------------------------------------------------------------------------

@dataclass
class Context:
    market: str
    closes: pd.DataFrame
    f: dict[str, pd.DataFrame]
    names: dict[str, str]
    sectors: dict[str, str]
    bench: pd.Series

    def row(self, name: str, t: pd.Timestamp) -> pd.Series:
        df = self.f.get(name)
        if df is None or t not in df.index:
            return pd.Series(dtype=float)
        return df.loc[t]

    def tradable(self, t: pd.Timestamp) -> pd.Series:
        return self.closes.loc[t].notna()

    def first_date(self, panels: tuple[str, ...], min_names: int = 20) -> pd.Timestamp:
        """First date every listed panel has data for at least min_names companies."""
        start = self.closes.index[0]
        for p in panels:
            df = self.f.get(p)
            if df is None:
                return self.closes.index[-1]
            ok = df.notna().sum(axis=1) >= min_names
            start = max(start, ok.idxmax() if ok.any() else self.closes.index[-1])
        return start


_ctx: dict[str, tuple[float, Context]] = {}
_ctx_lock = threading.Lock()


def stock_context(market: str, progress=lambda m: None) -> Context:
    with _ctx_lock:
        hit = _ctx.get(f"stocks:{market}")
        if hit and time.time() - hit[0] < CONTEXT_TTL:
            return hit[1]
        progress("Loading the market's 300 largest companies…")
        uni = data.universe(market, tournament.UNIVERSE)
        if uni.empty:
            raise RuntimeError(f"No companies available for market {market}")
        progress("Loading 5 years of daily prices…")
        closes, volumes = data.daily(list(uni["symbol"]))
        f = methods.features(closes, volumes)
        sma200 = closes.rolling(200, min_periods=150).mean()
        f["sma50"] = closes.rolling(50, min_periods=40).mean()
        f["sma150"] = closes.rolling(150, min_periods=120).mean()
        f["sma200"] = sma200
        f["sma200_rising"] = (sma200 > sma200.shift(21)).astype(float).where(sma200.notna())
        f["vol_surge"] = volumes.rolling(10, min_periods=8).mean() / volumes.rolling(100, min_periods=70).mean()
        f.update(fundamentals_history.panels(fundamentals_history.load(list(closes.columns), progress), closes))
        for k in ("q_roe", "q_leverage", "net_margin", "q_gpa"):
            if k in f:
                f[f"{k}_1y_ago"] = f[k].shift(252)
        bench_sym = BENCHMARK[market][0]
        b, _ = data.daily([bench_sym])
        bench = b[bench_sym].reindex(closes.index).ffill() if bench_sym in b else closes.mean(axis=1)
        names = dict(zip(uni["symbol"], uni["name"]))
        sectors = dict(zip(uni["symbol"], uni["sector"])) if "sector" in uni else {}
        ctx = Context(market, closes, f, names, sectors, bench)
        _ctx[f"stocks:{market}"] = (time.time(), ctx)
        return ctx


def allocation_context(market: str) -> Context:
    with _ctx_lock:
        hit = _ctx.get(f"alloc:{market}")
        if hit and time.time() - hit[0] < CONTEXT_TTL:
            return hit[1]
        assets = {s: n for st in STRATEGIES.values() if st.kind == "allocation" for s, n in st.assets.get(market, {}).items()}
        bench_sym = BENCHMARK[market][0]
        closes, _ = data.daily(sorted(set(assets) | {bench_sym}), period="10y")
        closes = closes.ffill(limit=5)
        f = {"close": closes, "sma210": closes.rolling(210, min_periods=200).mean(), "ret_252": closes / closes.shift(252) - 1}
        ctx = Context(market, closes, f, assets | {bench_sym: BENCHMARK[market][1]}, {}, closes[bench_sym])
        _ctx[f"alloc:{market}"] = (time.time(), ctx)
        return ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)


def _pick(ctx: Context, t, score: pd.Series, eligible: pd.Series, n: int, weighting: str = "equal") -> pd.Series:
    """Top-n by score among eligible, tradable names. If fewer than half of n qualify, the rest stays in cash."""
    ok = eligible.reindex(score.index).fillna(False).astype(bool) & ctx.tradable(t).reindex(score.index).fillna(False) & score.notna()
    top = score[ok].nlargest(n)
    if top.empty:
        return pd.Series(dtype=float)
    slots = max(len(top), max(5, n // 2))
    if weighting == "inverse_vol":
        iv = 1 / ctx.row("vol_63", t).reindex(top.index).clip(lower=0.05).fillna(0.3)
        return iv / iv.sum() * len(top) / slots
    return pd.Series(1 / slots, index=top.index)


def _true(ctx, t) -> pd.Series:
    return pd.Series(True, index=ctx.closes.columns)


def _market_uptrend(ctx: Context, t) -> bool:
    b = ctx.bench.loc[:t].dropna()
    return len(b) < 200 or b.iloc[-1] > b.iloc[-200:].mean()


# ---------------------------------------------------------------------------
# Famous stock pickers (rules in their published style)
# ---------------------------------------------------------------------------

def buffett(ctx, t, p):
    roe, lev, ey, beta = ctx.row("q_roe", t), ctx.row("q_leverage", t), ctx.row("earnings_yield", t), ctx.row("beta_126", t)
    elig = (roe > 0.15) & (lev < 0.5) & (ey > 0)
    score = _rank(ctx.row("quality", t)) + _rank(ey) + _rank(-beta)
    return _pick(ctx, t, score, elig, p.get("top_n", 15))


def greenblatt(ctx, t, p):
    ey, roe = ctx.row("earnings_yield", t), ctx.row("q_roe", t)
    skip = {"Financial Services", "Utilities"}
    not_fin = pd.Series({s: ctx.sectors.get(s) not in skip for s in ctx.closes.columns})
    score = _rank(ey) + _rank(roe)
    return _pick(ctx, t, score, (ey > 0) & (roe > 0) & not_fin, p.get("top_n", 20))


def graham(ctx, t, p):
    ey, lev, margin, eg = ctx.row("earnings_yield", t), ctx.row("q_leverage", t), ctx.row("net_margin", t), ctx.row("eps_growth", t)
    elig = (ey > 1 / 15) & (lev < 0.4) & (margin > 0) & (eg > -0.5)
    return _pick(ctx, t, _rank(ey) + _rank(-lev), elig, p.get("top_n", 20))


def lynch(ctx, t, p):
    ey, eg = ctx.row("earnings_yield", t), ctx.row("eps_growth", t)
    peg = (1 / ey) / (eg * 100)
    elig = (ey > 0) & (eg >= 0.10) & (eg <= 0.60) & (peg < 1.5)
    return _pick(ctx, t, -peg, elig, p.get("top_n", 15))


def oneil(ctx, t, p):
    if not _market_uptrend(ctx, t):
        return pd.Series(dtype=float)  # "M": stay out when the market is in a downtrend
    eg, rg, hi, mom = ctx.row("eps_growth", t), ctx.row("rev_growth", t), ctx.row("high_52w", t), ctx.row("mom_12_1", t)
    rs = _rank(mom)
    elig = (eg > 0.20) & (rg > 0.10) & (hi >= 0.85) & (rs >= 0.80)
    return _pick(ctx, t, rs + _rank(eg), elig, p.get("top_n", 10))


def minervini(ctx, t, p):
    c, s50, s150, s200 = ctx.closes.loc[t], ctx.row("sma50", t), ctx.row("sma150", t), ctx.row("sma200", t)
    rs = _rank(ctx.row("mom_12_1", t))
    elig = ((c > s50) & (s50 > s150) & (s150 > s200) & (ctx.row("sma200_rising", t) == 1)
            & (ctx.row("low_52w", t) >= 0.30) & (ctx.row("high_52w", t) >= 0.75) & (rs >= 0.70))
    return _pick(ctx, t, rs, elig, p.get("top_n", 12))


def darvas(ctx, t, p):
    elig = (ctx.row("high_52w", t) >= 0.97) & (ctx.row("vol_surge", t) >= 1.3) & (ctx.row("dist_sma200", t) > 0)
    return _pick(ctx, t, ctx.row("vol_surge", t), elig, p.get("top_n", 10))


def piotroski(ctx, t, p):
    roe, eg, rg = ctx.row("q_roe", t), ctx.row("eps_growth", t), ctx.row("rev_growth", t)
    lev, lev0 = ctx.row("q_leverage", t), ctx.row("q_leverage_1y_ago", t)
    nm, nm0 = ctx.row("net_margin", t), ctx.row("net_margin_1y_ago", t)
    gp, gp0 = ctx.row("q_gpa", t), ctx.row("q_gpa_1y_ago", t)
    points = [roe > 0, eg > 0, rg > 0, lev < lev0, nm > nm0, gp > gp0]
    fscore = sum(x.astype(float) for x in points)
    ey = ctx.row("earnings_yield", t)
    cheap = _rank(ey) >= 0.6
    return _pick(ctx, t, fscore + _rank(ey) * 0.5, cheap & (fscore >= 5) & (ey > 0), p.get("top_n", 20))


def asness(ctx, t, p):
    return _pick(ctx, t, ctx.row("mom_12_1", t), _true(ctx, t), p.get("top_n", 20))


def van_vliet(ctx, t, p):
    vol = ctx.row("vol_252", t)
    calm = _rank(-vol) >= 0.5
    return _pick(ctx, t, _rank(ctx.row("mom_12_1", t)), calm, p.get("top_n", 20))


# ---------------------------------------------------------------------------
# Custom strategy builder
# ---------------------------------------------------------------------------

FACTORS = {
    "momentum": ("Momentum (12-1 month return)", "mom_12_1", 1),
    "value": ("Value (earnings yield)", "earnings_yield", 1),
    "quality": ("Quality (profitability, safety)", "quality", 1),
    "low_vol": ("Low volatility", "vol_252", -1),
    "growth": ("Revenue growth", "rev_growth", 1),
    "trend": ("Trend (distance above 200-day avg)", "dist_sma200", 1),
    "near_high": ("Near 52-week high", "high_52w", 1),
    "reversal": ("1-month reversal (buy recent losers)", "ret_21", -1),
}
FUNDAMENTAL_FACTORS = {"value": "earnings_yield", "quality": "quality", "growth": "rev_growth"}


def parse_custom(params: dict) -> dict:
    """Normalise custom-strategy params from the query string."""
    raw = params.get("factors", "momentum:1,quality:1")
    if isinstance(raw, dict):  # already parsed
        raw = ",".join(f"{k}:{v}" for k, v in raw.items())
    factors: dict[str, float] = {}
    for part in str(raw).split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            k = k.strip()
            try:
                w = float(v)
            except ValueError:
                continue
            if k in FACTORS and 0 < w <= 5:
                factors[k] = w
    if not factors:
        raise ValueError("Pick at least one factor (e.g. factors=momentum:1,value:1)")
    top = int(params.get("top_n", 20))
    if not 3 <= top <= 50:
        raise ValueError("top_n must be between 3 and 50")
    rebalance = str(params.get("rebalance", "M")).upper()
    if rebalance not in ("W", "M", "Q"):
        raise ValueError("rebalance must be W, M or Q")
    weighting = str(params.get("weighting", "equal"))
    if weighting not in ("equal", "inverse_vol"):
        raise ValueError("weighting must be equal or inverse_vol")
    truthy = lambda v: str(v).lower() in ("1", "true", "yes", "on")
    return {"factors": factors, "top_n": top, "rebalance": rebalance, "weighting": weighting,
            "trend_filter": truthy(params.get("trend_filter", False)), "regime_filter": truthy(params.get("regime_filter", False))}


def custom(ctx, t, p):
    if p["regime_filter"] and not _market_uptrend(ctx, t):
        return pd.Series(dtype=float)
    score = sum(_rank(ctx.row(FACTORS[k][1], t) * FACTORS[k][2]).fillna(0.5) * w for k, w in p["factors"].items())
    elig = _true(ctx, t)
    for k in p["factors"]:
        elig &= ctx.row(FACTORS[k][1], t).notna()
    if p["trend_filter"]:
        elig &= ctx.row("dist_sma200", t) > 0
    return _pick(ctx, t, score, elig, p["top_n"], p["weighting"])


PRICE_MOMENTUM = {"asness", "minervini", "darvas", "oneil", "van_vliet"}  # rules that buy recent winners


def momentum_biased(st: "Strategy", params: dict) -> bool:
    """Strategies that buy recent winners gain the most from testing on today's (surviving) large companies."""
    return st.key in PRICE_MOMENTUM or (st.key == "custom" and bool({"momentum", "trend", "near_high"} & set(params.get("factors", {}))))


def custom_strategy(p: dict) -> "Strategy":
    labels = [f"{FACTORS[k][0]} ×{w:g}" for k, w in p["factors"].items()]
    rules = [f"Rank the 300 largest companies by: {', '.join(labels)}.", f"Hold the top {p['top_n']}, "
             + ("weighted by inverse volatility." if p["weighting"] == "inverse_vol" else "equally weighted.")]
    if p["trend_filter"]:
        rules.append("Only stocks above their 200-day average.")
    if p["regime_filter"]:
        rules.append("Go to cash when the market index is below its 200-day average.")
    return Strategy("custom", "Your strategy", "You", "custom", "Custom", "A strategy you designed.", rules, p["rebalance"],
                    lambda ctx, t, _: custom(ctx, t, p), needs=tuple(FUNDAMENTAL_FACTORS[k] for k in p["factors"] if k in FUNDAMENTAL_FACTORS),
                    top_n=p["top_n"], show=[(FACTORS[k][1], FACTORS[k][0].split(" (")[0], "pct" if k != "near_high" else "ratio") for k in p["factors"]])


# ---------------------------------------------------------------------------
# Classic allocation portfolios (ETFs)
# ---------------------------------------------------------------------------

def _static(weights: dict[str, dict[str, float]]):
    def select(ctx, t, p):
        w = pd.Series(weights[ctx.market])
        ok = ctx.closes.loc[t].reindex(w.index).notna()
        return w[ok] / w[ok].sum() if ok.any() else pd.Series(dtype=float)
    return select


ANTONACCI = {"in": {"risky": ["NIFTYBEES.NS", "MON100.NS"], "cash": "LIQUIDBEES.NS", "safe": "LTGILTBEES.NS"},
             "us": {"risky": ["SPY", "VEU"], "cash": "BIL", "safe": "AGG"}}


def antonacci(ctx, t, p):
    a = ANTONACCI[ctx.market]
    r = ctx.row("ret_252", t)
    risky = r.reindex(a["risky"]).dropna()
    if risky.empty:
        return pd.Series(dtype=float)
    best = risky.idxmax()
    if risky[best] > r.get(a["cash"], 0):
        return pd.Series({best: 1.0})
    return pd.Series({a["safe"]: 1.0}) if pd.notna(ctx.closes.loc[t].get(a["safe"])) else pd.Series(dtype=float)


FABER = {"in": (["NIFTYBEES.NS", "JUNIORBEES.NS", "MON100.NS", "GOLDBEES.NS", "LTGILTBEES.NS"], "LIQUIDBEES.NS"),
         "us": (["VTI", "VEU", "IEF", "VNQ", "DBC"], "BIL")}


def faber(ctx, t, p):
    assets, cash = FABER[ctx.market]
    c, sma = ctx.closes.loc[t], ctx.row("sma210", t)
    live = [a for a in assets if pd.notna(c.get(a))]
    if not live:
        return pd.Series(dtype=float)
    w = {a: 1 / len(live) for a in live if c[a] > sma.get(a, np.inf)}
    rest = 1 - sum(w.values())
    if rest > 1e-9:
        w[cash] = w.get(cash, 0) + rest
    return pd.Series(w)


ETF_NAMES = {
    "VTI": "Vanguard Total US Stock Market", "SPY": "S&P 500", "VEU": "Vanguard All-World ex-US", "VXUS": "Vanguard Total International",
    "TLT": "US Treasuries 20+ yr", "IEF": "US Treasuries 7-10 yr", "IEI": "US Treasuries 3-7 yr", "BND": "Vanguard Total Bond Market",
    "AGG": "iShares Core US Aggregate Bond", "GLD": "Gold", "DBC": "Commodities", "VNQ": "US Real Estate (REITs)", "BIL": "US T-Bills (cash)",
    "SHY": "US Treasuries 1-3 yr", "NIFTYBEES.NS": "Nifty 50 ETF", "JUNIORBEES.NS": "Nifty Next 50 ETF", "MON100.NS": "Nasdaq 100 ETF (India-listed)",
    "GOLDBEES.NS": "Gold ETF", "LIQUIDBEES.NS": "Liquid ETF (cash)", "LTGILTBEES.NS": "Long-term Gilt ETF",
}


def _names(*syms: str) -> dict[str, str]:
    return {s: ETF_NAMES[s] for s in syms}


# ---------------------------------------------------------------------------
# The library
# ---------------------------------------------------------------------------

S = Strategy
STRATEGIES: dict[str, Strategy] = {s.key: s for s in [
    S("buffett", "Quality at a fair price", "Warren Buffett", "stocks", "Quality value",
      "Wonderful businesses at fair prices: high return on equity, little debt, reasonable valuation, steady shares.",
      ["Return on equity above 15%", "Debt below 50% of assets", "Profitable (positive earnings)",
       "Rank by quality + cheapness + low market sensitivity (beta), as in “Buffett's Alpha” (Frazzini, Kabiller & Pedersen)",
       "Hold the top 15 equally, review quarterly"],
      "Q", buffett, source="Frazzini, Kabiller & Pedersen (2018), Buffett's Alpha", needs=("quality", "earnings_yield"), top_n=15,
      show=[("q_roe", "ROE", "pct"), ("q_leverage", "Debt/assets", "pct"), ("earnings_yield", "P/E", "pe"), ("beta_126", "Beta", "num")]),
    S("greenblatt", "Magic Formula", "Joel Greenblatt", "stocks", "Value + quality",
      "Good companies at bargain prices: rank every stock on earnings yield and return on capital, buy the best combined.",
      ["Rank by earnings yield (cheapness)", "Rank by return on capital (ROE used as the proxy)", "Add the two ranks; buy the top 20",
       "Skip banks, insurers and utilities, as in the book", "Review quarterly (Greenblatt holds each stock about a year)"],
      "Q", greenblatt, source="Greenblatt (2005), The Little Book That Beats the Market", needs=("earnings_yield",), top_n=20,
      show=[("earnings_yield", "P/E", "pe"), ("q_roe", "ROE", "pct")]),
    S("graham", "Defensive value", "Benjamin Graham", "stocks", "Deep value",
      "Buffett's teacher: cheap, conservatively financed, profitable companies with a margin of safety.",
      ["P/E below 15", "Debt below 40% of assets", "Profitable, earnings not collapsing", "Rank by cheapness and low debt; hold 20, review quarterly",
       "Simplified: Graham's dividend-record and book-value tests need data we don't have point-in-time"],
      "Q", graham, source="Graham (1949), The Intelligent Investor", needs=("earnings_yield",), top_n=20,
      show=[("earnings_yield", "P/E", "pe"), ("q_leverage", "Debt/assets", "pct"), ("net_margin", "Net margin", "pct")]),
    S("lynch", "Growth at a reasonable price", "Peter Lynch", "stocks", "GARP",
      "Fast-growing companies whose price hasn't caught up: PEG ratio (P/E ÷ growth) below 1.5, lowest first.",
      ["Earnings growing 10–60% a year", "PEG = P/E ÷ EPS growth below 1.5", "Buy the 15 lowest PEGs, review quarterly"],
      "Q", lynch, source="Lynch (1989), One Up on Wall Street", needs=("earnings_yield",), top_n=15,
      show=[("earnings_yield", "P/E", "pe"), ("eps_growth", "EPS growth", "pct")]),
    S("oneil", "CAN SLIM", "William O'Neil", "stocks", "Growth momentum",
      "Market leaders with surging earnings near new highs, bought only while the overall market is rising.",
      ["Earnings growth above 20% and sales growth above 10% (C, A)", "Price within 15% of its 52-week high (N)",
       "Top 20% relative strength (L)", "Cash when the index is below its 200-day average (M)", "Hold the top 10, review monthly"],
      "M", oneil, source="O'Neil (1988), How to Make Money in Stocks", needs=("earnings_yield",), top_n=10,
      show=[("eps_growth", "EPS growth", "pct"), ("rev_growth", "Sales growth", "pct"), ("mom_12_1", "12-m return", "pct")]),
    S("minervini", "Trend Template", "Mark Minervini", "stocks", "Momentum / trend",
      "US Investing Champion's stage-2 uptrend filter: only stocks in strong, confirmed uptrends with top relative strength.",
      ["Price > 50-day > 150-day > 200-day average", "200-day average rising for a month", "At least 30% above the 52-week low and within 25% of the high",
       "Relative strength in the top 30%", "Hold the 12 strongest, review weekly (a stock that breaks the template is sold)"],
      "W", minervini, source="Minervini (2013), Trade Like a Stock Market Wizard", top_n=12,
      show=[("mom_12_1", "12-m return", "pct"), ("high_52w", "% of 52-w high", "ratio")]),
    S("darvas", "Box breakout", "Nicolas Darvas", "stocks", "Breakout",
      "The dancer who made $2M in the 1950s: buy stocks breaking to new highs on heavy volume.",
      ["Price within 3% of its 52-week high", "Volume 30%+ above normal", "Above the 200-day average", "Hold the 10 biggest volume surges, review weekly"],
      "W", darvas, source="Darvas (1960), How I Made $2,000,000 in the Stock Market", top_n=10,
      show=[("high_52w", "% of 52-w high", "ratio"), ("vol_surge", "Volume vs normal", "x")]),
    S("piotroski", "F-Score value", "Joseph Piotroski", "stocks", "Value + improving fundamentals",
      "Among cheap stocks, buy the ones whose finances are improving; avoid the value traps.",
      ["Cheapest 40% by earnings yield", "Score 1 point each: profitable, EPS up, sales up, less debt, higher margin, higher gross profitability",
       "Need 5 of 6 points; hold 20, review quarterly", "Simplified: 6 of Piotroski's 9 tests are possible with our data"],
      "Q", piotroski, source="Piotroski (2000), Value Investing: The Use of Historical Financial Statement Information", needs=("earnings_yield",), top_n=20,
      show=[("earnings_yield", "P/E", "pe"), ("eps_growth", "EPS growth", "pct"), ("q_roe", "ROE", "pct")]),
    S("asness", "Momentum", "Cliff Asness (AQR)", "stocks", "Momentum",
      "The best-documented anomaly: last year's winners (skipping the last month) keep winning for a while.",
      ["Rank by 12-month return, skipping the latest month", "Hold the top 20 equally, review monthly"],
      "M", asness, source="Jegadeesh & Titman (1993); Asness, Moskowitz & Pedersen (2013)", top_n=20,
      show=[("mom_12_1", "12-1 m return", "pct"), ("vol_252", "Volatility", "pct")]),
    S("van_vliet", "Conservative formula", "Pim van Vliet (Robeco)", "stocks", "Low volatility",
      "“High returns from low risk”: the calmest half of the market, then the ones with the best momentum.",
      ["Keep the 50% of stocks with the lowest 1-year volatility", "Rank them by 12-1 month momentum", "Hold the top 20, review quarterly"],
      "Q", van_vliet, source="Van Vliet & Blitz (2018), The Conservative Formula", top_n=20,
      show=[("vol_252", "Volatility", "pct"), ("mom_12_1", "12-1 m return", "pct")]),
    S("dalio", "All Weather", "Ray Dalio", "allocation", "Risk parity",
      "Built to do OK in every economic season (growth, recession, inflation, deflation), with a lot of bonds.",
      ["US: 30% stocks, 40% long Treasuries, 15% medium Treasuries, 7.5% gold, 7.5% commodities",
       "India: 30% Nifty 50, 55% long gilts, 15% gold", "Rebalance quarterly"],
      "Q", _static({"us": {"VTI": .30, "TLT": .40, "IEI": .15, "GLD": .075, "DBC": .075}, "in": {"NIFTYBEES.NS": .30, "LTGILTBEES.NS": .55, "GOLDBEES.NS": .15}}),
      source="Bridgewater; Tony Robbins (2014), Money: Master the Game",
      assets={"us": _names("VTI", "TLT", "IEI", "GLD", "DBC"), "in": _names("NIFTYBEES.NS", "LTGILTBEES.NS", "GOLDBEES.NS")}),
    S("bogle", "Index & chill (60/40)", "Jack Bogle", "allocation", "Passive",
      "Vanguard's founder: own the whole market cheaply, add bonds to sleep at night, and never trade.",
      ["US: 60% total US stock market, 40% total bond market", "India: 50% Nifty 50, 10% Nifty Next 50, 40% gilts", "Rebalance once a year"],
      "A", _static({"us": {"VTI": .60, "BND": .40}, "in": {"NIFTYBEES.NS": .50, "JUNIORBEES.NS": .10, "LTGILTBEES.NS": .40}}),
      source="Bogle (2007), The Little Book of Common Sense Investing",
      assets={"us": _names("VTI", "BND"), "in": _names("NIFTYBEES.NS", "JUNIORBEES.NS", "LTGILTBEES.NS")}),
    S("antonacci", "Dual Momentum (GEM)", "Gary Antonacci", "allocation", "Trend + momentum",
      "Hold whichever stock market is strongest, but only while it beats cash; otherwise hide in bonds.",
      ["Compare home vs international stocks by 12-month return; pick the stronger", "Only if it beat cash (T-bills / liquid ETF) over 12 months",
       "Otherwise hold bonds", "Check monthly"],
      "M", antonacci, source="Antonacci (2014), Dual Momentum Investing",
      assets={"us": _names("SPY", "VEU", "BIL", "AGG"), "in": _names("NIFTYBEES.NS", "MON100.NS", "LIQUIDBEES.NS", "LTGILTBEES.NS")}),
    S("faber", "Ivy / GTAA timing", "Meb Faber", "allocation", "Trend following",
      "Five asset classes, each held only while it's above its 10-month average; this sidestepped most of 2008.",
      ["Equal 20% in 5 asset classes (stocks, international, bonds, real estate/gold, commodities)",
       "Each one only while above its 10-month moving average, else cash", "Check monthly"],
      "M", faber, source="Faber (2007), A Quantitative Approach to Tactical Asset Allocation",
      assets={"us": _names("VTI", "VEU", "IEF", "VNQ", "DBC", "BIL"), "in": _names("NIFTYBEES.NS", "JUNIORBEES.NS", "MON100.NS", "GOLDBEES.NS", "LTGILTBEES.NS", "LIQUIDBEES.NS")}),
    S("browne", "Permanent Portfolio", "Harry Browne", "allocation", "All-season",
      "Four equal parts for prosperity, recession, inflation and deflation. Simple and very steady.",
      ["25% stocks, 25% long government bonds, 25% gold, 25% cash", "Rebalance once a year"],
      "A", _static({"us": {"VTI": .25, "TLT": .25, "GLD": .25, "BIL": .25}, "in": {"NIFTYBEES.NS": .25, "LTGILTBEES.NS": .25, "GOLDBEES.NS": .25, "LIQUIDBEES.NS": .25}}),
      source="Browne (1987), Why the Best-Laid Investment Plans Usually Go Wrong",
      assets={"us": _names("VTI", "TLT", "GLD", "BIL"), "in": _names("NIFTYBEES.NS", "LTGILTBEES.NS", "GOLDBEES.NS", "LIQUIDBEES.NS")}),
]}


def get(key: str, params: dict | None = None) -> Strategy:
    if key == "custom":
        return custom_strategy(parse_custom(params or {}))
    from research import gurus  # noqa: PLC0415  (gurus imports this module)

    if key in gurus.GURUS:
        return gurus.strategy(key)
    if key not in STRATEGIES:
        raise KeyError(key)
    return STRATEGIES[key]


def catalogue() -> list[dict]:
    from research import gurus  # noqa: PLC0415

    builder = custom_strategy(parse_custom({})).meta() | {
        "name": "Build your own", "summary": "Combine factors, filters and rebalancing into your own rules, then backtest and paper trade them.",
        "rules": ["Choose factors and their weights", "Choose how many stocks to hold and how often to rebalance",
                  "Optional: only stocks above their 200-day average; cash when the market is falling"]}
    return [s.meta() for s in STRATEGIES.values()] + [gurus.strategy(k).meta() for k in gurus.GURUS] + [builder]
