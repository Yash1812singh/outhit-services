"""Ranks every company in a market for long-term and swing horizons, and summarises sectors.

Long-term score (0-100) is a multi-factor model. Each factor is a percentile
rank within the market (value is ranked within the sector, so banks are not
compared with software companies on P/E):

    value 25 · momentum 20 · growth 15 · trend 15 · analyst view 10 · dividend 5 · quality 10*

*Quality (ROE, debt, margins, cash flow, growth) needs per-company data, so it
is fetched only for the top candidates in a second "deep check" pass, which
also flags red flags and adds the technical timing signal.

Value + quality + momentum is the best-documented combination in the
academic factor literature, but it has long losing stretches and no
guarantee. We show every input so the user can judge.
"""

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from market_data.db import SessionLocal
from market_data.models import FundamentalsCache, StockQuote
from market_data.services import market_data, screener
from market_data.services import signals as sig
from market_data.universe import Asset

log = logging.getLogger(__name__)

PRE_WEIGHTS = {"value": 0.25, "momentum": 0.20, "growth": 0.15, "trend": 0.15, "analyst": 0.10, "dividend": 0.05}
QUALITY_WEIGHT = 0.10 / (1 - 0.10)  # quality's share once added on top of the pre-score
DEEP_CHECK = 40
# Size tiers by market-cap rank within the market (SEBI's large/mid/small-cap convention).
SIZE_TIERS = {"large": (1, 100), "mid": (101, 250), "small": (251, 10**9), "large_mid": (1, 250), "all": (1, 10**9)}
MAX_PER_SECTOR = 3
FUNDAMENTALS_TTL = timedelta(hours=24)
CACHE_TTL = 1800

_cache: dict[tuple, tuple[float, object]] = {}
_cache_lock = threading.Lock()


def _cached(key: tuple, fn):
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL:
            return hit[1]
    value = fn()
    with _cache_lock:
        _cache[key] = (time.time(), value)
    return value


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


# ---------------------------------------------------------------------------
# Market-wide scoring from screener data
# ---------------------------------------------------------------------------


def _analyst_score(rating: str | float | None) -> float:
    """'2.1 - Buy' -> 0..1 where 1 is Strong Buy (1.0) and 0 is Sell (5.0)."""
    if not isinstance(rating, str):
        return np.nan
    m = re.match(r"\s*([\d.]+)", rating)
    return float(np.clip((5 - float(m.group(1))) / 4, 0, 1)) if m else np.nan


def frame(rows: list[StockQuote]) -> pd.DataFrame:
    df = pd.DataFrame(
        [{"symbol": r.symbol, "name": r.name, "sector": r.sector, "exchange": r.exchange, "currency": r.currency, **r.data} for r in rows]
    )
    if df.empty:
        return df
    num = ["price", "change_pct", "volume", "avg_volume_3m", "market_cap", "pe", "forward_pe", "pb", "eps_ttm", "eps_forward",
           "dividend_yield_pct", "change_52w_pct", "high_52w", "low_52w", "sma50", "sma200"]
    for c in num:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["turnover"] = df["price"] * df["avg_volume_3m"]
    # One row per company: share classes (PBR / PBR-A, GOOG / GOOGL) share a name; keep the most traded.
    df = df.sort_values("turnover", ascending=False).drop_duplicates("name").reset_index(drop=True)
    return df


def score_market(df: pd.DataFrame) -> pd.DataFrame:
    """Adds factor columns (0..1) and `score` (0..100) to the screener frame."""
    if df.empty:
        return df
    df = df.copy()
    rank = lambda s: s.rank(pct=True)  # noqa: E731

    ey = (1 / df["pe"]).where(df["pe"] > 0)
    bp = (1 / df["pb"]).where(df["pb"] > 0)
    v_pe = ey.groupby(df["sector"]).transform(rank)
    v_pb = bp.groupby(df["sector"]).transform(rank)
    value = pd.concat([v_pe, v_pb], axis=1).mean(axis=1)
    df["f_value"] = value.where(df["pe"] > 0, 0.1)  # loss-makers rank near the bottom on value

    growth = (df["eps_forward"] / df["eps_ttm"] - 1).where((df["eps_ttm"] > 0) & (df["eps_forward"] > 0)).clip(-0.5, 1.0)
    df["eps_growth_est"] = growth
    df["f_growth"] = rank(growth)
    df["f_momentum"] = rank(df["change_52w_pct"])
    above = df["price"] / df["sma200"] - 1
    df["f_trend"] = 0.5 * rank(above) + 0.5 * (df["sma50"] > df["sma200"]).astype(float).where(df["sma200"].notna())
    df["f_dividend"] = rank(df["dividend_yield_pct"].fillna(0))
    df["f_analyst"] = df["analyst_rating"].map(_analyst_score) if "analyst_rating" in df else np.nan

    cols = [f"f_{k}" for k in PRE_WEIGHTS]
    w = pd.Series({f"f_{k}": v for k, v in PRE_WEIGHTS.items()})
    vals = df[cols]
    weights = vals.notna().mul(w, axis=1)
    df["score"] = (vals.fillna(0).mul(w, axis=1).sum(axis=1) / weights.sum(axis=1).replace(0, np.nan) * 100).round(1)

    # Liquidity: the least-traded quarter of the market is too thin to recommend.
    df["liquid"] = df["turnover"] >= df["turnover"].quantile(0.25)
    df["cap_rank"] = df["market_cap"].rank(ascending=False, method="first")
    df["size_tier"] = np.select([df["cap_rank"] <= 100, df["cap_rank"] <= 250], ["Large cap", "Mid cap"], "Small cap")
    return df


class NoData(Exception):
    pass


def market_frame(code: str) -> tuple[pd.DataFrame, object]:
    st = screener.ensure_market(code)
    if not st or st.companies == 0:
        raise NoData(f"Our data source has no company listings for {screener.MARKETS_BY_CODE[code].name} right now.")
    key = ("frame", code, st.refreshed_at.isoformat())
    return _cached(key, lambda: score_market(frame(screener.load(code)))), st


def _reasons(r: pd.Series) -> list[str]:
    out = []
    if r.get("f_value", 0) >= 0.7 and r.get("pe") and r["pe"] > 0:
        out.append(f"Cheap for its sector (P/E {r['pe']:.1f})")
    if r.get("f_momentum", 0) >= 0.7 and pd.notna(r.get("change_52w_pct")):
        out.append(f"Strong 1-year performance ({r['change_52w_pct']:+.0f}%)")
    if pd.notna(r.get("eps_growth_est")) and r["eps_growth_est"] >= 0.1:
        out.append(f"Earnings expected to grow {r['eps_growth_est']:.0%}")
    if r.get("f_trend", 0) >= 0.75:
        out.append("In a long-term uptrend (above 200-day average)")
    if pd.notna(r.get("dividend_yield_pct")) and r["dividend_yield_pct"] >= 2.5:
        out.append(f"Dividend yield {r['dividend_yield_pct']:.1f}%")
    if r.get("f_analyst", 0) >= 0.7:
        out.append(f"Analysts rate it {str(r.get('analyst_rating')).split('-')[-1].strip()}")
    return out


# ---------------------------------------------------------------------------
# Deep check (per-company fundamentals)
# ---------------------------------------------------------------------------

INFO_KEYS = ["returnOnEquity", "debtToEquity", "profitMargins", "revenueGrowth", "earningsGrowth", "freeCashflow",
             "currentRatio", "industry", "longBusinessSummary", "website", "fullTimeEmployees", "beta", "pegRatio",
             "bookValue", "totalDebt"]


def fundamentals(symbol: str) -> dict:
    with SessionLocal() as db:
        hit = db.get(FundamentalsCache, symbol)
        if hit and datetime.now(timezone.utc) - screener._aware(hit.fetched_at) < FUNDAMENTALS_TTL:
            return hit.data
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as exc:
        log.info("info failed for %s: %s", symbol, exc)
        return hit.data if hit else {}
    data = {k: info.get(k) for k in INFO_KEYS}
    if data.get("longBusinessSummary"):
        data["longBusinessSummary"] = data["longBusinessSummary"][:600]
    with SessionLocal() as db:
        db.merge(FundamentalsCache(symbol=symbol, data=data, fetched_at=datetime.now(timezone.utc)))
        db.commit()
    return data


def _num(x) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def quality(f: dict, sector: str, row: pd.Series | dict | None = None) -> tuple[float | None, list[str], list[str]]:
    """Returns (quality 0..1, strengths, red flags).

    Yahoo's per-company data is patchy outside the US (ROE and free cash flow are
    often missing for Indian stocks), so ROE and debt/equity fall back to values
    derived from the screener's P/E, P/B and market cap."""
    row = row if row is not None else {}
    parts, good, flags = [], [], []
    roe, de, margin = f.get("returnOnEquity"), f.get("debtToEquity"), f.get("profitMargins")
    rev, fcf = f.get("revenueGrowth"), f.get("freeCashflow")
    financial = sector == "Financial Services"  # leverage is the business model for banks
    pe, pb, mcap = _num(row.get("pe")), _num(row.get("pb")), _num(row.get("market_cap"))

    book = _num(f.get("bookValue"))
    if book is not None and book < 0:
        flags.append("Negative shareholder equity (owes more than it owns)")
    if roe is None and pe and pb and pe > 0 and pb > 0:
        roe = pb / pe  # P/B / P/E = earnings / book equity
    if de is None and not financial and pb and pb > 0 and mcap and f.get("totalDebt") is not None:
        de = f["totalDebt"] / (mcap / pb) * 100
    if margin is not None and margin > 0.5 and sector not in ("Financial Services", "Real Estate"):
        flags.append(f"Unusually high profit margin ({margin:.0%}) - may include one-off gains")
        margin = 0.2  # don't let a one-off inflate the quality score

    if roe is not None:
        parts.append(float(np.clip(roe / 0.25, 0, 1)))
        if roe >= 0.18:
            good.append(f"High return on equity ({roe:.0%})")
        elif roe < 0.05:
            flags.append(f"Low return on equity ({roe:.0%})")
    if de is not None and not financial:
        parts.append(float(np.clip(1 - de / 200, 0, 1)))
        if de > 200:
            flags.append(f"Heavy debt (debt/equity {de / 100:.1f}x)")
        elif de < 30:
            good.append("Very little debt")
    if margin is not None:
        parts.append(float(np.clip(margin / 0.2, 0, 1)))
        if margin < 0:
            flags.append("Loss-making")
    if rev is not None:
        parts.append(float(np.clip((rev + 0.05) / 0.25, 0, 1)))
        if rev < -0.05:
            flags.append(f"Revenue shrinking ({rev:.0%})")
        elif rev >= 0.15:
            good.append(f"Revenue growing {rev:.0%}")
    if fcf is not None and not financial:
        parts.append(1.0 if fcf > 0 else 0.0)
        if fcf < 0:
            flags.append("Negative free cash flow")
    return (float(np.mean(parts)) if parts else None), good, flags


def _timing(symbols: list[str], meta: dict[str, pd.Series]) -> dict[str, dict]:
    hist = market_data.get_history(symbols)
    out = {}
    for s, frame_ in hist.items():
        r = meta[s]
        a = sig.analyse(Asset(s, r["name"], "equity", "", r["currency"]), frame_)
        if not a.ready:
            continue
        bt = sig.backtest(a.close, a.scores)
        out[s] = {
            "score": round(a.score, 1),
            "signal": sig.label_for(a.score),
            "confidence": sig.confidence(a, bt),
            "volatility": sig.ind.annual_volatility(a.close),
            "backtest_accuracy": bt.get("directional_accuracy") if bt.get("available") else None,
        }
    return out


def _roe_from_row(r: pd.Series) -> float | None:
    pe, pb = _num(r.get("pe")), _num(r.get("pb"))
    return pb / pe if pe and pb and pe > 0 and pb > 0 else None


def _metrics(r: pd.Series) -> dict:
    keys = ["price", "change_pct", "market_cap", "pe", "forward_pe", "pb", "dividend_yield_pct", "change_52w_pct", "eps_growth_est", "analyst_rating"]
    return {k: (None if pd.isna(r.get(k)) else r.get(k)) for k in keys}


def _diversify(df: pd.DataFrame, n: int, per_sector: int) -> pd.DataFrame:
    return df.groupby("sector", group_keys=False).head(per_sector).head(n)


def long_term_picks(code: str, sector: str | None = None, size: str = "large_mid", n: int = 15) -> dict:
    def build():
        df, st = market_frame(code)
        lo, hi = SIZE_TIERS[size]
        in_tier = df["cap_rank"].between(lo, hi)
        pool = df[df["liquid"] & in_tier & df["score"].notna()]
        if sector:
            pool = pool[pool["sector"] == sector]
        pool = pool.sort_values("score", ascending=False)
        candidates = _diversify(pool, DEEP_CHECK, 8 if sector else 5)
        with ThreadPoolExecutor(max_workers=8) as ex:
            funds = dict(zip(candidates["symbol"], ex.map(fundamentals, candidates["symbol"])))

        picks = []
        for _, r in candidates.iterrows():
            q, good, flags = quality(funds.get(r["symbol"], {}), r["sector"], r)
            final = r["score"] if q is None else (r["score"] + QUALITY_WEIGHT * q * 100) / (1 + QUALITY_WEIGHT)
            picks.append({"row": r, "quality": q, "strengths": _reasons(r) + good, "flags": flags, "final": round(float(final), 1), "f": funds.get(r["symbol"], {})})
        picks = [p for p in picks if len(p["flags"]) < 2]
        picks.sort(key=lambda p: p["final"], reverse=True)

        chosen, per = [], {}
        for p in picks:
            sec = p["row"]["sector"]
            if per.get(sec, 0) >= (n if sector else MAX_PER_SECTOR):
                continue
            per[sec] = per.get(sec, 0) + 1
            chosen.append(p)
            if len(chosen) == n:
                break

        timing = _timing([p["row"]["symbol"] for p in chosen], {p["row"]["symbol"]: p["row"] for p in chosen})
        return {
            "market": code,
            "horizon": "long_term",
            "size": size,
            "as_of": st.refreshed_at.isoformat() if st else None,
            "universe_size": int(len(df)),
            "screened": int(len(pool)),
            "method": "Multi-factor score (value, momentum, growth, trend, analyst view, dividend) within the chosen size tier (large = top 100 by market value, mid = 101-250), then a quality check (ROE, debt, margins, revenue growth, free cash flow) on the top candidates. Companies with 2+ red flags are excluded; max 3 per sector for diversification.",
            "picks": [
                {
                    "symbol": p["row"]["symbol"],
                    "name": p["row"]["name"],
                    "sector": p["row"]["sector"],
                    "size_tier": p["row"]["size_tier"],
                    "industry": p["f"].get("industry"),
                    "currency": p["row"]["currency"],
                    "score": p["final"],
                    "factor_scores": {k: (None if pd.isna(p["row"].get(f"f_{k}")) else round(float(p["row"][f"f_{k}"]) * 100)) for k in PRE_WEIGHTS} | {"quality": None if p["quality"] is None else round(p["quality"] * 100)},
                    "metrics": _metrics(p["row"]) | {
                        "roe": p["f"].get("returnOnEquity") or _roe_from_row(p["row"]),
                        "debt_to_equity": p["f"].get("debtToEquity"),
                        "profit_margin": p["f"].get("profitMargins"),
                        "revenue_growth": p["f"].get("revenueGrowth"),
                    },
                    "strengths": p["strengths"],
                    "red_flags": p["flags"],
                    "timing": timing.get(p["row"]["symbol"]),
                    "about": p["f"].get("longBusinessSummary"),
                }
                for p in chosen
            ],
        }

    return _cached(("lt", code, sector, size, n), build)


def swing_picks(code: str, n: int = 15, universe: int = 150) -> dict:
    """1-4 week ideas: the technical signal on the most liquid stocks, weighted by how well it has worked on each one."""

    def build():
        df, st = market_frame(code)
        liquid = df[df["liquid"]].sort_values("turnover", ascending=False).head(universe)
        meta = {r["symbol"]: r for _, r in liquid.iterrows()}
        timing = _timing(list(meta), meta)
        weight = {"High": 1.0, "Medium": 0.8, "Low": 0.5}
        rows = []
        for s, t in timing.items():
            r = meta[s]
            weekly = t["volatility"] / np.sqrt(52) if t["volatility"] else None
            rows.append({
                "symbol": s,
                "name": r["name"],
                "sector": r["sector"],
                "currency": r["currency"],
                "price": r["price"],
                "change_pct": None if pd.isna(r["change_pct"]) else r["change_pct"],
                "signal": t["signal"],
                "score": t["score"],
                "confidence": t["confidence"],
                "backtest_accuracy": t["backtest_accuracy"],
                "rank_score": t["score"] * weight[t["confidence"]],
                "stop_loss": None if not weekly else round(float(r["price"] * (1 - 2 * weekly)), 2),
                "target": None if not weekly else round(float(r["price"] * (1 + 3 * weekly)), 2),
            })
        rows.sort(key=lambda x: x["rank_score"], reverse=True)
        bullish = [x for x in rows if x["score"] >= sig.BULLISH_AT][:n]
        bearish = [x for x in reversed(rows) if x["score"] <= sig.BEARISH_AT][:n]
        return {
            "market": code,
            "horizon": "swing",
            "as_of": st.refreshed_at.isoformat() if st else None,
            "universe_size": len(timing),
            "method": f"Outhit trend/momentum signal on the {len(timing)} most-traded stocks, ranked by score x backtested confidence. Stop ~2 weekly standard deviations below price, target ~3 above (1.5:1 reward/risk).",
            "picks": bullish,
            "avoid": bearish,
        }

    return _cached(("swing", code, n), build)


def sectors(code: str) -> dict:
    df, st = market_frame(code)
    out = []
    for name, g in df.groupby("sector"):
        breadth = (g["price"] > g["sma200"]).where(g["sma200"].notna()).mean()
        top = g[g["liquid"]].sort_values("score", ascending=False).head(3)
        out.append({
            "sector": name,
            "companies": int(len(g)),
            "market_cap": float(g["market_cap"].sum()),
            "median_change_pct": _f(g["change_pct"].median()),
            "median_change_52w_pct": _f(g["change_52w_pct"].median()),
            "median_pe": _f(g.loc[g["pe"] > 0, "pe"].median()),
            "median_dividend_yield_pct": _f(g["dividend_yield_pct"].median()),
            "breadth_above_200d": _f(breadth),
            "avg_score": _f(g["score"].mean()),
            "top": [{"symbol": r["symbol"], "name": r["name"], "score": _f(r["score"])} for _, r in top.iterrows()],
        })
    # Sector strength: half medium-term momentum, half breadth (share of stocks in uptrends).
    s = pd.DataFrame(out)
    if not s.empty:
        s["strength"] = (0.5 * s["median_change_52w_pct"].rank(pct=True) + 0.5 * s["breadth_above_200d"].rank(pct=True)) * 100
        out = s.sort_values("strength", ascending=False).to_dict("records")
    return {"market": code, "as_of": st.refreshed_at.isoformat() if st else None, "sectors": out}


def _f(x) -> float | None:
    return None if x is None or pd.isna(x) else float(x)


SORTS = {"market_cap", "change_pct", "change_52w_pct", "pe", "dividend_yield_pct", "score", "name"}


def browse(code: str, sector: str | None, q: str | None, sort: str, desc: bool, page: int, page_size: int) -> dict:
    df, st = market_frame(code)
    view = df
    if sector:
        view = view[view["sector"] == sector]
    if q:
        ql = q.lower()
        view = view[view["name"].str.lower().str.contains(ql, regex=False) | view["symbol"].str.lower().str.contains(ql, regex=False)]
    sort = sort if sort in SORTS else "market_cap"
    view = view.sort_values(sort, ascending=not desc, na_position="last")
    total = len(view)
    page_rows = view.iloc[(page - 1) * page_size : page * page_size]
    cols = ["symbol", "name", "sector", "exchange", "currency", "price", "change_pct", "market_cap", "pe", "pb",
            "dividend_yield_pct", "change_52w_pct", "analyst_rating", "score", "size_tier", "market_state", "delay_minutes"]
    return {
        "market": code,
        "as_of": st.refreshed_at.isoformat() if st else None,
        "total": total,
        "page": page,
        "page_size": page_size,
        "rows": [{c: (None if pd.isna(r.get(c)) else r.get(c)) for c in cols} for _, r in page_rows.iterrows()],
    }
