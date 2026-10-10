"""Earnings surprises (post-earnings-announcement drift) and analyst revisions.

- Surprise history: ~6 years of quarterly results with the consensus estimate,
  so "how big was the latest surprise?" can be known on every past date and
  tested in the tournament. The signal lasts 63 trading days after each report.
- Analyst revisions (current only): how many analysts raised vs cut their EPS
  estimates over the last 30 days, the % change in the consensus estimate over
  90 days, and the shift in buy-ratings over 3 months. Research shows analyst
  revision momentum is one of the more durable short-term signals; we can't
  backtest it here (no free revision history), so it is weighted modestly and
  logged in the pick ledger to be judged on live results.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf

from research import data

log = logging.getLogger(__name__)
SURPRISE_LIFE = 63


def _surprises(symbol: str) -> pd.DataFrame | None:
    def load():
        try:
            ed = yf.Ticker(symbol).get_earnings_dates(limit=24)
        except Exception as exc:
            log.info("earnings dates failed for %s: %s", symbol, exc)
            return None
        if ed is None or ed.empty or "Surprise(%)" not in ed:
            return None
        ed = ed.dropna(subset=["Surprise(%)"])
        ed.index = pd.DatetimeIndex(ed.index).tz_localize(None).normalize()
        return ed[["EPS Estimate", "Reported EPS", "Surprise(%)"]].sort_index()

    return data.cached(f"earnings:{symbol}", 3 * 24 * 3600, load)


def surprise_panel(symbols: list[str], closes: pd.DataFrame, progress=lambda m: None) -> pd.DataFrame | None:
    progress(f"Loading earnings-surprise history for {len(symbols)} companies…")
    with ThreadPoolExecutor(max_workers=6) as ex:
        hist = dict(zip(symbols, ex.map(_surprises, symbols)))
    idx = closes.index
    cols = {}
    for sym, ed in hist.items():
        if ed is None or ed.empty or sym not in closes:
            continue
        s = pd.Series(np.nan, index=idx)
        for d, row in ed.iterrows():
            pos = idx.searchsorted(d + pd.Timedelta(days=1))  # known from the next trading day
            if pos < len(idx):
                s.iloc[pos : pos + SURPRISE_LIFE] = float(np.clip(row["Surprise(%)"], -50, 50))
        cols[sym] = s
    if len(cols) < 20:
        return None
    return pd.DataFrame(cols).reindex(columns=closes.columns)


def revisions(symbol: str) -> dict | None:
    def load():
        try:
            t = yf.Ticker(symbol)
            rev, trend, rec = t.eps_revisions, t.eps_trend, t.recommendations_summary
        except Exception as exc:
            log.info("revisions failed for %s: %s", symbol, exc)
            return None
        out = {}
        if rev is not None and not rev.empty:
            row = rev.loc["+1y"] if "+1y" in rev.index else rev.iloc[0]
            up, down = float(row.get("upLast30days") or 0), float(row.get("downLast30days") or 0)
            out["up_30d"], out["down_30d"] = up, down
            out["revision_ratio"] = (up - down) / (up + down) if up + down else 0.0
        if trend is not None and not trend.empty:
            row = trend.loc["+1y"] if "+1y" in trend.index else trend.iloc[0]
            cur, old = row.get("current"), row.get("90daysAgo")
            if cur and old and old > 0:
                out["eps_estimate_change_90d"] = float(cur / old - 1)
        if rec is not None and not rec.empty and {"strongBuy", "buy", "hold", "sell", "strongSell"} <= set(rec.columns):
            r = rec.set_index("period") if "period" in rec.columns else rec
            share = (r["strongBuy"] + r["buy"]) / r[["strongBuy", "buy", "hold", "sell", "strongSell"]].sum(axis=1).replace(0, np.nan)
            if "0m" in share.index and "-3m" in share.index:
                out["buy_share"] = float(share["0m"])
                out["buy_share_change_3m"] = float(share["0m"] - share["-3m"])
        if not out:
            return None
        parts = [out.get("revision_ratio"), np.tanh(5 * out["eps_estimate_change_90d"]) if "eps_estimate_change_90d" in out else None,
                 np.tanh(5 * out["buy_share_change_3m"]) if "buy_share_change_3m" in out else None]
        parts = [p for p in parts if p is not None and np.isfinite(p)]
        out["score"] = float(np.mean(parts)) if parts else 0.0  # -1 (downgrades) .. +1 (upgrades)
        return out

    return data.cached(f"revisions:{symbol}", 24 * 3600, load)


def revisions_many(symbols: list[str]) -> dict[str, dict | None]:
    with ThreadPoolExecutor(max_workers=6) as ex:
        return dict(zip(symbols, ex.map(revisions, symbols)))
