"""Point-in-time fundamentals: the quality and value factors through history.

Annual income statements and balance sheets (4-5 fiscal years per company) are
turned into daily (date x symbol) panels where each figure only becomes
"known" 90 days after the fiscal year ends, roughly when annual results are
public. That avoids look-ahead and lets the tournament test the quality and
value factors on history, not just today.

Quality follows Novy-Marx (2013) and Asness, Frazzini & Pedersen's
"Quality minus Junk": profitability (gross profit / assets, ROE, ROA),
safety (low leverage) and growth in profitability, each ranked within the
market and averaged. Value is the earnings yield (EPS / price).
"""

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf

from research import data

log = logging.getLogger(__name__)

REPORT_LAG_DAYS = 90
TTL = 7 * 24 * 3600


def _statements(symbol: str) -> list[dict]:
    def load():
        try:
            t = yf.Ticker(symbol)
            inc, bs = t.income_stmt, t.balance_sheet
        except Exception as exc:
            log.info("statements failed for %s: %s", symbol, exc)
            return []
        if inc is None or inc.empty:
            return []
        # Some companies report in a different currency from their share price (e.g. INFY.NS
        # files in USD): ratios are fine, but EPS / price would be meaningless, so drop EPS then.
        try:
            info = t.get_info() or {}
            same_ccy = not info.get("financialCurrency") or info.get("financialCurrency") == info.get("currency")
        except Exception:
            same_ccy = True
        out = []
        for col in inc.columns:
            def g(df, *rows):
                for r in rows:
                    if df is not None and r in df.index and col in df.columns and pd.notna(df.at[r, col]):
                        return float(df.at[r, col])
                return None
            out.append({
                "period_end": pd.Timestamp(col).tz_localize(None) if pd.Timestamp(col).tzinfo else pd.Timestamp(col),
                "revenue": g(inc, "Total Revenue", "Operating Revenue"),
                "gross_profit": g(inc, "Gross Profit"),
                "operating_income": g(inc, "Operating Income", "Total Operating Income As Reported"),
                "net_income": g(inc, "Net Income", "Net Income Common Stockholders"),
                "eps": g(inc, "Diluted EPS", "Basic EPS") if same_ccy else None,
                "assets": g(bs, "Total Assets"),
                "equity": g(bs, "Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"),
                "debt": g(bs, "Total Debt"),
            })
        return out

    return data.cached(f"statements:{symbol}", TTL, load)


def load(symbols: list[str], progress=lambda m: None) -> dict[str, list[dict]]:
    progress(f"Loading 5 years of annual financial statements for {len(symbols)} companies…")
    with ThreadPoolExecutor(max_workers=6) as ex:
        return dict(zip(symbols, ex.map(_statements, symbols)))


def panels(statements: dict[str, list[dict]], closes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Daily panels: quality (composite rank), earnings_yield, and the quality components."""
    idx = closes.index
    raw: dict[str, dict[str, pd.Series]] = {k: {} for k in ("gpa", "roe", "roa", "leverage", "gpa_growth", "eps", "rev_growth", "eps_growth", "rev_accel", "net_margin")}
    for sym, rows in statements.items():
        if sym not in closes.columns or not rows:
            continue
        rows = sorted([r for r in rows if r["assets"]], key=lambda r: r["period_end"])
        if not rows:
            continue
        known = [r["period_end"] + pd.Timedelta(days=REPORT_LAG_DAYS) for r in rows]
        vals = {k: [] for k in raw}
        for i, r in enumerate(rows):
            gp = r["gross_profit"] if r["gross_profit"] is not None else r["operating_income"]
            gpa = gp / r["assets"] if gp is not None else None
            prev = rows[i - 1] if i else None
            prev_gp = None
            if prev and prev["assets"]:
                pg = prev["gross_profit"] if prev["gross_profit"] is not None else prev["operating_income"]
                prev_gp = pg / prev["assets"] if pg is not None else None
            vals["gpa"].append(gpa)
            vals["roe"].append(r["net_income"] / r["equity"] if r["net_income"] is not None and r["equity"] and r["equity"] > 0 else None)
            vals["roa"].append(r["net_income"] / r["assets"] if r["net_income"] is not None else None)
            vals["leverage"].append(r["debt"] / r["assets"] if r["debt"] is not None else None)
            vals["gpa_growth"].append(gpa - prev_gp if gpa is not None and prev_gp is not None else None)
            vals["eps"].append(r["eps"])
            rev, prev_rev = r["revenue"], prev["revenue"] if prev else None
            growth = rev / prev_rev - 1 if rev and prev_rev and prev_rev > 0 else None
            vals["rev_growth"].append(growth)
            prev_growth = vals["rev_growth"][-2] if len(vals["rev_growth"]) > 1 else None
            vals["rev_accel"].append(growth - prev_growth if growth is not None and prev_growth is not None else None)
            pe, pp = r["net_income"], prev["net_income"] if prev else None
            vals["eps_growth"].append((pe - pp) / abs(pp) if pe is not None and pp else None)
            vals["net_margin"].append(r["net_income"] / rev if r["net_income"] is not None and rev else None)
        for k, v in vals.items():
            s = pd.Series(v, index=pd.DatetimeIndex(known), dtype=float).dropna()
            s = s[~s.index.duplicated(keep="last")]
            if len(s):
                raw[k][sym] = s.reindex(idx.union(s.index)).ffill().reindex(idx)
    out: dict[str, pd.DataFrame] = {}
    frames = {k: pd.DataFrame(v).reindex(columns=closes.columns) for k, v in raw.items()}
    ranks = []
    for k, sign in (("gpa", 1), ("roe", 1), ("roa", 1), ("leverage", -1), ("gpa_growth", 1)):
        f = frames[k]
        if f.notna().any().any():
            ranks.append((sign * f).rank(axis=1, pct=True))
    if ranks:
        stacked = np.stack([r.to_numpy() for r in ranks])
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices are expected (no data yet)
            q = np.nanmean(stacked, axis=0)
        cover = np.sum(~np.isnan(stacked), axis=0)
        q[cover < 2] = np.nan  # need at least two components
        out["quality"] = pd.DataFrame(q, index=idx, columns=closes.columns)
    eps = frames["eps"]
    if eps.notna().any().any():
        out["earnings_yield"] = eps / closes
    out.update({f"q_{k}": frames[k] for k in ("gpa", "roe", "leverage")})
    out.update({k: frames[k].clip(-1, 3) for k in ("rev_growth", "eps_growth", "rev_accel") if frames[k].notna().any().any()})
    if frames["net_margin"].notna().any().any():
        out["net_margin"] = frames["net_margin"].clip(-1, 1)
    return out


def coverage(p: dict[str, pd.DataFrame]) -> dict:
    q = p.get("quality")
    if q is None:
        return {"companies": 0, "from": None}
    have = q.notna().sum(axis=1)
    start = have[have >= 20].index
    return {"companies": int(q.iloc[-1].notna().sum()), "from": str(start[0].date()) if len(start) else None}
