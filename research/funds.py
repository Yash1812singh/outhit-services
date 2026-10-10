"""Indian mutual fund ranking, category by category.

Data: AMFI's daily NAV file (every scheme in India, with its SEBI category) and
mfapi.in for full NAV history. Only Direct-plan Growth options are ranked
(lowest cost, no payouts distorting returns).

Each fund with at least 3 years of history is scored within its category on:
  3-year Sortino ratio (return per unit of downside risk)   30%
  3-year CAGR                                                20%
  Consistency: share of rolling 1-year periods (last 3y)
               in which it beat the category median          20%
  5-year CAGR (if available)                                 15%
  Maximum drawdown over 3 years (smaller is better)          15%

This is the same family of metrics fund-rating agencies use. Evidence that past
fund performance persists is weak, especially for large-cap funds, which is why
a low-cost index fund is shown as the benchmark choice in every equity category.
"""

import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pandas as pd

from research import data, jobs

log = logging.getLogger(__name__)

AMFI_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
MFAPI = "https://api.mfapi.in/mf/{code}"
RISK_FREE = 0.065
TTL = 24 * 3600

CATEGORIES = {
    "large_cap": ("Large cap", "Equity Scheme - Large Cap Fund"),
    "flexi_cap": ("Flexi cap", "Equity Scheme - Flexi Cap Fund"),
    "large_mid_cap": ("Large & mid cap", "Equity Scheme - Large & Mid Cap Fund"),
    "mid_cap": ("Mid cap", "Equity Scheme - Mid Cap Fund"),
    "small_cap": ("Small cap", "Equity Scheme - Small Cap Fund"),
    "multi_cap": ("Multi cap", "Equity Scheme - Multi Cap Fund"),
    "elss": ("ELSS (tax saver)", "Equity Scheme - ELSS"),
    "value": ("Value / contra", "Equity Scheme - Value Fund"),
    "focused": ("Focused", "Equity Scheme - Focused Fund"),
    "index": ("Index funds", "Other Scheme - Index Funds"),
    "aggressive_hybrid": ("Aggressive hybrid", "Hybrid Scheme - Aggressive Hybrid Fund"),
    "balanced_advantage": ("Balanced advantage", "Hybrid Scheme - Dynamic Asset Allocation or Balanced Advantage"),
    "short_duration": ("Short duration debt", "Debt Scheme - Short Duration Fund"),
    "corporate_bond": ("Corporate bond", "Debt Scheme - Corporate Bond Fund"),
    "gilt": ("Gilt", "Debt Scheme - Gilt Fund"),
    "liquid": ("Liquid", "Debt Scheme - Liquid Fund"),
}
EXCLUDE = re.compile(r"idcw|dividend|bonus|segregated|payout|reinvest", re.I)


def scheme_list() -> pd.DataFrame:
    def load():
        text = httpx.get(AMFI_URL, timeout=60, follow_redirects=True).text
        rows, category = [], None
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if ";" not in line:
                m = re.match(r"Open Ended Schemes\((.+)\)", line)
                if m:
                    category = m.group(1).strip()
                continue
            parts = line.split(";")
            if len(parts) < 8 or not parts[0].isdigit() or category is None:
                continue
            code, name, plan, option, nav, date = parts[0], parts[3], parts[4], parts[5], parts[6], parts[7]
            rows.append({"code": int(code), "name": name.strip(), "plan": plan, "option": option, "category": category, "nav": nav, "date": date})
        return pd.DataFrame(rows)

    return data.cached("amfi:schemes", TTL, load)


def _history(code: int) -> pd.Series | None:
    def load():
        try:
            r = httpx.get(MFAPI.format(code=code), timeout=30)
            d = r.json().get("data", [])
        except Exception as exc:
            log.info("mfapi failed for %s: %s", code, exc)
            return None
        if not d:
            return None
        s = pd.Series({pd.to_datetime(x["date"], format="%d-%m-%Y"): float(x["nav"]) for x in d if float(x["nav"]) > 0})
        return s.sort_index()

    return data.cached(f"mfnav:{code}", TTL, load)


def _cagr(s: pd.Series, years: float) -> float | None:
    start = s[s.index <= s.index[-1] - pd.DateOffset(days=round(365.25 * years))]
    if start.empty:
        return None
    return float((s.iloc[-1] / start.iloc[-1]) ** (1 / years) - 1)


def _metrics(s: pd.Series) -> dict | None:
    s3 = s[s.index >= s.index[-1] - pd.DateOffset(years=3)]
    if len(s3) < 500 or s.index[0] > s.index[-1] - pd.DateOffset(years=3):
        return None
    r = s3.pct_change().dropna()
    down = r[r < 0]
    cagr3 = _cagr(s, 3)
    ann_vol = float(r.std() * math.sqrt(252))
    downside = float(down.std() * math.sqrt(252)) if len(down) > 10 else None
    dd = float((s3 / s3.cummax() - 1).min())
    return {
        "return_1y": _cagr(s, 1), "cagr_3y": cagr3, "cagr_5y": _cagr(s, 5),
        "volatility_3y": ann_vol,
        "sharpe_3y": (cagr3 - RISK_FREE) / ann_vol if cagr3 is not None and ann_vol else None,
        "sortino_3y": (cagr3 - RISK_FREE) / downside if cagr3 is not None and downside else None,
        "max_drawdown_3y": dd,
        "nav": float(s.iloc[-1]), "nav_date": str(s.index[-1].date()),
        "history_years": round((s.index[-1] - s.index[0]).days / 365.25, 1),
    }


def rank_category(key: str, progress=lambda m: None) -> dict:
    label, amfi_name = CATEGORIES[key]
    schemes = scheme_list()
    pool = schemes[schemes["category"].str.contains(re.escape(amfi_name), case=False)
                   & schemes["plan"].str.contains("direct", case=False) & schemes["option"].str.contains("growth", case=False)
                   & ~schemes["name"].str.contains(EXCLUDE)]
    progress(f"Fetching NAV history for {len(pool)} {label} funds…")
    with ThreadPoolExecutor(max_workers=8) as ex:
        hist = dict(zip(pool["code"], ex.map(_history, pool["code"])))

    rows = []
    for _, f in pool.iterrows():
        s = hist.get(f["code"])
        if s is None or len(s) < 300:
            continue
        m = _metrics(s)
        if m:
            rows.append({"code": int(f["code"]), "name": f["name"], **m})
    if not rows:
        return {"status": "ready", "category": key, "label": label, "funds": [], "note": "No funds with 3+ years of history."}
    df = pd.DataFrame(rows)

    # Consistency: rolling 1-year returns vs the category median, over the last 3 years.
    navs = pd.DataFrame({r["code"]: hist[r["code"]] for r in rows}).sort_index().ffill()
    navs = navs[navs.index >= navs.index[-1] - pd.DateOffset(years=4)]
    roll = navs / navs.shift(252) - 1
    roll = roll[roll.index >= roll.index[-1] - pd.DateOffset(years=3)].iloc[::5]
    beat = roll.gt(roll.median(axis=1), axis=0).where(roll.notna())
    df["consistency"] = df["code"].map(beat.mean())

    rank = lambda c, asc=False: df[c].rank(pct=True, ascending=not asc) if df[c].notna().any() else pd.Series(0.5, index=df.index)  # noqa: E731
    parts = {"sortino_3y": 0.30, "cagr_3y": 0.20, "consistency": 0.20, "cagr_5y": 0.15, "max_drawdown_3y": 0.15}
    score = pd.Series(0.0, index=df.index)
    weight = pd.Series(0.0, index=df.index)
    for c, w in parts.items():
        r = df[c].rank(pct=True)  # higher is better for all (drawdown is negative, so less negative ranks higher)
        score += r.fillna(0) * w
        weight += r.notna() * w
    df["score"] = (score / weight * 100).round(1)
    df = df.sort_values("score", ascending=False)
    median = {c: float(df[c].median()) for c in ("return_1y", "cagr_3y", "cagr_5y", "max_drawdown_3y", "volatility_3y") if df[c].notna().any()}
    return {
        "status": "ready",
        "category": key,
        "label": label,
        "funds_ranked": int(len(df)),
        "category_median": median,
        "funds": [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()} for r in df.head(15).to_dict("records")],
        "method": "Direct-plan Growth funds with 3+ years of history, ranked within the category: 3y Sortino 30%, 3y CAGR 20%, rolling 1-year consistency vs category median 20%, 5y CAGR 15%, 3y max drawdown 15%.",
        "caveat": "Past fund performance is only weakly predictive. Check the expense ratio and fund manager tenure before investing; in large-cap, a low-cost index fund has historically beaten most active funds.",
    }


def get(key: str) -> dict:
    if key not in CATEGORIES:
        raise KeyError(key)
    return jobs.run(f"funds:{key}", lambda p: rank_category(key, p), TTL)
