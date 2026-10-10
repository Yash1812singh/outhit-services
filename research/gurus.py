"""Famous investors' real portfolios, from the 13F filings US fund managers must make every quarter.

A 13F lists a manager's US-listed stock holdings at quarter end and is published up to 45 days later,
so copying it always means buying what they owned 1.5-4.5 months ago. The backtest is honest about
that: each filing is "bought" the trading day after it became public, and held until the next one.

Data: SEC EDGAR (filings, holdings) and OpenFIGI (CUSIP -> ticker). Both are free and rate-limited,
and results are cached on disk (filings never change once published).
"""

import json
import logging
import threading
import time
import xml.etree.ElementTree as ET

import httpx
import pandas as pd

from research import data
from research.config import get_settings
from research.strategies import Context, Strategy

log = logging.getLogger(__name__)

# key: (person, fund, SEC CIK, style, one-liner)
GURUS: dict[str, tuple[str, str, int, str, str]] = {
    "buffett_13f": ("Warren Buffett", "Berkshire Hathaway", 1067983, "Quality value",
                    "The world's most famous investor: a concentrated book of wonderful businesses held for years."),
    "ackman_13f": ("Bill Ackman", "Pershing Square", 1336528, "Activist, concentrated",
                   "8-12 high-quality, simple, predictable businesses, often pushing management to improve."),
    "li_lu_13f": ("Li Lu", "Himalaya Capital", 1709323, "Deep value, concentrated",
                  "Charlie Munger's chosen manager: a handful of deeply researched, undervalued compounders."),
    "klarman_13f": ("Seth Klarman", "Baupost Group", 1061768, "Margin-of-safety value",
                    "Author of “Margin of Safety”: patient value investing with a big cash cushion."),
    "druckenmiller_13f": ("Stanley Druckenmiller", "Duquesne Family Office", 1536411, "Macro + growth",
                          "Soros's former partner with no losing year in 30: big bets on the strongest trends."),
    "tepper_13f": ("David Tepper", "Appaloosa", 1656456, "Contrarian, macro",
                   "Famous for buying banks at the 2009 bottom; bold, contrarian and macro-aware."),
    "pabrai_13f": ("Mohnish Pabrai", "Dalal Street LLC", 1549575, "Cloned deep value",
                   "“Heads I win, tails I don't lose much”: few, very cheap bets, openly cloning other greats."),
    "terry_smith_13f": ("Terry Smith", "Fundsmith", 1569205, "Quality growth",
                        "“Buy good companies, don't overpay, do nothing”: high-return-on-capital compounders."),
}
MIN_WEIGHT = 0.005  # positions below 0.5% are left out (and the rest re-weighted)
FILINGS_TTL = 12 * 3600
_sec_lock, _figi_lock = threading.Lock(), threading.Lock()
_last_sec = [0.0]
_last_figi = [0.0]
CUSIP_FILE = data.CACHE_DIR / "cusip_tickers.json"


def _sec(url: str) -> httpx.Response:
    with _sec_lock:  # SEC allows 10 requests/second
        wait = 0.15 - (time.time() - _last_sec[0])
        if wait > 0:
            time.sleep(wait)
        _last_sec[0] = time.time()
    r = httpx.get(url, headers={"User-Agent": get_settings().sec_user_agent}, timeout=30, follow_redirects=True)
    r.raise_for_status()
    return r


def _filing_list(cik: int) -> list[dict]:
    def load():
        rec = _sec(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()["filings"]["recent"]
        rows = [{"accession": rec["accessionNumber"][i], "filed": rec["filingDate"][i], "period": rec["reportDate"][i]}
                for i, f in enumerate(rec["form"]) if f == "13F-HR"]
        latest: dict[str, dict] = {}
        for r in sorted(rows, key=lambda r: r["filed"]):
            latest[r["period"]] = r  # a re-filed report for the same quarter replaces the earlier one
        return sorted(latest.values(), key=lambda r: r["filed"], reverse=True)

    return data.cached(f"13f:list:{cik}", FILINGS_TTL, load)


def _holdings(cik: int, accession: str) -> list[dict]:
    """Aggregated holdings of one filing: [{cusip, issuer, value, shares}], options excluded."""

    def load():
        folder = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}"
        items = _sec(f"{folder}/index.json").json()["directory"]["item"]
        xml_name = next(i["name"] for i in items if i["name"].lower().endswith(".xml") and "primary_doc" not in i["name"].lower())
        root = ET.fromstring(_sec(f"{folder}/{xml_name}").content)
        ns = {"n": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {"n": ""}
        agg: dict[str, dict] = {}
        for row in root.findall("n:infoTable", ns) if ns["n"] else root.findall("infoTable"):
            def val(tag: str) -> str:
                el = row.find(f".//n:{tag}", ns) if ns["n"] else row.find(f".//{tag}")
                return (el.text or "").strip() if el is not None else ""
            if val("putCall"):
                continue
            cusip = val("cusip").upper()
            h = agg.setdefault(cusip, {"cusip": cusip, "issuer": val("nameOfIssuer").title(), "value": 0.0, "shares": 0.0})
            h["value"] += float(val("value") or 0)
            if val("sshPrnamtType") == "SH":
                h["shares"] += float(val("sshPrnamt") or 0)
        return sorted(agg.values(), key=lambda h: -h["value"])

    return data.cached(f"13f:holdings:{accession}", 365 * 24 * 3600, load)


def _cusip_tickers(cusips: list[str]) -> dict[str, str | None]:
    """CUSIP -> Yahoo ticker via OpenFIGI (25 requests/min, 10 CUSIPs each without an API key)."""
    with _figi_lock:
        known: dict[str, str | None] = json.loads(CUSIP_FILE.read_text()) if CUSIP_FILE.exists() else {}
        todo = [c for c in dict.fromkeys(cusips) if c not in known]
        for i in range(0, len(todo), 10):
            chunk = todo[i : i + 10]
            wait = 2.5 - (time.time() - _last_figi[0])
            if wait > 0:
                time.sleep(wait)
            _last_figi[0] = time.time()
            try:
                r = httpx.post("https://api.openfigi.com/v3/mapping", timeout=30,
                               json=[{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in chunk])
                if r.status_code == 429:
                    time.sleep(60)
                    r = httpx.post("https://api.openfigi.com/v3/mapping", timeout=30,
                                   json=[{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in chunk])
                r.raise_for_status()
                for c, res in zip(chunk, r.json()):
                    hits = [d for d in res.get("data", []) if d.get("marketSector") == "Equity" and d.get("ticker")]
                    known[c] = hits[0]["ticker"].replace("/", "-").replace(" ", "-") if hits else None
            except Exception as exc:
                log.warning("OpenFIGI lookup failed: %s", exc)
                break  # try the rest next time
            CUSIP_FILE.write_text(json.dumps(known))
        return {c: known.get(c) for c in cusips}


def filings(key: str, years: int = 5, progress=lambda m: None) -> list[dict]:
    """Filings newest first: {filed, period, accession, holdings: [{ticker, issuer, weight, value, shares}], mapped_weight}."""
    person, fund, cik, *_ = GURUS[key]
    progress(f"Reading {fund}'s 13F filings from SEC EDGAR…")
    cutoff = (pd.Timestamp.today() - pd.DateOffset(years=years, months=4)).strftime("%Y-%m-%d")
    rows = [r for r in _filing_list(cik) if r["filed"] >= cutoff]
    out = []
    for n, r in enumerate(rows, 1):
        progress(f"Reading {fund}'s holdings: filing {n} of {len(rows)}…")
        out.append(r | {"holdings": _holdings(cik, r["accession"])})
    cusips = [h["cusip"] for f in out for h in f["holdings"]]
    progress(f"Matching {len(set(cusips))} securities to stock tickers…")
    tickers = _cusip_tickers(cusips)
    for f in out:
        total = sum(h["value"] for h in f["holdings"]) or 1
        for h in f["holdings"]:
            h["ticker"] = tickers.get(h["cusip"])
            h["weight"] = h["value"] / total
        f["mapped_weight"] = sum(h["weight"] for h in f["holdings"] if h["ticker"])
    return out


def _weights(filing: dict, priced: set[str] | None = None) -> pd.Series:
    w: dict[str, float] = {}
    for h in filing["holdings"]:
        if h["ticker"] and (priced is None or h["ticker"] in priced):
            w[h["ticker"]] = w.get(h["ticker"], 0) + h["weight"]
    s = pd.Series(w, dtype=float)
    s = s[s >= MIN_WEIGHT]
    return s / s.sum() if s.sum() > 0 else s


def context(key: str, years: int, progress=lambda m: None) -> tuple[Context, list[dict]]:
    fl = filings(key, years, progress)
    if not fl:
        raise RuntimeError("No 13F filings found for this investor in the period.")
    tickers = sorted({h["ticker"] for f in fl for h in f["holdings"] if h["ticker"] and h["weight"] >= MIN_WEIGHT / 2})
    progress(f"Loading prices for {len(tickers)} stocks…")
    closes, _ = data.daily(tickers + ["SPY"], period="10y")
    names = {h["ticker"]: h["issuer"] for f in reversed(fl) for h in f["holdings"] if h["ticker"]}
    ctx = Context("us", closes.drop(columns=["SPY"], errors="ignore"), {}, names, {}, closes["SPY"] if "SPY" in closes else closes.mean(axis=1))
    return ctx, fl


def strategy(key: str, fl: list[dict] | None = None) -> Strategy:
    person, fund, cik, style, line = GURUS[key]

    def select(ctx: Context, t, p):
        live = [f for f in (fl or []) if pd.Timestamp(f["filed"]) <= t]
        if not live:
            return pd.Series(dtype=float)
        priced = set(ctx.closes.columns[ctx.closes.loc[t].notna()])
        return _weights(max(live, key=lambda f: f["filed"]), priced)

    return Strategy(
        key, f"{fund} portfolio", person, "guru", style, line,
        [f"Copy {fund}'s latest 13F filing (its US stock holdings), weighted by position size",
         "Buy the day after each filing is published (up to 45 days after the quarter ends)",
         "Positions under 0.5% are skipped; options and bonds are not included"],
        "F", select, markets=("us",), source=f"SEC EDGAR 13F filings, CIK {cik}",
    )


def changes(new: dict, old: dict | None) -> list[dict]:
    """What changed between two filings, by share count: new, added, trimmed, sold."""
    if not old:
        return []
    prev = {h["cusip"]: h for h in old["holdings"]}
    cur = {h["cusip"]: h for h in new["holdings"]}
    out = []
    for c, h in cur.items():
        p = prev.get(c)
        if p is None:
            out.append({"action": "new", "issuer": h["issuer"], "ticker": h["ticker"], "weight": h["weight"], "change": None})
        elif p["shares"] and h["shares"] and abs(h["shares"] / p["shares"] - 1) >= 0.05:
            out.append({"action": "added" if h["shares"] > p["shares"] else "trimmed", "issuer": h["issuer"], "ticker": h["ticker"],
                        "weight": h["weight"], "change": h["shares"] / p["shares"] - 1})
    for c, p in prev.items():
        if c not in cur:
            out.append({"action": "sold", "issuer": p["issuer"], "ticker": p["ticker"], "weight": 0.0, "change": -1.0})
    order = {"new": 0, "added": 1, "trimmed": 2, "sold": 3}
    return sorted(out, key=lambda x: (order[x["action"]], -(x["weight"] or 0)))
