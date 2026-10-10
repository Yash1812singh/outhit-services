"""India governance signals from NSE disclosures.

- Promoter pledging (quarterly shareholding filings): the share of promoter
  holdings pledged as loan collateral. Indian studies link heavy pledging to
  higher crash risk and weaker performance; falling prices can force lenders
  to sell (as in Satyam).
- Insider trades (SEBI PIT disclosures): market purchases by promoters,
  promoter group and directors in the last 180 days (a confidence signal), and
  pledge creations / invocations (a warning sign).
"""

import logging
import re
from datetime import datetime, timedelta

import httpx
from urllib.parse import quote

from research import data

log = logging.getLogger(__name__)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/",
}
INSIDERS = re.compile(r"promoter|director|key managerial", re.I)


def _norm(name: str) -> str:
    n = re.sub(r"\b(limited|ltd|the|co|company|corporation|corp|india|of)\b", " ", name.lower())
    return re.sub(r"[^a-z0-9]+", "", n)


def _nse(path: str) -> dict:
    with httpx.Client(headers=HEADERS, timeout=40, follow_redirects=True) as c:
        r = c.get(f"https://www.nseindia.com/api/{path}")
        r.raise_for_status()
        return r.json()


def pledges() -> dict[str, dict]:
    """Normalised company name -> latest promoter pledge data."""

    def load():
        rows = _nse("corporate-pledgedata?index=equities").get("data", [])
        out = {}
        for r in rows:
            try:
                out[_norm(r["comName"])] = {
                    "company": r["comName"],
                    "pledged_pct_of_promoter": float(r.get("percSharesPledged") or 0),
                    "promoter_holding_pct": float((r.get("percPromoterHolding") or "0").strip() or 0),
                    "as_of": r.get("shp"),
                }
            except ValueError:
                continue
        return out

    try:
        return data.cached("nse:pledges", 24 * 3600, load)
    except Exception as exc:
        log.warning("pledge data unavailable: %s", exc)
        return {}


def insider_activity(symbol: str, company: str) -> dict | None:
    base = symbol.split(".")[0]

    def load():
        try:
            d = _nse(f"corporates-pit?index=equities&symbol={quote(base)}&issuer={quote(company)}")
        except Exception as exc:
            log.info("PIT failed for %s: %s", symbol, exc)
            return None
        cutoff = datetime.now() - timedelta(days=180)
        buy = sell = 0.0
        invoked = created = 0
        for r in d.get("data", []):
            try:
                when = datetime.strptime(r.get("acqfromDt") or r.get("intimDt"), "%d-%b-%Y")
            except (TypeError, ValueError):
                continue
            if when < cutoff:
                continue
            mode, kind, who = r.get("acqMode") or "", r.get("tdpTransactionType") or "", r.get("personCategory") or ""
            value = float(r.get("secVal") or 0)
            if "Invoke" in kind or "invoc" in mode.lower():
                invoked += 1
            elif mode == "Pledge Creation":
                created += 1
            elif INSIDERS.search(who) and mode == "Market Purchase" and kind == "Buy":
                buy += value
            elif INSIDERS.search(who) and mode == "Market Sale" and kind == "Sell":
                sell += value
        return {"insider_buy_value_180d": buy, "insider_sell_value_180d": sell, "pledges_created_180d": created, "pledges_invoked_180d": invoked}

    return data.cached(f"pit:{symbol}", 24 * 3600, load)


def assess(symbol: str, company: str, market: str) -> dict:
    """Governance score 0..1 (0.5 neutral) with strengths and red flags. India only; neutral elsewhere."""
    if market != "in":
        return {"score": 0.5, "strengths": [], "flags": [], "available": False}
    p = pledges().get(_norm(company))
    act = insider_activity(symbol, company) or {}
    score, strengths, flags = 0.5, [], []
    if p:
        pct = p["pledged_pct_of_promoter"]
        if pct >= 50:
            score -= 0.35
            flags.append(f"Promoters have pledged {pct:.0f}% of their shares")
        elif pct >= 20:
            score -= 0.2
            flags.append(f"Promoters have pledged {pct:.0f}% of their shares")
        elif pct == 0 and p["promoter_holding_pct"] > 0:
            score += 0.1
            strengths.append("No promoter shares pledged")
    if act.get("pledges_invoked_180d"):
        score -= 0.3
        flags.append("Lenders invoked pledged promoter shares in the last 6 months")
    elif act.get("pledges_created_180d"):
        score -= 0.1
    net = act.get("insider_buy_value_180d", 0) - act.get("insider_sell_value_180d", 0)
    if act.get("insider_buy_value_180d", 0) > 0 and net > 0:
        score += 0.2
        strengths.append("Promoters/directors bought shares on the market in the last 6 months")
    elif net < 0 and abs(net) > 5e7:
        score -= 0.05
    return {"score": max(0.0, min(1.0, score)), "strengths": strengths, "flags": flags, "available": True, "pledge": p, "insider": act}
