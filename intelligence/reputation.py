"""Company reputation score (0-100).

Built from three independent sources:
- News: average tone of coverage over the last ~4 months, and controversies
  (fraud, regulator probes, lawsuits, defaults, management exits, recalls),
  with older items counting less (45-day half-life).
- Governance: ISS governance risk ratings published via Yahoo Finance
  (audit, board, compensation, shareholder rights; 1 = low risk, 10 = high).
- Analyst consensus (1 = Strong Buy ... 5 = Sell).
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import yfinance as yf

from intelligence import news
from intelligence.db import Company, SessionLocal, aware, now

log = logging.getLogger(__name__)

GOV_KEYS = {"overallRisk": "Overall", "auditRisk": "Audit", "boardRisk": "Board", "compensationRisk": "Pay", "shareHolderRightsRisk": "Shareholder rights"}
GOV_TTL = timedelta(days=7)
HALF_LIFE_DAYS = 45


def governance(symbol: str) -> dict | None:
    with SessionLocal() as db:
        c = db.get(Company, symbol)
        if c is None:
            return None
        g = c.governance or {}
        fetched = g.get("fetched_at")
        if fetched and now() - aware(datetime.fromisoformat(fetched)) < GOV_TTL:
            return g
        try:
            info = yf.Ticker(symbol).info or {}
        except Exception as exc:
            log.info("governance lookup failed for %s: %s", symbol, exc)
            return g or None
        g = {k: info.get(k) for k in GOV_KEYS} | {"fetched_at": now().isoformat()}
        c.governance = g
        db.commit()
        return g


def _analyst(rating: str | None) -> float | None:
    if not rating:
        return None
    try:
        return float(rating.split("-")[0].strip())
    except ValueError:
        return None


def score(symbol: str) -> dict:
    items = news.recent(symbol, days=120)
    t = now()
    weights = np.array([0.5 ** ((t - aware(i.published_at)).days / HALF_LIFE_DAYS) for i in items]) if items else np.array([])
    tone = float(np.average([i.sentiment for i in items], weights=weights)) if items else None

    controversies = []
    penalty = 0.0
    for i, w in zip(items, weights):
        for e in i.events or []:
            if e.get("controversy"):
                penalty += abs(e["impact"]) * w
                controversies.append({"label": e["label"], "title": i.title, "url": i.url, "published_at": aware(i.published_at).isoformat()})

    with SessionLocal() as db:
        c = db.get(Company, symbol)
        rating = c.analyst_rating if c else None
    gov = governance(symbol) or {}
    overall = gov.get("overallRisk")
    analyst = _analyst(rating)

    parts = []
    value = 60.0
    if tone is not None:
        value += 20 * tone
        parts.append({"factor": "News tone", "effect": round(20 * tone, 1), "detail": f"Average tone of {len(items)} articles: {tone:+.2f} (-1 very negative, +1 very positive)"})
    pen = min(40.0, 12 * penalty)
    if controversies:
        value -= pen
        parts.append({"factor": "Controversies", "effect": round(-pen, 1), "detail": f"{len(controversies)} controversy headline(s) in the last 4 months"})
    if overall is not None:
        g_eff = (5.5 - overall) * 3
        value += g_eff
        parts.append({"factor": "Governance (ISS)", "effect": round(g_eff, 1), "detail": f"Overall governance risk {overall}/10 (lower is better)"})
    if analyst is not None:
        a_eff = (3 - analyst) * 4
        value += a_eff
        parts.append({"factor": "Analyst consensus", "effect": round(a_eff, 1), "detail": f"Rating {rating}"})

    value = float(np.clip(value, 0, 100))
    grade = "Excellent" if value >= 80 else "Good" if value >= 65 else "Mixed" if value >= 45 else "Poor" if value >= 30 else "Very poor"
    return {
        "symbol": symbol,
        "score": round(value, 1),
        "grade": grade,
        "articles": len(items),
        "news_tone": tone,
        "controversies": controversies[:10],
        "governance": {GOV_KEYS[k]: gov.get(k) for k in GOV_KEYS if gov.get(k) is not None},
        "analyst_rating": rating,
        "breakdown": parts,
        "coverage_note": None if len(items) >= 10 else "Limited news coverage so far; the score firms up as more articles are collected.",
    }
