"""The Outhit AI outlook: one score per company that fuses every signal and
updates as prices and news change.

Components (each scaled to -1..+1):

  quant       The main API's machine-learning model (probability of beating the
              market over ~1 month). Its weight depends on how well it tested
              in that market: 0.40 meaningful, 0.30 modest, 0.10 no edge, 0 if
              worse than chance. Falls back to the technical trend signal (0.20)
              for companies outside the model's universe.
  news        Tone of the last 7 days of headlines from the sentiment model,
              recency-weighted and shrunk toward 0 when there are few articles. 0.25
  events      Specific events in the last 14 days (results beats/misses,
              guidance changes, up/downgrades, probes, defaults...). 0.15
  reputation  Long-run reputation score (news tone, controversies, governance,
              analysts). 0.10
  momentum    Today's live price action vs. the volume-weighted average price. 0.10

Score = 100 x sum(weight x value) / (sum of all maximum weights). A component that
is missing (no recent news, model with no edge) therefore pulls the score toward
neutral rather than letting the remaining signals speak louder than they deserve.

The weights are an informed judgement, not fitted: there is no free archive of
historical news to fit them on. That is why every outlook is logged in the
prediction ledger and scored against what actually happens (see ledger.py).
"""

import logging
import threading
from datetime import timedelta

import numpy as np

from intelligence import backend, live, news, reputation
from intelligence.db import Company, SessionLocal, aware, now

log = logging.getLogger(__name__)

QUANT_WEIGHT = {"meaningful": 0.40, "modest": 0.30, "none": 0.10, "negative": 0.0}
TECH_WEIGHT = 0.20
WEIGHTS = {"news": 0.25, "events": 0.15, "reputation": 0.10, "momentum": 0.10}
MAX_WEIGHT = QUANT_WEIGHT["meaningful"] + sum(WEIGHTS.values())  # = 1.0
NEWS_HALF_LIFE_DAYS = 2.5
EVENT_HALF_LIFE_DAYS = 5

_latest: dict[str, dict] = {}
_lock = threading.Lock()


def label(score: float) -> str:
    if score >= 25:
        return "Bullish"
    if score >= 10:
        return "Mildly bullish"
    if score > -10:
        return "Neutral"
    if score > -25:
        return "Mildly bearish"
    return "Bearish"


def _quant(c: Company) -> dict | None:
    q = backend.quant(c.market)
    if q:
        row = next((r for r in q["all"] if r["symbol"] == c.symbol), None)
        if row:
            level = q["validation"]["verdict_level"]
            p = row["probability_outperform"]
            return {
                "key": "quant", "label": "Price-pattern ML model", "weight": QUANT_WEIGHT[level],
                "value": float(np.clip((p - 0.5) / 0.15, -1, 1)),
                "detail": f"{p:.0%} chance of beating the market over 1 month (model tested '{level}' in this market).",
            }
    t = backend.technical(c.symbol)
    if t and t.get("score") is not None:
        return {
            "key": "quant", "label": "Technical trend", "weight": TECH_WEIGHT,
            "value": float(np.clip(t["score"] / 100, -1, 1)),
            "detail": f"Trend/momentum signal {t['score']:+.0f} ({t['signal']}, {t.get('confidence', '?')} confidence).",
        }
    return None


def _news(items: list) -> dict | None:
    t = now()
    week = [i for i in items if t - aware(i.published_at) <= timedelta(days=7)]
    if not week:
        return None
    w = np.array([0.5 ** ((t - aware(i.published_at)).total_seconds() / 86400 / NEWS_HALF_LIFE_DAYS) * i.confidence for i in week])
    # Prefer the LLM's reading of each headline (better at context) and fall back to the trained model.
    llm_read = sum(1 for i in week if i.llm_scored)
    tone = float(np.average([i.llm_score if i.llm_scored else i.sentiment for i in week], weights=w))
    shrunk = tone * len(week) / (len(week) + 4)
    pos = sum(1 for i in week if i.label == "bullish")
    neg = sum(1 for i in week if i.label == "bearish")
    return {
        "key": "news", "label": "News sentiment (7 days)", "weight": WEIGHTS["news"], "value": float(np.clip(shrunk * 1.5, -1, 1)),
        "detail": f"{len(week)} articles ({llm_read} read by the LLM): {pos} positive, {neg} negative; weighted tone {tone:+.2f}.",
    }


def _events(items: list) -> dict | None:
    t = now()
    total, seen = 0.0, []
    for i in items:
        age = (t - aware(i.published_at)).total_seconds() / 86400
        if age > 14:
            continue
        for e in i.events or []:
            total += e["impact"] * 0.5 ** (age / EVENT_HALF_LIFE_DAYS)
            if e["label"] not in seen:
                seen.append(e["label"])
    if not seen:
        return None
    return {
        "key": "events", "label": "Corporate events (14 days)", "weight": WEIGHTS["events"], "value": float(np.tanh(total)),
        "detail": "Detected: " + ", ".join(seen[:5]) + ".",
    }


def _momentum(q: dict | None) -> dict | None:
    if not q or not q.get("vwap") or not q.get("trading"):
        return None
    dev = q["price"] / q["vwap"] - 1
    return {
        "key": "momentum", "label": "Today's price action", "weight": WEIGHTS["momentum"], "value": float(np.clip(dev / 0.01, -1, 1)),
        "detail": f"{q['change_pct']:+.2f}% today; price is {dev:+.2%} vs. today's volume-weighted average.",
    }


def compute(symbol: str) -> dict | None:
    with SessionLocal() as db:
        c = db.get(Company, symbol)
    if c is None:
        return None
    items = news.recent(symbol, days=14)
    rep = reputation.score(symbol)
    quote = live.get([symbol]).get(symbol)

    comps = [x for x in (_quant(c), _news(items), _events(items), _momentum(quote)) if x]
    comps.append({
        "key": "reputation", "label": "Reputation", "weight": WEIGHTS["reputation"], "value": (rep["score"] - 50) / 50,
        "detail": f"{rep['grade']} ({rep['score']:.0f}/100).",
    })
    # Divide by the weight a fully-covered company would have, not by what's available:
    # a missing signal counts as neutral instead of inflating the ones that remain.
    total_w = sum(x["weight"] for x in comps)
    score = 100 * sum(x["weight"] * x["value"] for x in comps) / MAX_WEIGHT
    for x in comps:
        x["contribution"] = round(100 * x["weight"] * x["value"] / MAX_WEIGHT, 1)

    direction = np.sign(score)
    agree = sum(x["weight"] for x in comps if np.sign(x["value"]) == direction and direction != 0) / total_w if total_w else 0
    coverage = total_w / MAX_WEIGHT
    if abs(score) >= 10 and agree >= 0.7 and coverage >= 0.6:
        confidence = "High"
    elif abs(score) >= 10 and agree >= 0.55:
        confidence = "Medium"
    else:
        confidence = "Low"

    with _lock:
        prev = _latest.get(symbol)
    result = {
        "symbol": symbol,
        "name": c.name,
        "market": c.market,
        "sector": c.sector,
        "currency": c.currency,
        "score": round(float(score), 1),
        "label": label(score),
        "confidence": confidence,
        "agreement": round(float(agree), 2),
        "coverage": round(float(coverage), 2),
        "components": sorted(comps, key=lambda x: abs(x["contribution"]), reverse=True),
        "reputation": {"score": rep["score"], "grade": rep["grade"]},
        "news_7d": sum(1 for i in items if now() - aware(i.published_at) <= timedelta(days=7)),
        "price": quote["price"] if quote else None,
        "change_pct": quote["change_pct"] if quote else None,
        "computed_at": now().isoformat(),
        "previous_score": prev["score"] if prev else None,
    }
    with _lock:
        _latest[symbol] = result
    return result


def latest(symbols: list[str] | None = None) -> dict[str, dict]:
    with _lock:
        return dict(_latest) if symbols is None else {s: _latest[s] for s in symbols if s in _latest}
