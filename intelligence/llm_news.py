"""LLM news reading (Groq).

Large language models predict next-day stock reactions to headlines where
older text models do not (Lopez-Lira & Tang, "Can ChatGPT Forecast Stock Price
Movements?"). Headlines are sent in batches of 20 with the company name, and the
model returns a score from -1 (bad news for the stock) to +1 (good news). Scores
are stored next to the trained model's, so both can be compared on live results.

Groq's free tier allows ~8,000 tokens per minute; a batch costs ~1,500, so the
loop paces itself and only scores the newest headlines first.
"""

import json
import logging
import time
from datetime import timedelta

import groq
from sqlalchemy import select

from intelligence.config import get_settings
from intelligence.db import Company, NewsItem, SessionLocal, now

log = logging.getLogger(__name__)
BATCH = 20
MAX_BATCHES_PER_RUN = 4
PAUSE = 15  # seconds between batches, to stay under the per-minute token limit

PROMPT = """You are an equity analyst. For each numbered headline, judge how the news is likely to move the named company's share price over the next few trading days.
Score from -1 (clearly bad for the stock) to +1 (clearly good); use 0 for routine or irrelevant news. Consider whether the news is already expected (results dates, generic market wraps are 0).
Reply only with JSON: {"scores":[{"i":1,"score":0.0}]}"""


def enabled() -> bool:
    return bool(get_settings().groq_api_key)


def _client() -> groq.Groq:
    return groq.Groq(api_key=get_settings().groq_api_key, max_retries=3, timeout=60)


def score_batch(items: list[tuple[str, str]]) -> list[float | None]:
    """items: (company name, headline). Returns one score per item (None if missing)."""
    lines = "\n".join(f"{i + 1}. [{company}] {title}" for i, (company, title) in enumerate(items))
    resp = _client().chat.completions.create(
        model=get_settings().groq_model,
        messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": lines}],
        temperature=0,
        max_completion_tokens=1500,
        response_format={"type": "json_object"},
    )
    try:
        scores = json.loads(resp.choices[0].message.content or "{}").get("scores", [])
    except json.JSONDecodeError:
        return [None] * len(items)
    by_i = {int(s["i"]): float(s["score"]) for s in scores if isinstance(s, dict) and "i" in s and "score" in s}
    return [max(-1.0, min(1.0, by_i[i + 1])) if (i + 1) in by_i else None for i in range(len(items))]


def run(max_batches: int = MAX_BATCHES_PER_RUN) -> int:
    """Score the newest unscored headlines from the last week."""
    if not enabled():
        return 0
    done = 0
    for b in range(max_batches):
        with SessionLocal() as db:
            rows = db.execute(
                select(NewsItem, Company.name).join(Company, Company.symbol == NewsItem.symbol)
                .where(NewsItem.llm_score.is_(None), NewsItem.published_at >= now() - timedelta(days=7))
                .order_by(NewsItem.published_at.desc()).limit(BATCH)
            ).all()
        if not rows:
            break
        try:
            scores = score_batch([(name, n.title) for n, name in rows])
        except groq.RateLimitError:
            log.info("Groq rate limit reached; will continue next run")
            break
        except Exception as exc:
            log.warning("LLM scoring failed: %s", exc)
            break
        with SessionLocal() as db:
            for (n, _), sc in zip(rows, scores):
                item = db.get(NewsItem, n.id)
                item.llm_score = sc if sc is not None else item.sentiment  # fall back so it isn't retried forever
                item.llm_scored = sc is not None
            db.commit()
        done += sum(s is not None for s in scores)
        if b < max_batches - 1:
            time.sleep(PAUSE)
    return done
