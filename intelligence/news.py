"""Collects company headlines from Google News and Yahoo Finance RSS, then scores
each one with the sentiment model and tags corporate events."""

import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import httpx
from sqlalchemy import select

from intelligence import events
from intelligence.db import Company, NewsItem, SessionLocal, now
from intelligence.sentiment import model as sentiment

log = logging.getLogger(__name__)

LOCALE = {"in": ("en-IN", "IN"), "us": ("en-US", "US"), "gb": ("en-GB", "GB"), "au": ("en-AU", "AU"), "ca": ("en-CA", "CA"), "sg": ("en-SG", "SG")}
SUFFIXES = re.compile(
    r"\b(limited|ltd\.?|inc\.?|incorporated|corporation|corp\.?|company|co\.?|plc|holdings?|group|s\.?a\.?|n\.?v\.?|ag|se|the|industries|enterprises?)\b",
    re.IGNORECASE,
)
MAX_AGE = timedelta(days=120)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; OuthitAI/1.0)"}


def short_name(name: str) -> str:
    """'Reliance Industries Limited' -> 'Reliance'; 'Apple Inc.' -> 'Apple'."""
    s = SUFFIXES.sub(" ", name)
    s = re.sub(r"[^\w&' .-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .-")
    words = s.split()
    return " ".join(words[:3]) if words else name


def _key(title: str) -> str:
    t = re.sub(r"\s+-\s+[^-]{2,60}$", "", title.lower())
    return re.sub(r"[^a-z0-9]+", " ", t).strip()[:200]


def _parse_rss(xml: str) -> list[dict]:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        src = it.find("source")
        try:
            published = parsedate_to_datetime(it.findtext("pubDate") or "")
        except (TypeError, ValueError):
            published = now()
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        if title and link:
            items.append({"title": title, "url": link, "source": src.text if src is not None else None, "published_at": published})
    return items


def _relevant(title: str, c: Company) -> bool:
    t = title.lower()
    first = short_name(c.name).split()[0].lower()
    ticker = c.symbol.split(".")[0].lower()
    return (len(first) >= 3 and first in t) or re.search(rf"\b{re.escape(ticker)}\b", t) is not None


def fetch(c: Company) -> list[dict]:
    hl, gl = LOCALE.get(c.market, ("en-US", "US"))
    query = quote_plus(f'"{short_name(c.name)}" stock OR shares')
    urls = [f"https://news.google.com/rss/search?q={query}&hl={hl}&gl={gl}&ceid={gl}:{hl.split('-')[0]}"]
    urls.append(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(c.symbol)}&region={gl}&lang=en-US")
    found: dict[str, dict] = {}
    with httpx.Client(timeout=20, headers=HEADERS, follow_redirects=True) as client:
        for url in urls:
            try:
                r = client.get(url)
                if r.status_code != 200:
                    continue
            except httpx.HTTPError as exc:
                log.info("news fetch failed %s: %s", url, exc)
                continue
            for item in _parse_rss(r.text):
                if now() - item["published_at"] > MAX_AGE or not _relevant(item["title"], c):
                    continue
                found.setdefault(_key(item["title"]), item)
    return [dict(v, key=k) for k, v in found.items()]


def refresh(symbol: str) -> int:
    """Fetch, score and store new headlines for one company. Returns how many were new."""
    with SessionLocal() as db:
        c = db.get(Company, symbol)
        if c is None:
            return 0
        items = fetch(c)
        known = set(db.scalars(select(NewsItem.key).where(NewsItem.symbol == symbol)).all())
        fresh = [i for i in items if i["key"] not in known]
        for item, s in zip(fresh, sentiment.score([i["title"] for i in fresh])):
            db.add(NewsItem(
                symbol=symbol, key=item["key"], title=item["title"], source=item["source"], url=item["url"],
                published_at=item["published_at"], sentiment=s["sentiment"], label=s["label"], confidence=s["confidence"],
                events=events.tag(item["title"]),
            ))
        c.news_checked_at = now()
        db.commit()
        return len(fresh)


def recent(symbol: str, days: int = 120) -> list[NewsItem]:
    with SessionLocal() as db:
        return list(db.scalars(
            select(NewsItem).where(NewsItem.symbol == symbol, NewsItem.published_at >= now() - timedelta(days=days)).order_by(NewsItem.published_at.desc())
        ).all())


def refresh_many(symbols: list[str], pause: float = 0.8) -> int:
    total = 0
    for s in symbols:
        try:
            total += refresh(s)
        except Exception as exc:  # one bad feed must not stop the cycle
            log.warning("news refresh failed for %s: %s", s, exc)
        time.sleep(pause)
    return total
