"""Background loops: live prices every minute, news every 10 minutes, outlooks every minute,
LLM headline scoring every 3 minutes, ledger scoring hourly, universe refresh every 6 hours."""

import logging
import threading
import time
from datetime import timedelta

from sqlalchemy import select

from intelligence import ledger, live, llm_news, news, outlook, registry
from intelligence.config import get_settings
from intelligence.db import Company, SessionLocal, aware, now

log = logging.getLogger(__name__)
_stop = threading.Event()
NEWS_BATCH = 50  # ~2 s per company; keeps ~170 tracked companies refreshed about every 40 minutes


def _every(seconds: int, fn, name: str, initial_delay: float = 0) -> None:
    def loop():
        if _stop.wait(initial_delay):
            return
        while not _stop.is_set():
            t = time.time()
            try:
                fn()
            except Exception:
                log.exception("%s failed", name)
            _stop.wait(max(5, seconds - (time.time() - t)))

    threading.Thread(target=loop, name=name, daemon=True).start()


def _prices() -> None:
    log.info("Live prices updated for %d symbols", live.poll())


def _outlooks() -> None:
    for s in registry.active_symbols():
        o = outlook.compute(s)
        if o:
            ledger.record(o)


def _news() -> None:
    active = set(registry.active_symbols())
    with SessionLocal() as db:
        rows = db.scalars(select(Company).where(Company.symbol.in_(active))).all()
    stale = sorted(rows, key=lambda c: c.news_checked_at.timestamp() if c.news_checked_at else 0)
    due = [c.symbol for c in stale if not c.news_checked_at or now() - aware(c.news_checked_at) > timedelta(minutes=20)]
    if due:
        added = news.refresh_many(due[:NEWS_BATCH])
        log.info("News: checked %d companies, %d new articles", min(len(due), NEWS_BATCH), added)


def start() -> None:
    s = get_settings()
    _stop.clear()
    _every(6 * 3600, lambda: log.info("Tracking %d companies", registry.bootstrap()), "bootstrap")
    _every(s.price_interval_seconds, _prices, "prices", initial_delay=10)
    _every(s.news_interval_seconds, _news, "news", initial_delay=15)
    _every(s.outlook_interval_seconds, _outlooks, "outlooks", initial_delay=30)
    _every(3600, ledger.evaluate, "ledger", initial_delay=120)
    _every(180, llm_news.run, "llm-news", initial_delay=60)


def stop() -> None:
    _stop.set()
