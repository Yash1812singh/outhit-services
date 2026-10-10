"""Keeps the heavy research warm for the tracked markets and scores past picks hourly."""

import logging
import threading
import time

from research import recommend, rockets, tournament
from research.config import get_settings

log = logging.getLogger(__name__)
_stop = threading.Event()


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


def _research() -> None:
    for m in get_settings().tracked_market_list:
        tournament.get(m)
        for h in ("long_term", "short_term"):
            recommend.get(m, h)
        for kind in ("rocket", "grower"):
            rockets.get(m, kind)


def start() -> None:
    _stop.clear()
    _every(1800, _research, "research", initial_delay=45)
    _every(3600, recommend.evaluate, "picks-ledger", initial_delay=180)


def stop() -> None:
    _stop.set()
