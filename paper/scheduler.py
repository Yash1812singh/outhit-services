"""Runs due copy-trade rebalances every minute and records end-of-day values hourly."""

import logging
import threading
import time

from paper import trading

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


def start() -> None:
    _stop.clear()
    _every(60, trading.due_rebalances, "copy-rebalance", initial_delay=20)
    _every(3600, trading.snapshot_all, "snapshots", initial_delay=120)


def stop() -> None:
    _stop.set()
