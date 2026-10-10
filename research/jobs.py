"""Background jobs with progress messages and on-disk results.

Expensive research (tournaments, fund rankings) runs in a thread; callers get
{"status": "running", "message": ...} until the result is ready, and finished
results survive restarts.
"""

import hashlib
import logging
import pickle
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from research.data import CACHE_DIR

log = logging.getLogger(__name__)


@dataclass
class Job:
    status: str = "running"
    message: str = "Starting…"
    started: float = field(default_factory=time.time)
    finished: float | None = None
    error: str | None = None


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def _path(key: str) -> Path:
    return CACHE_DIR / f"job-{hashlib.md5(key.encode()).hexdigest()}.pkl"


def _load(key: str, ttl: int):
    p = _path(key)
    if p.exists() and time.time() - p.stat().st_mtime < ttl:
        try:
            return pickle.loads(p.read_bytes())
        except Exception:
            p.unlink(missing_ok=True)
    return None


def run(key: str, fn, ttl: int) -> dict:
    """Return the cached result for `key`, or start `fn(progress)` in the background."""
    result = _load(key, ttl)
    if result is not None:
        return result
    with _lock:
        job = _jobs.get(key)
        if job and job.status == "error" and time.time() - (job.finished or 0) > 300:
            job = None  # allow a retry after a failure
        if job is None or job.status == "done":
            job = _jobs[key] = Job()

            def work():
                def progress(msg: str) -> None:
                    job.message = msg

                try:
                    out = fn(progress)
                    if isinstance(out, dict) and out.get("status") not in (None, "ready"):
                        job.status = "done"  # a dependency wasn't ready: don't cache, retry on the next call
                    else:
                        _path(key).write_bytes(pickle.dumps(out))
                        job.status = "done"
                except Exception as exc:
                    log.exception("job %s failed", key)
                    job.status, job.error = "error", str(exc)
                job.finished = time.time()

            threading.Thread(target=work, daemon=True, name=f"job:{key}").start()
    if job.status == "error":
        return {"status": "error", "message": job.error}
    result = _load(key, ttl)
    if result is not None:
        return result
    return {"status": "running", "message": job.message, "elapsed_seconds": round(time.time() - job.started)}


def peek(key: str, ttl: int):
    """The cached result if present, without starting anything."""
    return _load(key, ttl)
