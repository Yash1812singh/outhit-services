"""Repo-wide test setup: a shared internal token and a hard block on network access.

Service-specific settings (database URL, schedulers, API keys) live in each
service's tests/conftest.py, because every service's settings are lru_cached on
first import and must see their environment before that.
"""

import os
import socket

import pytest

os.environ["INTERNAL_TOKEN"] = "test-internal-token"

_real_connect = socket.socket.connect


class NetworkBlocked(OSError):
    pass


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Tests are offline: real connections (Yahoo, Groq, Google News, other services) fail fast."""

    def guarded(self, address, *args, **kwargs):
        if self.family == socket.AF_UNIX:
            return _real_connect(self, address, *args, **kwargs)
        raise NetworkBlocked(f"Network access is disabled in tests (tried {address!r})")

    monkeypatch.setattr(socket.socket, "connect", guarded)

    # yfinance talks through curl_cffi (libcurl), which bypasses Python sockets.
    import yfinance

    def blocked(*args, **kwargs):
        raise NetworkBlocked("yfinance is disabled in tests")

    for name in ("download", "Ticker", "Tickers", "Search", "screen"):
        if hasattr(yfinance, name):
            monkeypatch.setattr(yfinance, name, blocked)
