"""Client for the intelligence service: AI outlooks, live quotes, and tracking new symbols."""

from outhit_common.client import ServiceClient
from research.config import get_settings

_client = ServiceClient(get_settings().intelligence_url, "intelligence")


def latest(symbols: list[str]) -> dict[str, dict]:
    """Latest outlooks (one cached call for all tracked symbols, refreshed each minute)."""
    allo = _client.get("/internal/outlooks", ttl=60, timeout=20) or {}
    return {s: allo[s] for s in symbols if s in allo}


def quote(symbol: str) -> dict | None:
    data = _client.get("/api/ai/quotes", ttl=60, timeout=20, params={"symbols": symbol}) or {}
    return data.get(symbol)


def track(symbols: list[str]) -> None:
    _client.post("/internal/track", json=symbols, timeout=120)
