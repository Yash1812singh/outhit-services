"""Client for the market-data service (company universe, quant model, technical signal)."""

import logging
import time

import httpx

from intelligence.config import get_settings

log = logging.getLogger(__name__)
_cache: dict[str, tuple[float, object]] = {}


def _get(path: str, ttl: int, timeout: float = 60) -> dict | list | None:
    hit = _cache.get(path)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    try:
        r = httpx.get(get_settings().market_data_url + path, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        log.warning("Backend call %s failed: %s", path, exc)
        return hit[1] if hit else None
    _cache[path] = (time.time(), data)
    return data


def top_companies(market: str, n: int) -> list[dict]:
    data = _get(f"/api/stocks/{market}/companies?sort=market_cap&desc=true&page_size={n}", ttl=3600, timeout=120)
    return data.get("rows", []) if isinstance(data, dict) else []


def company(market: str, symbol: str) -> dict | None:
    data = _get(f"/api/stocks/{market}/companies?q={symbol}&page_size=5", ttl=3600)
    rows = data.get("rows", []) if isinstance(data, dict) else []
    return next((r for r in rows if r["symbol"] == symbol), None)


def quant(market: str, horizon: int = 21) -> dict | None:
    """The ML outperformance model for a market. Returns None while it is still training."""
    data = _get(f"/api/predict/{market}?horizon={horizon}", ttl=1800, timeout=30)
    if not isinstance(data, dict) or data.get("status") != "ready":
        _cache.pop(f"/api/predict/{market}?horizon={horizon}", None)  # retry soon
        return None
    return data


def technical(symbol: str) -> dict | None:
    data = _get(f"/api/market/asset/{symbol}", ttl=900)
    return data if isinstance(data, dict) and "score" in data else None
