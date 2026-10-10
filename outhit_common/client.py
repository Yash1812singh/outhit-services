"""Small cached HTTP client for calling other Outhit services."""

import logging
import time

import httpx

from outhit_common.internal import internal_token

log = logging.getLogger(__name__)


class ServiceClient:
    def __init__(self, base_url: str, name: str):
        self.base = base_url.rstrip("/")
        self.name = name
        self._cache: dict[str, tuple[float, object]] = {}

    def get(self, path: str, ttl: int = 0, timeout: float = 60, params: dict | None = None):
        key = path + "?" + str(sorted((params or {}).items()))
        hit = self._cache.get(key)
        if ttl and hit and time.time() - hit[0] < ttl:
            return hit[1]
        try:
            r = httpx.get(self.base + path, params=params, timeout=timeout, headers={"X-Internal-Token": internal_token()})
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            log.warning("%s call %s failed: %s", self.name, path, exc)
            return hit[1] if hit else None
        if ttl:
            self._cache[key] = (time.time(), data)
        return data

    def post(self, path: str, json: dict | list | None = None, timeout: float = 60):
        try:
            r = httpx.post(self.base + path, json=json, timeout=timeout, headers={"X-Internal-Token": internal_token()})
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            log.warning("%s call %s failed: %s", self.name, path, exc)
            return None

    def forget(self, path_prefix: str) -> None:
        for k in [k for k in self._cache if k.startswith(path_prefix)]:
            self._cache.pop(k, None)
