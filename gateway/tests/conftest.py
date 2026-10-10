import os
import shutil
import tempfile

# Must run before any gateway module is imported (settings are lru_cached).
_TMP = tempfile.mkdtemp(prefix="outhit-gateway-test-")
os.environ["GATEWAY_DATABASE_URL"] = f"sqlite:///{_TMP}/auth.db"
os.environ["GATEWAY_JWT_SECRET"] = "test-jwt-secret-that-is-long-enough-for-hs256"
os.environ["GATEWAY_RATE_LIMIT_AUTH"] = "10"
os.environ["GATEWAY_RATE_LIMIT_CHAT"] = "12"
os.environ["GATEWAY_RATE_LIMIT_DEFAULT"] = "600"

import httpx  # noqa: E402
import pytest  # noqa: E402

from gateway import main  # noqa: E402

SERVICE_URLS = {name: f"http://{name}.test" for name in main.SERVICES}


class Body(httpx.AsyncByteStream):
    """An unread response body, like a real network stream (httpx eagerly reads plain bytes
    content, and the gateway streams upstream responses with aiter_raw)."""

    def __init__(self, data: bytes):
        self.data = data

    async def __aiter__(self):
        yield self.data


def respond(status: int = 200, json=None, content: bytes = b"", content_type: str = "application/json") -> httpx.Response:
    import json as _json

    data = _json.dumps(json).encode() if json is not None else content
    return httpx.Response(status, headers={"content-type": content_type}, stream=Body(data))


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    main._hits.clear()
    yield
    main._hits.clear()


@pytest.fixture
def upstream(monkeypatch):
    """Point every service at an in-memory httpx MockTransport and record what the gateway sends."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        service = request.url.host.removesuffix(".test")
        if request.url.path == "/api/ai/stream":
            return respond(content=b"event: update\ndata: {}\n\n", content_type="text/event-stream")
        return respond(json={"service": service, "path": request.url.path, "query": request.url.query.decode()})

    for name, url in SERVICE_URLS.items():
        monkeypatch.setitem(main.SERVICES, name, url)
    monkeypatch.setattr(main, "_client", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return seen


@pytest.fixture(scope="session", autouse=True)
def _cleanup_db():
    yield
    main.auth.engine.dispose()
    shutil.rmtree(_TMP, ignore_errors=True)
