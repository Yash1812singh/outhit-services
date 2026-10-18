"""Outhit API gateway: the single public entry point.

- Login / sign-up and JWT verification (the only service that sees passwords).
- Routes each request to the service that owns it, adding the trusted user id.
- Streams Server-Sent Events (live prices) straight through.
- Per-client rate limits, stricter for login and chat.
- /api/health reports every service.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from gateway import auth
from gateway.config import get_settings
from outhit_common.internal import internal_token

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gateway")
settings = get_settings()
Path("data").mkdir(exist_ok=True)
auth.init_db()
if settings.jwt_secret.startswith("dev-only"):
    log.warning("GATEWAY_JWT_SECRET is not set; using an insecure development secret.")
if not settings.google_client_id:
    log.info("GATEWAY_GOOGLE_CLIENT_ID is not set; \"Sign in with Google\" is disabled.")

SERVICES = {
    "market-data": settings.market_data_url,
    "intelligence": settings.intelligence_url,
    "research": settings.research_url,
    "advisor": settings.advisor_url,
    "paper": settings.paper_url,
    "broker": settings.broker_url,
}
# Longest prefix wins.
ROUTES = sorted([
    ("/api/advisor", "advisor"),
    ("/api/market", "market-data"), ("/api/stocks", "market-data"), ("/api/global", "market-data"),
    ("/api/predict", "market-data"), ("/api/crypto", "market-data"),
    ("/api/ai/research", "research"), ("/api/ai/picks", "research"), ("/api/ai/funds", "research"),
    ("/api/ai/allocation", "research"), ("/api/ai/regime", "research"), ("/api/ai/portfolio", "research"),
    ("/api/ai/rockets", "research"), ("/api/ai/picks-track-record", "research"),
    ("/api/ai/strategies", "research"), ("/api/ai/backtest", "research"), ("/api/ai/gurus", "research"),
    ("/api/paper", "paper"), ("/api/broker", "broker"),
    ("/api/ai", "intelligence"),
], key=lambda r: len(r[0]), reverse=True)
HOP_HEADERS = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "upgrade", "x-internal-token", "x-user-id"}

_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _client
    _client = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=5), limits=httpx.Limits(max_connections=200))
    yield
    await _client.aclose()


app = FastAPI(title="Outhit API gateway", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(auth.router)


# ---------------------------------------------------------------------------
# Rate limiting (in memory, per client and route class)
# ---------------------------------------------------------------------------

_hits: dict[str, deque] = defaultdict(deque)


def _limited(key: str, limit: int) -> bool:
    now = time.monotonic()
    q = _hits[key]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= limit:
        return True
    q.append(now)
    return False


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    path = request.url.path
    who = request.client.host if request.client else "?"
    if path.startswith(("/api/auth/login", "/api/auth/register", "/api/auth/google")):
        cls, limit = "auth", settings.rate_limit_auth
    elif path.startswith("/api/advisor/chat"):
        cls, limit = "chat", settings.rate_limit_chat
    elif path.startswith("/api/broker/connect"):
        cls, limit = "auth", settings.rate_limit_auth
    elif path.startswith("/api/broker/orders") and request.method != "GET":
        cls, limit = "orders", settings.rate_limit_orders
    elif path.startswith("/api/ai/stream"):
        return await call_next(request)  # long-lived streams aren't counted per request
    else:
        cls, limit = "default", settings.rate_limit_default
    if request.method != "OPTIONS" and _limited(f"{cls}:{who}", limit):
        return JSONResponse({"detail": "Too many requests. Please wait a minute and try again."}, status_code=429)
    return await call_next(request)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health() -> dict:
    async def check(name: str, url: str):
        try:
            r = await _client.get(url + "/health", timeout=3)
            return name, {"status": "ok" if r.status_code == 200 else "error", **(r.json() if r.status_code == 200 else {})}
        except Exception as exc:
            return name, {"status": "down", "error": str(exc)[:100]}

    results = dict(await asyncio.gather(*(check(n, u) for n, u in SERVICES.items())))
    advisor = results.get("advisor", {})
    return {
        "status": "ok" if all(r["status"] == "ok" for r in results.values()) else "degraded",
        "services": results,
        "ai_advisor": bool(advisor.get("ai_provider")),
        "ai_provider": advisor.get("ai_provider"),
    }


# ---------------------------------------------------------------------------
# Proxy
# ---------------------------------------------------------------------------

@app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy(rest: str, request: Request):
    path = "/api/" + rest
    service = next((svc for prefix, svc in ROUTES if path == prefix or path.startswith(prefix + "/")), None)
    if service is None:
        return JSONResponse({"detail": "Not found"}, status_code=404)

    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
    headers["X-Internal-Token"] = internal_token()
    uid = auth.user_id_from_request(request)
    if uid is not None:
        headers["X-User-Id"] = str(uid)
    elif request.headers.get("authorization"):
        headers.pop("authorization", None)  # invalid/expired token: forward as anonymous

    url = httpx.URL(SERVICES[service] + path, query=request.url.query.encode())
    body = await request.body()
    try:
        upstream = await _client.send(_client.build_request(request.method, url, headers=headers, content=body), stream=True)
    except httpx.HTTPError as exc:
        log.warning("%s unavailable for %s: %s", service, path, exc)
        return JSONResponse({"detail": f"The {service} service is unavailable right now. Please try again shortly."}, status_code=503)

    out_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_HEADERS and k.lower() != "content-encoding"}
    if upstream.headers.get("content-type", "").startswith("text/event-stream"):
        out_headers["Cache-Control"] = "no-cache"
        out_headers["X-Accel-Buffering"] = "no"
    return StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code, headers=out_headers,
                             media_type=upstream.headers.get("content-type"), background=BackgroundTask(upstream.aclose))
