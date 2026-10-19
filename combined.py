"""All seven services in one process, for hosts that give you a single small server
(e.g. Render's free plan: one service, 512 MB, which sleeps when idle).

    uvicorn combined:app --port $PORT

The gateway answers at the root, exactly as when it runs alone. The other services are
mounted at /_svc/<name> and only answer requests from this machine; every service's
"*_URL" setting points there, so they still talk to each other over HTTP, unchanged.
Databases go to OUTHIT_DATA_DIR (default ./data), one file per service as before.
"""
from __future__ import annotations

import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

PORT = int(os.environ.get("PORT", "8001"))
DATA_DIR = Path(os.environ.get("OUTHIT_DATA_DIR", "data")).resolve()

# service name in URLs -> (package, settings prefix, database file)
SERVICES = {
    "market-data": ("market_data", "MARKET_DATA", "market_data.db"),
    "intelligence": ("intelligence", "INTELLIGENCE", "intelligence.db"),
    "research": ("research", "RESEARCH", "research.db"),
    "advisor": ("advisor", "ADVISOR", "advisor.db"),
    "paper": ("paper", "PAPER", "paper.db"),
    "broker": ("broker", "BROKER", "broker.db"),
}
# which other services each one calls (its "<PREFIX>_<SERVICE>_URL" settings)
CALLS = {
    "GATEWAY": list(SERVICES),
    "INTELLIGENCE": ["market-data"],
    "RESEARCH": ["market-data", "intelligence"],
    "ADVISOR": ["market-data", "intelligence", "research"],
    "PAPER": ["research", "intelligence"],
    "BROKER": ["intelligence"],
}


def _configure() -> None:
    """Point every database and service URL at this process. Settings are read when the
    services are imported, so this runs first; anything already set in the environment wins."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GATEWAY_DATABASE_URL", f"sqlite:///{DATA_DIR / 'auth.db'}")
    for _, prefix, db_file in SERVICES.values():
        os.environ.setdefault(f"{prefix}_DATABASE_URL", f"sqlite:///{DATA_DIR / db_file}")
    for prefix, targets in CALLS.items():
        for name in targets:
            os.environ.setdefault(f"{prefix}_{name.replace('-', '_').upper()}_URL", f"http://127.0.0.1:{PORT}/_svc/{name}")


_configure()

import hosted_backup  # noqa: E402

hosted_backup.restore(DATA_DIR)   # before the services open their databases

import asyncio  # noqa: E402
import hmac  # noqa: E402
from importlib import import_module  # noqa: E402

from outhit_common.internal import internal_token  # noqa: E402

from starlette.applications import Starlette  # noqa: E402
from starlette.responses import PlainTextResponse  # noqa: E402
from starlette.routing import Mount, Route  # noqa: E402

log = logging.getLogger("outhit.combined")
gateway = import_module("gateway.main").app
services = {name: import_module(f"{package}.main").app for name, (package, _, _) in SERVICES.items()}


@asynccontextmanager
async def lifespan(_app):
    # Mounted apps don't get startup/shutdown events on their own: run each one's here.
    stop = asyncio.Event()
    async with AsyncExitStack() as stack:
        for sub in [*services.values(), gateway]:
            await stack.enter_async_context(sub.router.lifespan_context(sub))
        log.info("all services started; data in %s", DATA_DIR)
        saver = asyncio.create_task(hosted_backup.run_forever(DATA_DIR, stop))
        yield
        stop.set()   # one last backup before the services close
        await saver


async def healthz(_request):
    """For the host's health check: answers instantly. /api/health asks every service in turn,
    which can take seconds on a busy, small CPU and get a healthy server restarted."""
    return PlainTextResponse("ok")


app = Starlette(
    routes=[Route("/healthz", healthz),
            *(Mount(f"/_svc/{name}", app=sub) for name, sub in services.items()), Mount("/", app=gateway)],
    lifespan=lifespan,
)


class InternalOnly:
    """/_svc/* is how the services reach each other; from outside, only the gateway is open.
    Every internal call (and every request the gateway forwards) carries INTERNAL_TOKEN, so
    that's what's checked: a client IP can be spoofed through proxy headers, the token can't."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] in ("http", "websocket") and path.startswith("/_svc/") and not path.endswith("/health"):
            token = dict(scope.get("headers") or []).get(b"x-internal-token", b"").decode()
            if not hmac.compare_digest(token, internal_token()):
                await PlainTextResponse("Not found", status_code=404)(scope, receive, send)
                return
        await self.inner(scope, receive, send)


app = InternalOnly(app)
